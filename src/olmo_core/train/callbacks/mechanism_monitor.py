"""Mechanism-driven monitor callback for preemptive training-instability detection.

Adapted from "Mechanism-Driven Monitors for Preemptive Detection of LLM Training
Instability" (arXiv:2606.28116). That work derives internal monitors from the
functional role of each critical module and shows they go abnormal thousands of
steps before the loss diverges. This callback ports the two core mechanisms into
the trainer's callback registry, complementing the reactive
:class:`StabilityMonitorCallback` (which only fires *after* loss/grad-norm spikes):

* **Low-precision attention** -- the spectral entropy of the QK bilinear product,
  i.e. the pre-softmax attention scores ``Q K^T``. Numerical faults (e.g. FP8
  overflow) concentrate this spectrum, so its entropy drops sharply well before
  the loss collapses.
* **MoE router** -- the normalized entropy of the expert-load distribution. Router
  / large-learning-rate faults that cause routing collapse drive this toward zero.

Each signal is tracked against a rolling baseline; a one-sided drop beyond
``threshold_std`` standard deviations is reported as a preemptive anomaly metric.

Scoping notes (Mode 2 -- adapted port): the core signals are implemented at full
fidelity, but auxiliary machinery is target-native. We reuse the trainer's existing
:class:`Trainer.record_metric` path and a parameter-free rolling-window anomaly
detector (in place of the paper's standalone detection framework), compute the QK
spectrum on a token sub-sample to bound cost, and deliberately omit the paper's
fault-injection benchmark -- evaluation belongs in a downstream PR.
"""

import functools as ft
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch

from ...doc_utils import beta_feature
from ..common import ReduceType
from .callback import Callback

log = logging.getLogger(__name__)

PREEMPTIVE_ATTN_ENTROPY_METRIC = "preemptive/attn/qk_spectral_entropy"
PREEMPTIVE_ATTN_ANOMALY_METRIC = "preemptive/attn/qk_entropy_anomaly"
PREEMPTIVE_MOE_ENTROPY_METRIC = "preemptive/moe/routing_entropy"
PREEMPTIVE_MOE_ANOMALY_METRIC = "preemptive/moe/routing_entropy_anomaly"


class _EntropyDropDetector:
    """Tracks a signal whose *drop* below a rolling baseline signals instability.

    Spectral / routing entropies are high when a module is healthy and collapse as
    a fault develops, so the anomaly is one-sided: ``value`` falling more than
    ``threshold_std`` standard deviations below the rolling mean.
    """

    def __init__(self, window_size: int, threshold_std: float):
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.history: Deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> Tuple[Optional[float], bool]:
        """
        :returns: ``(z_score, is_anomaly)``. ``z_score`` is ``None`` until the
            baseline window is full; thereafter it is the one-sided drop magnitude
            ``(mean - value) / std``.
        """
        if len(self.history) < self.window_size:
            self.history.append(value)
            return None, False

        mean = sum(self.history) / len(self.history)
        variance = sum((x - mean) ** 2 for x in self.history) / len(self.history)
        std = math.sqrt(variance)
        self.history.append(value)

        if std < 1e-10:
            return 0.0, False
        z_score = (mean - value) / std  # positive on a drop
        return z_score, z_score > self.threshold_std


@beta_feature
@dataclass
class MechanismMonitorCallback(Callback):
    """
    Preemptive, mechanism-driven stability monitor.

    Attaches forward hooks to attention (``w_q`` / ``w_k``) and MoE router modules
    and, each ``interval`` steps, records the QK spectral entropy and MoE routing
    entropy along with one-sided anomaly flags. Pair with
    :class:`StabilityMonitorCallback` for both early and reactive coverage.

    Metrics recorded (only once the baseline window is full, and only on monitored
    steps):

    - ``preemptive/attn/qk_spectral_entropy``: mean normalized spectral entropy of
      ``Q K^T`` across monitored attention layers (in ``[0, 1]``; ``1`` is healthy).
    - ``preemptive/attn/qk_entropy_anomaly``: ``1.0`` when the entropy drops beyond
      ``threshold_std`` below its rolling mean, else ``0.0``.
    - ``preemptive/moe/routing_entropy``: normalized entropy of the expert-load
      distribution across monitored routers (in ``[0, 1]``; ``1`` is balanced).
    - ``preemptive/moe/routing_entropy_anomaly``: ``1.0`` on a beyond-threshold
      routing-entropy drop, else ``0.0``.
    """

    enabled: bool = True
    """Master switch. When ``False`` no hooks are registered and no metrics are emitted."""

    interval: int = 10
    """How often (in optimizer steps) to record metrics. Forward hooks still capture every step but only record on interval."""

    max_tokens: int = 128
    """Maximum number of (sub-sampled) tokens used to compute the QK spectrum, bounding SVD cost."""

    max_layers: int = 8
    """Maximum number of attention layers to monitor. Layers are evenly sub-sampled."""

    window_size: int = 128
    """Number of monitored steps used as the rolling baseline for anomaly detection."""

    threshold_std: float = 5.0
    """One-sided drop, in standard deviations below the rolling mean, that flags an anomaly."""

    _handles: List[Any] = field(default_factory=list, repr=False)
    _q_buffer: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _step_entropies: List[float] = field(default_factory=list, repr=False)
    _step_routing_entropies: List[float] = field(default_factory=list, repr=False)
    _attn_detector: _EntropyDropDetector = field(init=False, repr=False)
    _moe_detector: _EntropyDropDetector = field(init=False, repr=False)

    def __post_init__(self):
        self._attn_detector = _EntropyDropDetector(self.window_size, self.threshold_std)
        self._moe_detector = _EntropyDropDetector(self.window_size, self.threshold_std)

    def post_attach(self):
        """Validate that we are attached to a transformer trainer."""
        if not self.enabled:
            return
        from ..train_module import TransformerTrainModule

        if not isinstance(self.trainer.train_module, TransformerTrainModule):
            raise ValueError(f"{type(self).__name__} only works with the TransformerTrainModule.")

    def pre_train(self):
        """Register forward hooks on attention and MoE router modules."""
        if not self.enabled:
            return

        from ..train_module import TransformerTrainModule

        assert isinstance(self.trainer.train_module, TransformerTrainModule)
        self._reset()
        self._attach_hooks(self.trainer.train_module.model)

    def _attach_hooks(self, model: torch.nn.Module):
        """Register forward hooks on the model's attention and MoE router modules."""
        from olmo_core.nn.attention import Attention
        from olmo_core.nn.moe import MoERouter

        attn_modules = [(n, m) for n, m in model.named_modules() if isinstance(m, Attention)]
        if len(attn_modules) > self.max_layers:
            stride = len(attn_modules) / self.max_layers
            attn_modules = [attn_modules[int(i * stride)] for i in range(self.max_layers)]

        for name, attn in attn_modules:
            self._handles.append(
                attn.w_q.register_forward_hook(ft.partial(self._q_hook, layer_id=name))
            )
            self._handles.append(
                attn.w_k.register_forward_hook(
                    ft.partial(
                        self._k_hook,
                        layer_id=name,
                        n_heads=attn.n_heads,
                        n_kv_heads=attn.n_kv_heads,
                        head_dim=attn.head_dim,
                    )
                )
            )

        n_routers = 0
        for name, module in model.named_modules():
            if isinstance(module, MoERouter):
                n_routers += 1
                self._handles.append(
                    module.register_forward_hook(ft.partial(self._router_hook, router_id=name))
                )

        log.info(
            "MechanismMonitor: monitoring %d attention layer(s) and %d MoE router(s).",
            len(attn_modules),
            n_routers,
        )

    @torch._dynamo.disable()
    def _q_hook(self, module, args, output, layer_id: str):
        """Stash the query projection for the matching key hook."""
        if not self.enabled or self.step % self.interval != 0:
            return
        if isinstance(output, torch.Tensor):
            self._q_buffer[layer_id] = output.detach()

    @torch._dynamo.disable()
    @torch.no_grad()
    def _k_hook(
        self,
        module,
        args,
        output,
        layer_id: str,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        """Combine the stashed query with this key projection to score the QK spectrum."""
        del module, args
        q = self._q_buffer.pop(layer_id, None)
        if not self.enabled or self.step % self.interval != 0:
            return
        if q is None or not isinstance(output, torch.Tensor):
            return
        entropy = self._qk_spectral_entropy(q, output.detach(), n_heads, n_kv_heads, head_dim)
        self._step_entropies.append(entropy)

    @torch._dynamo.disable()
    @torch.no_grad()
    def _router_hook(self, module, args, output, router_id: str):
        """Score routing balance from the expert-load histogram returned by the router."""
        del module, args, router_id
        if not self.enabled or self.step % self.interval != 0:
            return
        # MoERouter.forward returns (weights, indices, batch_size_per_expert, aux_loss).
        load = output[2] if isinstance(output, (tuple, list)) and len(output) > 2 else None
        if not isinstance(load, torch.Tensor) or load.numel() <= 1:
            return
        load = load.float()
        total = load.sum()
        if not torch.isfinite(total) or total <= 0:
            return
        probs = (load / total).clamp_min(1e-12)
        entropy = -(probs * probs.log()).sum() / math.log(load.numel())
        self._step_routing_entropies.append(float(entropy))

    def _qk_spectral_entropy(
        self,
        q_lin: torch.Tensor,
        k_lin: torch.Tensor,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> float:
        """Normalized spectral entropy of ``Q K^T`` averaged over heads and batch."""
        batch, seq = q_lin.shape[0], q_lin.shape[1]
        q = q_lin.float().reshape(batch, seq, n_heads, head_dim)
        k = k_lin.float().reshape(batch, seq, n_kv_heads, head_dim)

        if seq > self.max_tokens:
            step = max(1, seq // self.max_tokens)
            q, k = q[:, ::step], k[:, ::step]
        m = q.shape[1]

        if n_kv_heads != n_heads:
            if n_heads % n_kv_heads != 0:
                return float("nan")
            k = k.repeat_interleave(n_heads // n_kv_heads, dim=2)  # match GQA heads

        # (batch*heads, m, m) pre-softmax attention scores.
        qh = q.permute(0, 2, 1, 3).reshape(-1, m, head_dim)
        kh = k.permute(0, 2, 1, 3).reshape(-1, m, head_dim)
        scores = torch.bmm(qh, kh.transpose(1, 2))

        singular_values = torch.linalg.svdvals(scores).clamp_min(1e-12)
        spectrum = singular_values / singular_values.sum(dim=-1, keepdim=True)
        entropy = -(spectrum * spectrum.log()).sum(dim=-1) / math.log(m)
        return float(entropy.mean())

    def pre_optim_step(self):
        """Aggregate captured signals, update detectors, and record metrics."""
        if not self.enabled:
            return

        on_interval = self.step % self.interval == 0
        if on_interval and self._step_entropies:
            value = sum(self._step_entropies) / len(self._step_entropies)
            self._record(
                value,
                self._attn_detector,
                PREEMPTIVE_ATTN_ENTROPY_METRIC,
                PREEMPTIVE_ATTN_ANOMALY_METRIC,
            )
        self._step_entropies.clear()

        if on_interval and self._step_routing_entropies:
            value = sum(self._step_routing_entropies) / len(self._step_routing_entropies)
            self._record(
                value,
                self._moe_detector,
                PREEMPTIVE_MOE_ENTROPY_METRIC,
                PREEMPTIVE_MOE_ANOMALY_METRIC,
            )
        self._step_routing_entropies.clear()

    def _record(
        self,
        value: float,
        detector: _EntropyDropDetector,
        entropy_metric: str,
        anomaly_metric: str,
    ):
        z_score, anomaly = detector.update(value)
        self.trainer.record_metric(entropy_metric, value, reduce_type=ReduceType.mean)
        if z_score is not None:
            self.trainer.record_metric(
                anomaly_metric, 1.0 if anomaly else 0.0, reduce_type=ReduceType.max
            )

    def state_dict(self) -> Dict[str, Any]:
        """Save detector baselines for checkpoint resumption."""
        return {
            "attn_history": list(self._attn_detector.history),
            "moe_history": list(self._moe_detector.history),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore detector baselines from a checkpoint."""
        self._attn_detector.history = deque(
            state_dict.get("attn_history", []), maxlen=self.window_size
        )
        self._moe_detector.history = deque(
            state_dict.get("moe_history", []), maxlen=self.window_size
        )

    def close(self):
        """Remove all registered hooks."""
        self._reset()

    def _reset(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._q_buffer.clear()
        self._step_entropies.clear()
        self._step_routing_entropies.clear()
