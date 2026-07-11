import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from olmo_core.nn.attention import Attention
from olmo_core.nn.moe import MoELinearRouter
from olmo_core.train.callbacks import MechanismMonitorCallback
from olmo_core.train.callbacks.mechanism_monitor import (
    PREEMPTIVE_ATTN_ANOMALY_METRIC,
    PREEMPTIVE_ATTN_ENTROPY_METRIC,
    PREEMPTIVE_MOE_ENTROPY_METRIC,
    _EntropyDropDetector,
)


class _TinyAttnModel(nn.Module):
    """Two real attention layers so hook registration has something to monitor."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Attention(d_model=32, n_heads=4, n_kv_heads=2, bias=False),
                Attention(d_model=32, n_heads=4, n_kv_heads=2, bias=False),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _FakeTrainer:
    """Minimal stand-in capturing recorded metrics without a full trainer."""

    def __init__(self, model):
        self.train_module = SimpleNamespace(model=model)
        self.global_step = 0
        self.recorded = {}

    def record_metric(self, name, value, reduce_type=None, **kwargs):
        if isinstance(value, torch.Tensor):
            value = float(value.item())
        self.recorded[name] = float(value)


def _attach(callback, model):
    callback._trainer = _FakeTrainer(model)
    callback._attach_hooks(model)
    return callback._trainer


def test_attention_qk_spectral_entropy_recorded():
    """A real forward through Attention produces a finite, normalized QK spectral entropy."""
    torch.manual_seed(0)
    model = _TinyAttnModel().eval()
    callback = MechanismMonitorCallback(enabled=True, interval=1, window_size=4)
    trainer = _attach(callback, model)

    with torch.no_grad():
        model(torch.randn(2, 16, 32))
    callback.pre_optim_step()

    assert PREEMPTIVE_ATTN_ENTROPY_METRIC in trainer.recorded
    value = trainer.recorded[PREEMPTIVE_ATTN_ENTROPY_METRIC]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_moe_routing_entropy_recorded():
    """A real router forward produces a finite, normalized routing entropy."""
    router = MoELinearRouter(d_model=16, num_experts=8, top_k=2).eval()
    callback = MechanismMonitorCallback(enabled=True, interval=1, window_size=4)
    trainer = _attach(callback, router)

    with torch.no_grad():
        router(torch.randn(3, 12, 16))
    callback.pre_optim_step()

    assert PREEMPTIVE_MOE_ENTROPY_METRIC in trainer.recorded
    value = trainer.recorded[PREEMPTIVE_MOE_ENTROPY_METRIC]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_preemptive_anomaly_on_entropy_drop():
    """The detector fires on a sharp one-sided entropy drop, not during the healthy baseline."""
    detector = _EntropyDropDetector(window_size=5, threshold_std=2.0)

    for value in [0.90, 0.92, 0.88, 0.91, 0.89]:  # healthy, high entropy
        z_score, anomaly = detector.update(value)
        assert z_score is None  # baseline window not yet full
        assert not anomaly

    # A collapse (entropy -> ~0) is a large one-sided drop.
    z_score, anomaly = detector.update(0.10)
    assert z_score is not None
    assert z_score > 2.0
    assert anomaly


def test_anomaly_metric_emitted_on_drop():
    """The wired _record path emits the anomaly metric (1.0) once a drop clears the baseline."""
    model = _TinyAttnModel().eval()
    callback = MechanismMonitorCallback(enabled=True, interval=1, window_size=5, threshold_std=2.0)
    trainer = _attach(callback, model)

    for value in [0.90, 0.92, 0.88, 0.91, 0.89]:
        callback._record(
            value,
            callback._attn_detector,
            PREEMPTIVE_ATTN_ENTROPY_METRIC,
            PREEMPTIVE_ATTN_ANOMALY_METRIC,
        )
    # Anomaly flag is only emitted once the baseline window is full.
    assert PREEMPTIVE_ATTN_ANOMALY_METRIC not in trainer.recorded

    callback._record(
        0.10,
        callback._attn_detector,
        PREEMPTIVE_ATTN_ENTROPY_METRIC,
        PREEMPTIVE_ATTN_ANOMALY_METRIC,
    )
    assert trainer.recorded[PREEMPTIVE_ATTN_ANOMALY_METRIC] == pytest.approx(1.0)


def test_state_dict_round_trip():
    """Detector baselines survive a checkpoint save/load cycle."""
    callback = MechanismMonitorCallback(window_size=5, threshold_std=2.0)
    for value in [0.90, 0.92, 0.88, 0.91, 0.89]:
        callback._attn_detector.update(value)
    callback._moe_detector.update(0.5)

    restored = MechanismMonitorCallback(window_size=5, threshold_std=2.0)
    restored.load_state_dict(callback.state_dict())

    assert list(restored._attn_detector.history) == [0.90, 0.92, 0.88, 0.91, 0.89]
    assert list(restored._moe_detector.history) == [0.5]


def test_disabled_callback_is_noop():
    """When disabled, hooks are not registered and no metrics are emitted."""
    model = _TinyAttnModel().eval()
    callback = MechanismMonitorCallback(enabled=False)
    callback._trainer = _FakeTrainer(model)

    with torch.no_grad():
        model(torch.randn(1, 8, 32))
    callback.pre_optim_step()

    assert callback._handles == []
    assert callback._trainer.recorded == {}


# ---------- Coverage-gap tests (untested_new_surface reduction) ----------


def test_gqa_indivisible_head_count_returns_nan():
    """If ``n_heads`` is not a multiple of ``n_kv_heads`` the QK spectrum can't be
    reconstructed, so the entropy path must short-circuit to NaN rather than
    silently broadcast into garbage."""
    callback = MechanismMonitorCallback(enabled=True, interval=1, window_size=4)
    # Fabricate q / k linear outputs with n_heads=3, n_kv_heads=2 (indivisible).
    batch, seq, head_dim = 1, 8, 4
    q_lin = torch.randn(batch, seq, 3 * head_dim)
    k_lin = torch.randn(batch, seq, 2 * head_dim)

    entropy = callback._qk_spectral_entropy(q_lin, k_lin, n_heads=3, n_kv_heads=2, head_dim=head_dim)

    assert math.isnan(entropy)


def test_layer_subsampling_caps_monitored_hooks():
    """When more attention layers exist than ``max_layers``, hooks are attached to
    an evenly-strided subset and the extra layers are left untouched."""

    class _DeepAttnModel(nn.Module):
        def __init__(self, n_layers: int):
            super().__init__()
            self.layers = nn.ModuleList(
                [Attention(d_model=32, n_heads=4, n_kv_heads=2, bias=False) for _ in range(n_layers)]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = _DeepAttnModel(n_layers=10).eval()
    callback = MechanismMonitorCallback(enabled=True, interval=1, max_layers=3, window_size=4)
    _attach(callback, model)

    # Two hooks per monitored layer (w_q + w_k), zero for MoE routers on an attn-only model.
    assert len(callback._handles) == 2 * 3


def test_token_subsampling_handles_long_sequences():
    """Passing more tokens than ``max_tokens`` must not blow up the SVD — the code
    strides down to ``<= max_tokens`` and still returns a finite normalized entropy."""
    torch.manual_seed(3)
    callback = MechanismMonitorCallback(enabled=True, interval=1, max_tokens=16, window_size=4)
    batch, seq, head_dim, n_heads, n_kv_heads = 1, 256, 4, 4, 4  # seq >> max_tokens
    q_lin = torch.randn(batch, seq, n_heads * head_dim)
    k_lin = torch.randn(batch, seq, n_kv_heads * head_dim)

    entropy = callback._qk_spectral_entropy(
        q_lin, k_lin, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim
    )

    assert math.isfinite(entropy)
    assert 0.0 <= entropy <= 1.0


def test_detector_zero_variance_never_flags_anomaly():
    """A perfectly flat baseline has ``std == 0``; the detector's short-circuit
    must return no anomaly no matter how far the new value has drifted."""
    detector = _EntropyDropDetector(window_size=3, threshold_std=2.0)

    # Warm the baseline with three identical readings.
    for _ in range(3):
        z_score, anomaly = detector.update(0.5)
    # A subsequent zero-variance window: ``std < 1e-10`` guard fires.
    z_score, anomaly = detector.update(0.5)
    assert z_score == 0.0
    assert not anomaly
    # Even a huge nominal drop cannot be flagged while variance is zero.
    z_score, anomaly = detector.update(0.0)
    assert z_score == 0.0
    assert not anomaly


def test_close_removes_hooks_and_is_idempotent():
    """After ``close()`` the callback has no live hooks, subsequent forward passes
    record nothing, and a second ``close()`` is a no-op rather than an error."""
    torch.manual_seed(4)
    model = _TinyAttnModel().eval()
    callback = MechanismMonitorCallback(enabled=True, interval=1, window_size=4)
    trainer = _attach(callback, model)

    # Sanity: hooks were installed and fire on a forward pass.
    with torch.no_grad():
        model(torch.randn(1, 12, 32))
    callback.pre_optim_step()
    assert PREEMPTIVE_ATTN_ENTROPY_METRIC in trainer.recorded
    assert callback._handles  # non-empty
    prev_recorded = dict(trainer.recorded)

    callback.close()
    assert callback._handles == []

    # After close, the model still runs but no new metrics are captured.
    with torch.no_grad():
        model(torch.randn(1, 12, 32))
    callback.pre_optim_step()
    assert trainer.recorded == prev_recorded

    # Idempotent: second close does not raise.
    callback.close()
    assert callback._handles == []


def test_load_state_dict_truncates_history_beyond_window_size():
    """When restoring a checkpoint whose history is longer than the current
    ``window_size``, ``deque(maxlen=…)`` keeps only the most recent entries."""
    callback = MechanismMonitorCallback(window_size=3, threshold_std=2.0)

    # Longer-history checkpoint (e.g. taken with a bigger window_size).
    callback.load_state_dict(
        {
            "attn_history": [0.10, 0.20, 0.30, 0.40, 0.50],
            "moe_history": [0.7, 0.8],
        }
    )

    # deque(maxlen=3) drops the oldest entries and keeps the most recent 3.
    assert list(callback._attn_detector.history) == [0.30, 0.40, 0.50]
    # A history shorter than window_size passes through unchanged.
    assert list(callback._moe_detector.history) == [0.7, 0.8]
