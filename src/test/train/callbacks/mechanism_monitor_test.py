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
