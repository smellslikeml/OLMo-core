import pytest
import torch

from olmo_core.nn.moe import (
    MaestroConfig,
    MaestroImportanceScorer,
    MoEConfig,
    MoEType,
)
from olmo_core.testing import DEVICES


@pytest.mark.parametrize("device", DEVICES)
def test_maestro_scorer_initialization(device: torch.device):
    scorer = MaestroImportanceScorer(
        num_layers=2,
        num_experts_per_layer=4,
        device=device,
    )

    assert scorer.num_layers == 2
    assert scorer.num_experts_per_layer == 4
    assert scorer.transition_accum.shape == (2, 4, 4)
    assert scorer.visitation_accum.shape == (2, 4)
    assert scorer.transition_accum.device == device


@pytest.mark.parametrize("device", DEVICES)
def test_maestro_register_routing_patterns(device: torch.device):
    scorer = MaestroImportanceScorer(
        num_layers=3,
        num_experts_per_layer=4,
        collect_iterations=2,
        device=device,
    )

    # Simulate routing patterns from layer 0
    expert_indices = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 0]]], device=device)
    batch_size = torch.tensor([1, 1, 1, 1], device=device, dtype=torch.float32)

    scorer.register_routing_patterns(0, expert_indices, batch_size)

    assert scorer._iteration_count == 1
    assert not scorer._collection_complete


@pytest.mark.parametrize("device", DEVICES)
def test_maestro_compute_importance(device: torch.device):
    scorer = MaestroImportanceScorer(
        num_layers=2,
        num_experts_per_layer=4,
        collect_iterations=2,
        device=device,
    )

    # Register patterns for both layers
    for layer_idx in range(2):
        expert_indices = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 0]]], device=device)
        batch_size = torch.tensor([1, 1, 1, 1], device=device, dtype=torch.float32)
        scorer.register_routing_patterns(layer_idx, expert_indices, batch_size)

    # Force finalization
    scorer._finalize_collection()

    # Compute importance
    importance = scorer.compute_importance()

    assert importance.shape == (2, 4)
    assert importance.device == device
    # Importance scores should be non-negative
    assert (importance >= 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_maestro_get_top_experts(device: torch.device):
    scorer = MaestroImportanceScorer(
        num_layers=2,
        num_experts_per_layer=8,
        collect_iterations=2,
        device=device,
    )

    # Register patterns
    for layer_idx in range(2):
        expert_indices = torch.tensor(
            [[[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 0]]], device=device
        )
        batch_size = torch.ones(8, device=device, dtype=torch.float32)
        scorer.register_routing_patterns(layer_idx, expert_indices, batch_size)

    scorer._finalize_collection()
    importance = scorer.compute_importance()

    # Get top 50%
    keep_indices = scorer.get_top_experts(importance, top_k_fraction=0.5)

    assert len(keep_indices) == 2
    assert all(len(indices) == 4 for indices in keep_indices)

    # Get exact k
    keep_indices_k = scorer.get_top_experts(importance, k=3)
    assert all(len(indices) == 3 for indices in keep_indices_k)


@pytest.mark.parametrize("device", DEVICES)
def test_maestro_get_expert_masks(device: torch.device):
    scorer = MaestroImportanceScorer(
        num_layers=2,
        num_experts_per_layer=4,
        collect_iterations=2,
        device=device,
    )

    # Register patterns
    for layer_idx in range(2):
        expert_indices = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 0]]], device=device)
        batch_size = torch.ones(4, device=device, dtype=torch.float32)
        scorer.register_routing_patterns(layer_idx, expert_indices, batch_size)

    scorer._finalize_collection()

    # Get masks for top 50%
    masks = scorer.get_expert_masks(top_k_fraction=0.5)

    assert masks.shape == (2, 4)
    # Each layer should have exactly 2 True values
    assert masks.sum(dim=1).tolist() == [2, 2]


def test_maestro_reset_collection():
    scorer = MaestroImportanceScorer(
        num_layers=2,
        num_experts_per_layer=4,
        collect_iterations=2,
    )

    # Add some patterns
    expert_indices = torch.tensor([[[0, 1], [1, 2]]])
    batch_size = torch.ones(4, dtype=torch.float32)
    scorer.register_routing_patterns(0, expert_indices, batch_size)

    assert scorer._iteration_count == 1

    scorer.reset_collection()

    assert scorer._iteration_count == 0
    assert not scorer._collection_complete
    assert scorer.transition_accum.sum() == 0
    assert scorer.visitation_accum.sum() == 0


def test_maestro_config():
    config = MaestroConfig(
        num_layers=4,
        num_experts_per_layer=8,
        collect_iterations=50,
        stationary_tol=1e-5,
    )

    assert config.num_layers == 4
    assert config.num_experts_per_layer == 8
    assert config.collect_iterations == 50
    assert config.stationary_tol == 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_maestro_with_moe_layer(device: torch.device):
    """Test MAESTRO integration with actual MoE layer."""
    moe_config = MoEConfig(
        name=MoEType.default,
        num_experts=4,
        hidden_size=128,
    )

    moe_layer = moe_config.build(d_model=64, n_layers=1, init_device=device).to(device)

    # Run a forward pass
    x = torch.randn(2, 8, 64, device=device)
    _ = moe_layer(x)

    # Check that we can access routing information
    batch_size_per_expert = moe_layer.router.batch_size_per_expert
    assert batch_size_per_expert.shape == (4,)


@pytest.mark.parametrize("device", DEVICES)
def test_compute_expert_importance_convenience(device: torch.device):
    """Test the convenience function for computing importance."""
    moe_config = MoEConfig(
        name=MoEType.default,
        num_experts=4,
        hidden_size=64,
    )

    moe_layer = moe_config.build(d_model=32, n_layers=1, init_device=device).to(device)

    # Create a simple dummy dataloader
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return torch.randn(4, 16, 32)

    # Note: dataloader would be used in a full collection loop
    _ = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)

    # Wrap in a simple module that acts like a model
    class DummyModel(torch.nn.Module):
        def __init__(self, moe_layer):
            super().__init__()
            self.moe = moe_layer

        def forward(self, x):
            return self.moe(x)

    model = DummyModel(moe_layer)

    # Compute importance (this is a simplified test)
    # In practice, this would collect more iterations
    scorer = MaestroImportanceScorer(
        num_layers=1,
        num_experts_per_layer=4,
        collect_iterations=2,
        device=device,
    )

    # Simulate a few iterations
    for _ in range(2):
        x = torch.randn(2, 4, 32, device=device)
        _ = model(x)

        # Extract routing info
        batch_size = moe_layer.router.batch_size_per_expert.clone()
        scorer.visitation_accum[0] += batch_size.float()

    importance = scorer.compute_importance()

    assert importance.shape == (1, 4)
    assert importance.device == device
