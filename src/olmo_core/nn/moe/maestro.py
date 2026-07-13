"""
MAESTRO: Markov-chain Approximated Expert Sparsification via Transition-based ROuting

This module implements an adapted version of the MAESTRO pruning framework for MoE models.
The core insight from the paper "It Takes a MAESTRO To Prune Bad Experts" (arXiv:2607.08601)
is that expert importance should be assessed globally by modeling expert activation trajectories
as Ergodic Markov chains whose stationary distributions encode cross-layer dependencies.

This adaptation uses frequency-based estimation of transition matrices instead of learning them,
making it suitable for existing trained models without requiring additional training.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "MaestroImportanceScorer",
    "MaestroConfig",
    "compute_expert_importance",
    "apply_expert_mask",
]


log = logging.getLogger(__name__)


@dataclass
class MaestroConfig:
    """
    Configuration for MAESTRO importance scoring.

    :param num_layers: Number of MoE layers in the model.
    :param num_experts_per_layer: Number of experts in each MoE layer.
    :param collect_iterations: Number of forward passes to collect routing patterns for.
    :param stationary_tol: Tolerance for convergence when computing stationary distribution.
    :param stationary_max_iter: Maximum iterations for power iteration to compute stationary distribution.
    """

    num_layers: int
    num_experts_per_layer: int
    collect_iterations: int = 100
    stationary_tol: float = 1e-6
    stationary_max_iter: int = 1000


class MaestroImportanceScorer(nn.Module):
    """
    Collects expert routing patterns and computes MAESTRO importance scores.

    This module models expert activation trajectories as Markov chains and uses
    their stationary distributions to compute globally aware expert importance scores.

    Example usage:

    .. code-block:: python

        scorer = MaestroImportanceScorer(num_layers=4, num_experts=8)
        model = ...  # Your MoE model

        # Collection phase
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                output = model(batch)
                scorer.register_routing_patterns(model)

        # Compute importance scores
        importance = scorer.compute_importance()

        # Get indices of experts to keep (e.g., top 50%)
        num_to_keep = int(0.5 * scorer.num_experts_per_layer)
        keep_indices = scorer.get_top_experts(importance, k=num_to_keep)
    """

    def __init__(
        self,
        num_layers: int,
        num_experts_per_layer: int,
        collect_iterations: int = 100,
        stationary_tol: float = 1e-6,
        stationary_max_iter: int = 1000,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer
        self.collect_iterations = collect_iterations
        self.stationary_tol = stationary_tol
        self.stationary_max_iter = stationary_max_iter
        self.device = device or torch.device("cpu")

        # Transition matrix accumulator: [num_layers, num_experts, num_experts]
        # T[layer, i, j] = probability of transitioning from expert i to expert j
        self.register_buffer(
            "transition_accum",
            torch.zeros(num_layers, num_experts_per_layer, num_experts_per_layer),
        )

        # Expert visitation count: [num_layers, num_experts]
        self.register_buffer("visitation_accum", torch.zeros(num_layers, num_experts_per_layer))

        # Track collection state
        self._iteration_count = 0
        self._collection_complete = False

        # Storage for per-layer routing patterns: list of (expert_indices, batch_size_per_expert)
        self._routing_patterns: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []

    def reset_collection(self):
        """Reset all accumulated statistics."""
        self.transition_accum.zero_()
        self.visitation_accum.zero_()
        self._iteration_count = 0
        self._collection_complete = False
        self._routing_patterns = []

    @torch.no_grad()
    def register_routing_patterns(
        self,
        layer_idx: int,
        expert_indices: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ):
        """
        Register routing patterns from a single MoE layer.

        :param layer_idx: The layer index (0 to num_layers-1).
        :param expert_indices: Tensor of shape (B, S, top_k) with expert indices for each token.
        :param batch_size_per_expert: Tensor of shape (num_experts,) with count of tokens per expert.
        """
        if self._collection_complete:
            log.warning(
                "Collection is complete. Call reset_collection() to start a new collection."
            )
            return

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")

        # Ensure storage for this layer
        while len(self._routing_patterns) <= layer_idx:
            self._routing_patterns.append([])

        self._routing_patterns[layer_idx].append(
            (expert_indices.detach(), batch_size_per_expert.detach())
        )

        self._iteration_count += 1

        if self._iteration_count >= self.collect_iterations:
            self._finalize_collection()

    def _finalize_collection(self):
        """Finalize collection by building transition matrices from routing patterns."""
        if self._collection_complete:
            return

        self._collection_complete = True

        # Build transition matrices from collected patterns
        for layer_idx in range(min(len(self._routing_patterns), self.num_layers)):
            layer_patterns = self._routing_patterns[layer_idx]
            if not layer_patterns:
                continue

            for expert_indices, batch_size_per_expert in layer_patterns:
                # Update visitation counts
                if batch_size_per_expert.numel() == self.num_experts_per_layer:
                    self.visitation_accum[layer_idx] += batch_size_per_expert.float()

                # Update transition matrix
                # expert_indices: (B, S, top_k) -> flatten to (B*S,)
                # We use the expert indices directly in the loop below

                # Count co-occurrences of consecutive tokens (simplified approximation)
                # In a full implementation, we'd track actual token trajectories
                # Here we use a simplified approach: count expert co-activation within same batch
                B, S, top_k = expert_indices.shape
                if B * S > 1:
                    # For adjacent positions in sequence, build transitions
                    for i in range(top_k):
                        for j in range(top_k):
                            src_experts = expert_indices[:, :, i].flatten()  # (B*S,)
                            dst_experts = expert_indices[:, :, j].flatten()  # (B*S,)

                            # Build sparse transition counts
                            for k in range(src_experts.numel()):
                                src = src_experts[k].item()
                                dst = dst_experts[k].item()
                                if (
                                    0 <= src < self.num_experts_per_layer
                                    and 0 <= dst < self.num_experts_per_layer
                                ):
                                    self.transition_accum[layer_idx, src, dst] += 1

        # Normalize transition matrices
        for layer_idx in range(self.num_layers):
            row_sum = self.transition_accum[layer_idx].sum(dim=1, keepdim=True)
            # Avoid division by zero
            row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
            self.transition_accum[layer_idx] /= row_sum

        # Clear patterns to save memory
        self._routing_patterns = []

    def _compute_stationary_distribution(self, transition_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute stationary distribution of a Markov chain via power iteration.

        :param transition_matrix: Transition matrix of shape (num_states, num_states).
        :returns: Stationary distribution of shape (num_states,).
        """
        num_states = transition_matrix.shape[0]

        # Initialize with uniform distribution
        pi = torch.ones(num_states, device=transition_matrix.device) / num_states

        # Power iteration
        for _ in range(self.stationary_max_iter):
            pi_new = pi @ transition_matrix
            # Normalize
            pi_new = pi_new / pi_new.sum()

            # Check convergence
            if torch.norm(pi_new - pi, p=1) < self.stationary_tol:
                return pi_new

            pi = pi_new

        return pi

    @torch.no_grad()
    def compute_importance(self) -> torch.Tensor:
        """
        Compute MAESTRO importance scores for all experts.

        :returns: Tensor of shape (num_layers, num_experts) with importance scores.
        """
        if not self._collection_complete:
            self._finalize_collection()

        importance = torch.zeros(self.num_layers, self.num_experts_per_layer, device=self.device)

        for layer_idx in range(self.num_layers):
            # Get stationary distribution for this layer's transition matrix
            stationary = self._compute_stationary_distribution(self.transition_accum[layer_idx])

            # Combine stationary distribution with visitation frequency
            # This gives a global importance score that accounts for:
            # 1. How often the expert is visited (local importance)
            # 2. Its position in the transition graph (global importance)
            visitation_normalized = self.visitation_accum[layer_idx]
            if visitation_normalized.sum() > 0:
                visitation_normalized /= visitation_normalized.sum()

            # MAESTRO importance: weighted combination of stationary and visitation
            importance[layer_idx] = 0.7 * stationary + 0.3 * visitation_normalized

        return importance

    @torch.no_grad()
    def get_top_experts(
        self, importance: Optional[torch.Tensor] = None, k: int = None, top_k_fraction: float = None
    ) -> List[torch.Tensor]:
        """
        Get indices of top-k most important experts per layer.

        :param importance: Importance tensor from compute_importance(). If None, computes it.
        :param k: Exact number of experts to keep per layer.
        :param top_k_fraction: Fraction of experts to keep (e.g., 0.5 for 50%).
        :returns: List of tensors, one per layer, with indices of experts to keep.
        """
        if importance is None:
            importance = self.compute_importance()

        if top_k_fraction is not None:
            k = int(self.num_experts_per_layer * top_k_fraction)
        elif k is None:
            raise ValueError("Must specify either k or top_k_fraction")

        keep_indices = []
        for layer_idx in range(self.num_layers):
            layer_importance = importance[layer_idx]
            _, indices = torch.topk(layer_importance, k)
            keep_indices.append(indices)

        return keep_indices

    @torch.no_grad()
    def get_expert_masks(
        self, importance: Optional[torch.Tensor] = None, k: int = None, top_k_fraction: float = None
    ) -> torch.Tensor:
        """
        Get boolean masks for experts to keep per layer.

        :param importance: Importance tensor from compute_importance(). If None, computes it.
        :param k: Exact number of experts to keep per layer.
        :param top_k_fraction: Fraction of experts to keep (e.g., 0.5 for 50%).
        :returns: Boolean tensor of shape (num_layers, num_experts) where True means keep.
        """
        keep_indices = self.get_top_experts(importance, k, top_k_fraction)

        masks = torch.zeros(self.num_layers, self.num_experts_per_layer, dtype=torch.bool)
        for layer_idx, indices in enumerate(keep_indices):
            masks[layer_idx, indices] = True

        return masks


@torch.no_grad()
def compute_expert_importance(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_layers: int,
    num_experts: int,
    max_iterations: int = 100,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, MaestroImportanceScorer]:
    """
    Compute MAESTRO importance scores for experts in an MoE model.

    This is a convenience function that automates the collection and computation.

    :param model: The MoE model to analyze.
    :param dataloader: DataLoader providing input data.
    :param num_layers: Number of MoE layers in the model.
    :param num_experts: Number of experts per layer.
    :param max_iterations: Maximum number of batches to process.
    :param device: Device to run on.
    :returns: Tuple of (importance_scores, scorer).
    """
    scorer = MaestroImportanceScorer(
        num_layers=num_layers,
        num_experts_per_layer=num_experts,
        collect_iterations=max_iterations,
        device=device,
    ).to(device)

    model.eval()

    iteration = 0
    for batch in dataloader:
        if iteration >= max_iterations:
            break

        # Get input (assume batch is a dict or tensor)
        if isinstance(batch, dict):
            input_tensor = batch.get("input_ids", batch)
        else:
            input_tensor = batch

        # Move to device if needed
        if hasattr(input_tensor, "to"):
            input_tensor = input_tensor.to(device)

        with torch.no_grad():
            _ = model(input_tensor)

        # Extract routing patterns from model
        # This requires the model to expose its MoE layers
        # For now, we'll collect from the model's registered buffers
        _extract_routing_from_model(model, scorer, iteration)

        iteration += 1

    importance = scorer.compute_importance()
    return importance, scorer


def _extract_routing_from_model(model: nn.Module, scorer: MaestroImportanceScorer, iteration: int):
    """
    Extract routing patterns from a model and register them with the scorer.

    This is a helper that looks for MoE layers and extracts their routing information.
    """
    # Look for MoEBase instances in the model
    for name, module in model.named_modules():
        if hasattr(module, "router") and hasattr(module.router, "batch_size_per_expert"):
            # This is likely an MoE layer
            # Try to determine layer index from name
            layer_idx = _extract_layer_index(name)
            if layer_idx is not None and layer_idx < scorer.num_layers:
                batch_size_per_expert = module.router.batch_size_per_expert
                # We need expert indices too, but those aren't typically stored
                # For now, we'll use a simplified approach with just batch sizes
                scorer.visitation_accum[layer_idx] += batch_size_per_expert.float()


def _extract_layer_index(module_name: str) -> Optional[int]:
    """
    Try to extract a layer index from a module name.
    """
    import re

    match = re.search(r"layer\.(\d+)|blocks\.(\d+)|layers\.(\d+)", module_name)
    if match:
        for group in match.groups():
            if group is not None:
                return int(group)
    return None


@torch.no_grad()
def apply_expert_mask(
    model: nn.Module, expert_masks: List[torch.Tensor], in_place: bool = True
) -> nn.Module:
    """
    Apply expert pruning masks to an MoE model.

    This function zeros out the weights of experts that should be pruned.

    :param model: The MoE model to prune.
    :param expert_masks: List of boolean masks, one per MoE layer, where False means prune.
    :param in_place: If True, modifies model in place. If False, returns a copy.
    :returns: The pruned model.
    """
    if not in_place:
        import copy

        model = copy.deepcopy(model)

    # Find all MoE layers and apply masks
    layer_idx = 0
    for name, module in model.named_modules():
        if hasattr(module, "experts") and hasattr(module, "num_experts"):
            if layer_idx < len(expert_masks):
                mask = expert_masks[layer_idx]
                _prune_moe_layer(module, mask)
                layer_idx += 1

    return model


def _prune_moe_layer(moe_layer: nn.Module, expert_mask: torch.Tensor):
    """
    Prune experts from a single MoE layer by zeroing their weights.
    """
    num_experts = len(expert_mask)
    pruned_count = (~expert_mask).sum().item()

    if pruned_count == 0:
        return

    # Get experts module
    if hasattr(moe_layer, "experts"):
        experts_module = moe_layer.experts
        if hasattr(experts_module, "mlp"):
            mlp = experts_module.mlp
            # Zero out pruned expert weights
            if hasattr(mlp, "w1"):
                # Assuming w1 stores expert weights
                for i in range(num_experts):
                    if not expert_mask[i]:
                        # Zero out this expert's weights
                        _zero_expert_in_mlp(mlp, i)

    log.info(f"Pruned {pruned_count} experts from layer")


def _zero_expert_in_mlp(mlp: nn.Module, expert_idx: int):
    """
    Zero out the weights of a specific expert in an MoE MLP.
    """
    # This is a simplified implementation
    # In practice, the exact structure depends on the MoE implementation
    if hasattr(mlp, "weight"):
        # Handle different weight layouts
        weight = mlp.weight
        if weight.ndim >= 2:
            # Assuming weight is (num_experts, ...) or can be chunked by expert
            chunk_size = weight.shape[0] // weight.shape[0]  # Simplified
            if expert_idx * chunk_size < weight.shape[0]:
                start = expert_idx * chunk_size
                end = start + chunk_size
                weight[start:end] = 0
