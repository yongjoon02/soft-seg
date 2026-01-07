"""
PCGrad: Gradient Surgery for Multi-Task Learning.

Based on "Gradient Surgery for Multi-Task Learning" (Yu et al., NeurIPS 2020)
https://arxiv.org/abs/2001.06782

This implementation provides gradient projection to resolve conflicts between
multiple loss components during multi-task/multi-objective optimization.
"""
import torch
from typing import List, Optional


class PCGrad:
    """
    Project Conflicting Gradients (PCGrad) optimizer wrapper.
    
    When multiple loss components have conflicting gradients (negative cosine similarity),
    PCGrad projects the conflicting gradients to remove negative interference.
    
    Usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        pcgrad = PCGrad(optimizer)
        
        # In training loop:
        losses = [loss1, loss2, loss3]  # Multiple loss components
        pcgrad.pc_backward(losses)
        optimizer.step()
    
    Args:
        optimizer: PyTorch optimizer to wrap
        reduction: How to combine gradients after projection ('mean' or 'sum')
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, reduction: str = 'mean'):
        self.optimizer = optimizer
        self.reduction = reduction
        assert reduction in ['mean', 'sum'], f"Invalid reduction: {reduction}"
    
    @staticmethod
    def _grad_to_vector(grads: List[torch.Tensor]) -> torch.Tensor:
        """Flatten list of gradient tensors into a single vector."""
        return torch.cat([g.reshape(-1) for g in grads if g is not None])
    
    @staticmethod
    def _vector_to_grad(vec: torch.Tensor, grad_shapes: List[tuple]) -> List[torch.Tensor]:
        """Reshape flattened gradient vector back to original shapes."""
        grads = []
        idx = 0
        for shape in grad_shapes:
            size = torch.prod(torch.tensor(shape)).item()
            grads.append(vec[idx:idx+size].view(shape))
            idx += size
        return grads
    
    @staticmethod
    def _project_conflicting(grad1: torch.Tensor, grad2: torch.Tensor) -> torch.Tensor:
        """
        Project grad1 onto the normal plane of grad2 if they conflict.
        
        Conflict is detected when cosine similarity < 0 (obtuse angle).
        Projection removes the component of grad1 in the direction of grad2.
        
        Args:
            grad1: First gradient vector
            grad2: Second gradient vector
        
        Returns:
            Projected grad1 (unchanged if no conflict)
        """
        # Compute cosine similarity
        dot_product = torch.dot(grad1, grad2)
        
        # If positive (aligned), no projection needed
        if dot_product >= 0:
            return grad1
        
        # Project grad1 onto normal plane of grad2
        # g1_proj = g1 - (g1 · g2 / ||g2||^2) * g2
        grad2_norm_sq = torch.dot(grad2, grad2)
        if grad2_norm_sq < 1e-12:  # Avoid division by zero
            return grad1
        
        projection = grad1 - (dot_product / grad2_norm_sq) * grad2
        return projection
    
    def pc_backward(self, losses: List[torch.Tensor]) -> None:
        """
        Compute gradients for each loss and apply gradient projection.
        
        This method:
        1. Computes gradients for each loss separately
        2. Projects conflicting gradients to remove negative interference
        3. Combines projected gradients (mean or sum)
        4. Writes combined gradients to model parameters
        
        Args:
            losses: List of scalar loss tensors
        """
        # Store parameter shapes for reconstruction
        param_shapes = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    param_shapes.append(p.shape)
        
        # Compute gradient for each loss independently
        grad_vecs = []
        for i, loss in enumerate(losses):
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True if i < len(losses) - 1 else False)
            
            # Extract gradients as flat vector
            grads = []
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        grads.append(p.grad.data.clone())
                    else:
                        # Parameter has no gradient (e.g., frozen)
                        grads.append(torch.zeros_like(p.data))
            
            grad_vec = self._grad_to_vector(grads)
            grad_vecs.append(grad_vec)
        
        # Apply pairwise gradient projection
        projected_grads = []
        for i, grad_i in enumerate(grad_vecs):
            # Start with original gradient
            g_i = grad_i.clone()
            
            # Project away conflicts with all other gradients
            for j, grad_j in enumerate(grad_vecs):
                if i != j:
                    g_i = self._project_conflicting(g_i, grad_j)
            
            projected_grads.append(g_i)
        
        # Combine projected gradients
        if self.reduction == 'mean':
            combined_grad = torch.stack(projected_grads).mean(dim=0)
        else:  # sum
            combined_grad = torch.stack(projected_grads).sum(dim=0)
        
        # Write combined gradient to parameters
        self.optimizer.zero_grad()
        grads_reshaped = self._vector_to_grad(combined_grad, param_shapes)
        
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.grad = grads_reshaped[idx].clone()
                    idx += 1
    
    def step(self, *args, **kwargs):
        """Forward step call to wrapped optimizer."""
        return self.optimizer.step(*args, **kwargs)
    
    def zero_grad(self, *args, **kwargs):
        """Forward zero_grad call to wrapped optimizer."""
        return self.optimizer.zero_grad(*args, **kwargs)
    
    @property
    def param_groups(self):
        """Forward param_groups access to wrapped optimizer."""
        return self.optimizer.param_groups


def compute_gradient_cosine_similarity(
    grad1: torch.Tensor, 
    grad2: torch.Tensor
) -> float:
    """
    Compute cosine similarity between two gradient vectors.
    
    Args:
        grad1: First gradient (flattened)
        grad2: Second gradient (flattened)
    
    Returns:
        Cosine similarity in range [-1, 1]
    """
    dot = torch.dot(grad1, grad2)
    norm1 = torch.linalg.norm(grad1)
    norm2 = torch.linalg.norm(grad2)
    
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    
    return (dot / (norm1 * norm2)).item()


def detect_gradient_conflicts(
    grads: List[torch.Tensor],
    threshold: float = 0.0
) -> dict:
    """
    Detect conflicts between multiple gradient vectors.
    
    Args:
        grads: List of flattened gradient vectors
        threshold: Cosine similarity threshold for conflict (default: 0.0)
    
    Returns:
        Dictionary with conflict statistics
    """
    n = len(grads)
    conflicts = []
    similarities = []
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_gradient_cosine_similarity(grads[i], grads[j])
            similarities.append(sim)
            
            if sim < threshold:
                conflicts.append((i, j, sim))
    
    return {
        'num_pairs': len(similarities),
        'num_conflicts': len(conflicts),
        'conflict_ratio': len(conflicts) / len(similarities) if similarities else 0.0,
        'mean_similarity': sum(similarities) / len(similarities) if similarities else 0.0,
        'conflicts': conflicts,  # List of (idx1, idx2, similarity)
    }
