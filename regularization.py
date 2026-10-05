"""
Regularization techniques for continual pre-training to mitigate catastrophic forgetting.

Implements:
- L2-SP: L2 regularization towards base model weights (Xuhong et al., 2018)
- LayerRangeL2SP: Layer-aware L2 regularization with different strengths for first/middle/last layers

Additional utilities:
- LayerFreezer: Freeze middle layers during training (Partial SFT/CPT)
- WeightReverter: Restore middle layers post-training (Train-then-Revert)

Usage:
    from regularization import L2SPRegularizer, LayerRangeL2SPRegularizer, RegularizedTrainer
    from regularization import LayerFreezer, WeightReverter, get_num_layers
    
    # Standard L2-SP regularization
    regularizer = L2SPRegularizer(
        base_model=model,
        lambda_l2=0.01,
        exclude_patterns=["embed", "lm_head"]
    )
    
    # Layer-aware regularization (protect first/last layers more)
    num_layers = get_num_layers(model)
    regularizer = LayerRangeL2SPRegularizer(
        lambda_first=0.1,      # Strong regularization on first layers (multilingual)
        lambda_middle=0.001,   # Weak regularization on middle layers (reasoning)
        lambda_last=0.1,       # Strong regularization on last layers (multilingual)
        first_layer_end=6,     # Layers 0-5 are "first"
        last_layer_start=24,   # Layers 24+ are "last"
        num_layers=num_layers,
        attention_only=True,   # Only regularize attention, not MLP
    )
    
    # Freeze middle layers (Partial SFT/CPT)
    freezer = LayerFreezer(first_n=6, last_n=4, num_layers=num_layers)
    freezer.freeze(model)
    
    # Train-then-Revert strategy
    reverter = WeightReverter(first_n=6, last_n=4, num_layers=num_layers)
    reverter.register(model)  # Before training
    # ... train ...
    reverter.revert(model)    # After training
    
    # Use RegularizedTrainer instead of SFTTrainer
    trainer = RegularizedTrainer(
        model=model,
        regularizer=regularizer,
        ...
    )
"""

import re
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from trl import SFTTrainer


# =============================================================================
# Helper Functions
# =============================================================================

def get_num_layers(model: nn.Module, override: Optional[int] = None) -> int:
    """
    Auto-detect the number of transformer layers in a model.
    
    Args:
        model: The model to inspect
        override: If provided, use this value instead of auto-detecting
        
    Returns:
        Number of transformer layers
    """
    if override is not None:
        print(f"[get_num_layers] Using override: {override} layers")
        return override
    
    # Unwrap DDP/FSDP if needed
    unwrapped = model.module if hasattr(model, 'module') else model
    
    # Try common model structures
    # Qwen, LLaMA, Gemma, Mistral: model.model.layers
    if hasattr(unwrapped, 'model') and hasattr(unwrapped.model, 'layers'):
        num_layers = len(unwrapped.model.layers)
        print(f"[get_num_layers] Auto-detected {num_layers} layers (model.model.layers)")
        return num_layers
    
    # GPT-2, GPT-Neo: model.transformer.h
    if hasattr(unwrapped, 'transformer') and hasattr(unwrapped.transformer, 'h'):
        num_layers = len(unwrapped.transformer.h)
        print(f"[get_num_layers] Auto-detected {num_layers} layers (model.transformer.h)")
        return num_layers
    
    # BLOOM: model.transformer.h
    if hasattr(unwrapped, 'transformer') and hasattr(unwrapped.transformer, 'h'):
        num_layers = len(unwrapped.transformer.h)
        print(f"[get_num_layers] Auto-detected {num_layers} layers (model.transformer.h)")
        return num_layers
    
    # Fallback: count layers from parameter names
    layer_indices = set()
    for name in unwrapped.state_dict().keys():
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            layer_indices.add(int(match.group(1)))
    
    if layer_indices:
        num_layers = max(layer_indices) + 1
        print(f"[get_num_layers] Auto-detected {num_layers} layers (from parameter names)")
        return num_layers
    
    raise ValueError(
        "Could not auto-detect number of layers. "
        "Please specify --num_layers explicitly."
    )


def _extract_layer_idx(param_name: str) -> Optional[int]:
    """
    Extract layer index from a parameter name.
    
    Args:
        param_name: Parameter name like 'model.layers.15.mlp.down_proj.weight'
        
    Returns:
        Layer index (e.g., 15) or None if not a layer parameter
    """
    # Match patterns like 'layers.15.' or 'model.layers.15.'
    match = re.search(r'layers\.(\d+)\.', param_name)
    if match:
        return int(match.group(1))
    return None


def _is_attention_param(param_name: str) -> bool:
    """Check if parameter belongs to attention (not MLP)."""
    attention_patterns = ['self_attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    return any(p in param_name for p in attention_patterns)


def _is_mlp_param(param_name: str) -> bool:
    """Check if parameter belongs to MLP."""
    mlp_patterns = ['mlp', 'gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2']
    return any(p in param_name for p in mlp_patterns)


class Regularizer(ABC):
    """
    Abstract base class for regularization techniques.
    
    All regularizers must implement:
    - penalty(model) -> torch.Tensor: Compute the regularization penalty
    
    Optionally implement:
    - penalty_with_inputs(model, inputs, outputs) for input-dependent regularization (e.g., KL)
    """
    
    @abstractmethod
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the regularization penalty for the current model state.
        
        Args:
            model: The model being trained
            
        Returns:
            Scalar tensor representing the regularization loss
        """
        pass
    
    def penalty_with_inputs(
        self, 
        model: nn.Module, 
        inputs: Dict[str, torch.Tensor],
        student_outputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute input-dependent regularization penalty (e.g., KL distillation).
        
        Default implementation just calls penalty() for non-input-dependent regularizers.
        Override this for regularizers that need forward pass (like KL).
        
        Args:
            model: The model being trained
            inputs: The input batch (input_ids, attention_mask, etc.)
            student_outputs: Optional pre-computed student logits
            
        Returns:
            Scalar tensor representing the regularization loss
        """
        return self.penalty(model)
    
    def requires_inputs(self) -> bool:
        """
        Whether this regularizer needs inputs for penalty computation.
        Override to return True for KL-style regularizers.
        """
        return False
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this regularizer for logging."""
        pass


class L2SPRegularizer(Regularizer):
    """
    L2-SP Regularization: Penalize deviation from base model weights.
    
    Loss = CE_loss + lambda * sum((theta - theta_base)^2)
    
    Reference: "Explicit Inductive Bias for Transfer Learning with 
    Convolutional Networks" (Li et al., 2018)
    """
    
    def __init__(
        self,
        lambda_l2: float = 0.01,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
    ):
        self.lambda_l2 = lambda_l2
        self.exclude_patterns = exclude_patterns or []
        self.include_patterns = include_patterns or []
        self.base_weights: Dict[str, torch.Tensor] = {}
        # Keep the persistent copy on CPU, but cache each tensor on the model
        # device after its first use.  Moving weights inside penalty() on every
        # microstep causes a repeated CPU -> GPU transfer.
        self._device_base_weights: Dict[str, torch.Tensor] = {}
        self._registered = False
        self._matched_params: List[str] = []
    
    def get_name(self) -> str:
        """Return the name of this regularizer for logging."""
        return "l2sp"
    
    def _should_regularize(self, param_name: str) -> bool:
        """Check if a parameter should be regularized based on patterns."""
        # Skip vision components from multimodal models (safety net)
        if 'vision_tower' in param_name or 'multi_modal_projector' in param_name:
            return False
        
        if self.include_patterns:
            if not any(re.search(p, param_name) for p in self.include_patterns):
                return False
        
        if any(re.search(p, param_name) for p in self.exclude_patterns):
            return False
        
        return True
    
    def register_base_model(self, model: torch.nn.Module) -> None:
        """Store base model weights on CPU for memory efficiency."""
        self.base_weights = {}
        self._device_base_weights = {}
        self._matched_params = []
        
        unwrapped_model = model
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        
        state_dict = unwrapped_model.state_dict()
        
        for name, param in unwrapped_model.named_parameters():
            if self._should_regularize(name):
                if name in state_dict:
                    self.base_weights[name] = state_dict[name].detach().clone().cpu()
                    self._matched_params.append(name)
                else:
                    alt_name = f"model.{name}" if not name.startswith("model.") else name.replace("model.", "", 1)
                    if alt_name in state_dict:
                        self.base_weights[name] = state_dict[alt_name].detach().clone().cpu()
                        self._matched_params.append(name)
        
        self._registered = True
        print(f"[L2SPRegularizer] Registered {len(self.base_weights)} parameters for regularization")
        print(f"[L2SPRegularizer] Lambda: {self.lambda_l2}")
        if len(self.base_weights) > 0:
            print(f"[L2SPRegularizer] Sample matched params: {self._matched_params[:5]}")
        else:
            print(f"[L2SPRegularizer] WARNING: No parameters matched!")
    
    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        """Compute L2 penalty between current and base weights."""
        if not self._registered:
            raise RuntimeError("Must call register_base_model() before computing penalty")
        
        if len(self.base_weights) == 0:
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        unwrapped_model = model
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        
        penalty = None
        device = next(unwrapped_model.parameters()).device
        
        for name, param in unwrapped_model.named_parameters():
            if name in self.base_weights:
                base_param = self._device_base_weights.get(name)
                if base_param is None or base_param.device != device:
                    base_param = self.base_weights[name].to(device=device)
                    self._device_base_weights[name] = base_param
                diff = (param - base_param).pow(2).sum()
                if penalty is None:
                    penalty = diff
                else:
                    penalty = penalty + diff
        
        if penalty is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return self.lambda_l2 * penalty


class LayerRangeL2SPRegularizer(Regularizer):
    """
    Layer-aware L2-SP Regularization with different strengths for different layer ranges.
    
    This regularizer applies stronger regularization to first/last transformer layers
    (which contain multilingual capabilities) and weaker regularization to middle layers
    (which contain task-specific reasoning).
    
    Loss = CE_loss + sum_i(lambda_i * (theta_i - theta_base_i)^2)
    
    where lambda_i varies based on which layer range parameter i belongs to.
    
    Reference: Based on research showing multilingual capabilities are concentrated
    in first and last transformer layers.
    """
    
    def __init__(
        self,
        lambda_first: float = 0.1,
        lambda_middle: float = 0.001,
        lambda_last: float = 0.1,
        first_layer_end: int = 6,
        last_layer_start: int = 24,
        num_layers: int = 28,
        attention_only: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize layer-range L2-SP regularizer.
        
        Args:
            lambda_first: Regularization strength for first layers (0 to first_layer_end-1)
            lambda_middle: Regularization strength for middle layers
            lambda_last: Regularization strength for last layers (last_layer_start to num_layers-1)
            first_layer_end: First N layers (exclusive index, e.g., 6 means layers 0-5)
            last_layer_start: Start of last layers (inclusive index, e.g., 24 means layers 24+)
            num_layers: Total number of transformer layers
            attention_only: If True, only regularize attention params (exclude MLP)
            exclude_patterns: Regex patterns to exclude from regularization
            include_patterns: Regex patterns to include (if set, only these are regularized)
        """
        self.lambda_first = lambda_first
        self.lambda_middle = lambda_middle
        self.lambda_last = lambda_last
        self.first_layer_end = first_layer_end
        self.last_layer_start = last_layer_start
        self.num_layers = num_layers
        self.attention_only = attention_only
        self.exclude_patterns = exclude_patterns or []
        self.include_patterns = include_patterns or []
        
        self.base_weights: Dict[str, torch.Tensor] = {}
        # CPU is the durable copy; this cache avoids transferring the same
        # tensor from CPU to GPU on every gradient-accumulation microstep.
        self._device_base_weights: Dict[str, torch.Tensor] = {}
        self._registered = False
        
        # Track params by region for logging
        self._first_layer_params: List[str] = []
        self._middle_layer_params: List[str] = []
        self._last_layer_params: List[str] = []
        self._non_layer_params: List[str] = []  # embeddings, lm_head, etc.
    
    def get_name(self) -> str:
        """Return the name of this regularizer for logging."""
        return "layer_range_l2sp"
    
    def _get_layer_region(self, layer_idx: int) -> str:
        """Determine which region a layer belongs to."""
        if layer_idx < self.first_layer_end:
            return "first"
        elif layer_idx >= self.last_layer_start:
            return "last"
        else:
            return "middle"
    
    def _get_layer_lambda(self, param_name: str) -> float:
        """Get the regularization lambda for a parameter based on its layer."""
        layer_idx = _extract_layer_idx(param_name)
        
        if layer_idx is None:
            # Non-layer parameter (embedding, lm_head, etc.)
            # Apply middle (weak) regularization by default
            return self.lambda_first
        
        region = self._get_layer_region(layer_idx)
        if region == "first":
            return self.lambda_first
        elif region == "last":
            return self.lambda_last
        else:
            return self.lambda_middle
    
    def _should_regularize(self, param_name: str) -> bool:
        """Check if a parameter should be regularized based on patterns and attention_only."""
        # Skip vision components from multimodal models (safety net)
        if 'vision_tower' in param_name or 'multi_modal_projector' in param_name:
            return False
        
        # If attention_only, exclude MLP parameters
        if self.attention_only and _is_mlp_param(param_name):
            return False
        
        # Check include patterns
        if self.include_patterns:
            if not any(re.search(p, param_name) for p in self.include_patterns):
                return False
        
        # Check exclude patterns
        if any(re.search(p, param_name) for p in self.exclude_patterns):
            return False
        
        return True
    
    def register_base_model(self, model: torch.nn.Module) -> None:
        """Store base model weights on CPU for memory efficiency."""
        self.base_weights = {}
        self._device_base_weights = {}
        self._first_layer_params = []
        self._middle_layer_params = []
        self._last_layer_params = []
        self._non_layer_params = []
        
        unwrapped_model = model
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        
        state_dict = unwrapped_model.state_dict()
        
        for name, param in unwrapped_model.named_parameters():
            if self._should_regularize(name):
                # A zero coefficient contributes no gradient.  Do not retain
                # or repeatedly compare against those base weights.
                if self._get_layer_lambda(name) == 0.0:
                    continue

                if name in state_dict:
                    self.base_weights[name] = state_dict[name].detach().clone().cpu()
                else:
                    alt_name = f"model.{name}" if not name.startswith("model.") else name.replace("model.", "", 1)
                    if alt_name in state_dict:
                        self.base_weights[name] = state_dict[alt_name].detach().clone().cpu()
                    else:
                        continue
                
                # Categorize by layer region
                layer_idx = _extract_layer_idx(name)
                if layer_idx is None:
                    self._non_layer_params.append(name)
                else:
                    region = self._get_layer_region(layer_idx)
                    if region == "first":
                        self._first_layer_params.append(name)
                    elif region == "last":
                        self._last_layer_params.append(name)
                    else:
                        self._middle_layer_params.append(name)
        
        self._registered = True
        
        # Verbose logging
        print(f"\n{'='*60}")
        print(f"[LayerRangeL2SPRegularizer] Configuration:")
        print(f"{'='*60}")
        print(f"  Total layers: {self.num_layers}")
        print(f"  First layers: 0-{self.first_layer_end-1} (lambda={self.lambda_first})")
        print(f"  Middle layers: {self.first_layer_end}-{self.last_layer_start-1} (lambda={self.lambda_middle})")
        print(f"  Last layers: {self.last_layer_start}-{self.num_layers-1} (lambda={self.lambda_last})")
        print(f"  Attention only: {self.attention_only}")
        print(f"\n[LayerRangeL2SPRegularizer] Registered parameters:")
        print(f"  First layer params: {len(self._first_layer_params)}")
        print(f"  Middle layer params: {len(self._middle_layer_params)}")
        print(f"  Last layer params: {len(self._last_layer_params)}")
        print(f"  Non-layer params: {len(self._non_layer_params)}")
        print(f"  Total: {len(self.base_weights)}")
        
        if self._first_layer_params:
            print(f"\n  Sample first layer params: {self._first_layer_params[:3]}")
        if self._middle_layer_params:
            print(f"  Sample middle layer params: {self._middle_layer_params[:3]}")
        if self._last_layer_params:
            print(f"  Sample last layer params: {self._last_layer_params[:3]}")
        
        if len(self.base_weights) == 0:
            print(f"\n  WARNING: No parameters matched for regularization!")
        print(f"{'='*60}\n")
    
    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        """Compute layer-weighted L2 penalty between current and base weights."""
        if not self._registered:
            raise RuntimeError("Must call register_base_model() before computing penalty")
        
        if len(self.base_weights) == 0:
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        unwrapped_model = model
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        
        penalty = None
        device = next(unwrapped_model.parameters()).device
        
        for name, param in unwrapped_model.named_parameters():
            if name in self.base_weights:
                layer_lambda = self._get_layer_lambda(name)
                base_param = self._device_base_weights.get(name)
                if base_param is None or base_param.device != device:
                    base_param = self.base_weights[name].to(device=device)
                    self._device_base_weights[name] = base_param
                diff = layer_lambda * (param - base_param).pow(2).sum()
                
                if penalty is None:
                    penalty = diff
                else:
                    penalty = penalty + diff
        
        if penalty is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return penalty


class RegularizedTrainer(SFTTrainer):
    """
    SFTTrainer with regularization support.
    """
    
    def __init__(self, regularizer: Optional[Regularizer] = None, **kwargs):
        super().__init__(**kwargs)
        self.regularizer = regularizer
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with regularization penalty."""
        # Get base loss from parent
        if return_outputs:
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        else:
            loss = super().compute_loss(
                model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch
            )
            outputs = None
        
        ce_loss = loss.detach().item()
        reg_loss = 0.0
        
        # Add regularization penalty
        if self.regularizer is not None:
            if self.regularizer.requires_inputs():
                if outputs is None:
                    _, outputs = super().compute_loss(
                        model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                    )
                penalty = self.regularizer.penalty_with_inputs(model, inputs, outputs)
            else:
                penalty = self.regularizer.penalty(model)
            
            reg_loss = penalty.detach().item()
            loss = loss + penalty
        
        # Log metrics
        if self.regularizer is not None:
            reg_type = type(self.regularizer).__name__.replace("Regularizer", "").lower()
            self.log({
                "loss/ce": ce_loss,
                f"loss/{reg_type}": reg_loss,
                "loss/total": ce_loss + reg_loss,
            })
        
        if return_outputs:
            return loss, outputs
        return loss


# =============================================================================
# Layer Freezing and Weight Reversion Utilities
# =============================================================================

class LayerFreezer:
    """
    Utility to freeze middle transformer layers during training (Partial SFT/CPT).
    
    Freezing middle layers preserves reasoning/task-specific capabilities
    while allowing first/last layers to adapt for multilingual capabilities.
    
    Usage:
        freezer = LayerFreezer(first_n=6, last_n=4, num_layers=28)
        freezer.freeze(model)    # Before training
        # ... train ...
        freezer.unfreeze(model)  # Before saving (optional, for complete checkpoint)
    """
    
    def __init__(
        self,
        first_n: int = 6,
        last_n: int = 4,
        num_layers: int = 28,
        attention_only: bool = False,
    ):
        """
        Initialize layer freezer.
        
        Args:
            first_n: Number of first layers to keep trainable
            last_n: Number of last layers to keep trainable
            num_layers: Total number of transformer layers
            attention_only: If True, only freeze MLP in middle layers (keep attention trainable)
        """
        self.first_n = first_n
        self.last_n = last_n
        self.num_layers = num_layers
        self.attention_only = attention_only
        self._frozen_params: List[str] = []
        self._frozen = False
        
        # Validate
        if first_n + last_n > num_layers:
            raise ValueError(
                f"first_n ({first_n}) + last_n ({last_n}) exceeds num_layers ({num_layers})"
            )
    
    def _should_freeze(self, param_name: str) -> bool:
        """Check if a parameter should be frozen based on layer index."""
        layer_idx = _extract_layer_idx(param_name)
        
        if layer_idx is None:
            # Non-layer parameters (embeddings, lm_head) - don't freeze
            return False
        
        # Check if in middle range
        middle_start = self.first_n
        middle_end = self.num_layers - self.last_n
        
        if layer_idx >= middle_start and layer_idx < middle_end:
            # This is a middle layer
            if self.attention_only:
                # Only freeze MLP, keep attention trainable
                return _is_mlp_param(param_name)
            else:
                # Freeze all middle layer params
                return True
        
        return False
    
    def freeze(self, model: nn.Module) -> None:
        """Freeze middle layer parameters."""
        self._frozen_params = []
        
        unwrapped_model = model
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        
        frozen_count = 0
        total_frozen_elements = 0
        
        for name, param in unwrapped_model.named_parameters():
            if self._should_freeze(name):
                param.requires_grad = False
                self._frozen_params.append(name)
                frozen_count += 1
                total_frozen_elements += param.numel()
        
        self._frozen = True
        
        # Verbose logging
        middle_start = self.first_n
        middle_end = self.num_layers - self.last_n
        
        print(f"\n{'='*60}")
        print(f"[LayerFreezer] Configuration:")
        print(f"{'='*60}")
        print(f"  Total layers: {self.num_layers}")
        print(f"  Trainable (first): layers 0-{self.first_n - 1}")
        print(f"  Frozen (middle): layers {middle_start}-{middle_end - 1}")
        print(f"  Trainable (last): layers {middle_end}-{self.num_layers - 1}")
        print(f"  Attention only freeze: {self.attention_only}")
        print(f"\n[LayerFreezer] Frozen parameters:")
        print(f"  Parameter tensors: {frozen_count}")
        print(f"  Total elements: {total_frozen_elements:,}")
        print(f"  Memory saved: ~{total_frozen_elements * 2 / 1e9:.2f} GB (bf16 grads)")
        
        if self._frozen_params:
            print(f"\n  Sample frozen params: {self._frozen_params[:5]}")
        
        # Count remaining trainable
        trainable_count = sum(1 for p in unwrapped_model.parameters() if p.requires_grad)
        trainable_elements = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
        print(f"\n[LayerFreezer] Remaining trainable:")
        print(f"  Parameter tensors: {trainable_count}")
        print(f"  Total elements: {trainable_elements:,}")
        print(f"{'='*60}\n")
    
    def unfreeze(self, model: nn.Module) -> None:
        """Unfreeze all previously frozen parameters."""
        if not self._frozen:
            print("[LayerFreezer] Warning: unfreeze() called but no parameters were frozen")
            return
        
        unwrapped_model = model
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        
        unfrozen_count = 0
        for name, param in unwrapped_model.named_parameters():
            if name in self._frozen_params:
                param.requires_grad = True
                unfrozen_count += 1
        
        print(f"[LayerFreezer] Unfroze {unfrozen_count} parameters")
        self._frozen_params = []
        self._frozen = False


class WeightReverter:
    """
    Utility to store and restore middle layer weights (Train-then-Revert strategy).
    
    This implements the insight that training all parameters followed by
    reverting middle layers often outperforms freezing them from the start.
    
    Usage:
        reverter = WeightReverter(first_n=6, last_n=4, num_layers=28)
        reverter.register(model)  # Before training - stores base weights
        # ... train all parameters ...
        reverter.revert(model)    # After training - restores middle layers
    """
    
    def __init__(
        self,
        first_n: int = 6,
        last_n: int = 4,
        num_layers: int = 28,
        attention_only: bool = False,
    ):
        """
        Initialize weight reverter.
        
        Args:
            first_n: Number of first layers to keep trained
            last_n: Number of last layers to keep trained
            num_layers: Total number of transformer layers
            attention_only: If True, only revert MLP weights (keep trained attention)
        """
        self.first_n = first_n
        self.last_n = last_n
        self.num_layers = num_layers
        self.attention_only = attention_only
        self._stored_weights: Dict[str, torch.Tensor] = {}
        self._registered = False
    
    def _should_revert(self, param_name: str) -> bool:
        """Check if a parameter should be reverted based on layer index."""
        layer_idx = _extract_layer_idx(param_name)
        
        if layer_idx is None:
            # Non-layer parameters - don't revert
            return False
        
        # Check if in middle range
        middle_start = self.first_n
        middle_end = self.num_layers - self.last_n
        
        if layer_idx >= middle_start and layer_idx < middle_end:
            # This is a middle layer
            if self.attention_only:
                # Only revert MLP, keep trained attention
                return _is_mlp_param(param_name)
            else:
                # Revert all middle layer params
                return True
        
        return False
    
    def register(self, model: nn.Module) -> None:
        """Store base model weights for middle layers on CPU."""
        self._stored_weights = {}
        
        unwrapped_model = model
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        
        stored_count = 0
        total_elements = 0
        memory_bytes = 0
        
        for name, param in unwrapped_model.named_parameters():
            if self._should_revert(name):
                # Clone to CPU for memory efficiency
                self._stored_weights[name] = param.detach().clone().cpu()
                stored_count += 1
                total_elements += param.numel()
                memory_bytes += param.numel() * param.element_size()
        
        self._registered = True
        
        # Verbose logging
        middle_start = self.first_n
        middle_end = self.num_layers - self.last_n
        
        print(f"\n{'='*60}")
        print(f"[WeightReverter] Configuration:")
        print(f"{'='*60}")
        print(f"  Total layers: {self.num_layers}")
        print(f"  Keep trained (first): layers 0-{self.first_n - 1}")
        print(f"  Will revert (middle): layers {middle_start}-{middle_end - 1}")
        print(f"  Keep trained (last): layers {middle_end}-{self.num_layers - 1}")
        print(f"  Attention only: {self.attention_only}")
        print(f"\n[WeightReverter] Stored weights for reversion:")
        print(f"  Parameter tensors: {stored_count}")
        print(f"  Total elements: {total_elements:,}")
        print(f"  CPU memory used: {memory_bytes / 1e9:.2f} GB")
        
        if self._stored_weights:
            sample_params = list(self._stored_weights.keys())[:5]
            print(f"\n  Sample stored params: {sample_params}")
        print(f"{'='*60}\n")
    
    def revert(self, model: nn.Module) -> None:
        """Restore middle layer weights from stored values."""
        if not self._registered:
            raise RuntimeError("Must call register() before revert()")
        
        if not self._stored_weights:
            print("[WeightReverter] Warning: No weights stored, nothing to revert")
            return
        
        unwrapped_model = model
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        
        device = next(unwrapped_model.parameters()).device
        reverted_count = 0
        
        with torch.no_grad():
            for name, param in unwrapped_model.named_parameters():
                if name in self._stored_weights:
                    stored_weight = self._stored_weights[name].to(device)
                    param.copy_(stored_weight)
                    reverted_count += 1
        
        print(f"\n{'='*60}")
        print(f"[WeightReverter] Reversion complete:")
        print(f"{'='*60}")
        print(f"  Reverted {reverted_count} parameter tensors")
        print(f"  Middle layers ({self.first_n}-{self.num_layers - self.last_n - 1}) restored to base weights")
        print(f"  First/last layers retain trained weights")
        print(f"{'='*60}\n")
    
    def clear(self) -> None:
        """Clear stored weights to free memory."""
        self._stored_weights = {}
        self._registered = False
        print("[WeightReverter] Cleared stored weights")


def create_regularizer(
    regularization_type: str,
    base_model: torch.nn.Module,
    lambda_reg: float = 0.01,
    exclude_patterns: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None,
    # Layer-range L2-SP specific parameters
    lambda_first: Optional[float] = None,
    lambda_middle: Optional[float] = None,
    lambda_last: Optional[float] = None,
    first_layer_end: int = 6,
    last_layer_start: int = 24,
    num_layers: int = 28,
    attention_only: bool = False,
) -> Optional[Regularizer]:
    """
    Factory function to create regularizers.
    
    Args:
        regularization_type: One of "none", "l2sp", "layer_range_l2sp"
        base_model: The base model to regularize against
        lambda_reg: Regularization strength for l2sp
        exclude_patterns: Layer name patterns to exclude
        include_patterns: Layer name patterns to include
        
        # Layer-range L2-SP specific:
        lambda_first: Regularization strength for first layers (default: 0.1)
        lambda_middle: Regularization strength for middle layers (default: 0.001)
        lambda_last: Regularization strength for last layers (default: 0.1)
        first_layer_end: Index where first layers end (exclusive)
        last_layer_start: Index where last layers start (inclusive)
        num_layers: Total number of transformer layers
        attention_only: If True, only regularize attention (not MLP)
    
    Returns:
        Regularizer instance or None if type is "none"
    """
    if regularization_type == "none":
        return None
    
    elif regularization_type == "l2sp":
        regularizer = L2SPRegularizer(
            lambda_l2=lambda_reg,
            exclude_patterns=exclude_patterns,
            include_patterns=include_patterns,
        )
        regularizer.register_base_model(base_model)
        return regularizer
    
    elif regularization_type == "layer_range_l2sp":
        # Use provided lambdas or defaults
        _lambda_first = lambda_first if lambda_first is not None else 0.1
        _lambda_middle = lambda_middle if lambda_middle is not None else 0.001
        _lambda_last = lambda_last if lambda_last is not None else 0.1
        
        regularizer = LayerRangeL2SPRegularizer(
            lambda_first=_lambda_first,
            lambda_middle=_lambda_middle,
            lambda_last=_lambda_last,
            first_layer_end=first_layer_end,
            last_layer_start=last_layer_start,
            num_layers=num_layers,
            attention_only=attention_only,
            exclude_patterns=exclude_patterns,
            include_patterns=include_patterns,
        )
        regularizer.register_base_model(base_model)
        return regularizer
    
    else:
        raise ValueError(f"Unknown regularization type: {regularization_type}. Supported: 'none', 'l2sp', 'layer_range_l2sp'")
