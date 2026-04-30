"""
Utility functions for model loading.

Handles multimodal Gemma models by extracting only the text sub-model,
avoiding loading vision tower and multi-modal projector weights.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def load_model(model_path: str, **kwargs):
    """
    Load a model for causal LM, stripping vision components from multimodal Gemma models.
    
    For Gemma models with vision components (Gemma3ForConditionalGeneration),
    this loads only the text sub-model (Gemma3ForCausalLM) using the text_config.
    The built-in base_model_prefix="language_model" handles weight key remapping
    automatically for HuggingFace checkpoints.
    
    For checkpoints saved from training the full multimodal model, we detect
    the "language_model" prefix in saved keys and remap them manually.
    
    For all other models (Qwen, LLaMA, etc.), falls back to AutoModelForCausalLM.
    
    Args:
        model_path: HuggingFace model name or local path
        **kwargs: Additional arguments passed to from_pretrained()
            Common: dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    
    Returns:
        Model with use_cache=False set
    """
    # Normalize dtype kwarg: accept both 'dtype' and 'torch_dtype'
    if 'dtype' in kwargs and 'torch_dtype' not in kwargs:
        kwargs['torch_dtype'] = kwargs.pop('dtype')
    elif 'dtype' in kwargs and 'torch_dtype' in kwargs:
        kwargs.pop('dtype')
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    if _is_multimodal_gemma(config):
        model = _load_gemma_text_only(model_path, config, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )
    
    model.config.use_cache = False
    return model


def _is_multimodal_gemma(config) -> bool:
    """Check if config corresponds to a multimodal Gemma model."""
    return (
        getattr(config, 'model_type', None) == 'gemma3'
        and hasattr(config, 'text_config')
        and hasattr(config, 'vision_config')
    )


def _load_gemma_text_only(model_path: str, config, **kwargs):
    """
    Load only the text sub-model from a multimodal Gemma checkpoint.
    
    Handles two cases:
    1. Original HuggingFace weights: base_model_prefix="language_model" remaps automatically
    2. Saved checkpoints from training the full model: keys have "model.language_model.layers.*"
       which need manual remapping to "model.layers.*"
    """
    from transformers import Gemma3ForCausalLM
    import json
    import os
    
    text_config = config.text_config
    print(f"[load_model] Detected multimodal Gemma model (model_type=gemma3)")
    print(f"[load_model] Extracting text-only model: {text_config.num_hidden_layers} layers, "
          f"hidden_size={text_config.hidden_size}")
    
    # Check if this is a saved checkpoint with full multimodal keys
    # by inspecting the saved state dict keys
    is_full_multimodal_checkpoint = _has_multimodal_keys(model_path)
    
    if is_full_multimodal_checkpoint:
        print(f"[load_model] Detected saved multimodal checkpoint — remapping keys manually")
        model = _load_from_multimodal_checkpoint(model_path, text_config, **kwargs)
    else:
        # Standard HuggingFace weights — base_model_prefix handles remapping
        print(f"[load_model] Loading from HuggingFace weights (base_model_prefix remapping)")
        model = Gemma3ForCausalLM.from_pretrained(
            model_path, config=text_config, trust_remote_code=True, **kwargs
        )
    
    print(f"[load_model] Loaded text-only Gemma model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"[load_model] Skipped vision tower and multi-modal projector")
    
    return model


def _has_multimodal_keys(model_path: str) -> bool:
    """
    Check if a saved checkpoint contains multimodal keys (language_model prefix).
    
    Inspects safetensors or pytorch_model index files to determine key structure
    without loading the full weights.
    """
    import os
    import json
    
    # Check safetensors index
    safetensors_index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(safetensors_index):
        with open(safetensors_index) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        # If any key has "language_model" prefix, it's a full multimodal checkpoint
        for key in list(weight_map.keys())[:20]:
            if "language_model" in key:
                return True
        return False
    
    # Check pytorch_model index
    pytorch_index = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.exists(pytorch_index):
        with open(pytorch_index) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        for key in list(weight_map.keys())[:20]:
            if "language_model" in key:
                return True
        return False
    
    # Check single safetensors file
    single_safetensors = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_safetensors):
        try:
            from safetensors import safe_open
            with safe_open(single_safetensors, framework="pt") as f:
                keys = list(f.keys())[:20]
                for key in keys:
                    if "language_model" in key:
                        return True
        except ImportError:
            pass
        return False
    
    # Check single pytorch_model file
    single_pytorch = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(single_pytorch):
        # Load only metadata, not full weights
        import torch
        state_dict = torch.load(single_pytorch, map_location="cpu", weights_only=True)
        for key in list(state_dict.keys())[:20]:
            if "language_model" in key:
                del state_dict
                return True
        del state_dict
        return False
    
    # Default: assume not multimodal (could be HuggingFace hub)
    return False


def _load_from_multimodal_checkpoint(model_path: str, text_config, **kwargs):
    """
    Load text-only model from a checkpoint saved as Gemma3ForConditionalGeneration.
    
    Manually remaps keys:
        model.language_model.layers.X.* -> model.layers.X.*
        model.language_model.embed_tokens.* -> model.embed_tokens.*
        model.language_model.norm.* -> model.norm.*
        lm_head.* -> lm_head.*  (unchanged)
    
    Skips vision_tower.* and multi_modal_projector.* keys.
    """
    from transformers import Gemma3ForCausalLM
    from safetensors.torch import load_file
    import glob
    import os
    
    # Create empty model with text config
    print(f"[load_model] Initializing empty Gemma3ForCausalLM...")
    model = Gemma3ForCausalLM(config=text_config)
    
    # Load state dict from checkpoint files
    print(f"[load_model] Loading checkpoint weights from: {model_path}")
    full_state_dict = {}
    
    # Try safetensors first
    safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if safetensors_files:
        for sf in safetensors_files:
            shard = load_file(sf)
            full_state_dict.update(shard)
            del shard
    else:
        # Fall back to pytorch_model
        pytorch_files = sorted(glob.glob(os.path.join(model_path, "pytorch_model*.bin")))
        for pf in pytorch_files:
            shard = torch.load(pf, map_location="cpu", weights_only=True)
            full_state_dict.update(shard)
            del shard
    
    # Remap keys: strip "language_model." from the path
    # model.language_model.layers.X.* -> model.layers.X.*
    # model.language_model.embed_tokens.* -> model.embed_tokens.*
    # model.language_model.norm.* -> model.norm.*
    remapped_state_dict = {}
    skipped_keys = []
    remapped_count = 0
    
    for key, value in full_state_dict.items():
        # Skip vision components entirely
        if any(skip in key for skip in ['vision_tower', 'multi_modal_projector']):
            skipped_keys.append(key)
            continue
        
        new_key = key
        # Remap language_model prefix
        if '.language_model.' in key:
            new_key = key.replace('.language_model.', '.', 1)
            remapped_count += 1
        elif key.startswith('language_model.'):
            new_key = key.replace('language_model.', '', 1)
            remapped_count += 1
        
        remapped_state_dict[new_key] = value
    
    print(f"[load_model] Remapped {remapped_count} keys (language_model.* -> *)")
    print(f"[load_model] Skipped {len(skipped_keys)} vision/projector keys")
    if skipped_keys:
        print(f"[load_model] Sample skipped: {skipped_keys[:3]}")
    
    # Load into model
    missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
    
    if missing:
        print(f"[load_model] WARNING: {len(missing)} missing keys:")
        for k in missing[:5]:
            print(f"  - {k}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    
    if unexpected:
        print(f"[load_model] WARNING: {len(unexpected)} unexpected keys:")
        for k in unexpected[:5]:
            print(f"  - {k}")
        if len(unexpected) > 5:
            print(f"  ... and {len(unexpected) - 5} more")
    
    # Move to correct dtype
    torch_dtype = kwargs.get('torch_dtype', None)
    if torch_dtype:
        model = model.to(dtype=torch_dtype)
    
    # Free the full state dict
    del full_state_dict
    del remapped_state_dict
    
    return model
