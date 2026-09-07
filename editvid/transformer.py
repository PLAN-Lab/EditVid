import inspect
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, apply_rotary_emb, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
logger = logging.get_logger(__name__)

def _get_projections(attn: 'Flux2Attention', hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)
    return (query, key, value, encoder_query, encoder_key, encoder_value)

def _get_fused_projections(attn: 'Flux2Attention', hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, 'to_added_qkv'):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
    return (query, key, value, encoder_query, encoder_key, encoder_value)

def _get_qkv_projections(attn: 'Flux2Attention', hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)

def _align_kv_append_lists(append_k_list: List[torch.Tensor], append_v_list: List[torch.Tensor], identity_k: torch.Tensor, identity_v: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if len(append_k_list) == 0 and len(append_v_list) > 0:
        append_k_list = [identity_k] * len(append_v_list)
    if len(append_v_list) == 0 and len(append_k_list) > 0:
        append_v_list = [identity_v] * len(append_k_list)
    if len(append_k_list) != len(append_v_list):
        raise ValueError('Mismatched append counts for key/value frame mode application.')
    return (append_k_list, append_v_list)

def _normalize_cached_reference(seq_x: torch.Tensor, reference: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if reference is None or not torch.is_tensor(reference):
        return None
    if reference.ndim != seq_x.ndim:
        return None
    if reference.shape[0] != 1 or reference.shape[1:] != seq_x.shape[1:]:
        return None
    return reference.to(device=seq_x.device, dtype=seq_x.dtype)

def _resolve_layer_chunk_kv_cache(chunk_kv_cache_input: Optional[Dict[Any, Dict[str, torch.Tensor]]], layer_index: Optional[int]) -> Optional[Dict[str, torch.Tensor]]:
    if chunk_kv_cache_input is None or layer_index is None:
        return None
    if not isinstance(chunk_kv_cache_input, dict):
        return None
    layer_cache = chunk_kv_cache_input.get(layer_index, None)
    if layer_cache is None:
        layer_cache = chunk_kv_cache_input.get(str(layer_index), None)
    if not isinstance(layer_cache, dict):
        return None
    return layer_cache

def _get_chunk_kv_cache_references(chunk_kv_cache_input: Optional[Dict[Any, Dict[str, torch.Tensor]]], layer_index: Optional[int], latent_k: torch.Tensor, latent_v: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    layer_cache = _resolve_layer_chunk_kv_cache(chunk_kv_cache_input=chunk_kv_cache_input, layer_index=layer_index)
    if layer_cache is None:
        return (None, None, None, None)
    ith_k = _normalize_cached_reference(latent_k, layer_cache.get('ith_k'))
    last_k = _normalize_cached_reference(latent_k, layer_cache.get('last_k'))
    ith_v = _normalize_cached_reference(latent_v, layer_cache.get('ith_v'))
    last_v = _normalize_cached_reference(latent_v, layer_cache.get('last_v'))
    return (ith_k, last_k, ith_v, last_v)

def _collect_chunk_kv_cache(chunk_kv_cache_output: Optional[Dict[int, Dict[str, torch.Tensor]]], layer_index: Optional[int], latent_k: torch.Tensor, latent_v: torch.Tensor, frame_idx: int, cache_device: Union[str, torch.device]='cpu') -> None:
    if chunk_kv_cache_output is None or layer_index is None:
        return
    if latent_k.ndim != 4 or latent_v.ndim != 4:
        return
    if latent_k.shape != latent_v.shape:
        return
    num_frames = int(latent_k.shape[0])
    if num_frames <= 0:
        return
    ref_idx = int(max(0, min(frame_idx, num_frames - 1)))
    last_idx = num_frames - 1
    try:
        target_device = torch.device(cache_device) if not isinstance(cache_device, torch.device) else cache_device
    except (TypeError, RuntimeError):
        target_device = torch.device('cpu')
    chunk_kv_cache_output[layer_index] = {'ith_k': latent_k[ref_idx:ref_idx + 1].detach().to(target_device).clone(), 'ith_v': latent_v[ref_idx:ref_idx + 1].detach().to(target_device).clone(), 'last_k': latent_k[last_idx:last_idx + 1].detach().to(target_device).clone(), 'last_v': latent_v[last_idx:last_idx + 1].detach().to(target_device).clone()}

def _apply_frame_mode_single(seq_x: torch.Tensor, frame_operation: str, frame_idx: int, frame_interval: int=8, context_prev_frames: int=1, append_all_prev_context: bool=False, detach_references: bool=False, cached_ith_reference: Optional[torch.Tensor]=None, cached_last_reference: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    num_frames = seq_x.shape[0]
    ref_idx = int(max(0, min(frame_idx, num_frames - 1)))
    interval = max(1, int(frame_interval))
    prev_context = num_frames - 1 if int(context_prev_frames) <= 0 else int(context_prev_frames)
    prev_context = max(1, min(prev_context, max(1, num_frames - 1)))
    frame_indices = torch.arange(num_frames, device=seq_x.device, dtype=torch.long)
    ref_seq_x = seq_x.detach() if detach_references else seq_x
    cached_ith_reference = _normalize_cached_reference(ref_seq_x, cached_ith_reference)
    cached_last_reference = _normalize_cached_reference(ref_seq_x, cached_last_reference)

    def _select_key_frames(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        key_indices = _get_key_frame_indices(x.shape[0], interval, x.device)
        return (x.index_select(0, key_indices), key_indices)

    def _select_all_key_frames(x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        unique_key_indices = _get_unique_key_frame_indices(x.shape[0], interval, x.device)
        key_values = [x[int(key_idx.item()):int(key_idx.item()) + 1].expand_as(x) for key_idx in unique_key_indices]
        return (key_values, unique_key_indices)

    def _select_prev_shift(x: torch.Tensor, prev_offset: int) -> torch.Tensor:
        prev_indices = (frame_indices - int(prev_offset)).clamp(min=0)
        return x.index_select(0, prev_indices)

    def _mask_invalid_frames(x: torch.Tensor, valid_frames: torch.Tensor) -> torch.Tensor:
        return x * valid_frames[:, None, None, None].to(x.dtype)
    prev_shifts = []
    prev_valid_masks = []
    for prev_offset in range(1, prev_context + 1):
        shift = _select_prev_shift(ref_seq_x, prev_offset)
        valid = frame_indices >= prev_offset
        if cached_last_reference is not None:
            shift = torch.where(valid[:, None, None, None], shift, cached_last_reference.expand_as(ref_seq_x))
            valid = torch.ones_like(valid, dtype=torch.bool)
        prev_shifts.append(shift)
        prev_valid_masks.append(valid)
    prev_shifts_masked = [_mask_invalid_frames(shift, valid) for shift, valid in zip(prev_shifts, prev_valid_masks)]
    prev_sum = torch.stack(prev_shifts_masked, dim=0).sum(dim=0)
    prev_valid_count = torch.stack(prev_valid_masks, dim=0).sum(dim=0)
    prev_valid_count_clamped = prev_valid_count.clamp(min=1).to(seq_x.dtype)
    prev_mean_valid = prev_sum / prev_valid_count_clamped[:, None, None, None]
    has_prev = prev_valid_count > 0
    prev_mean_replace = torch.where(has_prev[:, None, None, None], prev_mean_valid, seq_x)
    prev_mean_append = _mask_invalid_frames(prev_mean_valid, has_prev)
    if frame_operation == 'prev':
        return (prev_mean_replace, [])
    ith_reference = cached_ith_reference if cached_ith_reference is not None else ref_seq_x[ref_idx:ref_idx + 1]
    if frame_operation == 'ith':
        return (ith_reference.expand_as(ref_seq_x), [])
    if frame_operation == 'key_frames':
        key_values, _ = _select_key_frames(ref_seq_x)
        return (key_values, [])
    if frame_operation == 'append':
        ith_append = ith_reference.expand_as(ref_seq_x)
        ith_valid_frames = torch.ones_like(frame_indices, dtype=torch.bool) if cached_ith_reference is not None else frame_indices != ref_idx
        return (seq_x, [_mask_invalid_frames(ith_append, ith_valid_frames)])
    if frame_operation == 'key_frame_append':
        key_values, key_indices = _select_key_frames(ref_seq_x)
        return (seq_x, [_mask_invalid_frames(key_values, frame_indices != key_indices)])
    if frame_operation == 'key_frame_all_append':
        key_values_list, unique_key_indices = _select_all_key_frames(ref_seq_x)
        return (seq_x, [_mask_invalid_frames(key_values, frame_indices != int(key_idx.item())) for key_values, key_idx in zip(key_values_list, unique_key_indices)])
    if frame_operation == 'prev_append':
        if append_all_prev_context:
            return (seq_x, prev_shifts_masked)
        return (seq_x, [prev_mean_append])
    if frame_operation == 'key_frame_prev_append':
        key_values, key_indices = _select_key_frames(ref_seq_x)
        key_append = _mask_invalid_frames(key_values, frame_indices != key_indices)
        if append_all_prev_context:
            return (seq_x, [key_append] + prev_shifts_masked)
        return (seq_x, [key_append, prev_mean_append])
    if frame_operation == 'key_frame_all_append_prev_append':
        key_values_list, unique_key_indices = _select_all_key_frames(ref_seq_x)
        key_appends = [_mask_invalid_frames(key_values, frame_indices != int(key_idx.item())) for key_values, key_idx in zip(key_values_list, unique_key_indices)]
        if append_all_prev_context:
            return (seq_x, key_appends + prev_shifts_masked)
        return (seq_x, key_appends + [prev_mean_append])
    if frame_operation == 'ith_prev_append':
        ith_valid_frames = torch.ones_like(frame_indices, dtype=torch.bool) if cached_ith_reference is not None else frame_indices != ref_idx
        ith_append = _mask_invalid_frames(ith_reference.expand_as(ref_seq_x), ith_valid_frames)
        if append_all_prev_context:
            return (seq_x, [ith_append] + prev_shifts_masked)
        return (seq_x, [ith_append, prev_mean_append])
    raise NotImplementedError("Only 'prev', 'ith', 'key_frames', 'append', 'key_frame_append', 'key_frame_all_append', 'prev_append', 'key_frame_prev_append', 'key_frame_all_append_prev_append', and 'ith_prev_append' frame_operation values are implemented.")

def _get_key_frame_indices(num_frames: int, frame_interval: int, device: torch.device) -> torch.Tensor:
    interval = max(1, int(frame_interval))
    frame_indices = torch.arange(num_frames, device=device, dtype=torch.long)
    chunk_starts = frame_indices // interval * interval
    center_offset = max(interval // 2 - 1, 0)
    max_valid_offset = (num_frames - chunk_starts - 1).clamp(min=0)
    key_offsets = torch.minimum(torch.full_like(chunk_starts, center_offset), max_valid_offset)
    return chunk_starts + key_offsets

def _get_unique_key_frame_indices(num_frames: int, frame_interval: int, device: torch.device) -> torch.Tensor:
    key_indices = _get_key_frame_indices(num_frames=num_frames, frame_interval=frame_interval, device=device)
    if key_indices.numel() == 0:
        return key_indices
    return torch.unique(key_indices, sorted=True)

def _get_key_frame_all_reference_names(num_frames: int, frame_interval: int, device: torch.device) -> List[str]:
    key_indices = _get_unique_key_frame_indices(num_frames=num_frames, frame_interval=frame_interval, device=device)
    return [f'key_frame_all_{int(key_idx.item())}' for key_idx in key_indices]

def _get_reference_frame_indices(num_frames: int, frame_operation: str, frame_idx: int, frame_interval: int, device: torch.device, context_prev_frames: int=1) -> Dict[str, torch.Tensor]:
    references: Dict[str, torch.Tensor] = {}
    prev_context = num_frames - 1 if int(context_prev_frames) <= 0 else int(context_prev_frames)
    prev_context = max(1, min(prev_context, max(1, num_frames - 1)))
    frame_indices = torch.arange(num_frames, device=device, dtype=torch.long)
    ith_idx = int(max(0, min(frame_idx, num_frames - 1)))
    ith_indices = torch.full((num_frames,), ith_idx, device=device, dtype=torch.long)
    key_indices = _get_key_frame_indices(num_frames, frame_interval, device)
    if frame_operation in {'prev', 'prev_append', 'ith_prev_append', 'key_frame_prev_append', 'key_frame_all_append_prev_append'}:
        for prev_offset in range(1, prev_context + 1):
            prev_indices = (frame_indices - prev_offset).clamp(min=0)
            references[f'prev_{prev_offset}'] = prev_indices
    if frame_operation in {'ith', 'append', 'ith_prev_append'}:
        references['ith'] = ith_indices
    if frame_operation in {'key_frames', 'key_frame_append', 'key_frame_prev_append'}:
        references['key_frames'] = key_indices
    if frame_operation in {'key_frame_all_append', 'key_frame_all_append_prev_append'}:
        key_frame_all_names = _get_key_frame_all_reference_names(num_frames=num_frames, frame_interval=frame_interval, device=device)
        for ref_name in key_frame_all_names:
            key_frame_idx = int(ref_name.split('_')[-1])
            references[ref_name] = torch.full((num_frames,), key_frame_idx, device=device, dtype=torch.long)
    return references

def _get_append_reference_names(frame_operation: str, references: Dict[str, torch.Tensor], append_all_prev_context: bool) -> List[str]:
    key_frame_all_ref_names = [ref_name for ref_name in references.keys() if str(ref_name).startswith('key_frame_all_')]

    def _prev_key(name: str) -> int:
        try:
            return int(str(name).split('_')[-1])
        except (TypeError, ValueError):
            return 0
    prev_ref_names = sorted([ref_name for ref_name in references.keys() if str(ref_name).startswith('prev_')], key=_prev_key)
    append_prev_ref_names = prev_ref_names if append_all_prev_context else prev_ref_names[:1]
    append_maps = {'append': ['ith'], 'key_frame_append': ['key_frames'], 'key_frame_all_append': key_frame_all_ref_names, 'prev_append': append_prev_ref_names, 'ith_prev_append': ['ith'] + append_prev_ref_names, 'key_frame_prev_append': ['key_frames'] + append_prev_ref_names, 'key_frame_all_append_prev_append': key_frame_all_ref_names + append_prev_ref_names}
    return append_maps.get(frame_operation, [])

def _format_quantiles(x: torch.Tensor) -> str:
    if x.numel() == 0:
        return '{}'
    quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0], device=x.device)
    values = torch.quantile(x, quantiles).detach().cpu().tolist()
    labels = ['0', '25', '50', '75', '90', '99', '100']
    parts = [f'{label}:{value:.4f}' for label, value in zip(labels, values)]
    return '{' + ', '.join(parts) + '}'

def _detach_post_attn_cyclic_correspondence(correspondence: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().to('cpu') if torch.is_tensor(value) else value for key, value in correspondence.items()}

def _move_post_attn_cyclic_correspondence(correspondence: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device=device) if torch.is_tensor(value) else value for key, value in correspondence.items()}

def _dropout_cyclic_valid_positions(valid_positions: torch.Tensor, dropout_p: float) -> torch.Tensor:
    num_valid = int(valid_positions.numel())
    if num_valid == 0 or dropout_p <= 0.0:
        return valid_positions[:0]
    if dropout_p >= 1.0:
        return valid_positions
    keep_count = int(round(dropout_p * num_valid))
    keep_count = max(1, min(keep_count, num_valid))
    keep_order = torch.randperm(num_valid, device=valid_positions.device)[:keep_count]
    return valid_positions.index_select(0, keep_order)

def _compute_post_attn_pair_cyclic_correspondence(reference_hidden_states: torch.Tensor, current_hidden_states: torch.Tensor, latent_hw: Tuple[int, int], tau: float, delta: float, dropout_p: float) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = reference_hidden_states.shape[0]
    latent_h, latent_w = (int(latent_hw[0]), int(latent_hw[1]))
    if seq_len != latent_h * latent_w:
        raise ValueError(f'Expected seq_len={seq_len} to match latent_hw product {latent_h * latent_w}.')
    tau_value = float(tau)
    delta_value = float(delta)
    dropout_value = float(dropout_p)
    if dropout_value < 0.0 or dropout_value > 1.0:
        raise ValueError(f'`cyclic_corr_dropout_p` must be in [0, 1], got {dropout_p}.')
    if delta_value < 0.0:
        raise ValueError(f'`cyclic_corr_delta` must be >= 0, got {delta}.')
    reference_features = F.normalize(reference_hidden_states.float().reshape(seq_len, -1), dim=-1)
    current_features = F.normalize(current_hidden_states.float().reshape(seq_len, -1), dim=-1)
    similarity = torch.matmul(reference_features, current_features.transpose(0, 1))
    max_scores, argmax_positions = similarity.max(dim=-1)
    reverse_argmax = similarity.max(dim=0).indices.to(torch.long)
    forward_valid = max_scores >= tau_value
    argmax_positions = argmax_positions.to(torch.long)
    gather_positions = argmax_positions.clamp(min=0, max=max(seq_len - 1, 0))
    cyclic_indices = reverse_argmax.gather(0, gather_positions)
    token_positions = torch.arange(seq_len, device=reference_hidden_states.device, dtype=torch.long)
    token_coords = torch.stack([torch.div(token_positions, latent_w, rounding_mode='floor'), token_positions % latent_w], dim=-1).to(torch.float32)
    cyclic_coords = token_coords[cyclic_indices]
    cyclic_valid = ((token_coords - cyclic_coords).pow(2).sum(dim=-1) < delta_value * delta_value) & forward_valid
    dropped_valid = torch.zeros_like(cyclic_valid)
    keep_positions = _dropout_cyclic_valid_positions(torch.nonzero(cyclic_valid, as_tuple=False).flatten(), dropout_value)
    if keep_positions.numel() > 0:
        dropped_valid[keep_positions] = True
    masked_argmax_positions = torch.where(dropped_valid, argmax_positions, torch.full_like(argmax_positions, -1))
    return (masked_argmax_positions, dropped_valid)

def _compute_post_attn_global_anchor_cyclic_correspondence(latent_hidden_states: torch.Tensor, latent_hw: Tuple[int, int], tau: float, delta: float, dropout_p: float, cached_anchor_features: Optional[torch.Tensor]=None) -> Optional[Dict[str, torch.Tensor]]:
    if latent_hidden_states.ndim != 4:
        return None
    num_frames, seq_len = latent_hidden_states.shape[:2]
    if seq_len == 0:
        return None
    if cached_anchor_features is None and num_frames <= 1:
        return None
    if cached_anchor_features is not None:
        anchor_features = cached_anchor_features.to(device=latent_hidden_states.device, dtype=latent_hidden_states.dtype)
        if anchor_features.ndim != 3 or anchor_features.shape[0] != seq_len:
            raise ValueError(f'`cached_anchor_features` must have shape [seq_len, num_heads, head_dim] matching the current latent hidden states; got {tuple(anchor_features.shape)} vs seq_len={seq_len}.')
        target_frame_indices = list(range(num_frames))
    else:
        anchor_features = latent_hidden_states[0]
        target_frame_indices = list(range(1, num_frames))
    reference_frame_indices = torch.full((num_frames,), -1, device=latent_hidden_states.device, dtype=torch.long)
    argmax_positions = torch.full((num_frames, seq_len), -1, device=latent_hidden_states.device, dtype=torch.long)
    valid_mask = torch.zeros((num_frames, seq_len), device=latent_hidden_states.device, dtype=torch.bool)
    for frame_idx in target_frame_indices:
        frame_argmax, frame_valid = _compute_post_attn_pair_cyclic_correspondence(reference_hidden_states=anchor_features, current_hidden_states=latent_hidden_states[frame_idx], latent_hw=latent_hw, tau=tau, delta=delta, dropout_p=dropout_p)
        reference_frame_indices[frame_idx] = -1 if cached_anchor_features is not None else 0
        argmax_positions[frame_idx] = frame_argmax
        valid_mask[frame_idx] = frame_valid
    return {'reference_frame_indices': reference_frame_indices, 'argmax_positions': argmax_positions, 'valid_mask': valid_mask, 'uses_cached_anchor': bool(cached_anchor_features is not None)}

def _apply_post_attn_global_anchor_cyclic_token_replacement(latent_hidden_states: torch.Tensor, latent_hw: Optional[Tuple[int, int]]=None, tau: float=0.0, delta: float=0.0, dropout_p: float=0.0, cached_anchor_features: Optional[torch.Tensor]=None, correspondence: Optional[Dict[str, torch.Tensor]]=None) -> torch.Tensor:
    if latent_hidden_states.ndim != 4:
        return latent_hidden_states
    num_frames, seq_len = latent_hidden_states.shape[:2]
    if seq_len == 0:
        return latent_hidden_states
    if cached_anchor_features is None and num_frames <= 1:
        return latent_hidden_states
    if cached_anchor_features is not None:
        anchor_features = cached_anchor_features.to(device=latent_hidden_states.device, dtype=latent_hidden_states.dtype)
        if anchor_features.ndim != 3 or anchor_features.shape[0] != seq_len:
            raise ValueError(f'`cached_anchor_features` must have shape [seq_len, num_heads, head_dim] matching the current latent hidden states; got {tuple(anchor_features.shape)} vs seq_len={seq_len}.')
        target_frame_indices = list(range(num_frames))
    else:
        anchor_features = latent_hidden_states[0]
        target_frame_indices = list(range(1, num_frames))
    updated_hidden_states = latent_hidden_states.clone()
    for frame_idx in target_frame_indices:
        if correspondence is not None:
            corr = _move_post_attn_cyclic_correspondence(correspondence, latent_hidden_states.device)
            valid_mask_row = corr['valid_mask'][frame_idx].to(torch.bool)
            argmax_row = corr['argmax_positions'][frame_idx].to(torch.long)
        else:
            argmax_row, valid_mask_row = _compute_post_attn_pair_cyclic_correspondence(reference_hidden_states=anchor_features, current_hidden_states=updated_hidden_states[frame_idx], latent_hw=latent_hw, tau=tau, delta=delta, dropout_p=dropout_p)
        reference_positions = torch.nonzero(valid_mask_row, as_tuple=False).flatten()
        if reference_positions.numel() == 0:
            continue
        target_positions = argmax_row.index_select(0, reference_positions)
        target_positions = target_positions.clamp(min=0, max=max(seq_len - 1, 0))
        current_hidden_states = updated_hidden_states[frame_idx].clone()
        current_hidden_states[target_positions] = anchor_features.index_select(0, reference_positions)
        updated_hidden_states[frame_idx] = current_hidden_states
    return updated_hidden_states

def _maybe_collect_block_cyclic_correspondence(*, hidden_states, layer_key, joint_attention_kwargs, latent_start):
    if not joint_attention_kwargs or not joint_attention_kwargs.get('cyclic_corr_collect_maps', False):
        return
    maps_output = joint_attention_kwargs.get('cyclic_corr_maps_output')
    if maps_output is None:
        return
    maps_output.setdefault('attempted_layers', set()).add(int(layer_key))
    extraction_layer = joint_attention_kwargs.get('cyclic_corr_extraction_layer_index')
    if extraction_layer is not None and int(layer_key) != int(extraction_layer):
        return
    latent_hw = joint_attention_kwargs.get('latent_hw')
    if latent_hw is None:
        return
    latent_seq_len = int(latent_hw[0]) * int(latent_hw[1])
    latent_end = min(int(latent_start) + latent_seq_len, hidden_states.shape[1])
    if latent_end <= int(latent_start):
        return
    latent_hidden_states = hidden_states[:, int(latent_start):latent_end].unsqueeze(2)
    anchor_features_in = joint_attention_kwargs.get('cyclic_corr_inversion_anchor_features_in')
    cached_anchor = None if anchor_features_in is None else anchor_features_in.get(int(layer_key))
    anchor_collect = joint_attention_kwargs.get('cyclic_corr_inversion_anchor_collect')
    if anchor_collect is not None and int(layer_key) not in anchor_collect and (latent_hidden_states.shape[0] > 0):
        anchor_collect[int(layer_key)] = latent_hidden_states[0].detach().clone()
    correspondence = _compute_post_attn_global_anchor_cyclic_correspondence(latent_hidden_states=latent_hidden_states, latent_hw=latent_hw, tau=joint_attention_kwargs.get('cyclic_corr_tau', 0.4), delta=joint_attention_kwargs.get('cyclic_corr_delta', 1.5), dropout_p=joint_attention_kwargs.get('cyclic_corr_dropout_p', 0.5), cached_anchor_features=cached_anchor)
    if correspondence is not None:
        maps_output.setdefault('seen_layers', set()).add(int(layer_key))
        maps_output.setdefault('layers', {}).setdefault(int(layer_key), _detach_post_attn_cyclic_correspondence(correspondence))

def _select_rotary_embedding(image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]], absolute_positions: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if image_rotary_emb is None:
        return image_rotary_emb
    cos, sin = image_rotary_emb
    selected_positions = absolute_positions.to(cos.device).to(torch.long)
    if selected_positions.numel() == 0:
        return image_rotary_emb
    selected_positions = selected_positions.clamp(min=0, max=max(cos.shape[0] - 1, 0))
    return (cos.index_select(0, selected_positions), sin.index_select(0, selected_positions))

class Flux2SwiGLU(nn.Module):
    """
    Flux 2 uses a SwiGLU-style activation in the transformer feedforward sub-blocks, but with the linear projection
    layer fused into the first linear layer of the FF sub-block. Thus, this module has no trainable parameters.
    """

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        x = self.gate_fn(x1) * x2
        return x

class Flux2FeedForward(nn.Module):

    def __init__(self, dim: int, dim_out: Optional[int]=None, mult: float=3.0, inner_dim: Optional[int]=None, bias: bool=False):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x

class Flux2AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, 'scaled_dot_product_attention'):
            raise ImportError(f'{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.')

    def __call__(self, attn: 'Flux2Attention', hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor=None, attention_mask: Optional[torch.Tensor]=None, image_rotary_emb: Optional[torch.Tensor]=None, latent_hw: Optional[Tuple[int, int]]=None, video_hw: Optional[Tuple[int, int]]=None, text_seq_len: Optional[int]=None, frame_operation: Optional[str]=None, stable_layers: Optional[List[int]]=None, frame_operation_layers: Optional[List[int]]=None, post_attn_layers: Optional[List[int]]=None, frame_idx: int=0, frame_interval: int=8, context_prev_frames: int=1, append_all_prev_context: bool=False, chunk_kv_cache_input: Optional[Dict[Any, Dict[str, torch.Tensor]]]=None, chunk_kv_cache_output: Optional[Dict[int, Dict[str, torch.Tensor]]]=None, collect_chunk_kv_cache: bool=False, chunk_kv_cache_device: str='cpu', cyclic_corr_token_replacement: bool=False, cyclic_corr_tau: float=0.8, cyclic_corr_delta: float=1.0, cyclic_corr_dropout_p: float=1.0, cyclic_corr_anchor_frame_num: int=0, cyclic_corr_maps_input: Optional[Dict[Any, Any]]=None, cyclic_corr_maps_output: Optional[Dict[Any, Any]]=None, cyclic_corr_collect_maps: bool=False, cyclic_corr_extraction_layer_index: Optional[int]=None, cyclic_corr_apply_replacement: bool=True, cyclic_corr_global_anchor_features_in: Optional[Dict[Any, torch.Tensor]]=None, cyclic_corr_global_anchor_collect: Optional[Dict[Any, torch.Tensor]]=None, cyclic_corr_inversion_anchor_features_in: Optional[Dict[Any, torch.Tensor]]=None, cyclic_corr_inversion_anchor_collect: Optional[Dict[Any, torch.Tensor]]=None) -> torch.Tensor:
        latent_seq_len = None
        video_seq_len = None
        effective_frame_layers = frame_operation_layers if frame_operation_layers is not None else stable_layers
        effective_post_attn_layers = post_attn_layers if post_attn_layers is not None else stable_layers
        if effective_frame_layers is not None and attn.layer_index is not None and (attn.layer_index not in effective_frame_layers):
            frame_operation = None
        if effective_post_attn_layers is not None and attn.layer_index is not None and (attn.layer_index not in effective_post_attn_layers):
            cyclic_corr_token_replacement = False
        if latent_hw is not None:
            latent_seq_len = int(latent_hw[0]) * int(latent_hw[1])
        if video_hw is not None:
            video_seq_len = int(video_hw[0]) * int(video_hw[1])
        if text_seq_len is None and encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
        original_query_seq_len = query.shape[1]
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        if attn.added_kv_proj_dim is not None and frame_operation is not None:
            if latent_seq_len is not None:
                start = text_seq_len or 0
                end = start + latent_seq_len
                end = min(end, query.shape[1], key.shape[1], value.shape[1])
                latent_q = query[:, start:end]
                latent_k = key[:, start:end]
                latent_v = value[:, start:end]
                cached_ith_k, cached_last_k, cached_ith_v, cached_last_v = _get_chunk_kv_cache_references(chunk_kv_cache_input=chunk_kv_cache_input, layer_index=attn.layer_index, latent_k=latent_k, latent_v=latent_v)
                latent_base_q, latent_q_appends = (latent_q, [])
                latent_base_k, latent_k_appends = _apply_frame_mode_single(latent_k, frame_operation, frame_idx, frame_interval, context_prev_frames=context_prev_frames, append_all_prev_context=append_all_prev_context, cached_ith_reference=cached_ith_k, cached_last_reference=cached_last_k)
                latent_base_v, latent_v_appends = _apply_frame_mode_single(latent_v, frame_operation, frame_idx, frame_interval, context_prev_frames=context_prev_frames, append_all_prev_context=append_all_prev_context, cached_ith_reference=cached_ith_v, cached_last_reference=cached_last_v)
                latent_k_appends, latent_v_appends = _align_kv_append_lists(latent_k_appends, latent_v_appends, identity_k=latent_k, identity_v=latent_v)
                prefix_q = query[:, :start]
                prefix_k = key[:, :start]
                prefix_v = value[:, :start]
                text_q_appends: List[torch.Tensor] = []
                text_k_appends: List[torch.Tensor] = []
                text_v_appends: List[torch.Tensor] = []
                text_k_appends, text_v_appends = _align_kv_append_lists(text_k_appends, text_v_appends, identity_k=prefix_k, identity_v=prefix_v)
                suffix_q = query[:, end:]
                suffix_k = key[:, end:]
                suffix_v = value[:, end:]
                query = torch.cat([prefix_q, latent_base_q, suffix_q], dim=1)
                key = torch.cat([prefix_k, latent_base_k, suffix_k], dim=1)
                value = torch.cat([prefix_v, latent_base_v, suffix_v], dim=1)
                all_q_appends = text_q_appends + latent_q_appends
                all_k_appends = text_k_appends + latent_k_appends
                all_v_appends = text_v_appends + latent_v_appends
                if len(all_q_appends) > 0:
                    query = torch.cat([query] + all_q_appends, dim=1)
                if len(all_k_appends) > 0:
                    key = torch.cat([key] + all_k_appends, dim=1)
                    value = torch.cat([value] + all_v_appends, dim=1)
            else:
                raise NotImplementedError('`frame_operation` requires `latent_hw` so latent sequence length can be inferred.')
        elif frame_operation is not None:
            if latent_seq_len is not None:
                start = 0
                end = start + latent_seq_len
                end = min(end, query.shape[1], key.shape[1], value.shape[1])
                latent_q = query[:, start:end]
                latent_k = key[:, start:end]
                latent_v = value[:, start:end]
                cached_ith_k, cached_last_k, cached_ith_v, cached_last_v = _get_chunk_kv_cache_references(chunk_kv_cache_input=chunk_kv_cache_input, layer_index=attn.layer_index, latent_k=latent_k, latent_v=latent_v)
                latent_base_q, latent_q_appends = (latent_q, [])
                latent_base_k, latent_k_appends = _apply_frame_mode_single(latent_k, frame_operation, frame_idx, frame_interval, context_prev_frames=context_prev_frames, append_all_prev_context=append_all_prev_context, cached_ith_reference=cached_ith_k, cached_last_reference=cached_last_k)
                latent_base_v, latent_v_appends = _apply_frame_mode_single(latent_v, frame_operation, frame_idx, frame_interval, context_prev_frames=context_prev_frames, append_all_prev_context=append_all_prev_context, cached_ith_reference=cached_ith_v, cached_last_reference=cached_last_v)
                latent_k_appends, latent_v_appends = _align_kv_append_lists(latent_k_appends, latent_v_appends, identity_k=latent_k, identity_v=latent_v)
                query = torch.cat([latent_base_q, query[:, end:]], dim=1)
                key = torch.cat([latent_base_k, key[:, end:]], dim=1)
                value = torch.cat([latent_base_v, value[:, end:]], dim=1)
                if len(latent_q_appends) > 0:
                    query = torch.cat([query] + latent_q_appends, dim=1)
                if len(latent_k_appends) > 0:
                    key = torch.cat([key] + latent_k_appends, dim=1)
                    value = torch.cat([value] + latent_v_appends, dim=1)
            else:
                raise NotImplementedError('`frame_operation` requires `latent_hw` so latent sequence length can be inferred.')
        if collect_chunk_kv_cache and frame_operation is not None and (latent_seq_len is not None):
            latent_start = text_seq_len or 0 if attn.added_kv_proj_dim is not None else 0
            latent_end = min(latent_start + latent_seq_len, key.shape[1], value.shape[1])
            if latent_end > latent_start:
                _collect_chunk_kv_cache(chunk_kv_cache_output=chunk_kv_cache_output, layer_index=attn.layer_index, latent_k=key[:, latent_start:latent_end], latent_v=value[:, latent_start:latent_end], frame_idx=frame_idx, cache_device=chunk_kv_cache_device)
        hidden_states = dispatch_attention_fn(query, key, value, attn_mask=attention_mask, backend=self._attention_backend, parallel_config=self._parallel_config)
        if hidden_states.shape[1] != original_query_seq_len:
            hidden_states = hidden_states[:, :original_query_seq_len]
        if cyclic_corr_token_replacement:
            if latent_seq_len is None:
                raise NotImplementedError('EditVid correspondence injection requires latent_hw.')
            latent_start = text_seq_len or 0 if attn.added_kv_proj_dim is not None else 0
            latent_end = min(latent_start + latent_seq_len, hidden_states.shape[1])
            if latent_end > latent_start:
                latent_hidden_states = hidden_states[:, latent_start:latent_end]
                layer_key = int(attn.layer_index) if attn.layer_index is not None else 0
                cached_anchor_for_layer = None
                if cyclic_corr_global_anchor_features_in is not None:
                    cached_anchor_for_layer = cyclic_corr_global_anchor_features_in.get(layer_key)
                if cyclic_corr_global_anchor_collect is not None and layer_key not in cyclic_corr_global_anchor_collect and (latent_hidden_states.shape[0] > 0):
                    cyclic_corr_global_anchor_collect[layer_key] = latent_hidden_states[0].detach().clone()
                if cyclic_corr_collect_maps and cyclic_corr_maps_output is not None:
                    cyclic_corr_maps_output.setdefault('attempted_layers', set()).add(layer_key)
                collect_this_layer = cyclic_corr_extraction_layer_index is None or layer_key == int(cyclic_corr_extraction_layer_index)
                if cyclic_corr_collect_maps and collect_this_layer and cyclic_corr_maps_output is not None:
                    cyclic_corr_maps_output.setdefault('seen_layers', set()).add(layer_key)
                    collected_correspondence = _compute_post_attn_global_anchor_cyclic_correspondence(
                        latent_hidden_states=latent_hidden_states,
                        latent_hw=latent_hw,
                        tau=cyclic_corr_tau,
                        delta=cyclic_corr_delta,
                        dropout_p=cyclic_corr_dropout_p,
                        cached_anchor_features=cached_anchor_for_layer,
                    )
                    if collected_correspondence is not None:
                        collected_correspondence['reference_strategy'] = 'global_anchor'
                        cyclic_corr_maps_output.setdefault('layers', {}).setdefault(
                            layer_key, _detach_post_attn_cyclic_correspondence(collected_correspondence)
                        )
                layers_input = (cyclic_corr_maps_input or {}).get('layers', {})
                correspondence = layers_input.get(layer_key)
                if correspondence is None:
                    correspondence = (cyclic_corr_maps_input or {}).get('default', None)
                if cyclic_corr_apply_replacement and correspondence is not None:
                    latent_hidden_states = _apply_post_attn_global_anchor_cyclic_token_replacement(latent_hidden_states=latent_hidden_states, latent_hw=latent_hw, tau=cyclic_corr_tau, delta=cyclic_corr_delta, dropout_p=cyclic_corr_dropout_p, cached_anchor_features=cached_anchor_for_layer, correspondence=correspondence)
                hidden_states = torch.cat([hidden_states[:, :latent_start], latent_hidden_states, hidden_states[:, latent_end:]], dim=1)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes([encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            return (hidden_states, encoder_hidden_states)
        else:
            return hidden_states

class Flux2Attention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = Flux2AttnProcessor
    _available_processors = [Flux2AttnProcessor]

    def __init__(self, query_dim: int, heads: int=8, dim_head: int=64, dropout: float=0.0, bias: bool=False, added_kv_proj_dim: Optional[int]=None, added_proj_bias: Optional[bool]=True, out_bias: bool=True, eps: float=1e-05, out_dim: int=None, elementwise_affine: bool=True, processor=None):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.use_bias = bias
        self.dropout = dropout
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))
        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)
        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)
        self.layer_index: Optional[int] = None
        self.layer_index: Optional[int] = None

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, image_rotary_emb: Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(f'joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored.')
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)

class Flux2ParallelSelfAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, 'scaled_dot_product_attention'):
            raise ImportError(f'{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.')

    def __call__(self, attn: 'Flux2ParallelSelfAttention', hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, image_rotary_emb: Optional[torch.Tensor]=None, latent_hw: Optional[Tuple[int, int]]=None, video_hw: Optional[Tuple[int, int]]=None, text_seq_len: Optional[int]=None, frame_operation: Optional[str]=None, stable_layers: Optional[List[int]]=None, frame_operation_layers: Optional[List[int]]=None, post_attn_layers: Optional[List[int]]=None, frame_idx: int=0, frame_interval: int=8, context_prev_frames: int=1, append_all_prev_context: bool=False, chunk_kv_cache_input: Optional[Dict[Any, Dict[str, torch.Tensor]]]=None, chunk_kv_cache_output: Optional[Dict[int, Dict[str, torch.Tensor]]]=None, collect_chunk_kv_cache: bool=False, chunk_kv_cache_device: str='cpu', cyclic_corr_token_replacement: bool=False, cyclic_corr_tau: float=0.8, cyclic_corr_delta: float=1.0, cyclic_corr_dropout_p: float=1.0, cyclic_corr_anchor_frame_num: int=0, cyclic_corr_maps_input: Optional[Dict[Any, Any]]=None, cyclic_corr_maps_output: Optional[Dict[Any, Any]]=None, cyclic_corr_collect_maps: bool=False, cyclic_corr_extraction_layer_index: Optional[int]=None, cyclic_corr_apply_replacement: bool=True, cyclic_corr_global_anchor_features_in: Optional[Dict[Any, torch.Tensor]]=None, cyclic_corr_global_anchor_collect: Optional[Dict[Any, torch.Tensor]]=None, cyclic_corr_inversion_anchor_features_in: Optional[Dict[Any, torch.Tensor]]=None, cyclic_corr_inversion_anchor_collect: Optional[Dict[Any, torch.Tensor]]=None) -> torch.Tensor:
        latent_seq_len = None
        video_seq_len = None
        effective_frame_layers = frame_operation_layers if frame_operation_layers is not None else stable_layers
        effective_post_attn_layers = post_attn_layers if post_attn_layers is not None else stable_layers
        if effective_frame_layers is not None and attn.layer_index is not None and (attn.layer_index not in effective_frame_layers):
            frame_operation = None
        if effective_post_attn_layers is not None and attn.layer_index is not None and (attn.layer_index not in effective_post_attn_layers):
            cyclic_corr_token_replacement = False
        if latent_hw is not None:
            latent_seq_len = int(latent_hw[0]) * int(latent_hw[1])
        if video_hw is not None:
            video_seq_len = int(video_hw[0]) * int(video_hw[1])
        if text_seq_len is None:
            text_seq_len = 0
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        original_query_seq_len = query.shape[1]
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        if frame_operation is not None:
            if latent_seq_len is not None:
                start = text_seq_len or 0
                end = start + latent_seq_len
                end = min(end, query.shape[1], key.shape[1], value.shape[1])
                latent_q = query[:, start:end]
                latent_k = key[:, start:end]
                latent_v = value[:, start:end]
                cached_ith_k, cached_last_k, cached_ith_v, cached_last_v = _get_chunk_kv_cache_references(chunk_kv_cache_input=chunk_kv_cache_input, layer_index=attn.layer_index, latent_k=latent_k, latent_v=latent_v)
                latent_base_q, latent_q_appends = (latent_q, [])
                latent_base_k, latent_k_appends = _apply_frame_mode_single(latent_k, frame_operation, frame_idx, frame_interval, context_prev_frames=context_prev_frames, append_all_prev_context=append_all_prev_context, cached_ith_reference=cached_ith_k, cached_last_reference=cached_last_k)
                latent_base_v, latent_v_appends = _apply_frame_mode_single(latent_v, frame_operation, frame_idx, frame_interval, context_prev_frames=context_prev_frames, append_all_prev_context=append_all_prev_context, cached_ith_reference=cached_ith_v, cached_last_reference=cached_last_v)
                latent_k_appends, latent_v_appends = _align_kv_append_lists(latent_k_appends, latent_v_appends, identity_k=latent_k, identity_v=latent_v)
                prefix_q = query[:, :start]
                prefix_k = key[:, :start]
                prefix_v = value[:, :start]
                text_q_appends: List[torch.Tensor] = []
                text_k_appends: List[torch.Tensor] = []
                text_v_appends: List[torch.Tensor] = []
                text_k_appends, text_v_appends = _align_kv_append_lists(text_k_appends, text_v_appends, identity_k=prefix_k, identity_v=prefix_v)
                suffix_q = query[:, end:]
                suffix_k = key[:, end:]
                suffix_v = value[:, end:]
                query = torch.cat([prefix_q, latent_base_q, suffix_q], dim=1)
                key = torch.cat([prefix_k, latent_base_k, suffix_k], dim=1)
                value = torch.cat([prefix_v, latent_base_v, suffix_v], dim=1)
                all_q_appends = text_q_appends + latent_q_appends
                all_k_appends = text_k_appends + latent_k_appends
                all_v_appends = text_v_appends + latent_v_appends
                if len(all_q_appends) > 0:
                    query = torch.cat([query] + all_q_appends, dim=1)
                if len(all_k_appends) > 0:
                    key = torch.cat([key] + all_k_appends, dim=1)
                    value = torch.cat([value] + all_v_appends, dim=1)
            else:
                raise NotImplementedError('`frame_operation` requires `latent_hw` so latent sequence length can be inferred.')
        if collect_chunk_kv_cache and frame_operation is not None and (latent_seq_len is not None):
            latent_start = text_seq_len or 0
            latent_end = min(latent_start + latent_seq_len, key.shape[1], value.shape[1])
            if latent_end > latent_start:
                _collect_chunk_kv_cache(chunk_kv_cache_output=chunk_kv_cache_output, layer_index=attn.layer_index, latent_k=key[:, latent_start:latent_end], latent_v=value[:, latent_start:latent_end], frame_idx=frame_idx, cache_device=chunk_kv_cache_device)
        hidden_states = dispatch_attention_fn(query, key, value, attn_mask=attention_mask, backend=self._attention_backend, parallel_config=self._parallel_config)
        if hidden_states.shape[1] != original_query_seq_len:
            hidden_states = hidden_states[:, :original_query_seq_len]
        if cyclic_corr_token_replacement:
            if latent_seq_len is None:
                raise NotImplementedError('EditVid correspondence injection requires latent_hw.')
            latent_start = text_seq_len or 0
            latent_end = min(latent_start + latent_seq_len, hidden_states.shape[1])
            if latent_end > latent_start:
                latent_hidden_states = hidden_states[:, latent_start:latent_end]
                layer_key = int(attn.layer_index) if attn.layer_index is not None else 0
                cached_anchor_for_layer = None
                if cyclic_corr_global_anchor_features_in is not None:
                    cached_anchor_for_layer = cyclic_corr_global_anchor_features_in.get(layer_key)
                if cyclic_corr_global_anchor_collect is not None and layer_key not in cyclic_corr_global_anchor_collect and (latent_hidden_states.shape[0] > 0):
                    cyclic_corr_global_anchor_collect[layer_key] = latent_hidden_states[0].detach().clone()
                if cyclic_corr_collect_maps and cyclic_corr_maps_output is not None:
                    cyclic_corr_maps_output.setdefault('attempted_layers', set()).add(layer_key)
                collect_this_layer = cyclic_corr_extraction_layer_index is None or layer_key == int(cyclic_corr_extraction_layer_index)
                if cyclic_corr_collect_maps and collect_this_layer and cyclic_corr_maps_output is not None:
                    cyclic_corr_maps_output.setdefault('seen_layers', set()).add(layer_key)
                    collected_correspondence = _compute_post_attn_global_anchor_cyclic_correspondence(
                        latent_hidden_states=latent_hidden_states,
                        latent_hw=latent_hw,
                        tau=cyclic_corr_tau,
                        delta=cyclic_corr_delta,
                        dropout_p=cyclic_corr_dropout_p,
                        cached_anchor_features=cached_anchor_for_layer,
                    )
                    if collected_correspondence is not None:
                        collected_correspondence['reference_strategy'] = 'global_anchor'
                        cyclic_corr_maps_output.setdefault('layers', {}).setdefault(
                            layer_key, _detach_post_attn_cyclic_correspondence(collected_correspondence)
                        )
                layers_input = (cyclic_corr_maps_input or {}).get('layers', {})
                correspondence = layers_input.get(layer_key)
                if correspondence is None:
                    correspondence = (cyclic_corr_maps_input or {}).get('default', None)
                if cyclic_corr_apply_replacement and correspondence is not None:
                    latent_hidden_states = _apply_post_attn_global_anchor_cyclic_token_replacement(latent_hidden_states=latent_hidden_states, latent_hw=latent_hw, tau=cyclic_corr_tau, delta=cyclic_corr_delta, dropout_p=cyclic_corr_dropout_p, cached_anchor_features=cached_anchor_for_layer, correspondence=correspondence)
                hidden_states = torch.cat([hidden_states[:, :latent_start], latent_hidden_states, hidden_states[:, latent_end:]], dim=1)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)
        return hidden_states

class Flux2ParallelSelfAttention(torch.nn.Module, AttentionModuleMixin):
    """
    Flux 2 parallel self-attention for the Flux 2 single-stream transformer blocks.

    This implements a parallel transformer block, where the attention QKV projections are fused to the feedforward (FF)
    input projections, and the attention output projections are fused to the FF output projections. See the [ViT-22B
    paper](https://arxiv.org/abs/2302.05442) for a visual depiction of this type of transformer block.
    """
    _default_processor_cls = Flux2ParallelSelfAttnProcessor
    _available_processors = [Flux2ParallelSelfAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(self, query_dim: int, heads: int=8, dim_head: int=64, dropout: float=0.0, bias: bool=False, out_bias: bool=True, eps: float=1e-05, out_dim: int=None, elementwise_affine: bool=True, mlp_ratio: float=4.0, mlp_mult_factor: int=2, processor=None):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.use_bias = bias
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(query_dim * self.mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor
        self.to_qkv_mlp_proj = torch.nn.Linear(self.query_dim, self.inner_dim * 3 + self.mlp_hidden_dim * self.mlp_mult_factor, bias=bias)
        self.mlp_act_fn = Flux2SwiGLU()
        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.to_out = torch.nn.Linear(self.inner_dim + self.mlp_hidden_dim, self.out_dim, bias=out_bias)
        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, image_rotary_emb: Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(f'joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored.')
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb, **kwargs)

class Flux2SingleTransformerBlock(nn.Module):

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float=3.0, eps: float=1e-06, bias: bool=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux2ParallelSelfAttention(query_dim=dim, dim_head=attention_head_dim, heads=num_attention_heads, out_dim=dim, bias=bias, out_bias=bias, eps=eps, mlp_ratio=mlp_ratio, mlp_mult_factor=2, processor=Flux2ParallelSelfAttnProcessor())

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor], temb_mod_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, joint_attention_kwargs: Optional[Dict[str, Any]]=None, split_hidden_states: bool=False, text_seq_len: Optional[int]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        mod_shift, mod_scale, mod_gate = temb_mod_params
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb, **joint_attention_kwargs)
        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        if split_hidden_states:
            encoder_hidden_states, hidden_states = (hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:])
            return (encoder_hidden_states, hidden_states)
        else:
            return hidden_states

class Flux2TransformerBlock(nn.Module):

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float=3.0, eps: float=1e-06, bias: bool=False):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux2Attention(query_dim=dim, added_kv_proj_dim=dim, dim_head=attention_head_dim, heads=num_attention_heads, out_dim=dim, bias=bias, added_proj_bias=bias, out_bias=bias, eps=eps, processor=Flux2AttnProcessor())
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb_mod_params_img: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], temb_mod_params_txt: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, joint_attention_kwargs: Optional[Dict[str, Any]]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa
        attention_outputs = self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states, image_rotary_emb=image_rotary_emb, **joint_attention_kwargs)
        attn_output, context_attn_output = attention_outputs
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output
        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return (encoder_hidden_states, hidden_states)

class Flux2PosEmbed(nn.Module):

    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == 'mps'
        is_npu = ids.device.type == 'npu'
        freqs_dtype = torch.float32 if is_mps or is_npu else torch.float64
        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(self.axes_dim[i], pos[..., i], theta=self.theta, repeat_interleave_real=True, use_real=True, freqs_dtype=freqs_dtype)
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return (freqs_cos, freqs_sin)

class Flux2TimestepGuidanceEmbeddings(nn.Module):

    def __init__(self, in_channels: int=256, embedding_dim: int=6144, bias: bool=False, guidance_embeds: bool=True):
        super().__init__()
        self.time_proj = Timesteps(num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=in_channels, time_embed_dim=embedding_dim, sample_proj_bias=bias)
        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(in_channels=in_channels, time_embed_dim=embedding_dim, sample_proj_bias=bias)
        else:
            self.guidance_embedder = None

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(timestep.dtype))
        if guidance is not None and self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(guidance.dtype))
            time_guidance_emb = timesteps_emb + guidance_emb
            return time_guidance_emb
        else:
            return timesteps_emb

class Flux2Modulation(nn.Module):

    def __init__(self, dim: int, mod_param_sets: int=2, bias: bool=False):
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        mod = self.act_fn(temb)
        mod = self.linear(mod)
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * self.mod_param_sets, dim=-1)
        return tuple((mod_params[3 * i:3 * (i + 1)] for i in range(self.mod_param_sets)))

class EditVidTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, FluxTransformer2DLoadersMixin, CacheMixin, AttentionMixin):
    """
    The Transformer model introduced in Flux 2.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `8`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `48`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `48`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `15360`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `True`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(32, 32, 32, 32)`):
            The dimensions to use for the rotary positional embeddings.
    """
    _supports_gradient_checkpointing = True
    _no_split_modules = ['Flux2TransformerBlock', 'Flux2SingleTransformerBlock']
    _skip_layerwise_casting_patterns = ['pos_embed', 'norm']
    _repeated_blocks = ['Flux2TransformerBlock', 'Flux2SingleTransformerBlock']
    _cp_plan = {'': {'hidden_states': ContextParallelInput(split_dim=1, expected_dims=3, split_output=False), 'encoder_hidden_states': ContextParallelInput(split_dim=1, expected_dims=3, split_output=False), 'img_ids': ContextParallelInput(split_dim=1, expected_dims=3, split_output=False), 'txt_ids': ContextParallelInput(split_dim=1, expected_dims=3, split_output=False)}, 'proj_out': ContextParallelOutput(gather_dim=1, expected_dims=3)}

    @register_to_config
    def __init__(self, patch_size: int=1, in_channels: int=128, out_channels: Optional[int]=None, num_layers: int=8, num_single_layers: int=48, attention_head_dim: int=128, num_attention_heads: int=48, joint_attention_dim: int=15360, timestep_guidance_channels: int=256, mlp_ratio: float=3.0, axes_dims_rope: Tuple[int, ...]=(32, 32, 32, 32), rope_theta: int=2000, eps: float=1e-06, guidance_embeds: bool=True):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(in_channels=timestep_guidance_channels, embedding_dim=self.inner_dim, bias=False, guidance_embeds=guidance_embeds)
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)
        self.transformer_blocks = nn.ModuleList([Flux2TransformerBlock(dim=self.inner_dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim, mlp_ratio=mlp_ratio, eps=eps, bias=False) for _ in range(num_layers)])
        self.single_transformer_blocks = nn.ModuleList([Flux2SingleTransformerBlock(dim=self.inner_dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim, mlp_ratio=mlp_ratio, eps=eps, bias=False) for _ in range(num_single_layers)])
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor=None, timestep: torch.LongTensor=None, img_ids: torch.Tensor=None, txt_ids: torch.Tensor=None, guidance: torch.Tensor=None, joint_attention_kwargs: Optional[Dict[str, Any]]=None, return_dict: bool=True) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop('scale', 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        elif joint_attention_kwargs is not None and joint_attention_kwargs.get('scale', None) is not None:
            logger.warning('Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.')
        num_txt_tokens = encoder_hidden_states.shape[1]
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)[0]
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0), torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0))
        for index_block, block in enumerate(self.transformer_blocks):
            block.attn.layer_index = index_block
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states, double_stream_mod_img, double_stream_mod_txt, concat_rotary_emb, joint_attention_kwargs)
            else:
                block_kwargs = joint_attention_kwargs
                encoder_hidden_states, hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb_mod_params_img=double_stream_mod_img, temb_mod_params_txt=double_stream_mod_txt, image_rotary_emb=concat_rotary_emb, joint_attention_kwargs=block_kwargs)
            _maybe_collect_block_cyclic_correspondence(hidden_states=hidden_states, layer_key=index_block, joint_attention_kwargs=joint_attention_kwargs, latent_start=0)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        offset = len(self.transformer_blocks)
        for index_block, block in enumerate(self.single_transformer_blocks):
            block.attn.layer_index = offset + index_block
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, None, single_stream_mod, concat_rotary_emb, joint_attention_kwargs)
            else:
                block_kwargs = joint_attention_kwargs
                hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=None, temb_mod_params=single_stream_mod, image_rotary_emb=concat_rotary_emb, joint_attention_kwargs=block_kwargs)
            _maybe_collect_block_cyclic_correspondence(hidden_states=hidden_states, layer_key=offset + index_block, joint_attention_kwargs=joint_attention_kwargs, latent_start=num_txt_tokens)
        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
