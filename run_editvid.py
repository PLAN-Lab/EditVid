import argparse
import csv
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
repo_root_for_import = Path(__file__).resolve().parent
vendor_src = repo_root_for_import / '.vendor' / 'diffusers' / 'src'
if vendor_src.exists() and str(vendor_src) not in sys.path:
    sys.path.insert(0, str(vendor_src))
from editvid import EditVidPipeline, EditVidTransformer2DModel
BASE_DEFAULTS: Dict[str, Any] = {'model_id': 'black-forest-labs/FLUX.2-klein-9B', 'manifest_json': str(repo_root_for_import / 'examples' / 'manifest.json'), 'source_videos_dir': str(repo_root_for_import / 'examples' / 'source_videos'), 'output_root': str(repo_root_for_import / 'outputs'), 'run_name': 'editvid', 'device': 'cuda', 'dtype': 'bfloat16', 'cpu_offload': True, 'seed': 14, 'chunk_size': 24, 'fallback_chunk_sizes': [16], 'max_frames': None, 'guidance_scale': 1.0, 'num_inference_steps': 4, 'max_sequence_length': 512, 'attention_step_indices': [0, 1, 2, 3], 'frame_operation_step_indices': [0, 1, 2, 3], 'post_attn_step_indices': [0, 1, 2, 3], 'source_inversion': True, 'source_inversion_prompt_col': 'source_prompt', 'source_inversion_guidance_scale': 1.0, 'height': None, 'width': None, 'vital_layers': None, 'frame_operation_layers': None, 'post_attn_layers': list(range(12)), 'frame_operation': 'prev_append', 'frame_idx': 0, 'frame_interval': 8, 'context_prev_frames': 1, 'append_all_prev_context': False, 'cyclic_corr_token_replacement': True, 'cyclic_corr_tau': 0.4, 'cyclic_corr_delta': 1.5, 'cyclic_corr_dropout_p': 0.5, 'cyclic_corr_anchor_frame_num': 0, 'cyclic_corr_extraction_layer_index': 6, 'cyclic_corr_extraction_timestep': 0.25, 'cache_chunk_kv': True, 'chunk_kv_cache_device': 'cpu', 'latent_anchor_strength': 0.6, 'latent_anchor_step_indices': [0, 1, 2], 'latent_anchor_feather': 3, 'latent_anchor_source': 'step_diff', 'latent_anchor_mask_low_quantile': 0.25, 'latent_anchor_mask_high_quantile': 0.65, 'latent_anchor_mask_gamma': 0.7, 'fallback_fps': 8.0, 'save_frames': True, 'skip_existing': True, 'save_run_metadata': True}
DTYPE_MAP = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}

def parse_int_list(value: str) -> List[int]:
    parts = [part.strip() for part in str(value).split(',') if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError('Expected a comma-separated list of integers.')
    try:
        return [int(part) for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f'Could not parse integer list: {value}') from error

def add_tri_state_flag(parser: argparse.ArgumentParser, name: str, help_text: str):
    dest = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f'--{name}', dest=dest, action='store_true', help=help_text)
    disable_text = help_text
    if help_text.lower().startswith('enable '):
        disable_text = help_text[len('Enable '):]
    group.add_argument(f'--no-{name}', dest=dest, action='store_false', help=f'Disable {disable_text}')
    parser.set_defaults(**{dest: None})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate temporally consistent video edits with EDITVID and FLUX2-Klein.')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--manifest-json', type=str, default=None)
    parser.add_argument('--source-videos-dir', type=str, default=None)
    parser.add_argument('--output-root', type=str, default=None)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--results-manifest', type=str, default='')
    parser.add_argument('--rows', type=str, default='', help="Optional row selection like '0,1,2' or '3-10'.")
    parser.add_argument('--max-rows', type=int, default=-1)
    add_tri_state_flag(parser, 'skip-existing', 'Enable skipping samples with existing output video.')
    add_tri_state_flag(parser, 'save-run-metadata', 'Enable saving prompt/config metadata per output sample.')
    parser.add_argument('--model-id', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dtype', choices=tuple(DTYPE_MAP.keys()), default=None)
    add_tri_state_flag(parser, 'cpu-offload', 'Enable model CPU offload.')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--chunk-size', type=int, default=None)
    parser.add_argument('--fallback-chunk-sizes', type=parse_int_list, default=None, help='Smaller chunk sizes to retry after generation errors. Default: 16.')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--max-sequence-length', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--vital-layers', type=parse_int_list, default=None)
    parser.add_argument('--frame-idx', type=int, default=None)
    parser.add_argument('--frame-interval', type=int, default=None)
    parser.add_argument('--cyclic-corr-anchor-frame-num', type=int, default=None)
    parser.add_argument('--fallback-fps', type=float, default=None)
    add_tri_state_flag(parser, 'save-frames', 'Save generated frames as PNG images.')
    parser.add_argument('--latent-anchor-source', choices=('step_diff',), default=None)
    parser.add_argument('--guidance-scale', type=float, default=None, help='Paper default: 1.0.')
    parser.add_argument('--num-inference-steps', type=int, default=None, help='Paper default: 4.')
    parser.add_argument('--attention-step-indices', type=parse_int_list, default=None, help='Paper default: 0,1,2,3.')
    parser.add_argument('--frame-operation-step-indices', type=parse_int_list, default=None, help='Denoising steps for sparse causal KV memory. Paper default: 0,1,2,3.')
    parser.add_argument('--post-attn-step-indices', type=parse_int_list, default=None, help='Denoising steps for global token injection. Paper default: 0,1,2,3.')
    parser.add_argument('--frame-operation-layers', type=parse_int_list, default=None, help='KV-memory layers. Omit to use all transformer blocks, as in the paper.')
    parser.add_argument('--post-attn-layers', type=parse_int_list, default=None, help='Global token-injection layers. Paper default: 0 through 11.')
    parser.add_argument('--source-inversion-prompt-col', type=str, default=None)
    parser.add_argument('--source-inversion-guidance-scale', type=float, default=None, help='Paper default: 1.0.')
    parser.add_argument('--cyclic-corr-extraction-layer-index', type=int, default=None, help='Correspondence extraction block. Paper default: penultimate double-stream block (index 6).')
    parser.add_argument('--cyclic-corr-extraction-timestep', type=float, default=None, help='Paper default: 0.25.')
    parser.add_argument('--cyclic-corr-tau', type=float, default=None, help='Correspondence confidence threshold. Paper default: 0.4.')
    parser.add_argument('--cyclic-corr-delta', type=float, default=None, help='Cycle-consistency spatial radius. Paper default: 1.5.')
    parser.add_argument('--cyclic-corr-dropout-p', type=float, default=None, help='Paper default: 0.5.')
    parser.add_argument('--latent-anchor-strength', type=float, default=None)
    parser.add_argument('--latent-anchor-step-indices', type=parse_int_list, default=None)
    parser.add_argument('--latent-anchor-feather', type=int, default=None, help='Soft-mask smoothing radius. Release default: 3.')
    parser.add_argument('--latent-anchor-mask-low-quantile', type=float, default=None, help='Paper default: 0.25.')
    parser.add_argument('--latent-anchor-mask-high-quantile', type=float, default=None, help='Paper default: 0.65.')
    parser.add_argument('--latent-anchor-mask-gamma', type=float, default=None, help='Paper default: 0.7.')
    return parser.parse_args()

def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(BASE_DEFAULTS)
    for key, value in vars(args).items():
        if key in {'rows', 'max_rows', 'results_manifest', 'local_rank'}:
            continue
        if value is not None:
            config[key] = value
    return config

def parse_rows_spec(spec: str) -> Optional[List[int]]:
    spec = (spec or '').strip()
    if not spec:
        return None
    result = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            a, b = chunk.split('-', 1)
            a, b = (int(a), int(b))
            step = 1 if b >= a else -1
            result.extend(list(range(a, b + step, step)))
        else:
            result.append(int(chunk))
    seen = set()
    out = []
    for x in result:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _resolve_cuda_device(device_str: str) -> Optional[torch.device]:
    if not torch.cuda.is_available():
        return None
    try:
        resolved = torch.device(device_str)
    except Exception:
        return torch.device('cuda', torch.cuda.current_device())
    if resolved.type != 'cuda':
        return None
    if resolved.index is None:
        return torch.device('cuda', torch.cuda.current_device())
    return resolved

def _release_generation_memory(device_str: str) -> None:
    gc.collect()
    cuda_device = _resolve_cuda_device(device_str)
    if cuda_device is None:
        return
    try:
        torch.cuda.synchronize(cuda_device)
    except Exception:
        pass
    try:
        with torch.cuda.device(cuda_device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

def _reset_cpu_offload_hooks(pipe: EditVidPipeline, cpu_offload_enabled: bool) -> None:
    if not cpu_offload_enabled:
        return
    try:
        pipe.maybe_free_model_hooks()
    except Exception:
        pass

def _get_worker_info(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, int]:
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', str(rank)))
    if world_size > 1 and (args.device is None or str(args.device).strip() == 'cuda'):
        config['device'] = f'cuda:{local_rank}'
    resolved_cuda = _resolve_cuda_device(str(config['device']))
    if resolved_cuda is not None:
        torch.cuda.set_device(resolved_cuda)
    return {'rank': rank, 'world_size': world_size, 'local_rank': local_rank}

def _build_chunk_size_candidates(chunk_size: int, fallback_chunk_sizes: List[int]) -> List[int]:
    candidates: List[int] = []
    seen = set()
    for candidate in [int(chunk_size)] + [int(x) for x in fallback_chunk_sizes]:
        if candidate <= 0 or candidate > int(chunk_size) or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates

def _per_rank_manifest_path(manifest_path: Path, rank: int) -> Path:
    return manifest_path.with_name(f'{manifest_path.stem}.rank{rank:02d}{manifest_path.suffix}')

def _per_rank_done_path(manifest_path: Path, rank: int) -> Path:
    return manifest_path.with_name(f'{manifest_path.stem}.rank{rank:02d}.done')

def _write_and_maybe_merge_manifest(rows_out: List[Dict[str, Any]], field_order: List[str], manifest_path: Path, worker_rank: int, worker_world_size: int) -> None:
    import pandas as pd
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.DataFrame(rows_out)
    for col in field_order:
        if col not in manifest_df.columns:
            manifest_df[col] = ''
    manifest_df = manifest_df[field_order + [c for c in manifest_df.columns if c not in field_order]]
    if worker_world_size <= 1:
        manifest_df.to_csv(manifest_path, index=False)
        print(f'[INFO] Wrote generation manifest: {manifest_path}')
        return
    part_manifest_path = _per_rank_manifest_path(manifest_path, worker_rank)
    done_path = _per_rank_done_path(manifest_path, worker_rank)
    manifest_df.to_csv(part_manifest_path, index=False)
    done_path.write_text('done\n', encoding='utf-8')
    print(f'[INFO][rank={worker_rank}] Wrote shard manifest: {part_manifest_path}')
    if worker_rank != 0:
        return
    wait_deadline = time.time() + 24 * 60 * 60
    while time.time() < wait_deadline:
        missing = [rank for rank in range(worker_world_size) if not _per_rank_done_path(manifest_path, rank).exists()]
        if not missing:
            break
        time.sleep(2.0)
    else:
        raise TimeoutError(f'Timed out waiting for shard completion markers under {manifest_path.parent}')
    merged_parts = []
    for rank in range(worker_world_size):
        rank_manifest_path = _per_rank_manifest_path(manifest_path, rank)
        merged_parts.append(pd.read_csv(rank_manifest_path))
    merged_manifest_df = pd.concat(merged_parts, ignore_index=True)
    if 'row_index' in merged_manifest_df.columns:
        merged_manifest_df = merged_manifest_df.sort_values(['row_index', 'worker_rank'], kind='stable')
    merged_manifest_df.to_csv(manifest_path, index=False)
    print(f'[INFO] Wrote merged generation manifest: {manifest_path}')

def get_video_fps(path: str, default: float=8.0) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and fps > 0:
        return float(fps)
    return float(default)

def load_first_video_frames(path: str, num_frames: Optional[int]=None) -> List[Image.Image]:
    try:
        import imageio.v3 as iio
        frames: List[Image.Image] = []
        for frame_idx, frame in enumerate(iio.imiter(path)):
            if num_frames is not None and frame_idx >= num_frames:
                break
            frames.append(Image.fromarray(frame))
        if not frames:
            raise ValueError(f'No frames decoded from video: {path}')
        return frames
    except Exception:
        cap = cv2.VideoCapture(path)
        frames = []
        frame_idx = 0
        while True:
            if num_frames is not None and frame_idx >= num_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            frame_idx += 1
        cap.release()
        if not frames:
            raise ValueError(f'No frames decoded from video: {path}')
        return frames

def _write_video_with_ffmpeg(frames: List[Image.Image], out_path: str, fps: float, codec: str='libx264'):
    ffmpeg_bin = shutil.which('ffmpeg')
    if ffmpeg_bin is None:
        raise RuntimeError('ffmpeg executable not found in PATH.')
    width, height = frames[0].size
    for idx, frame in enumerate(frames):
        if frame.size != (width, height):
            raise ValueError(f'Inconsistent frame size at index {idx}: expected {(width, height)}, got {frame.size}.')
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ffmpeg_frames_', dir=str(out_path_obj.parent)) as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for idx, img in enumerate(frames):
            img.convert('RGB').save(tmp_dir_path / f'{idx:06d}.png', format='PNG')
        cmd = [ffmpeg_bin, '-y', '-hide_banner', '-loglevel', 'error', '-framerate', str(float(fps)), '-i', str(tmp_dir_path / '%06d.png'), '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black,format=yuv420p', '-c:v', codec]
        if codec == 'libx264':
            cmd.extend(['-preset', 'medium', '-crf', '18'])
        elif codec == 'mpeg4':
            cmd.extend(['-q:v', '2'])
        cmd.extend(['-movflags', '+faststart', out_path])
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='replace').strip()
            raise RuntimeError(f"ffmpeg failed with codec={codec}: {stderr or 'no stderr output'}")

def img_list_to_video(frames: List[Image.Image], out_path: str, fps: float):
    if not frames:
        raise ValueError('No frames to write to video.')
    first_error: Optional[Exception] = None
    try:
        _write_video_with_ffmpeg(frames, out_path, fps, codec='libx264')
        return
    except Exception as error:
        first_error = error
    try:
        _write_video_with_ffmpeg(frames, out_path, fps, codec='mpeg4')
        return
    except Exception as second_error:
        raise RuntimeError(f'Failed to write video to {out_path}: libx264={repr(first_error)}; mpeg4={repr(second_error)}') from second_error

def build_attention_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the fixed attention configuration used by EditVid."""
    return {'frame_operation': 'prev_append', 'frame_idx': config['frame_idx'], 'frame_interval': config['frame_interval'], 'context_prev_frames': 1, 'append_all_prev_context': False, 'cyclic_corr_token_replacement': True, 'cyclic_corr_tau': config['cyclic_corr_tau'], 'cyclic_corr_delta': config['cyclic_corr_delta'], 'cyclic_corr_dropout_p': config['cyclic_corr_dropout_p'], 'cyclic_corr_anchor_frame_num': config['cyclic_corr_anchor_frame_num'], 'cyclic_corr_extraction_layer_index': config['cyclic_corr_extraction_layer_index'], 'cyclic_corr_extraction_timestep': config['cyclic_corr_extraction_timestep'], 'chunk_kv_cache_device': 'cpu'}

def _extract_sample_id(*texts: str, fallback_index: int) -> str:
    for text in texts:
        if not text:
            continue
        match = re.search('sample_\\d{5}', str(text))
        if match:
            return match.group(0)
    return f'sample_{fallback_index:05d}'

def _resolve_existing_path(raw_path: str, repo_root: Path, manifest_dir: Path) -> Optional[Path]:
    if not raw_path:
        return None
    p = Path(raw_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(repo_root / p)
        candidates.append(manifest_dir / p)
        candidates.append(p)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None

def _resolve_source_video_path(entry: Dict[str, Any], source_videos_dir: Path, repo_root: Path, manifest_dir: Path, sample_id: str) -> Path:
    raw = str(entry.get('source_video', '') or entry.get('source_video_path', '') or '').strip()
    resolved = _resolve_existing_path(raw, repo_root=repo_root, manifest_dir=manifest_dir)
    if resolved is not None:
        return resolved
    basename = Path(raw).name if raw else ''
    candidates = [source_videos_dir / f'{sample_id}_source.mp4']
    if basename:
        candidates.append(source_videos_dir / basename)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve source video for sample_id={sample_id}. raw='{raw}', source_videos_dir='{source_videos_dir}'")

def _resolve_edit_prompt(entry: Dict[str, Any]) -> str:
    for key in ['edit_prompt', 'editing_prompt', 'prompt']:
        value = entry.get(key, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    raise ValueError('Manifest row has no non-empty prompt in keys: edit_prompt/editing_prompt/prompt')

def _generate_video_for_chunk_size(pipe: EditVidPipeline, video_frames: List[Image.Image], prompt: str, source_inversion_prompt: Optional[str], config: Dict[str, Any], chunk_size: int) -> List[Image.Image]:
    generated_frames: List[Image.Image] = []
    chunk_kv_cache: Optional[Dict[str, Any]] = None
    global_anchor_cache: Optional[Dict[str, Any]] = None
    generator = torch.Generator(device=config['device']).manual_seed(int(config['seed']))
    for start_idx in range(0, len(video_frames), int(chunk_size)):
        output = None
        images = None
        call_kwargs: Dict[str, Any] = {}
        attention_kwargs: Optional[Dict[str, Any]] = None
        ref_images = video_frames[start_idx:start_idx + int(chunk_size)]
        prompts = [prompt] * len(ref_images)
        try:
            _reset_cpu_offload_hooks(pipe, bool(config['cpu_offload']))
            attention_kwargs = build_attention_kwargs(config=config)
            call_kwargs = {'video_frames': ref_images, 'prompt': prompts, 'guidance_scale': config['guidance_scale'], 'num_inference_steps': config['num_inference_steps'], 'source_inversion': True, 'source_inversion_prompt': source_inversion_prompt, 'source_inversion_guidance_scale': config['source_inversion_guidance_scale'], 'generator': generator, 'max_sequence_length': config['max_sequence_length'], 'attention_step_indices': config['attention_step_indices'], 'frame_operation_step_indices': config['frame_operation_step_indices'], 'post_attn_step_indices': config['post_attn_step_indices']}
            if attention_kwargs:
                call_kwargs['attention_kwargs'] = attention_kwargs
            if config['height'] is not None:
                call_kwargs['height'] = config['height']
            if config['width'] is not None:
                call_kwargs['width'] = config['width']
            if config['vital_layers'] is not None:
                call_kwargs['vital_layers'] = config['vital_layers']
            if config['frame_operation_layers'] is not None:
                call_kwargs['frame_operation_layers'] = config['frame_operation_layers']
            if config['post_attn_layers'] is not None:
                call_kwargs['post_attn_layers'] = config['post_attn_layers']
            call_kwargs['chunk_kv_cache'] = chunk_kv_cache
            call_kwargs['return_chunk_kv_cache'] = True
            call_kwargs['cyclic_corr_global_anchor_cache'] = global_anchor_cache
            call_kwargs['return_cyclic_corr_global_anchor_cache'] = True
            if config.get('latent_anchor_strength', 0.0):
                call_kwargs['latent_anchor_strength'] = float(config['latent_anchor_strength'])
                call_kwargs['latent_anchor_feather'] = int(config.get('latent_anchor_feather') or 0)
                call_kwargs['latent_anchor_source'] = config.get('latent_anchor_source') or 'step_diff'
                call_kwargs['latent_anchor_mask_low_quantile'] = float(config.get('latent_anchor_mask_low_quantile', 0.0))
                call_kwargs['latent_anchor_mask_high_quantile'] = float(config.get('latent_anchor_mask_high_quantile', 1.0))
                call_kwargs['latent_anchor_mask_gamma'] = float(config.get('latent_anchor_mask_gamma', 1.0))
                if config.get('latent_anchor_step_indices') is not None:
                    call_kwargs['latent_anchor_step_indices'] = config['latent_anchor_step_indices']
            output = pipe(**call_kwargs)
            images = output.images
            chunk_kv_cache = output.chunk_kv_cache
            if global_anchor_cache is None:
                global_anchor_cache = getattr(output, 'cyclic_corr_global_anchor_cache', None)
            generated_frames.extend((image.copy() for image in images))
        except Exception:
            _reset_cpu_offload_hooks(pipe, bool(config['cpu_offload']))
            raise
        finally:
            images = None
            output = None
            attention_kwargs = None
            call_kwargs.clear()
            _reset_cpu_offload_hooks(pipe, bool(config['cpu_offload']))
            _release_generation_memory(config['device'])
    return generated_frames

def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    worker_info = _get_worker_info(args=args, config=config)
    worker_rank = int(worker_info['rank'])
    worker_world_size = int(worker_info['world_size'])
    worker_local_rank = int(worker_info['local_rank'])
    if config['chunk_size'] <= 0:
        raise ValueError('`--chunk-size` must be > 0.')
    manifest_path = Path(config['manifest_json']).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f'Manifest not found: {manifest_path}')
    repo_root = repo_root_for_import
    manifest_dir = manifest_path.parent
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_rows = json.load(f)
    if not isinstance(manifest_rows, list):
        raise ValueError(f'Manifest must be a JSON list: {manifest_path}')
    selected_indices = list(range(len(manifest_rows)))
    row_spec = parse_rows_spec(args.rows)
    if row_spec is not None:
        selected_indices = [idx for idx in row_spec if 0 <= idx < len(manifest_rows)]
    if args.max_rows is not None and int(args.max_rows) > 0:
        selected_indices = selected_indices[:int(args.max_rows)]
    if worker_world_size > 1:
        selected_indices = [idx for idx in selected_indices if idx % worker_world_size == worker_rank]
    source_videos_dir = Path(config['source_videos_dir']).resolve() if config['source_videos_dir'] else (manifest_dir / 'source_videos').resolve()
    output_root = Path(config['output_root']).resolve() / str(config['run_name'])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_out_path = Path(args.results_manifest).resolve() if args.results_manifest else output_root / 'generation_manifest.csv'
    dtype = DTYPE_MAP[config['dtype']]
    transformer = EditVidTransformer2DModel.from_pretrained(config['model_id'], subfolder='transformer', torch_dtype=dtype)
    pipe = EditVidPipeline.from_pretrained(config['model_id'], transformer=transformer, torch_dtype=dtype)
    if config['cpu_offload']:
        resolved_cuda = _resolve_cuda_device(str(config['device']))
        if resolved_cuda is not None and resolved_cuda.index is not None:
            pipe.enable_model_cpu_offload(gpu_id=int(resolved_cuda.index), device=resolved_cuda.type)
        else:
            pipe.enable_model_cpu_offload(device=config['device'])
    else:
        pipe.to(config['device'])
    chunk_size_candidates = _build_chunk_size_candidates(chunk_size=int(config['chunk_size']), fallback_chunk_sizes=list(config['fallback_chunk_sizes']))
    print(f'[INFO][rank={worker_rank}/{worker_world_size}] Selected rows: {len(selected_indices)}')
    print(f'[INFO][rank={worker_rank}] Source videos dir: {source_videos_dir}')
    print(f'[INFO][rank={worker_rank}] Output root: {output_root}')
    print(f"[INFO][rank={worker_rank}] Device: {config['device']} (local_rank={worker_local_rank})")
    print(f"[INFO][rank={worker_rank}] CPU offload: {config['cpu_offload']}")
    print(f'[INFO][rank={worker_rank}] Chunk size candidates: {chunk_size_candidates}')
    rows_out: List[Dict[str, Any]] = []
    field_order = ['row_index', 'sample_id', 'status', 'error', 'worker_rank', 'worker_world_size', 'source_video_path', 'edit_prompt', 'source_inversion', 'source_inversion_prompt_col', 'source_inversion_prompt', 'source_inversion_guidance_scale', 'output_dir', 'output_video_path', 'output_video_path_flat', 'saved_frames_dir', 'fps', 'num_source_frames', 'num_generated_frames', 'seed', 'requested_chunk_size', 'chunk_size', 'attempted_chunk_sizes', 'guidance_scale', 'num_inference_steps', 'frame_operation', 'frame_idx', 'attention_step_indices', 'post_attn_step_indices', 'post_attn_layers', 'vital_layers', 'cyclic_corr_token_replacement', 'cyclic_corr_extraction_layer_index', 'cyclic_corr_extraction_timestep', 'latent_anchor_strength', 'latent_anchor_step_indices', 'latent_anchor_feather', 'latent_anchor_source', 'latent_anchor_mask_low_quantile', 'latent_anchor_mask_high_quantile', 'latent_anchor_mask_gamma', 'runtime_sec']
    for local_i, row_index in enumerate(tqdm(selected_indices, desc='[GEN] Video-edit rows', dynamic_ncols=True), start=1):
        t0 = time.time()
        entry = manifest_rows[row_index]
        video_frames: List[Image.Image] = []
        generated_frames: List[Image.Image] = []
        sample_id = _extract_sample_id(entry.get('source_video', ''), entry.get('edited_video', ''), entry.get('source_video_path', ''), fallback_index=int(row_index))
        out_row: Dict[str, Any] = {'row_index': int(row_index), 'sample_id': sample_id, 'status': 'ok', 'error': '', 'worker_rank': worker_rank, 'worker_world_size': worker_world_size, 'source_video_path': '', 'edit_prompt': '', 'source_inversion': config['source_inversion'], 'source_inversion_prompt_col': config['source_inversion_prompt_col'], 'source_inversion_prompt': '', 'source_inversion_guidance_scale': config['source_inversion_guidance_scale'], 'output_dir': '', 'output_video_path': '', 'output_video_path_flat': '', 'saved_frames_dir': '', 'fps': '', 'num_source_frames': '', 'num_generated_frames': '', 'seed': config['seed'], 'requested_chunk_size': config['chunk_size'], 'chunk_size': '', 'attempted_chunk_sizes': ','.join(map(str, chunk_size_candidates)), 'guidance_scale': config['guidance_scale'], 'num_inference_steps': config['num_inference_steps'], 'frame_operation': config['frame_operation'] if config['frame_operation'] is not None else 'none', 'frame_idx': config['frame_idx'], 'attention_step_indices': '' if config['attention_step_indices'] is None else ','.join(map(str, config['attention_step_indices'])), 'post_attn_step_indices': '' if config['post_attn_step_indices'] is None else ','.join(map(str, config['post_attn_step_indices'])), 'post_attn_layers': '' if config['post_attn_layers'] is None else ','.join(map(str, config['post_attn_layers'])), 'vital_layers': '' if config['vital_layers'] is None else ','.join(map(str, config['vital_layers'])), 'cyclic_corr_token_replacement': config['cyclic_corr_token_replacement'], 'cyclic_corr_extraction_layer_index': config['cyclic_corr_extraction_layer_index'], 'cyclic_corr_extraction_timestep': config['cyclic_corr_extraction_timestep'], 'latent_anchor_strength': config['latent_anchor_strength'], 'latent_anchor_step_indices': '' if config['latent_anchor_step_indices'] is None else ','.join(map(str, config['latent_anchor_step_indices'])), 'latent_anchor_feather': config['latent_anchor_feather'], 'latent_anchor_source': config['latent_anchor_source'], 'latent_anchor_mask_low_quantile': config['latent_anchor_mask_low_quantile'], 'latent_anchor_mask_high_quantile': config['latent_anchor_mask_high_quantile'], 'latent_anchor_mask_gamma': config['latent_anchor_mask_gamma'], 'runtime_sec': 0.0}
        try:
            source_video_path = _resolve_source_video_path(entry=entry, source_videos_dir=source_videos_dir, repo_root=repo_root, manifest_dir=manifest_dir, sample_id=sample_id)
            prompt = _resolve_edit_prompt(entry)
            source_inversion_prompt = None
            if config['source_inversion']:
                source_inversion_prompt = str(entry.get(config['source_inversion_prompt_col'], '') or '').strip()
                if not source_inversion_prompt:
                    raise ValueError(f"Missing source inversion prompt in manifest key '{config['source_inversion_prompt_col']}' for row {row_index}.")
                out_row['source_inversion_prompt'] = source_inversion_prompt
            out_row['source_video_path'] = str(source_video_path)
            out_row['edit_prompt'] = prompt
            fps = get_video_fps(str(source_video_path), default=float(config['fallback_fps']))
            video_frames = load_first_video_frames(str(source_video_path), num_frames=config['max_frames'])
            if len(video_frames) == 0:
                raise ValueError(f'No source frames for row {row_index}')
            out_row['fps'] = float(fps)
            out_row['num_source_frames'] = len(video_frames)
            sample_dir = output_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            out_video_path = sample_dir / 'edited.mp4'
            out_video_path_flat = output_root / f'{sample_id}_edited.mp4'
            out_row['output_dir'] = str(sample_dir)
            out_row['output_video_path'] = str(out_video_path)
            out_row['output_video_path_flat'] = str(out_video_path_flat)
            out_row['saved_frames_dir'] = str(sample_dir / 'frames')
            if config['skip_existing'] and (out_video_path.exists() or out_video_path_flat.exists()):
                out_row['status'] = 'skipped_exists'
                out_row['chunk_size'] = config['chunk_size']
            else:
                generation_error: Optional[Exception] = None
                for attempt_idx, attempt_chunk_size in enumerate(chunk_size_candidates):
                    out_row['chunk_size'] = attempt_chunk_size
                    try:
                        generated_frames = _generate_video_for_chunk_size(pipe=pipe, video_frames=video_frames, prompt=prompt, source_inversion_prompt=source_inversion_prompt, config=config, chunk_size=attempt_chunk_size)
                        generation_error = None
                        break
                    except Exception as error:
                        generation_error = error
                        generated_frames.clear()
                        _release_generation_memory(config['device'])
                        if attempt_idx + 1 < len(chunk_size_candidates):
                            print(f'[WARN][rank={worker_rank}] row={row_index} failed with chunk_size={attempt_chunk_size}: {repr(error)}. Retrying with smaller chunk size.')
                        else:
                            raise RuntimeError(f'All chunk size attempts failed for row {row_index}: {chunk_size_candidates}') from error
                if generation_error is not None:
                    raise generation_error
                if config['save_frames']:
                    frames_dir = sample_dir / 'frames'
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    for frame_idx, image in enumerate(generated_frames):
                        image.save(frames_dir / f'{frame_idx:05d}.png')
                img_list_to_video(generated_frames, str(out_video_path), fps=float(fps))
                if out_video_path_flat != out_video_path:
                    shutil.copy2(out_video_path, out_video_path_flat)
                out_row['num_generated_frames'] = len(generated_frames)
            if config['save_run_metadata']:
                prompt_path = sample_dir / 'prompt.txt'
                prompt_path.write_text(f'{prompt}\n', encoding='utf-8')
                run_config = dict(config)
                run_config['source_video_path'] = str(source_video_path)
                run_config['source_prompt'] = str(entry.get('source_prompt', ''))
                run_config['target_prompt'] = str(entry.get('target_prompt', ''))
                run_config['sample_id'] = sample_id
                run_config['row_index'] = int(row_index)
                (sample_dir / 'run_config.json').write_text(json.dumps(run_config, indent=2), encoding='utf-8')
            if out_row['status'] == 'ok':
                print(f'[OK][rank={worker_rank}] ({local_i}/{len(selected_indices)}) row={row_index} sample={sample_id}')
        except Exception as error:
            out_row['status'] = 'error'
            out_row['error'] = repr(error)
            print(f'[ERR][rank={worker_rank}] row={row_index} sample={sample_id}: {repr(error)}')
        finally:
            generated_frames.clear()
            video_frames.clear()
            _release_generation_memory(config['device'])
            out_row['runtime_sec'] = time.time() - t0
            rows_out.append(out_row)
    _write_and_maybe_merge_manifest(rows_out=rows_out, field_order=field_order, manifest_path=manifest_out_path, worker_rank=worker_rank, worker_world_size=worker_world_size)
if __name__ == '__main__':
    main()
