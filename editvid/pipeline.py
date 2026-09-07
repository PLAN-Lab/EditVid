import inspect
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from diffusers.loaders import Flux2LoraLoaderMixin
from diffusers.models import AutoencoderKLFlux2
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from .pipeline_output import EditVidPipelineOutput
from .transformer import EditVidTransformer2DModel
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
logger = logging.get_logger(__name__)
SUPPORTED_FRAME_OPERATIONS = ('prev', 'ith', 'key_frames', 'append', 'key_frame_append', 'key_frame_all_append', 'prev_append', 'key_frame_prev_append', 'key_frame_all_append_prev_append', 'ith_prev_append')
EXAMPLE_DOC_STRING = '\n    Examples:\n        ```py\n        >>> import torch\n        >>> from diffusers import EditVidPipeline\n\n        >>> pipe = EditVidPipeline.from_pretrained(\n        ...     "black-forest-labs/FLUX.2-klein-base-9B", torch_dtype=torch.bfloat16\n        ... )\n        >>> pipe.to("cuda")\n        >>> prompt = "A cat holding a sign that says hello world"\n        >>> # Depending on the variant being used, the pipeline call will slightly vary.\n        >>> # Refer to the pipeline documentation for more details.\n        >>> image = pipe(prompt, num_inference_steps=50, guidance_scale=4.0).images[0]\n        >>> image.save("flux2_output.png")\n        ```\n'

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = (8.73809524e-05, 1.89833333)
    a2, b2 = (0.00016927, 0.45666666)
    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b
    return float(mu)

def retrieve_timesteps(scheduler, num_inference_steps: Optional[int]=None, device: Optional[Union[str, torch.device]]=None, timesteps: Optional[List[int]]=None, sigmas: Optional[List[float]]=None, **kwargs):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError('Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values')
    if timesteps is not None:
        accepts_timesteps = 'timesteps' in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timestep schedules. Please check whether you are using the correct scheduler.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = 'sigmas' in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas schedules. Please check whether you are using the correct scheduler.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return (timesteps, num_inference_steps)

def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator]=None, sample_mode: str='sample'):
    if hasattr(encoder_output, 'latent_dist') and sample_mode == 'sample':
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, 'latent_dist') and sample_mode == 'argmax':
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, 'latents'):
        return encoder_output.latents
    else:
        raise AttributeError('Could not access latents of provided encoder_output')

def _normalize_chunk_kv_cache_steps(chunk_kv_cache: Optional[Dict[str, Any]]) -> Optional[List[Dict[Any, Any]]]:
    if chunk_kv_cache is None:
        return None
    if not isinstance(chunk_kv_cache, dict):
        return None
    steps = chunk_kv_cache.get('steps', None)
    if not isinstance(steps, list):
        return None
    normalized_steps: List[Dict[Any, Any]] = []
    for step in steps:
        if isinstance(step, dict):
            normalized_steps.append(step)
        else:
            normalized_steps.append({})
    return normalized_steps

class EditVidPipeline(DiffusionPipeline, Flux2LoraLoaderMixin):
    """
    The Flux2 Klein pipeline for text-to-image generation.

    Reference:
    [https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)

    Args:
        transformer ([`EditVidTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLFlux2`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen3ForCausalLM`]):
            [Qwen3ForCausalLM](https://huggingface.co/docs/transformers/en/model_doc/qwen3#transformers.Qwen3ForCausalLM)
        tokenizer (`Qwen2TokenizerFast`):
            Tokenizer of class
            [Qwen2TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2TokenizerFast).
    """
    model_cpu_offload_seq = 'text_encoder->transformer->vae'
    _callback_tensor_inputs = ['latents', 'prompt_embeds']

    def __init__(self, scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKLFlux2, text_encoder: Qwen3ForCausalLM, tokenizer: Qwen2TokenizerFast, transformer: EditVidTransformer2DModel, is_distilled: bool=False):
        super().__init__()
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, transformer=transformer)
        self.register_to_config(is_distilled=is_distilled)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, 'vae', None) else 8
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

    @staticmethod
    def _get_qwen3_prompt_embeds(text_encoder: Qwen3ForCausalLM, tokenizer: Qwen2TokenizerFast, prompt: Union[str, List[str]], dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None, max_sequence_length: int=512, hidden_states_layers: List[int]=(9, 18, 27)):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        all_input_ids = []
        all_attention_masks = []
        for single_prompt in prompt:
            messages = [{'role': 'user', 'content': single_prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_sequence_length)
            all_input_ids.append(inputs['input_ids'])
            all_attention_masks.append(inputs['attention_mask'])
        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)
        output = text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
        return prompt_embeds

    @staticmethod
    def _prepare_text_ids(x: torch.Tensor, t_coord: Optional[torch.Tensor]=None):
        B, L, _ = x.shape
        out_ids = []
        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)
        return torch.stack(out_ids)

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor):
        """
        Generates 4D position coordinates (T, H, W, L) for latent tensors.

        Args:
            latents (torch.Tensor):
                Latent tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor:
                Position IDs tensor of shape (B, H*W, 4) All batches share the same coordinate structure: T=0,
                H=[0..H-1], W=[0..W-1], L=0
        """
        batch_size, _, height, width = latents.shape
        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)
        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_ids

    @staticmethod
    def _prepare_image_ids(image_latents: List[torch.Tensor], scale: int=10, start_idx: int=0):
        """
        Generates 4D time-space coordinates (T, H, W, L) for a sequence of image latents.

        This function creates a unique coordinate for every pixel/patch across all input latent with different
        dimensions.

        Args:
            image_latents (List[torch.Tensor]):
                A list of image latent feature tensors, typically of shape (C, H, W).
            scale (int, optional):
                A factor used to define the time separation (T-coordinate) between latents. T-coordinate for the i-th
                latent is: 'scale + scale * i'. Defaults to 10.

        Returns:
            torch.Tensor:
                The combined coordinate tensor. Shape: (1, N_total, 4) Where N_total is the sum of (H * W) for all
                input latents.

        Coordinate Components (Dimension 4):
            - T (Time): The unique index indicating which latent image the coordinate belongs to.
            - H (Height): The row index within that latent image.
            - W (Width): The column index within that latent image.
            - L (Seq. Length): A sequence length dimension, which is always fixed at 0 (size 1)
        """
        if not isinstance(image_latents, list):
            raise ValueError(f'Expected `image_latents` to be a list, got {type(image_latents)}.')
        t_coords = [scale + scale * (start_idx + t) for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]
        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape
            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)
        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)
        return image_latent_ids

    @staticmethod
    def _prepare_video_frame_ids(video_frame_latents: torch.Tensor, scale: int=10) -> torch.Tensor:
        """
        Generates 4D time-space coordinates (T, H, W, L) for per-sample video frame latents.

        The video frame condition uses the first image-condition time slot: `T = scale + 0 * scale` (e.g. 10).
        """
        if video_frame_latents.ndim != 4:
            raise ValueError(f'Expected `video_frame_latents` dims 4, got {video_frame_latents.ndim}.')
        batch_size, _, height, width = video_frame_latents.shape
        device = video_frame_latents.device
        t = torch.tensor([scale], device=device)
        h = torch.arange(height, device=device)
        w = torch.arange(width, device=device)
        l = torch.arange(1, device=device)
        frame_ids = torch.cartesian_prod(t, h, w, l)
        frame_ids = frame_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return frame_ids

    @staticmethod
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @staticmethod
    def _pack_latents(latents):
        """
        pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
        """
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @staticmethod
    def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> list[torch.Tensor]:
        """
        using position ids to scatter tokens into place
        """
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)
            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1
            flat_ids = h_ids * w + w_ids
            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)
        return torch.stack(x_list, dim=0)

    def encode_prompt(self, prompt: Union[str, List[str]], device: Optional[torch.device]=None, num_images_per_prompt: int=1, prompt_embeds: Optional[torch.Tensor]=None, max_sequence_length: int=512, text_encoder_out_layers: Tuple[int]=(9, 18, 27)):
        device = device or self._execution_device
        if prompt is None:
            prompt = ''
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompt_embeds = self._get_qwen3_prompt_embeds(text_encoder=self.text_encoder, tokenizer=self.tokenizer, prompt=prompt, device=device, max_sequence_length=max_sequence_length, hidden_states_layers=text_encoder_out_layers)
        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return (prompt_embeds, text_ids)

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if image.ndim != 4:
            raise ValueError(f'Expected image dims 4, got {image.ndim}.')
        image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode='argmax')
        image_latents = self._patchify_latents(image_latents)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std
        return image_latents

    def prepare_latents(self, batch_size, num_latents_channels, height, width, dtype, device, generator: Optional[Union[torch.Generator, List[torch.Generator]]], latents: Optional[torch.Tensor]=None):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.')
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(device)
        latents = self._pack_latents(latents)
        return (latents, latent_ids)

    def _decode_packed_latents_to_image_tensor(self, packed_latents: torch.Tensor, latent_ids: torch.Tensor) -> torch.Tensor:
        unpacked_latents = self._unpack_latents_with_ids(packed_latents, latent_ids)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked_latents.device, unpacked_latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(unpacked_latents.device, unpacked_latents.dtype)
        unpacked_latents = unpacked_latents * latents_bn_std + latents_bn_mean
        unpacked_latents = self._unpatchify_latents(unpacked_latents)
        return self.vae.decode(unpacked_latents.to(dtype=self.vae.dtype), return_dict=False)[0].float()

    def _encode_pil_images_to_packed_latents(self, images: List[PIL.Image.Image], latent_ids: torch.Tensor, *, device: torch.device, dtype: torch.dtype, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None) -> torch.Tensor:
        if not images:
            raise ValueError('Expected at least one image to encode.')
        latent_h = int(latent_ids[0, :, 1].max().item() + 1)
        latent_w = int(latent_ids[0, :, 2].max().item() + 1)
        height = latent_h * self.vae_scale_factor * 2
        width = latent_w * self.vae_scale_factor * 2
        packed_latents = []
        for idx, image in enumerate(images):
            frame = self.image_processor.preprocess(image.convert('RGB'), height=height, width=width, resize_mode='crop')
            frame_generator = generator[idx] if isinstance(generator, list) else generator
            frame = frame.to(device=device, dtype=self.vae.dtype)
            frame_latents = self._encode_vae_image(frame, generator=frame_generator)
            packed_latents.append(self._pack_latents(frame_latents).squeeze(0))
        return torch.stack(packed_latents, dim=0).to(device=device, dtype=dtype)

    @staticmethod
    def _tensor_to_pil_images(images: torch.Tensor) -> List[PIL.Image.Image]:
        images = images.detach().clamp(0, 1).cpu()
        return [PIL.Image.fromarray((image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)) for image in images]

    @staticmethod
    def _summarize_cyclic_corr_maps(cyclic_corr_maps: Optional[Dict[Any, Any]]) -> Tuple[int, int]:
        if not cyclic_corr_maps:
            return (0, 0)
        layers = cyclic_corr_maps.get('layers', {})
        if not isinstance(layers, dict):
            return (0, 0)
        valid_count = 0
        for correspondence in layers.values():
            if not isinstance(correspondence, dict):
                continue
            valid_mask = correspondence.get('valid_mask', None)
            if torch.is_tensor(valid_mask):
                valid_count += int(valid_mask.to(torch.bool).sum().item())
        return (len(layers), valid_count)

    def _register_rf_cyclic_corr_hooks(self, *, maps_output, attention_kwargs, latent_ids, text_ids):
        from .transformer import _compute_post_attn_global_anchor_cyclic_correspondence, _detach_post_attn_cyclic_correspondence
        latent_h = int(latent_ids[0, :, 1].max().item() + 1)
        latent_w = int(latent_ids[0, :, 2].max().item() + 1)
        latent_hw = (latent_h, latent_w)
        latent_seq_len = latent_h * latent_w
        text_seq_len = int(text_ids.shape[1])
        double_blocks = list(getattr(self.transformer, 'transformer_blocks', []))
        single_blocks = list(getattr(self.transformer, 'single_transformer_blocks', []))
        total_layers = len(double_blocks) + len(single_blocks)
        extraction_layer = attention_kwargs.get('cyclic_corr_extraction_layer_index')
        layer_indices = list(range(total_layers)) if extraction_layer is None else [int(extraction_layer)]
        handles = []

        def make_hook(layer_key, latent_start):

            def hook(_module, _inputs, output):
                hidden_states = output[1] if isinstance(output, tuple) else output
                if hidden_states is None or not torch.is_tensor(hidden_states):
                    return
                maps_output.setdefault('attempted_layers', set()).add(int(layer_key))
                latent_end = min(latent_start + latent_seq_len, hidden_states.shape[1])
                if latent_end <= latent_start:
                    return
                latent_hidden_states = hidden_states[:, latent_start:latent_end].unsqueeze(2)
                correspondence = _compute_post_attn_global_anchor_cyclic_correspondence(
                    latent_hidden_states=latent_hidden_states,
                    latent_hw=latent_hw,
                    tau=attention_kwargs.get('cyclic_corr_tau', 0.4),
                    delta=attention_kwargs.get('cyclic_corr_delta', 1.5),
                    dropout_p=attention_kwargs.get('cyclic_corr_dropout_p', 0.5),
                )
                if correspondence is not None:
                    correspondence['reference_strategy'] = 'global_anchor'
                    maps_output.setdefault('seen_layers', set()).add(int(layer_key))
                    maps_output.setdefault('layers', {})[int(layer_key)] = (
                        _detach_post_attn_cyclic_correspondence(correspondence)
                    )
            return hook
        for layer_key in layer_indices:
            if layer_key < 0 or layer_key >= total_layers:
                continue
            if layer_key < len(double_blocks):
                handles.append(double_blocks[layer_key].register_forward_hook(make_hook(layer_key, 0)))
            else:
                handles.append(single_blocks[layer_key - len(double_blocks)].register_forward_hook(make_hook(layer_key, text_seq_len)))
        return handles

    def _apply_latent_anchor(self, latents: torch.Tensor, z_src: torch.Tensor, latent_ids: torch.Tensor, strength: float, feather: int, mask_low_quantile: float, mask_high_quantile: float, mask_gamma: float) -> torch.Tensor:
        if strength <= 0.0:
            return latents
        latents_dtype = latents.dtype
        device = latents.device
        latents_f = latents.to(torch.float32)
        z_src_f = z_src.to(device=device, dtype=torch.float32)
        diff = (latents_f - z_src_f).abs().mean(dim=-1, keepdim=True)
        diff_low = torch.quantile(diff, float(mask_low_quantile), dim=1, keepdim=True)
        diff_high = torch.quantile(diff, float(mask_high_quantile), dim=1, keepdim=True)
        mask = (diff - diff_low) / (diff_high - diff_low).clamp(min=1e-06)
        mask = mask.clamp(0.0, 1.0)
        if not math.isclose(float(mask_gamma), 1.0):
            mask = mask.pow(float(mask_gamma))
        if feather > 0:
            mask_unpacked = self._unpack_latents_with_ids(mask, latent_ids)
            kernel_size = max(1, 2 * int(feather) + 1)
            sigma = max(0.001, float(feather) / 2.0)
            half = (kernel_size - 1) // 2
            offsets = torch.arange(kernel_size, device=device, dtype=torch.float32) - half
            kernel_1d = torch.exp(-offsets ** 2 / (2.0 * sigma * sigma))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
            kernel_y = kernel_1d.view(1, 1, kernel_size, 1)
            mask_unpacked = F.conv2d(mask_unpacked, kernel_x, padding=(0, half))
            mask_unpacked = F.conv2d(mask_unpacked, kernel_y, padding=(half, 0))
            mask = self._pack_latents(mask_unpacked)
            mask = mask.clamp(0.0, 1.0)
        m_eff = strength * (1.0 - mask)
        blended = (1.0 - m_eff) * latents_f + m_eff * z_src_f
        return blended.to(dtype=latents_dtype)

    @staticmethod
    def _resize_and_normalize_heatmap(heatmap: torch.Tensor, width: int, height: int) -> np.ndarray:
        if heatmap.ndim != 2:
            raise ValueError(f'Expected a 2D heatmap, got {tuple(heatmap.shape)}.')
        heatmap = heatmap.float().unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(heatmap, size=(height, width), mode='bilinear', align_corners=False)[0, 0]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max().clamp(min=1e-06)
        return heatmap.detach().cpu().numpy()

    @staticmethod
    def _overlay_heatmap_on_image(image: PIL.Image.Image, heatmap: np.ndarray, alpha: float) -> PIL.Image.Image:
        rgb = np.array(image.convert('RGB'), dtype=np.float32)
        heat = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
        heat_color = np.zeros_like(rgb)
        heat_color[..., 0] = 255.0 * heat
        blend = np.clip(rgb * (1.0 - alpha * heat[..., None]) + heat_color * (alpha * heat[..., None]), 0.0, 255.0)
        return PIL.Image.fromarray(blend.astype(np.uint8))

    def prepare_image_latents(self, images: List[torch.Tensor], batch_size, generator: torch.Generator, device, dtype, start_idx: int=0):
        image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            imagge_latent = self._encode_vae_image(image=image, generator=generator)
            image_latents.append(imagge_latent)
        image_latent_ids = self._prepare_image_ids(image_latents, start_idx=start_idx)
        packed_latents = []
        for latent in image_latents:
            packed = self._pack_latents(latent)
            packed = packed.squeeze(0)
            packed_latents.append(packed)
        image_latents = torch.cat(packed_latents, dim=0)
        image_latents = image_latents.unsqueeze(0)
        image_latents = image_latents.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.to(device)
        return (image_latents, image_latent_ids)

    def prepare_video_latents(self, video_frames: List[torch.Tensor], batch_size: int, generator: Optional[Union[torch.Generator, List[torch.Generator]]], device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(video_frames, list):
            raise ValueError(f'Expected `video_frames` to be a list, got {type(video_frames)}.')
        if len(video_frames) != batch_size:
            raise ValueError(f'Expected `video_frames` to have length {batch_size}, got {len(video_frames)}.')
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.')
        frame_latents = []
        for i, frame in enumerate(video_frames):
            frame = frame.to(device=device, dtype=dtype)
            frame_generator = generator[i] if isinstance(generator, list) else generator
            frame_latent = self._encode_vae_image(image=frame, generator=frame_generator)
            frame_latents.append(frame_latent)
        video_frame_latents = torch.cat(frame_latents, dim=0)
        video_frame_ids = self._prepare_video_frame_ids(video_frame_latents)
        video_frame_latents = self._pack_latents(video_frame_latents)
        return (video_frame_latents, video_frame_ids)

    def _predict_velocity(self, latents: torch.Tensor, latent_ids: torch.Tensor, timestep: torch.Tensor, prompt_embeds: torch.Tensor, text_ids: torch.Tensor, video_latents: Optional[torch.Tensor]=None, video_latent_ids: Optional[torch.Tensor]=None, negative_prompt_embeds: Optional[torch.Tensor]=None, negative_text_ids: Optional[torch.Tensor]=None, guidance_scale: float=1.0, attention_kwargs: Optional[Dict[str, Any]]=None) -> torch.Tensor:
        latent_model_input = latents.to(self.transformer.dtype)
        latent_image_ids = latent_ids
        transformer_attention_kwargs = dict(attention_kwargs) if attention_kwargs is not None else {}
        transformer_attention_kwargs.pop('cyclic_corr_extraction_timestep', None)
        if transformer_attention_kwargs:
            if 'latent_hw' not in transformer_attention_kwargs:
                latent_height = int(latent_ids[0, :, 1].max().item() + 1)
                latent_width = int(latent_ids[0, :, 2].max().item() + 1)
                transformer_attention_kwargs['latent_hw'] = (latent_height, latent_width)
            if 'text_seq_len' not in transformer_attention_kwargs:
                transformer_attention_kwargs['text_seq_len'] = text_ids.shape[1]
        if video_latents is not None:
            latent_model_input = torch.cat([latent_model_input, video_latents.to(self.transformer.dtype)], dim=1)
            latent_image_ids = torch.cat([latent_image_ids, video_latent_ids], dim=1)
            if transformer_attention_kwargs and 'video_hw' not in transformer_attention_kwargs and (video_latent_ids is not None):
                video_height = int(video_latent_ids[0, :, 1].max().item() + 1)
                video_width = int(video_latent_ids[0, :, 2].max().item() + 1)
                transformer_attention_kwargs['video_hw'] = (video_height, video_width)
        timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
        with self.transformer.cache_context('cond'):
            velocity = self.transformer(hidden_states=latent_model_input, timestep=timestep / self.scheduler.config.num_train_timesteps, guidance=None, encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_image_ids, joint_attention_kwargs=transformer_attention_kwargs, return_dict=False)[0]
        velocity = velocity[:, :latents.shape[1], :]
        do_cfg = guidance_scale > 1.0 and (not self.config.is_distilled)
        if do_cfg:
            if negative_prompt_embeds is None or negative_text_ids is None:
                raise ValueError('Negative prompt embeddings are required when source inversion guidance is > 1.')
            with self.transformer.cache_context('uncond'):
                negative_velocity = self.transformer(hidden_states=latent_model_input, timestep=timestep / self.scheduler.config.num_train_timesteps, guidance=None, encoder_hidden_states=negative_prompt_embeds, txt_ids=negative_text_ids, img_ids=latent_image_ids, joint_attention_kwargs=transformer_attention_kwargs, return_dict=False)[0]
            negative_velocity = negative_velocity[:, :latents.shape[1], :]
            velocity = negative_velocity + guidance_scale * (velocity - negative_velocity)
        return velocity

    def invert_source_latents_rf_solver(self, source_latents: torch.Tensor, latent_ids: torch.Tensor, sigmas: torch.Tensor, prompt_embeds: torch.Tensor, text_ids: torch.Tensor, video_latents: Optional[torch.Tensor]=None, video_latent_ids: Optional[torch.Tensor]=None, negative_prompt_embeds: Optional[torch.Tensor]=None, negative_text_ids: Optional[torch.Tensor]=None, guidance_scale: float=1.0, attention_kwargs: Optional[Dict[str, Any]]=None, trajectory_output: Optional[List[torch.Tensor]]=None) -> torch.Tensor:
        if sigmas.ndim != 1 or sigmas.numel() < 2:
            raise ValueError(f'Expected at least two 1D sigma values for source inversion, got {tuple(sigmas.shape)}.')
        latents_dtype = source_latents.dtype
        latents = source_latents.to(dtype=prompt_embeds.dtype)
        inv_sigmas = sigmas.to(device=latents.device, dtype=torch.float32).flip(0)
        if trajectory_output is not None:
            trajectory_output.append(source_latents.detach().to(dtype=latents_dtype).clone())
        extraction_step_index = None
        if attention_kwargs is not None and attention_kwargs.get('cyclic_corr_collect_maps', False):
            extraction_timestep = attention_kwargs.get('cyclic_corr_extraction_timestep', None)
            if extraction_timestep is not None:
                extraction_timestep_tensor = torch.tensor(float(extraction_timestep), device=inv_sigmas.device, dtype=inv_sigmas.dtype)
                extraction_step_index = int(torch.argmin((inv_sigmas[:-1] - extraction_timestep_tensor).abs()).item())
        for step_index, (sigma_curr, sigma_next) in enumerate(zip(inv_sigmas[:-1], inv_sigmas[1:])):
            delta = sigma_next - sigma_curr
            timestep = sigma_curr * self.scheduler.config.num_train_timesteps
            timestep_mid = (sigma_curr + 0.5 * delta) * self.scheduler.config.num_train_timesteps
            velocity_attention_kwargs = attention_kwargs
            midpoint_attention_kwargs = attention_kwargs
            collect_velocity_maps = False
            if attention_kwargs is not None and attention_kwargs.get('cyclic_corr_collect_maps', False):
                velocity_attention_kwargs = dict(attention_kwargs)
                midpoint_attention_kwargs = dict(attention_kwargs)
                if extraction_step_index is not None:
                    collect_velocity_maps = step_index == extraction_step_index
                    velocity_attention_kwargs['cyclic_corr_collect_maps'] = collect_velocity_maps
                else:
                    collect_velocity_maps = True
                midpoint_attention_kwargs['cyclic_corr_collect_maps'] = False
            hook_handles: List[Any] = []
            if collect_velocity_maps and velocity_attention_kwargs is not None:
                maps_output = velocity_attention_kwargs.get('cyclic_corr_maps_output', None)
                if maps_output is not None:
                    hook_handles = self._register_rf_cyclic_corr_hooks(maps_output=maps_output, attention_kwargs=velocity_attention_kwargs, latent_ids=latent_ids, text_ids=text_ids)
            try:
                velocity = self._predict_velocity(latents=latents, latent_ids=latent_ids, timestep=timestep, prompt_embeds=prompt_embeds, text_ids=text_ids, video_latents=video_latents, video_latent_ids=video_latent_ids, negative_prompt_embeds=negative_prompt_embeds, negative_text_ids=negative_text_ids, guidance_scale=guidance_scale, attention_kwargs=velocity_attention_kwargs)
            finally:
                for handle in hook_handles:
                    handle.remove()
            midpoint_latents = latents.to(torch.float32) + 0.5 * delta * velocity.to(torch.float32)
            midpoint_velocity = self._predict_velocity(latents=midpoint_latents.to(dtype=latents.dtype), latent_ids=latent_ids, timestep=timestep_mid, prompt_embeds=prompt_embeds, text_ids=text_ids, video_latents=video_latents, video_latent_ids=video_latent_ids, negative_prompt_embeds=negative_prompt_embeds, negative_text_ids=negative_text_ids, guidance_scale=guidance_scale, attention_kwargs=midpoint_attention_kwargs)
            first_order = (midpoint_velocity.to(torch.float32) - velocity.to(torch.float32)) / (0.5 * delta)
            latents = (latents.to(torch.float32) + delta * velocity.to(torch.float32) + 0.5 * delta * delta * first_order).to(dtype=latents_dtype)
            if trajectory_output is not None:
                trajectory_output.append(latents.detach().to(dtype=latents_dtype).clone())
        return latents

    def check_inputs(self, prompt, height, width, prompt_embeds=None, attention_kwargs=None, callback_on_step_end_tensor_inputs=None, guidance_scale=None):
        if height is not None and height % (self.vae_scale_factor * 2) != 0 or (width is not None and width % (self.vae_scale_factor * 2) != 0):
            logger.warning(f'`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly')
        if callback_on_step_end_tensor_inputs is not None and (not all((k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs))):
            raise ValueError(f'`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}')
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(f'Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.')
        elif prompt is None and prompt_embeds is None:
            raise ValueError('Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.')
        elif prompt is not None and (not isinstance(prompt, str) and (not isinstance(prompt, list))):
            raise ValueError(f'`prompt` has to be of type `str` or `list` but is {type(prompt)}')
        if attention_kwargs is not None:
            if not isinstance(attention_kwargs, dict):
                raise ValueError(f'`attention_kwargs` must be a dict when provided, got {type(attention_kwargs)}.')
            frame_operation = attention_kwargs.get('frame_operation', None)
            if frame_operation == 'none':
                frame_operation = None
            if frame_operation is not None and frame_operation not in SUPPORTED_FRAME_OPERATIONS:
                raise ValueError(f"`attention_kwargs['frame_operation']` must be one of {SUPPORTED_FRAME_OPERATIONS}, got {frame_operation}.")
        if guidance_scale > 1.0 and self.config.is_distilled:
            logger.warning(f'Guidance scale {guidance_scale} is ignored for step-wise distilled models.')

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and (not self.config.is_distilled)

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(self, image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]]=None, video_frames: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]]=None, prompt: Union[str, List[str]]=None, height: Optional[int]=None, width: Optional[int]=None, num_inference_steps: int=50, sigmas: Optional[List[float]]=None, guidance_scale: Optional[float]=4.0, num_images_per_prompt: int=1, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents: Optional[torch.Tensor]=None, source_inversion: bool=False, source_inversion_num_steps: Optional[int]=None, source_inversion_prompt: Optional[Union[str, List[str]]]=None, source_inversion_guidance_scale: float=1.0, vital_layers: Optional[List[int]]=None, frame_operation_layers: Optional[List[int]]=None, post_attn_layers: Optional[List[int]]=None, prompt_embeds: Optional[torch.Tensor]=None, negative_prompt_embeds: Optional[Union[str, List[str]]]=None, output_type: Optional[str]='pil', return_dict: bool=True, attention_kwargs: Optional[Dict[str, Any]]=None, chunk_kv_cache: Optional[Dict[str, Any]]=None, return_chunk_kv_cache: bool=False, attention_step_indices: Optional[List[int]]=None, frame_operation_step_indices: Optional[List[int]]=None, post_attn_step_indices: Optional[List[int]]=None, callback_on_step_end: Optional[Callable[[int, int, Dict], None]]=None, callback_on_step_end_tensor_inputs: List[str]=['latents'], max_sequence_length: int=512, latent_anchor_strength: float=0.0, latent_anchor_step_indices: Optional[List[int]]=None, latent_anchor_feather: int=3, latent_anchor_source: str='step_diff', latent_anchor_mask_low_quantile: float=0.0, latent_anchor_mask_high_quantile: float=1.0, latent_anchor_mask_gamma: float=1.0, cyclic_corr_global_anchor_cache: Optional[Dict[str, Any]]=None, return_cyclic_corr_global_anchor_cache: bool=False):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            video_frames (`PIL.Image.Image` or `List[PIL.Image.Image]`, *optional*):
                Per-sample video-frame conditioning. Must have a 1:1 correspondence with the number of samples
                generated, i.e. `len(prompt) * num_images_per_prompt` (or `prompt_embeds.shape[0] * num_images_per_prompt`).
                The video-frame condition occupies the first image-condition slot (idx=1, `T=10` when `scale=10`), so
                other `image` conditions start from idx=2.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality. For step-wise distilled models,
                `guidance_scale` is ignored.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            source_inversion (`bool`, *optional*, defaults to `False`):
                If `True`, initialize generation from RF-Solver-inverted source video latents instead of random noise.
                Requires `video_frames`.
            source_inversion_prompt (`str` or `List[str]`, *optional*):
                Prompt used while inverting the source video frames. If omitted, the target prompt embeddings are reused.
            source_inversion_guidance_scale (`float`, *optional*, defaults to `1.0`):
                Classifier-free guidance scale used only during source inversion.
            vital_layers (`List[int]`, *optional*):
                Legacy shared layer selector applied to both frame/KV operations and post-attention latent operations
                when the more specific selectors below are unset.
            frame_operation_layers (`List[int]`, *optional*):
                Layer indices (0-based across double then single blocks) where frame/KV operations should apply. If
                `None`, falls back to `vital_layers`.
            post_attn_layers (`List[int]`, *optional*):
                Layer indices (0-based across double then single blocks) where post-attention latent operations should
                apply. If `None`, falls back to `vital_layers`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Note that "" is used as the negative prompt in this pipeline.
                If not provided, will be generated from "".
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            chunk_kv_cache (`dict`, *optional*):
                Optional chunk-to-chunk KV cache returned by a previous call when `return_chunk_kv_cache=True`.
                The cache is consumed per denoising step and per transformer layer to provide cross-chunk frame
                references for frame-operation KV modes.
            return_chunk_kv_cache (`bool`, *optional*, defaults to `False`):
                If `True`, returns a chunk KV cache in the output under `chunk_kv_cache`, containing per-step/per-layer
                cached `ith` and `last` frame K/V tensors for the next chunk.
            attention_step_indices (`List[int]`, *optional*):
                Legacy global denoising step gate for `attention_kwargs`-based modifications. If `None`, attention
                modifications are applied on all steps. If `frame_operation_step_indices` or `post_attn_step_indices`
                are provided, those more specific schedules override this per subsystem.
            frame_operation_step_indices (`List[int]`, *optional*):
                Denoising step indices where frame/KV operations are applied. If `None`, falls back to
                `attention_step_indices`.
            post_attn_step_indices (`List[int]`, *optional*):
                Denoising step indices where post-attention latent operations are applied. If `None`, falls back to
                `attention_step_indices`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
        Examples:

        Returns:
            [`~pipelines.flux2.EditVidPipelineOutput`] or `tuple`: [`~pipelines.flux2.EditVidPipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """
        self.check_inputs(prompt=prompt, height=height, width=width, prompt_embeds=prompt_embeds, attention_kwargs=attention_kwargs, callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs, guidance_scale=guidance_scale)
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        base_attention_kwargs = dict(attention_kwargs) if attention_kwargs is not None else None
        if base_attention_kwargs is not None:
            if base_attention_kwargs.get('cyclic_corr_token_replacement', False) and (not source_inversion):
                raise ValueError('EditVid correspondence injection requires source_inversion=True.')
        self._current_timestep = None
        self._interrupt = False
        attention_step_indices_set = set(attention_step_indices) if attention_step_indices is not None else None
        frame_operation_step_indices_set = set(frame_operation_step_indices) if frame_operation_step_indices is not None else attention_step_indices_set
        post_attn_step_indices_set = set(post_attn_step_indices) if post_attn_step_indices is not None else attention_step_indices_set
        if frame_operation_step_indices_set is not None and any((step_idx < 0 for step_idx in frame_operation_step_indices_set)):
            raise ValueError(f'`frame_operation_step_indices` must contain only non-negative indices, got {frame_operation_step_indices}.')
        if post_attn_step_indices_set is not None and any((step_idx < 0 for step_idx in post_attn_step_indices_set)):
            raise ValueError(f'`post_attn_step_indices` must contain only non-negative indices, got {post_attn_step_indices}.')
        source_inversion_guidance_scale = float(source_inversion_guidance_scale)
        if source_inversion_guidance_scale <= 0:
            raise ValueError(f'`source_inversion_guidance_scale` must be > 0, got {source_inversion_guidance_scale}.')
        chunk_kv_cache_steps = _normalize_chunk_kv_cache_steps(chunk_kv_cache)
        if chunk_kv_cache is not None and chunk_kv_cache_steps is None:
            logger.warning("Ignoring `chunk_kv_cache` because it is not in the expected format {'steps': List[Dict[...]]}.")
        if source_inversion and video_frames is None:
            raise ValueError('`source_inversion=True` requires `video_frames`.')
        if source_inversion and latents is not None:
            raise ValueError('Pass either explicit `latents` or `source_inversion=True`, not both.')
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        effective_batch_size = batch_size * num_images_per_prompt
        prompt_embeds, text_ids = self.encode_prompt(prompt=prompt, prompt_embeds=prompt_embeds, device=device, num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length)
        negative_text_ids = None
        if self.do_classifier_free_guidance:
            negative_prompt = ''
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(prompt=negative_prompt, prompt_embeds=negative_prompt_embeds, device=device, num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length)
        if video_frames is not None and (not isinstance(video_frames, list)):
            video_frames = [video_frames]
        if video_frames is not None:
            if len(video_frames) == batch_size and num_images_per_prompt > 1:
                video_frames = [frame for frame in video_frames for _ in range(num_images_per_prompt)]
            if len(video_frames) != effective_batch_size:
                raise ValueError(f'Expected `video_frames` to have length {effective_batch_size} (one per generated sample), got {len(video_frames)}.')
        condition_video_frames = None
        if video_frames is not None:
            for img in video_frames:
                self.image_processor.check_image_input(img)
            condition_video_frames = []
            target_height, target_width = (height, width)
            if target_height is None or target_width is None:
                img0 = video_frames[0]
                w0, h0 = img0.size
                if w0 * h0 > 1024 * 1024:
                    img0 = self.image_processor._resize_to_target_area(img0, 1024 * 1024)
                    w0, h0 = img0.size
                multiple_of = self.vae_scale_factor * 2
                target_width = w0 // multiple_of * multiple_of
                target_height = h0 // multiple_of * multiple_of
            multiple_of = self.vae_scale_factor * 2
            target_width = int(target_width) // multiple_of * multiple_of
            target_height = int(target_height) // multiple_of * multiple_of
            for img in video_frames:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                img = self.image_processor.preprocess(img, height=target_height, width=target_width, resize_mode='crop')
                condition_video_frames.append(img)
            height = height or target_height
            width = width or target_width
        if image is not None and (not isinstance(image, list)):
            image = [image]
        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)
            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size
                multiple_of = self.vae_scale_factor * 2
                image_width = image_width // multiple_of * multiple_of
                image_height = image_height // multiple_of * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode='crop')
                condition_images.append(img)
                height = height or image_height
                width = width or image_width
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(batch_size=effective_batch_size, num_latents_channels=num_channels_latents, height=height, width=width, dtype=prompt_embeds.dtype, device=device, generator=generator, latents=latents)
        video_latents = None
        video_latent_ids = None
        if condition_video_frames is not None:
            video_generator = generator
            if isinstance(generator, list) and len(generator) == batch_size and (num_images_per_prompt > 1):
                video_generator = [g for g in generator for _ in range(num_images_per_prompt)]
            video_latents, video_latent_ids = self.prepare_video_latents(video_frames=condition_video_frames, batch_size=effective_batch_size, generator=video_generator, device=device, dtype=self.vae.dtype)
            video_latent_ids = video_latent_ids.to(device)
        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(images=condition_images, batch_size=effective_batch_size, generator=generator, device=device, dtype=self.vae.dtype, start_idx=1 if condition_video_frames is not None else 0)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, 'use_flow_sigmas') and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        latent_anchor_strength = float(latent_anchor_strength)
        if latent_anchor_strength < 0.0 or latent_anchor_strength > 1.0:
            raise ValueError(f'`latent_anchor_strength` must be in [0, 1], got {latent_anchor_strength}.')
        latent_anchor_active = latent_anchor_strength > 0.0
        if latent_anchor_active and str(latent_anchor_source) != 'step_diff':
            raise ValueError(f"`latent_anchor_source` only supports 'step_diff' currently, got {latent_anchor_source!r}.")
        if latent_anchor_step_indices is None:
            latent_anchor_step_indices_set = set(range(max(0, num_inference_steps - 1)))
        else:
            latent_anchor_step_indices_set = {int(x) for x in latent_anchor_step_indices}
        latent_anchor_feather = max(0, int(latent_anchor_feather))
        latent_anchor_mask_low_quantile = float(latent_anchor_mask_low_quantile)
        latent_anchor_mask_high_quantile = float(latent_anchor_mask_high_quantile)
        latent_anchor_mask_gamma = float(latent_anchor_mask_gamma)
        if not 0.0 <= latent_anchor_mask_low_quantile < latent_anchor_mask_high_quantile <= 1.0:
            raise ValueError(f'`latent_anchor_mask_low_quantile` and `latent_anchor_mask_high_quantile` must satisfy 0 <= low < high <= 1, got {latent_anchor_mask_low_quantile} and {latent_anchor_mask_high_quantile}.')
        if latent_anchor_mask_gamma <= 0.0:
            raise ValueError(f'`latent_anchor_mask_gamma` must be > 0, got {latent_anchor_mask_gamma}.')
        inversion_trajectory_clean_to_noise: Optional[List[torch.Tensor]] = [] if latent_anchor_active and source_inversion else None
        global_anchor_cache_in_steps: Optional[List[Dict[Any, torch.Tensor]]] = None
        global_anchor_cache_in_inversion: Optional[Dict[Any, torch.Tensor]] = None
        if cyclic_corr_global_anchor_cache is not None and isinstance(cyclic_corr_global_anchor_cache, dict):
            steps_in = cyclic_corr_global_anchor_cache.get('denoising_steps', None)
            if isinstance(steps_in, list):
                global_anchor_cache_in_steps = [s if isinstance(s, dict) else {} for s in steps_in]
            inv_in = cyclic_corr_global_anchor_cache.get('inversion_features', None)
            if isinstance(inv_in, dict):
                global_anchor_cache_in_inversion = inv_in
        global_anchor_cache_out_steps: Optional[List[Dict[Any, torch.Tensor]]] = [{} for _ in range(num_inference_steps)] if return_cyclic_corr_global_anchor_cache else None
        global_anchor_cache_out_inversion: Optional[Dict[Any, torch.Tensor]] = {} if return_cyclic_corr_global_anchor_cache else None
        self._last_source_inversion_time_sec = 0.0
        self._last_source_inversion_num_steps = 0
        if source_inversion:
            source_prompt_embeds = prompt_embeds
            source_text_ids = text_ids
            source_negative_prompt_embeds = None
            source_negative_text_ids = None
            if source_inversion_prompt is not None:
                source_prompt = source_inversion_prompt
                source_num_images_per_prompt = num_images_per_prompt
                if isinstance(source_prompt, str) and effective_batch_size > 1:
                    source_prompt = [source_prompt] * effective_batch_size
                    source_num_images_per_prompt = 1
                source_prompt_embeds, source_text_ids = self.encode_prompt(prompt=source_prompt, device=device, num_images_per_prompt=source_num_images_per_prompt, max_sequence_length=max_sequence_length)
            if source_inversion_guidance_scale > 1.0 and (not self.config.is_distilled):
                negative_source_prompt = ''
                if source_prompt_embeds.shape[0] > 1:
                    negative_source_prompt = [''] * source_prompt_embeds.shape[0]
                source_negative_prompt_embeds, source_negative_text_ids = self.encode_prompt(prompt=negative_source_prompt, device=device, num_images_per_prompt=1, max_sequence_length=max_sequence_length)
            if source_inversion_num_steps is None:
                inversion_sigmas = self.scheduler.sigmas[:len(timesteps) + 1]
            else:
                source_inversion_num_steps = int(source_inversion_num_steps)
                if source_inversion_num_steps < 1:
                    raise ValueError(f'`source_inversion_num_steps` must be >= 1 when inversion is enabled, got {source_inversion_num_steps}.')
                inversion_scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config)
                inversion_sigmas_arg = np.linspace(1.0, 1 / source_inversion_num_steps, source_inversion_num_steps)
                if hasattr(inversion_scheduler.config, 'use_flow_sigmas') and inversion_scheduler.config.use_flow_sigmas:
                    inversion_sigmas_arg = None
                inversion_mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=source_inversion_num_steps)
                retrieve_timesteps(inversion_scheduler, source_inversion_num_steps, device, sigmas=inversion_sigmas_arg, mu=inversion_mu)
                inversion_sigmas = inversion_scheduler.sigmas[:source_inversion_num_steps + 1]
            logger.info(f'[source_inversion] RF-Solver inversion for {video_latents.shape[0]} frame latents with {len(inversion_sigmas) - 1} steps.')
            source_inversion_attention_kwargs = None
            source_inversion_cyclic_corr_maps = None
            if base_attention_kwargs is not None:
                cyclic_keys = ('cyclic_corr_token_replacement', 'cyclic_corr_tau', 'cyclic_corr_delta', 'cyclic_corr_dropout_p', 'cyclic_corr_anchor_frame_num', 'cyclic_corr_extraction_layer_index', 'cyclic_corr_extraction_timestep')
                source_inversion_attention_kwargs = {key: base_attention_kwargs[key] for key in cyclic_keys if key in base_attention_kwargs}
                if source_inversion_attention_kwargs.get('cyclic_corr_token_replacement', False):
                    source_inversion_attention_kwargs['stable_layers'] = vital_layers
                    source_inversion_attention_kwargs['post_attn_layers'] = None
                    source_inversion_attention_kwargs['cyclic_corr_collect_maps'] = True
                    source_inversion_attention_kwargs['cyclic_corr_apply_replacement'] = False
                    source_inversion_cyclic_corr_maps = {}
                    source_inversion_attention_kwargs['cyclic_corr_maps_output'] = source_inversion_cyclic_corr_maps
                    if global_anchor_cache_in_inversion is not None:
                        source_inversion_attention_kwargs['cyclic_corr_inversion_anchor_features_in'] = global_anchor_cache_in_inversion
                    if global_anchor_cache_out_inversion is not None:
                        source_inversion_attention_kwargs['cyclic_corr_inversion_anchor_collect'] = global_anchor_cache_out_inversion
                else:
                    source_inversion_attention_kwargs = None
            inversion_t0 = None
            if source_inversion:
                if torch.cuda.is_available() and latents.device.type == 'cuda':
                    torch.cuda.synchronize(latents.device)
                inversion_t0 = time.perf_counter()
            inverted_source_latents = self.invert_source_latents_rf_solver(source_latents=video_latents, latent_ids=latent_ids, sigmas=inversion_sigmas, prompt_embeds=source_prompt_embeds, text_ids=source_text_ids, video_latents=video_latents, video_latent_ids=video_latent_ids, negative_prompt_embeds=source_negative_prompt_embeds, negative_text_ids=source_negative_text_ids, guidance_scale=source_inversion_guidance_scale, attention_kwargs=source_inversion_attention_kwargs, trajectory_output=inversion_trajectory_clean_to_noise)
            if source_inversion:
                latents = inverted_source_latents
                if torch.cuda.is_available() and latents.device.type == 'cuda':
                    torch.cuda.synchronize(latents.device)
                self._last_source_inversion_time_sec = time.perf_counter() - inversion_t0
                self._last_source_inversion_num_steps = len(inversion_sigmas) - 1
            if source_inversion_cyclic_corr_maps is not None and base_attention_kwargs is not None:
                collected_layers = source_inversion_cyclic_corr_maps.get('layers', {})
                if collected_layers and 'default' not in source_inversion_cyclic_corr_maps:
                    extraction_layer = base_attention_kwargs.get('cyclic_corr_extraction_layer_index', None)
                    if extraction_layer is not None and int(extraction_layer) in collected_layers:
                        source_inversion_cyclic_corr_maps['default'] = collected_layers[int(extraction_layer)]
                    elif len(collected_layers) == 1:
                        source_inversion_cyclic_corr_maps['default'] = next(iter(collected_layers.values()))
                num_corr_layers, num_valid_corr_tokens = self._summarize_cyclic_corr_maps(source_inversion_cyclic_corr_maps)
                if num_corr_layers == 0:
                    seen_layers = source_inversion_cyclic_corr_maps.get('seen_layers', [])
                    if isinstance(seen_layers, set):
                        seen_layers = sorted(seen_layers)
                    attempted_layers = source_inversion_cyclic_corr_maps.get('attempted_layers', [])
                    if isinstance(attempted_layers, set):
                        attempted_layers = sorted(attempted_layers)
                    raise RuntimeError(f'RF inversion correspondence strategy was requested, but no cyclic correspondence maps were collected. Check `--cyclic-corr-extraction-layer-index`, `--post-attn-layers`, and that `--cyclic-corr-token-replacement` is enabled. Seen extraction layers: {seen_layers}. Attempted cyclic layers: {attempted_layers}.')
                if num_valid_corr_tokens == 0:
                    logger.warning('[cyclic_corr] RF inversion collected maps from %d layer(s), but all valid masks are empty. Token replacement will be a no-op unless you lower `--cyclic-corr-tau`, increase `--cyclic-corr-delta`, or change the extraction timestep/layer.', num_corr_layers)
                else:
                    logger.info('[cyclic_corr] RF inversion collected %d valid token correspondences from %d layer(s).', num_valid_corr_tokens, num_corr_layers)
                base_attention_kwargs['cyclic_corr_maps_input'] = source_inversion_cyclic_corr_maps
        initial_latents = latents.clone()
        latent_anchor_forward_trajectory: Optional[List[torch.Tensor]] = None
        if latent_anchor_active:
            if inversion_trajectory_clean_to_noise is not None and len(inversion_trajectory_clean_to_noise) >= 2:
                latent_anchor_forward_trajectory = list(reversed(inversion_trajectory_clean_to_noise))
                expected_len = num_inference_steps + 1
                if len(latent_anchor_forward_trajectory) < expected_len:
                    logger.warning(f'[latent_anchor] RF inversion trajectory has {len(latent_anchor_forward_trajectory)} checkpoints but expected {expected_len}; falling back to linear interpolation.')
                    latent_anchor_forward_trajectory = None
            elif video_latents is None:
                logger.warning('[latent_anchor] enabled but no source video latents available; anchor will be a no-op.')
        if chunk_kv_cache_steps is not None and len(chunk_kv_cache_steps) != num_inference_steps:
            logger.warning(f'`chunk_kv_cache` has {len(chunk_kv_cache_steps)} steps but current call uses {num_inference_steps}; missing steps will be ignored.')
        output_chunk_kv_cache_steps = [{} for _ in range(num_inference_steps)] if return_chunk_kv_cache else None
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                step_latents = latents
                step_latent_ids = latent_ids
                step_video_latents = video_latents
                step_video_latent_ids = video_latent_ids
                step_image_latents = image_latents
                step_image_latent_ids = image_latent_ids
                step_prompt_embeds = prompt_embeds
                step_text_ids = text_ids
                step_negative_prompt_embeds = negative_prompt_embeds
                step_negative_text_ids = negative_text_ids
                timestep = t.expand(step_latents.shape[0]).to(step_latents.dtype)
                latent_model_input = step_latents.to(self.transformer.dtype)
                latent_image_ids = step_latent_ids
                latent_height = int(step_latent_ids[0, :, 1].max().item() + 1)
                latent_width = int(step_latent_ids[0, :, 2].max().item() + 1)
                latent_hw = (latent_height, latent_width)
                video_hw = None
                if step_video_latent_ids is not None:
                    video_height = int(step_video_latent_ids[0, :, 1].max().item() + 1)
                    video_width = int(step_video_latent_ids[0, :, 2].max().item() + 1)
                    video_hw = (video_height, video_width)
                text_seq_len = step_text_ids.shape[1]
                apply_frame_operation_mods = frame_operation_step_indices_set is None or i in frame_operation_step_indices_set
                apply_post_attn_mods = post_attn_step_indices_set is None or i in post_attn_step_indices_set
                apply_any_attention_mods = apply_frame_operation_mods or apply_post_attn_mods
                attention_kwargs = dict(base_attention_kwargs) if apply_any_attention_mods and base_attention_kwargs is not None else {}
                attention_kwargs.pop('cyclic_corr_extraction_timestep', None)
                step_chunk_kv_cache_in = None
                step_chunk_kv_cache_out = None
                if apply_frame_operation_mods:
                    if chunk_kv_cache_steps is not None and i < len(chunk_kv_cache_steps):
                        step_chunk_kv_cache_in = chunk_kv_cache_steps[i]
                    if return_chunk_kv_cache:
                        step_chunk_kv_cache_out = {}
                if apply_any_attention_mods:
                    if not apply_frame_operation_mods:
                        for key_name in ('frame_operation', 'frame_idx', 'frame_interval', 'context_prev_frames', 'append_all_prev_context'):
                            attention_kwargs.pop(key_name, None)
                    if not apply_post_attn_mods:
                        for key_name in ('cyclic_corr_token_replacement', 'cyclic_corr_tau', 'cyclic_corr_delta', 'cyclic_corr_dropout_p', 'cyclic_corr_anchor_frame_num', 'cyclic_corr_maps_input', 'cyclic_corr_maps_output', 'cyclic_corr_collect_maps', 'cyclic_corr_apply_replacement', 'cyclic_corr_global_anchor_features_in', 'cyclic_corr_global_anchor_collect'):
                            attention_kwargs.pop(key_name, None)
                    step_global_anchor_features_in: Optional[Dict[Any, torch.Tensor]] = None
                    if global_anchor_cache_in_steps is not None and i < len(global_anchor_cache_in_steps):
                        step_global_anchor_features_in = global_anchor_cache_in_steps[i] or None
                    step_global_anchor_collect: Optional[Dict[Any, torch.Tensor]] = None
                    if global_anchor_cache_out_steps is not None:
                        step_global_anchor_collect = global_anchor_cache_out_steps[i]
                    attention_kwargs.update(
                        {
                            'latent_hw': latent_hw,
                            'video_hw': video_hw,
                            'text_seq_len': text_seq_len,
                            'stable_layers': vital_layers,
                            'frame_operation_layers': frame_operation_layers,
                            'post_attn_layers': post_attn_layers,
                            'chunk_kv_cache_input': step_chunk_kv_cache_in,
                            'chunk_kv_cache_output': step_chunk_kv_cache_out,
                            'collect_chunk_kv_cache': bool(return_chunk_kv_cache and apply_frame_operation_mods),
                            'cyclic_corr_global_anchor_features_in': step_global_anchor_features_in,
                            'cyclic_corr_global_anchor_collect': step_global_anchor_collect,
                        }
                    )
                self._attention_kwargs = attention_kwargs
                if step_video_latents is not None:
                    latent_model_input = torch.cat([latent_model_input, step_video_latents.to(self.transformer.dtype)], dim=1)
                    latent_image_ids = torch.cat([latent_image_ids, step_video_latent_ids], dim=1)
                if step_image_latents is not None:
                    latent_model_input = torch.cat([latent_model_input, step_image_latents.to(self.transformer.dtype)], dim=1)
                    latent_image_ids = torch.cat([latent_image_ids, step_image_latent_ids], dim=1)
                cond_attention_kwargs = attention_kwargs
                if apply_frame_operation_mods and return_chunk_kv_cache:
                    cond_attention_kwargs = dict(attention_kwargs)
                    cond_attention_kwargs['collect_chunk_kv_cache'] = True
                with self.transformer.cache_context('cond'):
                    noise_pred = self.transformer(hidden_states=latent_model_input, timestep=timestep / 1000, guidance=None, encoder_hidden_states=step_prompt_embeds, txt_ids=step_text_ids, img_ids=latent_image_ids, joint_attention_kwargs=cond_attention_kwargs, return_dict=False)[0]
                noise_pred = noise_pred[:, :step_latents.size(1), :]
                if self.do_classifier_free_guidance:
                    uncond_attention_kwargs = attention_kwargs
                    if apply_frame_operation_mods and return_chunk_kv_cache:
                        uncond_attention_kwargs = dict(attention_kwargs)
                        uncond_attention_kwargs['collect_chunk_kv_cache'] = False
                    if uncond_attention_kwargs is attention_kwargs:
                        uncond_attention_kwargs = dict(attention_kwargs)
                    uncond_attention_kwargs['cyclic_corr_global_anchor_collect'] = None
                    with self.transformer.cache_context('uncond'):
                        neg_noise_pred = self.transformer(hidden_states=latent_model_input, timestep=timestep / 1000, guidance=None, encoder_hidden_states=step_negative_prompt_embeds, txt_ids=step_negative_text_ids, img_ids=latent_image_ids, joint_attention_kwargs=uncond_attention_kwargs, return_dict=False)[0]
                    neg_noise_pred = neg_noise_pred[:, :step_latents.size(1), :]
                if output_chunk_kv_cache_steps is not None:
                    output_chunk_kv_cache_steps[i] = step_chunk_kv_cache_out or {}
                if self.do_classifier_free_guidance:
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)
                if latent_anchor_active and video_latents is not None and (i in latent_anchor_step_indices_set):
                    if latent_anchor_forward_trajectory is not None and i + 1 < len(latent_anchor_forward_trajectory):
                        z_src_step = latent_anchor_forward_trajectory[i + 1].to(device=latents.device, dtype=latents.dtype)
                    else:
                        next_sigma_anchor = self.scheduler.sigmas[i + 1].to(device=latents.device, dtype=latents.dtype)
                        while next_sigma_anchor.ndim < latents.ndim:
                            next_sigma_anchor = next_sigma_anchor.unsqueeze(-1)
                        z_src_step = next_sigma_anchor * initial_latents.to(latents.dtype) + (1.0 - next_sigma_anchor) * video_latents.to(device=latents.device, dtype=latents.dtype)
                    pre_anchor_delta = (latents.float() - z_src_step.float()).abs().mean().item()
                    latents = self._apply_latent_anchor(latents=latents, z_src=z_src_step, latent_ids=latent_ids, strength=latent_anchor_strength, feather=latent_anchor_feather, mask_low_quantile=latent_anchor_mask_low_quantile, mask_high_quantile=latent_anchor_mask_high_quantile, mask_gamma=latent_anchor_mask_gamma)
                    logger.info(f'[latent_anchor] step={i} strength={latent_anchor_strength:.3f} feather={latent_anchor_feather} mask_q=({latent_anchor_mask_low_quantile:.3f},{latent_anchor_mask_high_quantile:.3f}) gamma={latent_anchor_mask_gamma:.3f} mean_abs(latents-z_src)_pre={pre_anchor_delta:.4f}')
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop('latents', latents)
                    prompt_embeds = callback_outputs.pop('prompt_embeds', prompt_embeds)
                if i == len(timesteps) - 1 or (i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()
        self._current_timestep = None
        latents = self._unpack_latents_with_ids(latents, latent_ids)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        if output_type == 'latent':
            image = latents
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()
        output_chunk_kv_cache = None
        if output_chunk_kv_cache_steps is not None:
            output_chunk_kv_cache = {'schema_version': 1, 'num_inference_steps': num_inference_steps, 'steps': output_chunk_kv_cache_steps}
        output_global_anchor_cache = None
        if return_cyclic_corr_global_anchor_cache:
            output_global_anchor_cache = {'schema_version': 1, 'num_inference_steps': num_inference_steps, 'denoising_steps': global_anchor_cache_out_steps or [], 'inversion_features': global_anchor_cache_out_inversion or {}}
        if not return_dict:
            outputs = [image]
            if output_chunk_kv_cache is not None:
                outputs.append(output_chunk_kv_cache)
            if output_global_anchor_cache is not None:
                outputs.append(output_global_anchor_cache)
            return tuple(outputs)
        return EditVidPipelineOutput(images=image, chunk_kv_cache=output_chunk_kv_cache, cyclic_corr_global_anchor_cache=output_global_anchor_cache)
