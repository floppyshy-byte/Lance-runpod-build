from __future__ import annotations

import base64
import gc
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import runpod

# Ensure /app is in path for Lance imports
sys.path.insert(0, "/app")

# Run setup first
from setup_models import setup_lance_models

setup_lance_models()

# Set CUDA allocator config
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import torch
from safetensors.torch import load_file
from transformers import set_seed
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

from config.config_factory import DataArguments, InferenceArguments, ModelArguments
from data.data_utils import add_special_tokens
from data.dataset_base import DataConfig, simple_custom_collate
from data.datasets_custom import ValidationDataset
from inference_lance import (
    apply_inference_defaults,
    clean_memory,
    init_from_model_path_if_needed,
    save_prompt_results,
    validate_on_fixed_batch,
)
from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer
from modeling.qwen2.modeling_qwen2 import Qwen2Config
from modeling.vae.wan.model import WanVideoVAE
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

# ---------------------------------------------------------------------------
# Constants (mirrored from common.gradio_utils.settings)
# ---------------------------------------------------------------------------
MODEL_VARIANT_VIDEO = "video"
MODEL_VARIANT_IMAGE = "image"
DEFAULT_VIT_TYPE = "qwen_2_5_vl_original"
DEFAULT_TIMESTEPS = 30
DEFAULT_TIMESTEP_SHIFT = 3.5
DEFAULT_CFG_TEXT_SCALE = 4.0
DEFAULT_RESOLUTION = "video_480p"
DEFAULT_IMAGE_RESOLUTION = "image_768res"
DEFAULT_HEIGHT = 352
DEFAULT_WIDTH = 640
DEFAULT_NUM_FRAMES = 97  # 8 seconds @ 12fps + 1
MAX_VIDEO_NUM_FRAMES = 12 * 10 + 1
TEXT_TEMPLATE = True
USE_KVCACHE = True

TASK_T2V = "t2v"
TASK_T2I = "t2i"
TASK_VIDEO_EDIT = "video_edit"
TASK_IMAGE_EDIT = "image_edit"
TASK_X2T_VIDEO = "x2t_video"
TASK_X2T_IMAGE = "x2t_image"
GENERATION_TASKS = {TASK_T2V, TASK_T2I, TASK_IMAGE_EDIT, TASK_VIDEO_EDIT}
UNDERSTANDING_TASKS = {TASK_X2T_VIDEO, TASK_X2T_IMAGE}
IMAGE_TASKS = {TASK_T2I, TASK_IMAGE_EDIT, TASK_X2T_IMAGE}
VIDEO_TASKS = {TASK_T2V, TASK_VIDEO_EDIT, TASK_X2T_VIDEO}
EDIT_TASKS = {TASK_IMAGE_EDIT, TASK_VIDEO_EDIT}

I2T_QA_SYSTEM_PROMPT = "View the image attentively and provide a suitable answer to the posed question."
V2T_QA_SYSTEM_PROMPT = "View the video  attentively and provide a suitable answer to the posed question."

PROMPT_JSON_FILENAME = "prompt.json"
RUN_RECORD_FILENAME = "generation_record.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_task(task: str) -> str:
    task = (task or TASK_T2V).strip().lower()
    mapping = {
        "video generation": TASK_T2V,
        "t2v": TASK_T2V,
        "text to video": TASK_T2V,
        "image generation": TASK_T2I,
        "t2i": TASK_T2I,
        "text to image": TASK_T2I,
        "video edit": TASK_VIDEO_EDIT,
        "video editing": TASK_VIDEO_EDIT,
        "image edit": TASK_IMAGE_EDIT,
        "image editing": TASK_IMAGE_EDIT,
        "video understanding": TASK_X2T_VIDEO,
        "v2t": TASK_X2T_VIDEO,
        "x2t_video": TASK_X2T_VIDEO,
        "image understanding": TASK_X2T_IMAGE,
        "i2t": TASK_X2T_IMAGE,
        "x2t_image": TASK_X2T_IMAGE,
    }
    result = mapping.get(task, "")
    if result not in GENERATION_TASKS | UNDERSTANDING_TASKS:
        raise ValueError(f"Unsupported task type: {task}")
    return result


def normalize_seed(seed: int) -> int:
    import random
    return random.randint(0, 2**31 - 1) if seed == -1 else int(seed)


def normalize_resolution_for_backend(resolution: str, task: str) -> str:
    internal_task = normalize_task(task)
    resolution = str(resolution or "").strip()
    if internal_task in IMAGE_TASKS:
        valid = {DEFAULT_IMAGE_RESOLUTION}
    elif internal_task in VIDEO_TASKS:
        valid = {"video_360p", "video_480p"}
    else:
        valid = {DEFAULT_RESOLUTION}
    return resolution if resolution in valid else (DEFAULT_IMAGE_RESOLUTION if internal_task in IMAGE_TASKS else DEFAULT_RESOLUTION)


def get_model_base_dir() -> Path:
    configured = os.getenv("LANCE_MODEL_BASE_DIR")
    return Path(configured).expanduser() if configured else Path("downloads")


def get_model_path(model_variant: str) -> Path:
    variant = model_variant.strip().lower()
    if variant in {"image", "t2i", "i2t"}:
        variant_dir = "Lance_3B"
        env_name = "LANCE_IMAGE_MODEL_PATH"
    else:
        variant_dir = "Lance_3B_Video"
        env_name = "LANCE_VIDEO_MODEL_PATH"
    configured = os.getenv(env_name)
    if configured:
        return Path(configured).expanduser()
    configured = os.getenv("LANCE_MODEL_PATH")
    if configured:
        return Path(configured).expanduser()
    return get_model_base_dir() / variant_dir


def create_request_json(
    task: str,
    prompt: str,
    input_video: Optional[str],
    input_image: Optional[str],
    system_prompt: Optional[str] = None,
    tmp_dir: Path = Path("/tmp/lance_inputs"),
) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prompt_file = tmp_dir / f"{task}_{timestamp}.json"

    if task == TASK_T2V:
        payload = {"000000.mp4": prompt}
    elif task == TASK_T2I:
        payload = {"000000.png": prompt}
    elif task == TASK_VIDEO_EDIT:
        if not input_video:
            raise ValueError("video_edit requires input_video")
        payload = {
            "000000": {
                "interleave_array": [prompt, input_video, input_video],
                "element_dtype_array": ["text", "video", "video"],
                "istarget_in_interleave": [0, 0, 1],
            }
        }
    elif task == TASK_IMAGE_EDIT:
        if not input_image:
            raise ValueError("image_edit requires input_image")
        payload = {
            "000000": {
                "interleave_array": [prompt, input_image, input_image],
                "element_dtype_array": ["text", "image", "image"],
                "istarget_in_interleave": [0, 0, 1],
            }
        }
    elif task == TASK_X2T_VIDEO:
        if not input_video:
            raise ValueError("x2t_video requires input_video")
        system_prompt = system_prompt or V2T_QA_SYSTEM_PROMPT
        payload = {
            "000000": {
                "interleave_array": [input_video, [system_prompt, prompt, ""]],
                "element_dtype_array": ["video", "text"],
                "istarget_in_interleave": [0, 1],
            }
        }
    elif task == TASK_X2T_IMAGE:
        if not input_image:
            raise ValueError("x2t_image requires input_image")
        system_prompt = system_prompt or I2T_QA_SYSTEM_PROMPT
        payload = {
            "000000": {
                "interleave_array": [input_image, [system_prompt, prompt, ""]],
                "element_dtype_array": ["image", "text"],
                "istarget_in_interleave": [0, 1],
            }
        }
    else:
        raise ValueError(f"Unsupported task type: {task}")

    with prompt_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return prompt_file


def build_save_dir(task: str, base_dir: Path = Path("/tmp/lance_results")) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{task}_{timestamp}_{int(time.time() * 1000) % 1000:03d}"


def find_generated_video(save_dir: Path) -> Optional[Path]:
    videos = sorted(save_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return videos[0] if videos else None


def find_generated_image(save_dir: Path) -> Optional[Path]:
    images = sorted(save_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return images[0] if images else None


def extract_text_result(save_dir: Path) -> str:
    prompt_result_path = save_dir / PROMPT_JSON_FILENAME
    if not prompt_result_path.exists():
        return ""
    with prompt_result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return ""
    first_value = next(iter(data.values()))
    return first_value if isinstance(first_value, str) else json.dumps(first_value, ensure_ascii=False)


def decode_base64_to_file(b64_data: str, suffix: str, tmp_dir: Path = Path("/tmp/lance_inputs")) -> str:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    data = base64.b64decode(b64_data)
    path = tmp_dir / f"input_{int(time.time() * 1000)}_{hash(b64_data) % 10000:04d}{suffix}"
    path.write_bytes(data)
    return str(path)


def encode_file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Lance Pipeline Wrapper
# ---------------------------------------------------------------------------

class LancePipeline:
    """Simplified pipeline for RunPod serverless (single GPU, one model variant)."""

    def __init__(self, device_id: int, model_variant: str) -> None:
        self._init_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self.initialized = False
        self.device = device_id
        self.model_variant = model_variant.strip().lower()
        if self.model_variant in {"image", "t2i", "i2t"}:
            self.model_variant = MODEL_VARIANT_IMAGE
        else:
            self.model_variant = MODEL_VARIANT_VIDEO

        self.model: Optional[Lance] = None
        self.vae_model: Optional[WanVideoVAE] = None
        self.vae_config = None
        self.tokenizer: Optional[Qwen2Tokenizer] = None
        self.new_token_ids: Optional[dict] = None
        self.image_token_id: Optional[int] = None
        self.base_model_args: Optional[ModelArguments] = None
        self.base_data_args: Optional[DataArguments] = None
        self.base_inference_args: Optional[InferenceArguments] = None

    def _log_stage(self, stage_name: str, start_time: float, extra: str = "") -> None:
        elapsed = time.perf_counter() - start_time
        suffix = f" | {extra}" if extra else ""
        print(f"[startup][gpu:{self.device}] {stage_name} done in {elapsed:.2f}s{suffix}", flush=True)

    def initialize(self) -> None:
        with self._init_lock:
            if self.initialized:
                return

            model_path = str(get_model_path(self.model_variant))
            print(f"[startup][gpu:{self.device}][{self.model_variant}] Using Lance model path: {model_path}", flush=True)

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable.")
            if self.device >= torch.cuda.device_count():
                raise RuntimeError(f"GPU {self.device} unavailable. Detected {torch.cuda.device_count()} GPU(s).")
            torch.cuda.set_device(self.device)

            model_args = ModelArguments(
                model_path=model_path,
                vit_type=DEFAULT_VIT_TYPE,
                llm_qk_norm=True,
                llm_qk_norm_und=True,
                llm_qk_norm_gen=True,
                tie_word_embeddings=False,
                max_num_frames=MAX_VIDEO_NUM_FRAMES,
                max_latent_size=64,
                latent_patch_size=[1, 1, 1],
            )
            data_args = DataArguments()
            inference_args = InferenceArguments(
                validation_num_timesteps=DEFAULT_TIMESTEPS,
                validation_timestep_shift=DEFAULT_TIMESTEP_SHIFT,
                copy_init_moe=True,
                visual_und=True,
                visual_gen=True,
                vae_model_type="wan",
                apply_qwen_2_5_vl_pos_emb=True,
                apply_chat_template=False,
                cfg_type=0,
                validation_data_seed=42,
                video_height=DEFAULT_HEIGHT,
                video_width=DEFAULT_WIDTH,
                num_frames=DEFAULT_NUM_FRAMES,
                task=TASK_T2V,
                save_path_gen=str(Path("/tmp/lance_results")),
                resolution=DEFAULT_RESOLUTION,
                text_template=TEXT_TEMPLATE,
                use_KVcache=USE_KVCACHE,
            )
            apply_inference_defaults(model_args, data_args, inference_args)
            inference_args.validation_noise_seed = inference_args.validation_data_seed

            self.base_model_args = model_args
            self.base_data_args = data_args
            self.base_inference_args = inference_args

            set_seed(inference_args.global_seed)

            stage_start = time.perf_counter()
            llm_config: Qwen2Config = Qwen2Config.from_json_file(str(Path(model_args.model_path) / "llm_config.json"))
            self._log_stage("LLM config load", stage_start)

            llm_config.layer_module = model_args.layer_module
            llm_config.qk_norm = model_args.llm_qk_norm
            llm_config.qk_norm_und = model_args.llm_qk_norm_und
            llm_config.qk_norm_gen = model_args.llm_qk_norm_gen
            llm_config.tie_word_embeddings = model_args.tie_word_embeddings
            llm_config.freeze_und = inference_args.freeze_und
            llm_config.apply_qwen_2_5_vl_pos_emb = inference_args.apply_qwen_2_5_vl_pos_emb

            stage_start = time.perf_counter()
            language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)
            self._log_stage("LLM weight init", stage_start)

            vit_model = None
            vit_config = None
            if inference_args.visual_und:
                stage_start = time.perf_counter()
                vit_config = Qwen2_5_VLVisionConfig.from_pretrained(model_args.vit_path)
                self._log_stage("VIT config load", stage_start)

                stage_start = time.perf_counter()
                vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
                vit_weights = load_file(str(Path(model_args.vit_path) / "vit.safetensors"))
                vit_model.load_state_dict(vit_weights, strict=True)
                self._log_stage("VIT weight load", stage_start)
                clean_memory(vit_weights)

            if inference_args.visual_gen:
                stage_start = time.perf_counter()
                vae_model = WanVideoVAE()
                vae_config = deepcopy(vae_model.vae_config)
                self._log_stage("VAE init", stage_start)
            else:
                vae_model = None
                vae_config = None

            config = LanceConfig(
                visual_gen=inference_args.visual_gen,
                visual_und=inference_args.visual_und,
                llm_config=llm_config,
                vit_config=vit_config if inference_args.visual_und else None,
                vae_config=vae_config if inference_args.visual_gen else None,
                latent_patch_size=model_args.latent_patch_size,
                max_num_frames=model_args.max_num_frames,
                max_latent_size=model_args.max_latent_size,
                vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
                connector_act=model_args.connector_act,
                interpolate_pos=model_args.interpolate_pos,
                timestep_shift=inference_args.timestep_shift,
            )
            model: Lance = Lance(
                language_model=language_model,
                vit_model=vit_model if inference_args.visual_und else None,
                vit_type=model_args.vit_type,
                config=config,
                training_args=inference_args,
            )

            stage_start = time.perf_counter()
            model = model.to(dtype=torch.bfloat16)
            self._log_stage("Lance model bf16 cast", stage_start)

            stage_start = time.perf_counter()
            tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
            tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
            self._log_stage("tokenizer load", stage_start, extra=f"num_new_tokens={num_new_tokens}")

            if inference_args.copy_init_moe:
                language_model.init_moe()

            init_from_model_path_if_needed(model, model_args)

            if num_new_tokens > 0:
                model.language_model.resize_token_embeddings(len(tokenizer))
                model.config.llm_config.vocab_size = len(tokenizer)
                model.language_model.config.vocab_size = len(tokenizer)

            if model_args.vit_type.lower() == "qwen2_5_vl":
                from common.model.hacks import hack_qwen2_5_vl_config
                language_model = hack_qwen2_5_vl_config(language_model)

            image_token_id = language_model.config.video_token_id
            new_token_ids.update({"image_token_id": image_token_id})
            model.update_tokenizer(tokenizer=tokenizer)

            if model_args.tie_word_embeddings:
                model.language_model.untie_lm_head()
                model.language_model.copy_new_token_rows_to_lm_head(num_new_tokens)
                model_args.tie_word_embeddings = False
                llm_config.tie_word_embeddings = False
            else:
                assert (
                    model.language_model.get_input_embeddings().weight.data.data_ptr()
                    != model.language_model.get_output_embeddings().weight.data.data_ptr()
                ), "tie_word_embeddings conflict"

            stage_start = time.perf_counter()
            model = model.to(device=self.device)
            self._log_stage("Lance model move to GPU", stage_start)
            model.eval()
            if vae_model is not None and hasattr(vae_model, "eval"):
                vae_model.eval()

            self.model = model
            self.vae_model = vae_model
            self.vae_config = vae_config
            self.tokenizer = tokenizer
            self.new_token_ids = new_token_ids
            self.image_token_id = image_token_id
            self.initialized = True
            print(
                f"[startup][gpu:{self.device}][{self.model_variant}] Lance model loaded and ready.",
                flush=True,
            )

    def _build_request_batch(
        self,
        prompt_file: Path,
        model_args: ModelArguments,
        data_args: DataArguments,
        inference_args: InferenceArguments,
    ):
        assert self.tokenizer is not None
        assert self.new_token_ids is not None
        assert self.vae_config is not None

        from common.utils.misc import tuple_mul

        dataset_config = DataConfig.from_yaml(str(prompt_file))
        if inference_args.visual_und:
            dataset_config.vit_patch_size = model_args.vit_patch_size
            dataset_config.vit_patch_size_temporal = model_args.vit_patch_size_temporal
            dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side
        if inference_args.visual_gen:
            vae_downsample = tuple_mul(
                tuple(model_args.latent_patch_size),
                (
                    self.vae_config.downsample_temporal,
                    self.vae_config.downsample_spatial,
                    self.vae_config.downsample_spatial,
                ),
            )
            dataset_config.latent_patch_size = model_args.latent_patch_size
            dataset_config.vae_downsample = vae_downsample
            dataset_config.max_latent_size = model_args.max_latent_size
            dataset_config.max_num_frames = model_args.max_num_frames

        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

        dataset_config.num_frames = inference_args.num_frames
        dataset_config.H = inference_args.video_height
        dataset_config.W = inference_args.video_width
        dataset_config.task = inference_args.task
        dataset_config.resolution = inference_args.resolution
        dataset_config.text_template = inference_args.text_template

        val_dataset = ValidationDataset(
            jsonl_path=str(prompt_file),
            tokenizer=self.tokenizer,
            data_args=data_args,
            model_args=model_args,
            training_args=inference_args,
            new_token_ids=self.new_token_ids,
            dataset_config=dataset_config,
            local_rank=0,
            world_size=1,
        )
        return simple_custom_collate([val_dataset[0]])

    def generate(
        self,
        task: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        input_video: Optional[str] = None,
        input_image: Optional[str] = None,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_frames: int = DEFAULT_NUM_FRAMES,
        seed: int = 42,
        resolution: str = DEFAULT_RESOLUTION,
        num_inference_steps: int = DEFAULT_TIMESTEPS,
        timestep_shift: float = DEFAULT_TIMESTEP_SHIFT,
        cfg_scale: float = DEFAULT_CFG_TEXT_SCALE,
    ):
        self.initialize()
        internal_task = normalize_task(task)
        prompt = (prompt or "").strip()

        if internal_task in GENERATION_TASKS and not prompt:
            raise ValueError("Prompt is required for generation tasks.")
        if internal_task in UNDERSTANDING_TASKS and not prompt:
            raise ValueError("Question is required for understanding tasks.")
        if internal_task in {TASK_VIDEO_EDIT, TASK_X2T_VIDEO} and not input_video:
            raise ValueError("input_video is required for this task.")
        if internal_task in {TASK_IMAGE_EDIT, TASK_X2T_IMAGE} and not input_image:
            raise ValueError("input_image is required for this task.")

        torch.cuda.set_device(self.device)
        actual_seed = normalize_seed(seed)
        prompt_file = create_request_json(
            task=internal_task,
            prompt=prompt,
            input_video=input_video,
            input_image=input_image,
            system_prompt=system_prompt,
        )
        save_dir = build_save_dir(internal_task)
        save_dir.mkdir(parents=True, exist_ok=True)

        request_model_args = deepcopy(self.base_model_args)
        request_model_args.cfg_text_scale = float(cfg_scale)

        request_data_args = deepcopy(self.base_data_args)
        request_data_args.val_dataset_config_file = str(prompt_file)

        request_inference_args = deepcopy(self.base_inference_args)
        request_inference_args.validation_num_timesteps = int(num_inference_steps)
        request_inference_args.validation_timestep_shift = float(timestep_shift)
        request_inference_args.validation_data_seed = actual_seed
        request_inference_args.validation_noise_seed = actual_seed
        request_inference_args.video_height = int(height)
        request_inference_args.video_width = int(width)
        request_inference_args.num_frames = int(num_frames)
        request_inference_args.resolution = normalize_resolution_for_backend(resolution, internal_task)
        request_inference_args.save_path_gen = str(save_dir)
        request_inference_args.task = internal_task
        request_inference_args.text_template = TEXT_TEMPLATE
        request_inference_args.prompt_data_dict = {}

        print(
            f"[inference] task={internal_task} gpu={self.device} seed={actual_seed} "
            f"size={height}x{width} frames={num_frames} resolution={request_inference_args.resolution}",
            flush=True,
        )

        with self._generate_lock:
            clean_memory()
            generate_start = time.perf_counter()
            val_data_cpu = self._build_request_batch(
                prompt_file=prompt_file,
                model_args=request_model_args,
                data_args=request_data_args,
                inference_args=request_inference_args,
            )
            validate_on_fixed_batch(
                fsdp_model=self.model,
                vae_model=self.vae_model,
                tokenizer=self.tokenizer,
                val_data_cpu=val_data_cpu,
                training_args=request_inference_args,
                model_args=request_model_args,
                inference_args=request_inference_args,
                new_token_ids=self.new_token_ids,
                image_token_id=self.image_token_id,
                device=self.device,
                save_source_video=False,
                save_path_gen=request_inference_args.save_path_gen,
                save_path_gt="",
            )
            elapsed = time.perf_counter() - generate_start
            save_prompt_results(request_inference_args.prompt_data_dict, request_inference_args.save_path_gen, None)
            clean_memory()

        video_path = find_generated_video(save_dir) if internal_task in {TASK_T2V, TASK_VIDEO_EDIT} else None
        image_path = find_generated_image(save_dir) if internal_task in {TASK_T2I, TASK_IMAGE_EDIT} else None
        text_result = extract_text_result(save_dir) if internal_task in UNDERSTANDING_TASKS else ""

        print(f"[inference] Completed in {elapsed:.2f}s", flush=True)

        return {
            "video_path": str(video_path) if video_path else None,
            "image_path": str(image_path) if image_path else None,
            "text_result": text_result,
            "elapsed_seconds": round(elapsed, 3),
            "save_dir": str(save_dir),
        }


# ---------------------------------------------------------------------------
# Global pipeline registry (lazy init per variant)
# ---------------------------------------------------------------------------

_pipelines: dict[str, LancePipeline] = {}
_pipeline_lock = threading.Lock()


def get_pipeline(model_variant: str) -> LancePipeline:
    variant = model_variant.strip().lower()
    if variant in {"image", "t2i", "i2t"}:
        key = MODEL_VARIANT_IMAGE
    else:
        key = MODEL_VARIANT_VIDEO

    with _pipeline_lock:
        if key not in _pipelines:
            _pipelines[key] = LancePipeline(device_id=0, model_variant=key)
        return _pipelines[key]


def task_to_variant(task: str) -> str:
    internal = normalize_task(task)
    if internal in IMAGE_TASKS:
        return MODEL_VARIANT_IMAGE
    return MODEL_VARIANT_VIDEO


# ---------------------------------------------------------------------------
# RunPod Handler
# ---------------------------------------------------------------------------

def handler(event):
    try:
        input_data = event.get("input", {})

        task = input_data.get("task", "t2v")
        prompt = input_data.get("prompt", "")
        system_prompt = input_data.get("system_prompt")
        height = int(input_data.get("height", DEFAULT_HEIGHT))
        width = int(input_data.get("width", DEFAULT_WIDTH))
        num_frames = int(input_data.get("num_frames", DEFAULT_NUM_FRAMES))
        seed = int(input_data.get("seed", 42))
        resolution = input_data.get("resolution", DEFAULT_RESOLUTION)
        num_inference_steps = int(input_data.get("num_inference_steps", DEFAULT_TIMESTEPS))
        timestep_shift = float(input_data.get("timestep_shift", DEFAULT_TIMESTEP_SHIFT))
        cfg_scale = float(input_data.get("cfg_scale", DEFAULT_CFG_TEXT_SCALE))

        # Decode base64 inputs if provided
        input_video = None
        input_image = None
        if "input_video_base64" in input_data:
            input_video = decode_base64_to_file(input_data["input_video_base64"], ".mp4")
        elif "input_video" in input_data and input_data["input_video"]:
            input_video = input_data["input_video"]

        if "input_image_base64" in input_data:
            input_image = decode_base64_to_file(input_data["input_image_base64"], ".png")
        elif "input_image" in input_data and input_data["input_image"]:
            input_image = input_data["input_image"]

        variant = task_to_variant(task)
        pipeline = get_pipeline(variant)

        result = pipeline.generate(
            task=task,
            prompt=prompt,
            system_prompt=system_prompt,
            input_video=input_video,
            input_image=input_image,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            resolution=resolution,
            num_inference_steps=num_inference_steps,
            timestep_shift=timestep_shift,
            cfg_scale=cfg_scale,
        )

        # Build response
        response = {
            "task": task,
            "seed": seed,
            "elapsed_seconds": result["elapsed_seconds"],
        }

        if result["video_path"]:
            video_path = Path(result["video_path"])
            response["output_type"] = "video"
            response["media_type"] = "video/mp4"
            response["output_base64"] = encode_file_to_base64(video_path)
            response["output_filename"] = video_path.name
        elif result["image_path"]:
            image_path = Path(result["image_path"])
            response["output_type"] = "image"
            response["media_type"] = "image/png"
            response["output_base64"] = encode_file_to_base64(image_path)
            response["output_filename"] = image_path.name
        elif result["text_result"]:
            response["output_type"] = "text"
            response["text"] = result["text_result"]
        else:
            response["output_type"] = "none"
            response["warning"] = "No output was generated."

        return response

    except Exception as exc:
        print(traceback.format_exc(), flush=True)
        return {"error": str(exc), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    print("[Handler] Starting Lance RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
