"""
Trình bao LLM - tải các model Qwen bằng HuggingFace transformers.
Hỗ trợ các vai trò executor, classifier, optimizer và verifier.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger("llm")
_lock = threading.Lock()
_loaded_models: Dict[str, "LLMWrapper"] = {}


def _normalize_device(device: str) -> str:
    device = str(device).strip().lower()
    if device == "gpu":
        return "cuda"
    return device


class LLMWrapper:
    """Trình bao mỏng quanh một checkpoint causal-LM của HuggingFace."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info(f"Đang tải tokenizer từ {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        requested_device = _normalize_device(device)
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA not available! Cannot load model {model_path} on CUDA. "
                "Please install a CUDA-enabled PyTorch build and check NVIDIA drivers."
            )
        target_device = "cuda" if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if target_device == "cuda" else torch.float32
        device_map = "auto" if target_device == "cuda" else None

        logger.info(f"Đang tải model từ {model_path} (device={target_device})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            quantization_config=None,
        )
        if target_device == "cpu":
            self.model.to("cpu")
        self.model.eval()
        logger.info(f"Đã tải model: {Path(model_path).name}")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_strings: Optional[List[str]] = None,
    ) -> str:
        """Sinh phản hồi cho prompt đầu vào."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = "\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in messages)
            text += "\n[ASSISTANT]:"

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature or self.temperature,
            top_p=top_p or self.top_p,
            do_sample=(temperature or self.temperature) > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
        )

        new_tokens = output[0][input_ids.shape[-1]:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if stop_strings:
            for stop in stop_strings:
                if stop in decoded:
                    decoded = decoded[: decoded.index(stop)]

        return decoded


def load_llm(variant: str = "executor", config_override: Optional[Dict] = None) -> LLMWrapper:
    """Tải và cache model theo vai trò: executor, classifier, optimizer, verifier."""
    global _loaded_models

    if variant in _loaded_models:
        return _loaded_models[variant]

    with _lock:
        if variant in _loaded_models:
            return _loaded_models[variant]

        cfg = get_config()
        if variant in ("executor", "classifier", "optimizer", "verifier"):
            section = cfg.get_section("llm").get(variant, {})
        else:
            raise ValueError(f"Vai trò LLM không hợp lệ: {variant}")

        if config_override:
            section.update(config_override)

        wrapper = LLMWrapper(
            model_path=section["model_path"],
            max_new_tokens=section.get("max_new_tokens", 1024),
            temperature=section.get("temperature", 0.7),
            top_p=section.get("top_p", 0.9),
            device=section.get("device", "cuda"),
        )
        _loaded_models[variant] = wrapper
        return wrapper


def unload_llm(variant: str) -> None:
    """Gỡ model khỏi cache để giải phóng bộ nhớ."""
    if variant in _loaded_models:
        del _loaded_models[variant]
        torch.cuda.empty_cache()
        logger.info(f"Đã gỡ model: {variant}")
