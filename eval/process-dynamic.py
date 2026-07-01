import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import signal
import statistics
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress FFmpeg/codec and transformers warning noise before video imports.
os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")
os.environ.setdefault("AV_LOG_LEVEL", "error")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
from datasets import load_dataset
from tqdm import tqdm
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*unused.*")
warnings.filterwarnings("ignore", message=".*av1.*")
warnings.filterwarnings("ignore", message=".*Missing Sequence Header.*")
warnings.filterwarnings("ignore", message=".*mmco: unref short failure.*")
warnings.filterwarnings("ignore", message=".*h264.*")
warnings.filterwarnings("ignore", message=".*codec.*")

from sage.main import SAGE
from sage.src.context_vlm import ContextVLM
from sage.src.functions.utils.temporal import get_video_duration
from sage.src.functions.utils.transcribe import transcribe_video
from sage.src.functions.utils.utils import check_api_health
from sage.utils.utils import SAMPLED_BASELINE_MINERVA_PROMPT, SAMPLED_BASELINE_PROMPT

TOOL_CALL_MODEL = os.environ.get("TOOL_CALL_MODEL", "None")
VLLM_CLIENT_URL = os.environ.get("VLLM_CLIENT_URL", "None")
TRANSCRIBE_API_URL = os.environ.get("TRANSCRIBE_API_URL", "None")
USE_ASR = os.environ.get("USE_ASR", "True").lower() == "true"

MINERVA_DURATION_BUCKETS = [
    (0, 600, "0-600s"),
    (600, float("inf"), "600+s"),
]

SAGE_BENCH_DURATION_BUCKETS = [
    (0, 60, "0-60s"),
    (60, 180, "60-180s"),
    (180, 300, "180-300s"),
    (300, 600, "300-600s"),
    (600, 1200, "600-1200s"),
    (1200, 2400, "1200-2400s"),
    (2400, float("inf"), "2400+s"),
]

# Native Gemini Developer API paid-tier prices for gemini-2.5-flash.
# Input rate applies to text/image/video input tokens; output includes thinking tokens.
GEMINI_25_FLASH_INPUT_USD_PER_1M = 0.30
GEMINI_25_FLASH_OUTPUT_USD_PER_1M = 2.50


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def check_tool_call_model() -> None:
    if TOOL_CALL_MODEL and TOOL_CALL_MODEL != "None":
        urls = [url.strip() for url in VLLM_CLIENT_URL.split(",") if url.strip()]
        for url in urls:
            check_api_health(url, "VLLM client")


def split_list(items: List[Any], n: int) -> List[List[Any]]:
    if len(items) == 0:
        return [[] for _ in range(n)]
    if len(items) < n:
        return [[items[i]] if i < len(items) else [] for i in range(n)]
    chunk_size = math.ceil(len(items) / n)
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_chunk(items: List[Any], n: int, k: int) -> List[Any]:
    chunks = split_list(items, n)
    return [] if k >= len(chunks) else chunks[k]


@contextmanager
def timeout(seconds: int):
    if not hasattr(signal, "SIGALRM"):
        yield  # no-op on Windows; rely on per-sample retry logic
        return
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def count_id_tokens(value: Any) -> int:
    """Count actual integer token IDs in a nested token-ID structure."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return 1
    if isinstance(value, (list, tuple)):
        return sum(count_id_tokens(item) for item in value)
    return 0


def extract_local_token_usage(complete_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract token IDs returned by SAGE-MM when run_inference(return_ids=True) is enabled."""
    turns = []
    if not isinstance(complete_results, dict):
        return {
            "local_token_turns": [],
            "local_prompt_tokens": None,
            "local_completion_tokens": None,
            "local_total_tokens": None,
        }

    for idx, context_item in enumerate(complete_results.get("context_vlm", []) or []):
        if isinstance(context_item, dict):
            turns.append({
                "turn_type": "context_vlm",
                "turn_index": idx,
                "prompt_tokens": count_id_tokens(context_item.get("prompt_ids")),
                "completion_tokens": count_id_tokens(context_item.get("completion_ids")),
            })

    for idx, reasoner_item in enumerate(complete_results.get("iterative_reasoner", []) or []):
        if isinstance(reasoner_item, dict) and (
            "prompt_ids" in reasoner_item or "completion_ids" in reasoner_item
        ):
            turns.append({
                "turn_type": "iterative_reasoner",
                "turn_index": idx,
                "prompt_tokens": count_id_tokens(reasoner_item.get("prompt_ids")),
                "completion_tokens": count_id_tokens(reasoner_item.get("completion_ids")),
            })

    if not turns:
        return {
            "local_token_turns": [],
            "local_prompt_tokens": None,
            "local_completion_tokens": None,
            "local_total_tokens": None,
        }

    prompt_tokens = sum(turn["prompt_tokens"] for turn in turns)
    completion_tokens = sum(turn["completion_tokens"] for turn in turns)
    return {
        "local_token_turns": turns,
        "local_prompt_tokens": prompt_tokens,
        "local_completion_tokens": completion_tokens,
        "local_total_tokens": prompt_tokens + completion_tokens,
    }


def extract_tool_metrics(
    complete_results: Optional[Dict[str, Any]],
    original_tool_names: set[str],
    declared_synthesized_names: set[str],
) -> Dict[str, Any]:
    counts: Counter = Counter()
    if isinstance(complete_results, dict):
        for step in complete_results.get("iterative_reasoner", []) or []:
            if not isinstance(step, dict):
                continue
            tool_calls = step.get("tool_calls", {}) or {}
            if isinstance(tool_calls, dict):
                for tool_key in tool_calls.keys():
                    base_name = str(tool_key).split("_#")[0]
                    if base_name != "None":
                        counts[base_name] += 1

    synthesized_counts = Counter()
    original_counts = Counter()
    for name, count in counts.items():
        if name in declared_synthesized_names or name not in original_tool_names:
            synthesized_counts[name] += count
        else:
            original_counts[name] += count

    turns = (
        int(complete_results.get("num_iterative_reasoner_calls", 0))
        if isinstance(complete_results, dict)
        else 0
    )
    context_calls = (
        len(complete_results.get("context_vlm", []) or [])
        if isinstance(complete_results, dict)
        else 0
    )
    reasoner_tool_vlm_calls = (
        counts.get("perform_reasoning", 0)
        + counts.get("identify_timestamps_visually", 0)
    )

    return {
        "num_turns": turns,
        "total_tool_calls": sum(counts.values()),
        "tool_call_counts": dict(counts),
        "tool_names": sorted(counts.keys()),
        "original_tool_calls": sum(original_counts.values()),
        "original_tool_call_counts": dict(original_counts),
        "original_tool_names": sorted(original_counts.keys()),
        "synthesized_tool_calls": sum(synthesized_counts.values()),
        "synthesized_tool_call_counts": dict(synthesized_counts),
        "synthesized_tool_names": sorted(synthesized_counts.keys()),
        "context_vlm_calls": context_calls,
        "iterative_reasoner_calls": turns,
        "visual_tool_vlm_calls": reasoner_tool_vlm_calls,
        "logical_vlm_calls_total": context_calls + turns + reasoner_tool_vlm_calls,
    }


def detect_empty_media_handoff_failure(
    complete_results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Detect the released-runtime failure mode where an extraction tool returns
    non-empty media paths but a later perform_reasoning tool call receives
    media_paths=[]. This records the behavior without changing inference.
    """
    extracted_media_paths: List[str] = []
    perform_reasoning_calls = 0
    empty_media_reasoning_calls = 0

    if isinstance(complete_results, dict):
        for step in complete_results.get("iterative_reasoner", []) or []:
            if not isinstance(step, dict):
                continue
            tool_calls = step.get("tool_calls", {}) or {}
            if not isinstance(tool_calls, dict):
                continue

            for tool_key, call in tool_calls.items():
                base_name = str(tool_key).split("_#")[0]
                call = call or {}
                result = call.get("result", {}) or {}
                arguments = call.get("arguments", {}) or {}

                if base_name == "extract_parts_from_timestamp" and isinstance(result, dict):
                    paths = result.get("media_paths", []) or []
                    if isinstance(paths, list) and paths:
                        extracted_media_paths.extend(paths)

                if base_name == "perform_reasoning":
                    perform_reasoning_calls += 1
                    media_paths = arguments.get("media_paths", []) or []
                    if isinstance(media_paths, list) and len(media_paths) == 0:
                        empty_media_reasoning_calls += 1

    extracted_media_available = len(extracted_media_paths) > 0
    empty_media_handoff_failure = (
        extracted_media_available and empty_media_reasoning_calls > 0
    )
    return {
        "extracted_media_available": extracted_media_available,
        "extracted_media_path_count": len(extracted_media_paths),
        "perform_reasoning_calls": perform_reasoning_calls,
        "empty_media_reasoning_calls": empty_media_reasoning_calls,
        "empty_media_handoff_failure": empty_media_handoff_failure,
    }


def get_usage_value(usage: Any, key: str) -> Optional[float]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


class _TrackedCompletions:
    """Proxy for OpenRouter/OpenAI-compatible tool calls; logs returned token/cost usage."""

    def __init__(self, inner: Any, processor: "Processor"):
        self._inner = inner
        self._processor = processor

    def create(self, *args, **kwargs):
        response = self._inner.create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sample_id": self._processor.current_sample_id,
            "model": kwargs.get("model"),
            "prompt_tokens": get_usage_value(usage, "prompt_tokens"),
            "completion_tokens": get_usage_value(usage, "completion_tokens"),
            "total_tokens": get_usage_value(usage, "total_tokens"),
            "cost": get_usage_value(usage, "cost"),
            "response_id": getattr(response, "id", None),
        }
        self._processor.current_tool_api_usage.append(entry)
        self._processor.run_tool_api_usage.append(entry)
        return response

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class _TrackedChat:
    def __init__(self, inner: Any, processor: "Processor"):
        self._inner = inner
        self.completions = _TrackedCompletions(inner.completions, processor)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class _TrackedOpenAIClient:
    def __init__(self, inner: Any, processor: "Processor"):
        self._inner = inner
        self.chat = _TrackedChat(inner.chat, processor)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def _usage_int(usage: Any, field_name: str) -> Optional[int]:
    value = get_usage_value(usage, field_name)
    return int(value) if isinstance(value, (int, float)) else None


def _gemini_flash_cost_usd(
    prompt_tokens: Optional[int],
    output_tokens: Optional[int],
    thinking_tokens: Optional[int],
) -> Optional[float]:
    if prompt_tokens is None and output_tokens is None and thinking_tokens is None:
        return None
    billed_input = prompt_tokens or 0
    billed_output = (output_tokens or 0) + (thinking_tokens or 0)
    return (
        billed_input * GEMINI_25_FLASH_INPUT_USD_PER_1M
        + billed_output * GEMINI_25_FLASH_OUTPUT_USD_PER_1M
    ) / 1_000_000


class _TrackedRequestsProxy:
    """Proxy only for sage.src.functions.utils.transcribe.requests, not all HTTP requests."""

    def __init__(self, inner: Any, processor: "Processor"):
        self._inner = inner
        self._processor = processor

    def post(self, url, *args, **kwargs):
        if str(url).rstrip("/").endswith("/transcribe"):
            payload = kwargs.get("json") or {}
            event = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "sample_id": self._processor.current_sample_id,
                "filepath": payload.get("filepath"),
                "url": str(url),
            }
            self._processor.run_whisper_events.append(event)
            if self._processor.current_sample_id is not None:
                self._processor.current_whisper_events.append(event)
        return self._inner.post(url, *args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def sum_present(values: List[Optional[float]]) -> Optional[float]:
    present = [value for value in values if isinstance(value, (int, float))]
    return sum(present) if present else None


class Processor:
    def __init__(
        self,
        model_name: str = "sage:allenai/SAGE-MM-Qwen3-VL-8B-SFT_RL", 
        method_label: str = "SAGE",
        num_sampled_frames: int = 128,
        gpu_idx: int = 0,
        benchmark: str = "sage_bench",
        tool_to_drop: Optional[str] = None,
        timeout_seconds: int = 300,
        max_num_iterative_reasoner_calls: int = 10,
        use_gemini_as_tool: bool = False,
        use_video: bool = True,
        collect_local_token_ids: bool = False,
        gpu_hourly_cost_usd: Optional[float] = None,
        synthesized_tool_names: Optional[List[str]] = None,
        clear_gemini_cache_before_run: bool = False,
    ):
        if not use_video:
            print("Not using video")

        if benchmark == "minerva_bench":
            self.benchmark = "minerva_bench"
            self.dataset_path = "data/minerva_videos/minerva.json"
        elif benchmark == "sage_bench":
            self.benchmark = "sage_bench"
            self.dataset_path = "allenai/SAGE-Bench"
        else:
            raise ValueError(f"Benchmark {benchmark} not supported")

        self.method_label = method_label
        self.num_sampled_frames = num_sampled_frames
        self.model_name = model_name.lower()
        self.gpu_idx = gpu_idx
        self.entry = 0
        self.timeout_seconds = timeout_seconds
        self.max_num_iterative_reasoner_calls = max_num_iterative_reasoner_calls
        self.tool_to_drop = tool_to_drop
        self.collect_local_token_ids = collect_local_token_ids
        self.gpu_hourly_cost_usd = gpu_hourly_cost_usd
        self.declared_synthesized_names = set()  # populated after model loads
        self.clear_gemini_cache_before_run = clear_gemini_cache_before_run

        self.current_sample_id: Optional[str] = None
        self.current_tool_api_usage: List[Dict[str, Any]] = []
        self.current_gemini_usage: List[Dict[str, Any]] = []
        self.current_gpt_usage: List[Dict[str, Any]] = []
        self.current_qwen_vllm_usage: List[Dict[str, Any]] = []
        self.current_whisper_events: List[Dict[str, Any]] = []
        self.run_tool_api_usage: List[Dict[str, Any]] = []
        self.run_gemini_usage: List[Dict[str, Any]] = []
        self.run_gpt_usage: List[Dict[str, Any]] = []
        self.run_qwen_vllm_usage: List[Dict[str, Any]] = []
        self.run_whisper_events: List[Dict[str, Any]] = []

        #fixed the tools to drop issue
        if tool_to_drop and tool_to_drop != "None":
            drop_tool_call_files = [tool_to_drop]
        else:
            drop_tool_call_files = []

        if "sage" in model_name.lower() and ("gemini" in model_name.lower() or "gpt" in model_name.lower()):
            self.model = SAGE(
                model_name,
                drop_tool_call_files=drop_tool_call_files,
                max_num_iterative_reasoner_calls=max_num_iterative_reasoner_calls,
                use_video=use_video,
            )
            self.model.context_vlm.delete_client_files()
        elif "qwen" in model_name.lower() or "molmo2" in model_name.lower():
            self.model = SAGE(
                model_name,
                drop_tool_call_files=drop_tool_call_files,
                max_num_iterative_reasoner_calls=max_num_iterative_reasoner_calls,
                use_gemini_as_tool=use_gemini_as_tool,
                use_video=use_video,
            )
        elif "gemini" in model_name.lower():
            self.model = ContextVLM(api_type=model_name, use_video=use_video)
            self.model.delete_client_files()
        elif "gpt" in model_name.lower():
            self.model = ContextVLM(api_type=model_name, use_video=use_video)
        else:
            raise ValueError(f"Model name {model_name} not supported")

        _output_label = model_name.split('/')[-1].replace(':', '-')
        self.output_file = f"{_output_label}_{self.benchmark}_results.jsonl"
        self.is_base_molmo2 = "molmo2" in model_name.lower() and "sage" not in model_name.lower()

        dispatcher = getattr(self.model, "dispatcher", {})
        self.original_tool_names = set(dispatcher.keys()) if isinstance(dispatcher, dict) else set()

        # Auto-detect synthesized tool names from the synthesized tools file
        _auto_synthesized: set[str] = set()
        _synth_file = os.path.join("sage", "src", "functions", "tools", "0_synthesized.py")
        if os.path.exists(_synth_file):
            import ast as _ast
            with open(_synth_file) as _f:
                _tree = _ast.parse(_f.read())
            _auto_synthesized = {n.name for n in _ast.walk(_tree) if isinstance(n, _ast.FunctionDef)}
        self.declared_synthesized_names = _auto_synthesized | set(synthesized_tool_names or [])
        # Remove synthesized names from original so they don't double-count
        self.original_tool_names -= self.declared_synthesized_names

        if self.clear_gemini_cache_before_run:
            try:
                from sage.src.api.gemini_cache import clear_cache as gemini_clear_cache
                gemini_clear_cache()
                print("Cleared Gemini response cache before run.")
            except Exception as exc:
                print(f"Warning: could not clear Gemini response cache: {exc}")

        self._install_tool_api_tracker()
        self._install_native_gemini_tracker()
        self._install_gpt_tool_tracker()
        self._install_passive_qwen_vllm_tracker()
        self._install_whisper_tracker()

    def _install_tool_api_tracker(self) -> None:
        client = getattr(getattr(self.model, "context_vlm", None), "client", None)
        tool_clients = getattr(client, "tool_call_clients", None)
        if tool_clients:
            client.tool_call_clients = [
                _TrackedOpenAIClient(existing_client, self)
                for existing_client in tool_clients
            ]

    def _install_native_gemini_tracker(self) -> None:
        """
        Track SAGE's Gemini tool backend when Gemini is routed through OpenRouter.

        The active sage.src.api.gemini.Gemini class creates an OpenAI-compatible
        OpenRouter client internally and calls:
            self.client.chat.completions.create(...)
        inside get_response(). We wrap that per-instance completion method only
        while get_response() runs, capture its returned usage/cost, then restore it.
        This records bookkeeping only; prompts, media, model selection, and returned
        answer text are unchanged.
        """
        try:
            import sage.src.api.response as response_module
            gemini_class = response_module.Gemini

            if getattr(gemini_class, "_table_logger_openrouter_gemini_patched", False):
                return

            original_get_response = gemini_class.get_response
            processor = self

            def tracked_get_response(gemini_instance, *args, **kwargs):
                call_start = time.perf_counter()
                logged_entry = None
                completions_obj = None
                original_create = None

                model_name = kwargs.get("model_name")
                if model_name is None and len(args) >= 4:
                    model_name = args[3]
                model_name = model_name or "gemini-2.5-flash"

                try:
                    completions_obj = gemini_instance.client.chat.completions
                    original_create = completions_obj.create

                    def tracked_create(*create_args, **create_kwargs):
                        nonlocal logged_entry
                        api_start = time.perf_counter()
                        response = original_create(*create_args, **create_kwargs)
                        api_seconds = time.perf_counter() - api_start
                        usage = getattr(response, "usage", None)

                        prompt_tokens = _usage_int(usage, "prompt_tokens")
                        output_tokens = _usage_int(usage, "completion_tokens")
                        total_tokens = _usage_int(usage, "total_tokens")
                        cost = get_usage_value(usage, "cost")
                        reported_cost = float(cost) if isinstance(cost, (int, float)) else None
                        estimated_cost = (
                            reported_cost
                            if reported_cost is not None
                            else _gemini_flash_cost_usd(prompt_tokens, output_tokens, None)
                        )

                        logged_entry = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "sample_id": processor.current_sample_id,
                            "model": create_kwargs.get("model", model_name),
                            "api_called": True,
                            "cache_hit_or_no_api_call": False,
                            "latency_seconds": api_seconds,
                            "usage_metadata_available": usage is not None,
                            "prompt_tokens": prompt_tokens,
                            "output_tokens": output_tokens,
                            "thinking_tokens": None,
                            "total_tokens": total_tokens,
                            "cached_content_tokens": None,
                            "openrouter_reported_cost_usd": reported_cost,
                            "estimated_cost_usd": estimated_cost,
                            "response_id": getattr(response, "id", None),
                        }
                        return response

                    completions_obj.create = tracked_create
                except Exception:
                    # Fallback: still run Gemini and log that no OpenRouter request
                    # could be intercepted through the available client shape.
                    completions_obj = None
                    original_create = None

                try:
                    result = original_get_response(gemini_instance, *args, **kwargs)
                finally:
                    if completions_obj is not None and original_create is not None:
                        completions_obj.create = original_create

                call_seconds = time.perf_counter() - call_start
                if processor.current_sample_id is not None:
                    if logged_entry is None:
                        # A cache hit returns before chat.completions.create(), or a
                        # different client implementation was used. No billing is assumed.
                        logged_entry = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "sample_id": processor.current_sample_id,
                            "model": model_name,
                            "api_called": False,
                            "cache_hit_or_no_api_call": True,
                            "latency_seconds": call_seconds,
                            "usage_metadata_available": False,
                            "prompt_tokens": None,
                            "output_tokens": None,
                            "thinking_tokens": None,
                            "total_tokens": None,
                            "cached_content_tokens": None,
                            "openrouter_reported_cost_usd": None,
                            "estimated_cost_usd": 0.0,
                            "response_id": None,
                        }
                    else:
                        logged_entry["get_response_wall_clock_seconds"] = call_seconds

                    processor.current_gemini_usage.append(logged_entry)
                    processor.run_gemini_usage.append(logged_entry)

                return result

            gemini_class.get_response = tracked_get_response
            gemini_class._table_logger_openrouter_gemini_patched = True
        except Exception as exc:
            print(f"Warning: could not install OpenRouter Gemini usage tracker: {exc}")

    def _install_gpt_tool_tracker(self) -> None:
        """
        Track GPT-4o tool calls selected by USE_GPT_AS_TOOL=True.

        This is bookkeeping only. It does not change tool selection, model inputs,
        model outputs, or released-evaluator labels. It temporarily intercepts an
        OpenAI-compatible chat.completions.create call exposed by the GPT instance,
        if present, and records returned usage/cost.
        """
        try:
            import sage.src.api.response as response_module
            gpt_class = response_module.GPT

            if getattr(gpt_class, "_table_logger_gpt_patched", False):
                return

            original_get_response = gpt_class.get_response
            processor = self

            def find_chat_client(gpt_instance):
                candidates = []
                for attr in ("client", "openai_client", "tool_call_client"):
                    if hasattr(gpt_instance, attr):
                        candidates.append(getattr(gpt_instance, attr))
                for value in getattr(gpt_instance, "__dict__", {}).values():
                    candidates.append(value)
                    if isinstance(value, (list, tuple)):
                        candidates.extend(value)
                for candidate in candidates:
                    try:
                        if hasattr(candidate, "chat") and hasattr(candidate.chat, "completions"):
                            if hasattr(candidate.chat.completions, "create"):
                                return candidate
                    except Exception:
                        pass
                return None

            def tracked_get_response(gpt_instance, *args, **kwargs):
                call_start = time.perf_counter()
                logged_entry = None
                completions_obj = None
                original_create = None
                requested_model = kwargs.get("model", "gpt:gpt-4o")

                try:
                    client_obj = find_chat_client(gpt_instance)
                    if client_obj is not None:
                        completions_obj = client_obj.chat.completions
                        original_create = completions_obj.create

                        def tracked_create(*create_args, **create_kwargs):
                            nonlocal logged_entry
                            api_start = time.perf_counter()
                            response = original_create(*create_args, **create_kwargs)
                            api_seconds = time.perf_counter() - api_start
                            usage = getattr(response, "usage", None)
                            cost = get_usage_value(usage, "cost")
                            reported_cost = float(cost) if isinstance(cost, (int, float)) else None
                            logged_entry = {
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "sample_id": processor.current_sample_id,
                                "model": create_kwargs.get("model", requested_model),
                                "api_called": True,
                                "cache_hit_or_no_api_call": False,
                                "latency_seconds": api_seconds,
                                "usage_metadata_available": usage is not None,
                                "prompt_tokens": _usage_int(usage, "prompt_tokens"),
                                "output_tokens": _usage_int(usage, "completion_tokens"),
                                "total_tokens": _usage_int(usage, "total_tokens"),
                                "openrouter_reported_cost_usd": reported_cost,
                                "estimated_cost_usd": reported_cost,
                                "response_id": getattr(response, "id", None),
                            }
                            return response

                        completions_obj.create = tracked_create
                except Exception:
                    completions_obj = None
                    original_create = None

                try:
                    result = original_get_response(gpt_instance, *args, **kwargs)
                finally:
                    if completions_obj is not None and original_create is not None:
                        completions_obj.create = original_create

                call_seconds = time.perf_counter() - call_start
                if processor.current_sample_id is not None:
                    if logged_entry is None:
                        logged_entry = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "sample_id": processor.current_sample_id,
                            "model": requested_model,
                            "api_called": False,
                            "cache_hit_or_no_api_call": True,
                            "latency_seconds": call_seconds,
                            "usage_metadata_available": False,
                            "prompt_tokens": None,
                            "output_tokens": None,
                            "total_tokens": None,
                            "openrouter_reported_cost_usd": None,
                            "estimated_cost_usd": None,
                            "response_id": None,
                        }
                    else:
                        logged_entry["get_response_wall_clock_seconds"] = call_seconds
                    processor.current_gpt_usage.append(logged_entry)
                    processor.run_gpt_usage.append(logged_entry)
                return result

            gpt_class.get_response = tracked_get_response
            gpt_class._table_logger_gpt_patched = True
        except Exception as exc:
            print(f"Warning: could not install GPT tool usage tracker: {exc}")

    def _install_passive_qwen_vllm_tracker(self) -> None:
        """
        Passively count local SAGE-MM/Qwen vLLM tokens without enabling SAGE's
        broken return_ids=True path.

        vLLM's normal RequestOutput includes prompt_token_ids and outputs[*].token_ids.
        This wrapper calls the original vLLM generation method unchanged, then reads
        token-id lengths from its returned object. It does not alter prompts, outputs,
        generation parameters, tool decisions, or evaluator labels.
        """
        try:
            client = getattr(getattr(self.model, "context_vlm", None), "client", None)
            engine = getattr(client, "vllm_engine", None)
            if engine is None:
                print("Warning: no vLLM engine found for passive Qwen token logging.")
                return
            if getattr(engine, "_table_logger_qwen_passive_patched", False):
                return

            processor = self
            method_name = "generate" if callable(getattr(engine, "generate", None)) else "chat"
            original_method = getattr(engine, method_name, None)
            if not callable(original_method):
                print("Warning: vLLM engine exposes neither generate nor chat; Qwen tokens unavailable.")
                return

            def tracked_vllm_call(*args, **kwargs):
                start = time.perf_counter()
                outputs = original_method(*args, **kwargs)
                elapsed = time.perf_counter() - start

                requests = outputs if isinstance(outputs, (list, tuple)) else [outputs]
                prompt_tokens = 0
                completion_tokens = 0
                request_count = 0
                for request_output in requests:
                    if request_output is None:
                        continue
                    request_count += 1
                    prompt_ids = getattr(request_output, "prompt_token_ids", None)
                    if prompt_ids is not None:
                        prompt_tokens += len(prompt_ids)
                    generations = getattr(request_output, "outputs", None) or []
                    for generation in generations:
                        token_ids = getattr(generation, "token_ids", None)
                        if token_ids is not None:
                            completion_tokens += len(token_ids)

                entry = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "sample_id": processor.current_sample_id,
                    "backend": "local_vllm_qwen_sage_mm",
                    "wrapped_method": method_name,
                    "request_count": request_count,
                    "latency_seconds": elapsed,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                if processor.current_sample_id is not None:
                    processor.current_qwen_vllm_usage.append(entry)
                    processor.run_qwen_vllm_usage.append(entry)
                return outputs

            setattr(engine, method_name, tracked_vllm_call)
            engine._table_logger_qwen_passive_patched = True
            print(f"Passive Qwen token logging installed on vLLM engine.{method_name}().")
        except Exception as exc:
            print(f"Warning: could not install passive Qwen/vLLM token tracker: {exc}")

    def _install_whisper_tracker(self) -> None:
        try:
            import sage.src.functions.utils.transcribe as transcribe_module
            if not isinstance(transcribe_module.requests, _TrackedRequestsProxy):
                transcribe_module.requests = _TrackedRequestsProxy(transcribe_module.requests, self)
        except Exception as exc:
            print(f"Warning: could not install Whisper request tracker: {exc}")

    def load_videos(self) -> List[Dict[str, Any]]:
        if self.benchmark == "minerva_bench":
            return self.load_minerva_videos()
        return self.load_sage_bench_videos()

    def _skip_completed(self, video_sets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding="utf-8") as handle:
                existing_ids = {
                    json.loads(line)["id"] for line in handle if line.strip()
                }
            video_sets = [row for row in video_sets if row["id"] not in existing_ids]
        return video_sets

    def load_sage_bench_videos(self) -> List[Dict[str, Any]]:
        """Load SAGE-Bench; native metadata has difficulty, modality, ques_type, and duration."""
        videos = load_dataset(self.dataset_path, split="test")
        video_sets = []
        skipped_missing = 0

        for v in tqdm(videos, total=len(videos), desc="Loading SAGE-Bench videos"):
            video_path = os.path.join(os.environ.get("VIDEO_DIR", "data/sage_bench_videos"), f"{v['video_id']}.mp4")
            if not os.path.exists(video_path):
                skipped_missing += 1
                continue

            video_duration_disk = get_video_duration(video_path)
            transcript_path = video_path.replace(".mp4", ".txt")
            transcript = ""
            if "sage" not in self.model_name:
                if not os.path.exists(transcript_path):
                    print(f"Transcript not found for {video_path}, transcribing...")
                    try:
                        transcript = str(transcribe_video(video_path))
                        with open(transcript_path, "w", encoding="utf-8") as handle:
                            handle.write(transcript)
                    except Exception as exc:
                        print(f"Error transcribing {video_path}: {exc}")
                else:
                    with open(transcript_path, "r", encoding="utf-8") as handle:
                        transcript = handle.read()

            raw_question = v["question"]
            question = raw_question
            is_mcq = v["ques_type"] == "mcq"
            if "sage" not in self.model_name:
                question = SAMPLED_BASELINE_PROMPT.replace("<<<question>>>", question)
                if USE_ASR:
                    question = question.replace("<<<asr>>>", transcript)

            duration_seconds = float(v["duration_seconds"])
            video_sets.append({
                "id": hashlib.md5(f"{v['question']}|{video_path}".encode()).hexdigest(),
                "path": video_path,
                "question": question if not (self.is_base_molmo2 and is_mcq) else (
                    question, "video_eval_multiple_choice"
                ),
                "answer": None,
                "full_answer": v["gt_answer"],
                "benchmark": "sage_bench",
                "video_id": v["video_id"],
                "raw_question": raw_question,
                "ques_type": str(v["ques_type"]).replace("-", "_"),
                "difficulty": v["difficulty"],
                "modality": v["modality"],
                "duration": duration_seconds,
                "duration_seconds": duration_seconds,
                "video_duration_original": v["video_duration"],
                "video_duration_seconds_disk": video_duration_disk,
                "transcript_path": transcript_path,
                "has_transcript_at_load": os.path.exists(transcript_path),
            })
        #ORIGINAL - COMMENTED OUT
        """
        if skipped_missing:
            print(f"Skipped {skipped_missing} SAGE-Bench samples because the video file is absent")
        print(f"Loaded {len(video_sets)} SAGE-Bench samples for evaluation")
        return self._skip_completed(video_sets)"""

        #NEW FOR DYNAMIC API
        if skipped_missing:
            print(f"Skipped {skipped_missing} SAGE-Bench samples because the video file is absent")
        # ---- holdout filter ----
        exclude_path = os.environ.get("EXCLUDE_IDS_FILE")
        if exclude_path and os.path.exists(exclude_path):
            with open(exclude_path) as f:
                exclude_ids = set(json.load(f))
            before = len(video_sets)
            video_sets = [r for r in video_sets if r["id"] not in exclude_ids]
            print(f"Holdout filter: dropped {before - len(video_sets)} of {before} samples (file: {exclude_path})")
        # ---- end holdout filter ----
        print(f"Loaded {len(video_sets)} SAGE-Bench samples for evaluation")
        return self._skip_completed(video_sets)


    def load_minerva_videos(self) -> List[Dict[str, Any]]:
        """Load MINERVA with native domain/category/question-type metadata."""
        with open(self.dataset_path, "r", encoding="utf-8") as handle:
            videos = json.load(handle)

        answer_id_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        video_sets = []
        skipped_missing = 0

        for v in tqdm(videos, total=len(videos), desc="Loading MINERVA videos"):
            video_path = os.path.join(
                "data", "minerva_videos", v["video_id"], f"{v['video_id']}.mp4"
            )
            if not os.path.exists(video_path):
                skipped_missing += 1
                continue

            video_duration = get_video_duration(video_path)
            transcript_path = video_path.replace(".mp4", ".txt")
            choices = "\n".join(
                f"({answer_id_to_letter[i]}) {v[f'answer_choice_{i}']}"
                for i in range(5)
            )

            if "sage" not in self.model_name:
                transcript = ""
                if os.path.exists(transcript_path):
                    with open(transcript_path, "r", encoding="utf-8") as handle:
                        transcript = handle.read()
                question = (
                    SAMPLED_BASELINE_MINERVA_PROMPT
                    .replace("<<<question>>>", v["question"])
                    .replace("<<<answer choices>>>", choices)
                )
                if USE_ASR:
                    question = question.replace("<<<asr>>>", transcript)
            else:
                question = (
                    v["question"] + "\nAnswer from the given options: \n" + choices + "\n"
                )

            answer_id = int(v["answer_id"])
            correct_answer = (
                answer_id_to_letter[answer_id] if "sage" in self.model_name else answer_id
            )
            video_sets.append({
                "id": v["key"],
                "path": video_path,
                "question": question if not self.is_base_molmo2 else (
                    question, "video_eval_multiple_choice"
                ),
                "answer": correct_answer,
                "full_answer": v[f"answer_choice_{answer_id}"],
                "benchmark": "minerva_bench",
                "video_id": v["video_id"],
                "raw_question": v["question"],
                "ques_type": v.get("question_type"),
                "question_type": v.get("question_type"),
                "domain": v.get("split"),
                "category": v.get("category"),
                "duration": video_duration,
                "duration_seconds": video_duration,
                "answer_id": answer_id,
                "answer_letter": answer_id_to_letter[answer_id],
                "answer_choices": {
                    answer_id_to_letter[i]: v.get(f"answer_choice_{i}") for i in range(5)
                },
                "ground_truth_reasoning": v.get("reasoning"),
                "difficulty": None,
                "modality": None,
                "transcript_path": transcript_path,
                "has_transcript_at_load": os.path.exists(transcript_path),
            })
        
        #ORIGINAL - COMMENTED OUT
        """
        if skipped_missing:
            print(f"Skipped {skipped_missing} MINERVA samples because the video file is absent")
        print(f"Loaded {len(video_sets)} MINERVA samples for evaluation")
        return self._skip_completed(video_sets)"""

        #NEW FOR DYNAMIC
        if skipped_missing:
            print(f"Skipped {skipped_missing} MINERVA samples because the video file is absent")
        # ---- holdout filter ----
        exclude_path = os.environ.get("EXCLUDE_IDS_FILE")
        if exclude_path and os.path.exists(exclude_path):
            with open(exclude_path) as f:
                exclude_ids = set(json.load(f))
            before = len(video_sets)
            video_sets = [r for r in video_sets if r["id"] not in exclude_ids]
            print(f"Holdout filter: dropped {before - len(video_sets)} of {before} samples (file: {exclude_path})")
        # ---- end holdout filter ----
        print(f"Loaded {len(video_sets)} MINERVA samples for evaluation")
        return self._skip_completed(video_sets)


    def process_video(self, v: Dict[str, Any], gpu_idx: int) -> Dict[str, Any]:
        max_attempts = 5 if "sage" in self.model_name else 10
        print(f"Processing video {v['id']}: {v['path']} for GPU {gpu_idx}")

        self.current_sample_id = v["id"]
        self.current_tool_api_usage = []
        self.current_gemini_usage = []
        self.current_gpt_usage = []
        self.current_qwen_vllm_usage = []
        self.current_whisper_events = []

        sample_start_time = time.perf_counter()
        transcript_path = v["path"].replace(".mp4", ".txt")
        transcript_before = os.path.exists(transcript_path)

        answer = None
        complete_results = None
        local_token_usage = {
            "local_token_turns": [],
            "local_prompt_tokens": None,
            "local_completion_tokens": None,
            "local_total_tokens": None,
        }

        question = v["question"]
        if not isinstance(question, str):
            assert len(question) == 2, "Question must be a tuple of (question, eval_style)"
            question, eval_style = question
        else:
            eval_style = None

        attempt = 0
        for attempt in range(max_attempts):
            try:
                with timeout(self.timeout_seconds):
                    if "sage" not in self.model_name and any(
                        name in self.model_name
                        for name in ["gemini", "gpt", "longrl", "video-thinker", "video-r1"]
                    ):
                        answer = self.model.answer(
                            v["path"],
                            question,
                            model_name=self.model_name,
                            num_sampled_frames=self.num_sampled_frames,
                        )
                    elif "sage" in self.model_name:
                        retry_temperature = 0.7 if attempt > 0 else 0.0
                        if self.collect_local_token_ids:
                            (
                                answer,
                                complete_results,
                                _completion_ids,
                                _prompt_ids,
                                _attention_mask,
                            ) = self.model.run_inference(
                                v["path"],
                                question,
                                model_name=self.model_name,
                                num_sampled_frames=self.num_sampled_frames,
                                temperature=retry_temperature,
                                return_ids=True,
                            )
                        else:
                            answer, complete_results = self.model.run_inference(
                                v["path"],
                                question,
                                model_name=self.model_name,
                                num_sampled_frames=self.num_sampled_frames,
                                temperature=retry_temperature,
                            )
                    else:
                        answer = self.model.context_vlm.client.get_response(
                            prompt=question,
                            media=[v["path"]],
                            media_type="video",
                            eval_style=eval_style,
                        )
                    break
            except TimeoutError as exc:
                print(f"Timeout on attempt {attempt + 1}/{max_attempts}: {exc}")
                if attempt >= max_attempts - 1:
                    answer = (
                        f"Could not produce an answer after {max_attempts} "
                        "attempts due to timeout"
                    )
                    complete_results = None
            except Exception as exc:
                if "FileStorageBytesPerProject" in str(exc):
                    self.model.context_vlm.delete_client_files()
                print(f"Error on attempt {attempt + 1}/{max_attempts}: {exc}")
                if attempt >= max_attempts - 1:
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    answer = (
                        f"Could not produce an answer after {max_attempts} "
                        f"attempts with error {exc}"
                    )
                    complete_results = None

        wall_clock_seconds = time.perf_counter() - sample_start_time
        if self.collect_local_token_ids:
            local_token_usage = extract_local_token_usage(complete_results)

        tool_metrics = extract_tool_metrics(
            complete_results,
            self.original_tool_names,
            self.declared_synthesized_names,
        )
        handoff_metrics = detect_empty_media_handoff_failure(complete_results)
        tool_api_prompt_tokens = sum_present(
            [row.get("prompt_tokens") for row in self.current_tool_api_usage]
        )
        tool_api_completion_tokens = sum_present(
            [row.get("completion_tokens") for row in self.current_tool_api_usage]
        )
        tool_api_total_tokens = sum_present(
            [row.get("total_tokens") for row in self.current_tool_api_usage]
        )
        tool_api_cost = sum_present([row.get("cost") for row in self.current_tool_api_usage])
        gemini_prompt_tokens = sum_present(
            [row.get("prompt_tokens") for row in self.current_gemini_usage]
        )
        gemini_output_tokens = sum_present(
            [row.get("output_tokens") for row in self.current_gemini_usage]
        )
        gemini_thinking_tokens = sum_present(
            [row.get("thinking_tokens") for row in self.current_gemini_usage]
        )
        gemini_total_tokens = sum_present(
            [row.get("total_tokens") for row in self.current_gemini_usage]
        )
        gemini_estimated_cost = sum_present(
            [row.get("estimated_cost_usd") for row in self.current_gemini_usage]
        )
        gemini_tool_invocations = len(self.current_gemini_usage)
        gemini_tool_actual_api_requests = sum(
            1 for row in self.current_gemini_usage if row.get("api_called") is True
        )
        gemini_tool_cache_hits_or_no_api_call = sum(
            1 for row in self.current_gemini_usage
            if row.get("cache_hit_or_no_api_call") is True
        )
        gpt_prompt_tokens = sum_present(
            [row.get("prompt_tokens") for row in self.current_gpt_usage]
        )
        gpt_output_tokens = sum_present(
            [row.get("output_tokens") for row in self.current_gpt_usage]
        )
        gpt_total_tokens = sum_present(
            [row.get("total_tokens") for row in self.current_gpt_usage]
        )
        gpt_estimated_cost = sum_present(
            [row.get("estimated_cost_usd") for row in self.current_gpt_usage]
        )
        gpt_tool_invocations = len(self.current_gpt_usage)
        gpt_tool_actual_api_requests = sum(
            1 for row in self.current_gpt_usage if row.get("api_called") is True
        )
        gpt_tool_cache_hits_or_no_api_call = sum(
            1 for row in self.current_gpt_usage
            if row.get("cache_hit_or_no_api_call") is True
        )
        qwen_sage_prompt_tokens = sum_present(
            [row.get("prompt_tokens") for row in self.current_qwen_vllm_usage]
        )
        qwen_sage_completion_tokens = sum_present(
            [row.get("completion_tokens") for row in self.current_qwen_vllm_usage]
        )
        qwen_sage_total_tokens = sum_present(
            [row.get("total_tokens") for row in self.current_qwen_vllm_usage]
        )
        qwen_sage_vllm_calls = len(self.current_qwen_vllm_usage)
        gpu_cost = (
            wall_clock_seconds * self.gpu_hourly_cost_usd / 3600
            if self.gpu_hourly_cost_usd is not None
            else None
        )

        # Qwen tokens are collected passively from normal vLLM outputs; return_ids stays False.
        local_token_usage = {
            "local_token_turns": self.current_qwen_vllm_usage,
            "local_prompt_tokens": qwen_sage_prompt_tokens,
            "local_completion_tokens": qwen_sage_completion_tokens,
            "local_total_tokens": qwen_sage_total_tokens,
        }
        tokens_are_complete = qwen_sage_total_tokens is not None
        logged_total_tokens = (
            qwen_sage_total_tokens
            + (tool_api_total_tokens or 0)
            + (gemini_total_tokens or 0)
            + (gpt_total_tokens or 0)
            if tokens_are_complete else None
        )

        result = {
            "path": v["path"],
            "question": question,
            "answer": answer,
            "correct_answer": v["answer"],
            "id": v["id"],
            "full_answer": v["full_answer"],
            "benchmark": self.benchmark,
            "method": self.method_label,
            "video_id": v.get("video_id"),
            "raw_question": v.get("raw_question"),
            "ques_type": v.get("ques_type"),
            "question_type": v.get("question_type"),
            "difficulty": v.get("difficulty"),
            "modality": v.get("modality"),
            "domain": v.get("domain"),
            "category": v.get("category"),
            "duration": v.get("duration"),
            "duration_seconds": v.get("duration_seconds"),
            "video_duration_original": v.get("video_duration_original"),
            "video_duration_seconds_disk": v.get("video_duration_seconds_disk"),
            "answer_id": v.get("answer_id"),
            "answer_letter": v.get("answer_letter"),
            "answer_choices": v.get("answer_choices"),
            "ground_truth_reasoning": v.get("ground_truth_reasoning"),
            "num_attempts": attempt + 1,
            "wall_clock_seconds": wall_clock_seconds,
            "gpu_hourly_cost_usd": self.gpu_hourly_cost_usd,
            "gpu_cost_usd": gpu_cost,
            "tool_call_model": TOOL_CALL_MODEL,
            "tool_api_prompt_tokens": tool_api_prompt_tokens,
            "tool_api_completion_tokens": tool_api_completion_tokens,
            "tool_api_total_tokens": tool_api_total_tokens,
            "tool_api_cost_usd": tool_api_cost,
            "tool_api_calls": len(self.current_tool_api_usage),
            "tool_api_usage": self.current_tool_api_usage,
            # A Gemini tool invocation can be served from SAGE's persistent Gemini cache.
            # "api_calls" counts actual OpenRouter requests; "invocations" includes cache hits.
            "gemini_tool_invocations": gemini_tool_invocations,
            "gemini_tool_api_calls": gemini_tool_actual_api_requests,
            "gemini_tool_actual_api_requests": gemini_tool_actual_api_requests,
            "gemini_tool_cache_hits_or_no_api_call": gemini_tool_cache_hits_or_no_api_call,
            "gemini_tool_prompt_tokens": gemini_prompt_tokens,
            "gemini_tool_output_tokens": gemini_output_tokens,
            "gemini_tool_thinking_tokens": gemini_thinking_tokens,
            "gemini_tool_total_tokens": gemini_total_tokens,
            "gemini_tool_estimated_cost_usd": gemini_estimated_cost,
            "gemini_tool_usage": self.current_gemini_usage,
            "gpt_tool_invocations": gpt_tool_invocations,
            "gpt_tool_api_calls": gpt_tool_actual_api_requests,
            "gpt_tool_actual_api_requests": gpt_tool_actual_api_requests,
            "gpt_tool_cache_hits_or_no_api_call": gpt_tool_cache_hits_or_no_api_call,
            "gpt_tool_prompt_tokens": gpt_prompt_tokens,
            "gpt_tool_output_tokens": gpt_output_tokens,
            "gpt_tool_total_tokens": gpt_total_tokens,
            "gpt_tool_estimated_cost_usd": gpt_estimated_cost,
            "gpt_tool_usage": self.current_gpt_usage,
            "collect_local_token_ids": False,
            "qwen_token_collection_mode": "passive_vllm_request_output",
            **local_token_usage,
            "qwen_sage_prompt_tokens": qwen_sage_prompt_tokens,
            "qwen_sage_completion_tokens": qwen_sage_completion_tokens,
            "qwen_sage_total_tokens": qwen_sage_total_tokens,
            "qwen_sage_vllm_calls": qwen_sage_vllm_calls,
            "qwen_sage_vllm_usage": self.current_qwen_vllm_usage,
            "tokens_are_complete": tokens_are_complete,
            "total_tokens_logged": logged_total_tokens,
            **tool_metrics,
            **handoff_metrics,
            "transcribe_api_url": TRANSCRIBE_API_URL,
            "had_transcript_before_sample": transcript_before,
            "has_transcript_after_sample": os.path.exists(transcript_path),
            "transcript_created_during_sample": (
                not transcript_before and os.path.exists(transcript_path)
            ),
            "whisper_invocations": len(self.current_whisper_events),
            "whisper_events": self.current_whisper_events,
        }
        if complete_results is not None:
            result["complete_results"] = complete_results

        with open(self.output_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
        print(f"Answer for video {v['id']} on GPU {gpu_idx}: {answer}")
        print(f"Wrote result for video {v['id']}")
        self.current_sample_id = None
        return result

    def process_all_videos(
        self,
        max_workers: int = 1,
        num_gpus: int = 1,
        gpu_idx: int = 0,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        video_sets = get_chunk(self.load_videos(), num_gpus, gpu_idx)
        if limit is not None:
            video_sets = video_sets[:limit]
        self.video_sets = video_sets

        start_time = time.perf_counter()
        progress_bar = tqdm(
            total=len(video_sets),
            desc=f"Processing videos on GPU {gpu_idx}",
            unit="sample",
            position=gpu_idx,
            leave=True,
        )
        for video in video_sets:
            check_tool_call_model()
            self.process_video(video, gpu_idx)
            self.entry += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                "current_video": video["id"],
                "elapsed": f"{time.perf_counter() - start_time:.1f}s",
            })
            print(f"let's move onto {self.entry}")
        progress_bar.close()

        elapsed = time.perf_counter() - start_time
        print(f"Loop completed for {self.method_label} on GPU {self.gpu_idx}")
        if "sage" not in self.model_name and "gemini" in self.model_name:
            self.model.delete_client_files()
        elif "gemini" in self.model_name:
            self.model.context_vlm.delete_client_files()

        print(f"Cleaning up for {self.method_label} on GPU {self.gpu_idx}")
        if "qwen" in self.model_name or "molmo2" in self.model_name:
            self.model.context_vlm.client.cleanup()
        elif "longrl" in self.model_name or "video-thinker" in self.model_name:
            self.model.client.cleanup()

        return {
            "processed_in_this_invocation": len(video_sets),
            "inference_wall_clock_seconds": elapsed,
            "whisper_events": self.run_whisper_events,
            "tool_api_usage": self.run_tool_api_usage,
            "gemini_tool_usage": self.run_gemini_usage,
            "gpt_tool_usage": self.run_gpt_usage,
            "qwen_sage_vllm_usage": self.run_qwen_vllm_usage,
        }


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def accuracy_groups(
    rows: List[Dict[str, Any]], key: str, ordered_values: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None:
            groups[str(value)].append(row)

    values = ordered_values or sorted(groups.keys())
    output = []
    for value in values:
        items = groups.get(value, [])
        if not items:
            continue
        correct = sum(1 for item in items if item.get("is_correct") is True)
        output.append({
            key: value,
            "total_questions": len(items),
            "correct": correct,
            "accuracy_percent": (correct / len(items)) * 100,
        })
    return output


def add_duration_bucket(rows: List[Dict[str, Any]], benchmark: str) -> None:
    buckets = MINERVA_DURATION_BUCKETS if benchmark == "minerva_bench" else SAGE_BENCH_DURATION_BUCKETS
    for row in rows:
        row["duration_bucket"] = None
        duration = row.get("duration_seconds", row.get("duration"))
        if isinstance(duration, (int, float)):
            for lower, upper, label in buckets:
                if lower <= duration < upper:
                    row["duration_bucket"] = label
                    break


def load_synthesis_metrics(args: argparse.Namespace) -> Dict[str, Any]:
    metrics = {
        "synthesis_num_sampled_queries": args.synthesis_num_sampled_queries,
        "synthesis_wall_clock_seconds": args.synthesis_wall_clock_seconds,
        "synthesis_prompt_tokens": args.synthesis_prompt_tokens,
        "synthesis_completion_tokens": args.synthesis_completion_tokens,
        "synthesis_total_tokens": (
            args.synthesis_prompt_tokens + args.synthesis_completion_tokens
            if args.synthesis_prompt_tokens is not None
            and args.synthesis_completion_tokens is not None
            else None
        ),
        "synthesis_cost_usd": args.synthesis_cost_usd,
    }
    if args.synthesis_metrics_json:
        with open(args.synthesis_metrics_json, "r", encoding="utf-8") as handle:
            metrics.update(json.load(handle))
    return metrics


def evaluate_and_write_reports(
    processor: Processor,
    run_info: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    eval_path = Path(__file__).with_name("evaluate_responses.py")
    spec = importlib.util.spec_from_file_location("sage_evaluate_responses", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import evaluator from {eval_path}")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)

    (
        accuracy,
        direct_answers,
        iterative_reasoner_occasions,
        direct_accuracy,
        iterative_reasoner_accuracy,
        evaluation_stats,
        per_sample_stats,
    ) = evaluator.evaluate_answers(processor.output_file)

    raw_rows = read_jsonl(processor.output_file)
    if len(raw_rows) != len(per_sample_stats):
        raise ValueError(
            "Raw JSONL and evaluated sample count do not match: "
            f"{len(raw_rows)} versus {len(per_sample_stats)}"
        )

    merged_rows = []
    for raw, evaluated in zip(raw_rows, per_sample_stats):
        merged = dict(raw)
        merged.update(evaluated)
        # Restore cost/metadata fields overwritten or not preserved by evaluator.
        for key in [
            "duration_seconds", "domain", "category", "question_type", "benchmark",
            "method", "wall_clock_seconds", "gpu_cost_usd", "tool_api_cost_usd",
            "num_turns", "total_tool_calls", "tool_call_counts", "tool_names",
            "original_tool_calls", "original_tool_names", "synthesized_tool_calls",
            "synthesized_tool_names", "local_prompt_tokens", "local_completion_tokens",
            "local_total_tokens", "qwen_token_collection_mode",
            "qwen_sage_prompt_tokens", "qwen_sage_completion_tokens",
            "qwen_sage_total_tokens", "qwen_sage_vllm_calls", "qwen_sage_vllm_usage",
            "tool_api_prompt_tokens", "tool_api_completion_tokens",
            "tool_api_total_tokens", "tool_api_cost_usd", "gemini_tool_invocations",
            "gemini_tool_api_calls", "gemini_tool_actual_api_requests",
            "gemini_tool_cache_hits_or_no_api_call",
            "gemini_tool_prompt_tokens", "gemini_tool_output_tokens",
            "gemini_tool_thinking_tokens", "gemini_tool_total_tokens",
            "gemini_tool_estimated_cost_usd", "gemini_tool_usage",
            "gpt_tool_invocations", "gpt_tool_api_calls", "gpt_tool_actual_api_requests",
            "gpt_tool_cache_hits_or_no_api_call", "gpt_tool_prompt_tokens",
            "gpt_tool_output_tokens", "gpt_tool_total_tokens",
            "gpt_tool_estimated_cost_usd", "gpt_tool_usage",
            "total_tokens_logged", "tokens_are_complete",
            "logical_vlm_calls_total", "context_vlm_calls", "iterative_reasoner_calls",
            "visual_tool_vlm_calls", "whisper_invocations",
            "extracted_media_available", "extracted_media_path_count",
            "perform_reasoning_calls", "empty_media_reasoning_calls",
            "empty_media_handoff_failure",
        ]:
            merged[key] = raw.get(key)
        merged_rows.append(merged)

    add_duration_bucket(merged_rows, processor.benchmark)
    synthesis = load_synthesis_metrics(args)

    total = len(merged_rows)
    correct = sum(1 for row in merged_rows if row.get("is_correct") is True)

    # Audit evaluator failures without changing released-evaluator correctness labels.
    null_evaluation_reasoning_rows = [
        row for row in merged_rows if row.get("evaluation_reasoning") is None
    ]
    null_evaluation_reasoning_count = len(null_evaluation_reasoning_rows)
    null_evaluation_reasoning_ids = [
        row.get("id") for row in null_evaluation_reasoning_rows
    ]

    total_wall_clock = sum(row.get("wall_clock_seconds", 0) or 0 for row in merged_rows)
    total_tools = sum(row.get("total_tool_calls", 0) or 0 for row in merged_rows)
    total_turns = sum(row.get("num_turns", 0) or 0 for row in merged_rows)
    total_gpu_cost = sum_present([row.get("gpu_cost_usd") for row in merged_rows])
    total_tool_api_cost = sum_present([row.get("tool_api_cost_usd") for row in merged_rows])
    total_gemini_tool_cost = sum_present(
        [row.get("gemini_tool_estimated_cost_usd") for row in merged_rows]
    )
    total_gpt_tool_cost = sum_present(
        [row.get("gpt_tool_estimated_cost_usd") for row in merged_rows]
    )
    inference_cost = sum_present([
        total_gpu_cost, total_tool_api_cost, total_gemini_tool_cost, total_gpt_tool_cost
    ])
    synthesis_cost = synthesis.get("synthesis_cost_usd")
    total_cost_with_synthesis = sum_present([inference_cost, synthesis_cost])

    complete_token_rows = [
        row for row in merged_rows if row.get("tokens_are_complete") is True
    ]
    qwen_sage_tokens_complete_for_all_samples = (
        len(complete_token_rows) == total and total > 0
    )
    qwen_sage_prompt_tokens_total = (
        sum(row.get("qwen_sage_prompt_tokens", 0) or 0 for row in complete_token_rows)
        if qwen_sage_tokens_complete_for_all_samples else None
    )
    qwen_sage_completion_tokens_total = (
        sum(row.get("qwen_sage_completion_tokens", 0) or 0 for row in complete_token_rows)
        if qwen_sage_tokens_complete_for_all_samples else None
    )
    qwen_sage_total_tokens = (
        sum(row.get("qwen_sage_total_tokens", 0) or 0 for row in complete_token_rows)
        if qwen_sage_tokens_complete_for_all_samples else None
    )
    total_tokens_all_samples = (
        sum(row.get("total_tokens_logged", 0) or 0 for row in complete_token_rows)
        if qwen_sage_tokens_complete_for_all_samples else None
    )

    duration_order = (
        [label for _, _, label in MINERVA_DURATION_BUCKETS]
        if processor.benchmark == "minerva_bench"
        else [label for _, _, label in SAGE_BENCH_DURATION_BUCKETS]
    )

    summary = {
        "method": processor.method_label,
        "benchmark": processor.benchmark,
        "total_questions": total,
        "correct": correct,
        "overall_accuracy_percent": (correct / total * 100) if total else None,
        "direct_answers": direct_answers,
        "direct_accuracy_percent": direct_accuracy * 100,
        "iterative_reasoner_occasions": iterative_reasoner_occasions,
        "iterative_reasoner_accuracy_percent": iterative_reasoner_accuracy * 100,
        "could_not_produce_count": evaluation_stats.get("could_not_produce_count"),
        "null_evaluation_reasoning_count": null_evaluation_reasoning_count,
        "null_evaluation_reasoning_rate_percent": (
            null_evaluation_reasoning_count / total * 100 if total else None
        ),
        "null_evaluation_reasoning_sample_ids": null_evaluation_reasoning_ids,
        "duration_wise_accuracy": accuracy_groups(
            merged_rows, "duration_bucket", duration_order
        ),
        "question_type_accuracy": accuracy_groups(merged_rows, "ques_type"),
        "difficulty_accuracy": accuracy_groups(merged_rows, "difficulty"),
        "modality_accuracy": accuracy_groups(merged_rows, "modality"),
        "domain_accuracy": (
            accuracy_groups(merged_rows, "domain")
            if processor.benchmark == "minerva_bench"
            else "Not available: SAGE-Bench provides no domain field."
        ),
        "category_accuracy": (
            accuracy_groups(merged_rows, "category")
            if processor.benchmark == "minerva_bench"
            else "Not available: SAGE-Bench provides no category field."
        ),
        "cost_metrics": {
            "total_wall_clock_seconds": total_wall_clock,
            "seconds_per_sample": total_wall_clock / total if total else None,
            "total_turns": total_turns,
            "average_turns_per_sample": total_turns / total if total else None,
            "total_tool_calls": total_tools,
            "average_tool_calls_per_sample": total_tools / total if total else None,
            "empty_media_handoff_failure_count": sum(
                1 for row in merged_rows if row.get("empty_media_handoff_failure") is True
            ),
            "empty_media_handoff_failure_rate_percent": (
                100 * sum(
                    1 for row in merged_rows if row.get("empty_media_handoff_failure") is True
                ) / total if total else None
            ),
            "tool_using_samples": sum(
                1 for row in merged_rows if (row.get("total_tool_calls", 0) or 0) > 0
            ),
            "empty_media_failure_rate_among_tool_samples_percent": (
                100 * sum(
                    1 for row in merged_rows if row.get("empty_media_handoff_failure") is True
                ) / sum(
                    1 for row in merged_rows if (row.get("total_tool_calls", 0) or 0) > 0
                )
                if sum(
                    1 for row in merged_rows if (row.get("total_tool_calls", 0) or 0) > 0
                ) else None
            ),
            "empty_media_reasoning_calls_total": sum(
                row.get("empty_media_reasoning_calls", 0) or 0 for row in merged_rows
            ),
            "tool_call_counts": dict(Counter(
                name
                for row in merged_rows
                for name, count in (row.get("tool_call_counts") or {}).items()
                for _ in range(count)
            )),
            "original_tool_call_counts": dict(Counter(
                name
                for row in merged_rows
                for name, count in (row.get("original_tool_call_counts") or {}).items()
                for _ in range(count)
            )),
            "synthesized_tool_call_counts": dict(Counter(
                name
                for row in merged_rows
                for name, count in (row.get("synthesized_tool_call_counts") or {}).items()
                for _ in range(count)
            )),
            "tool_api_prompt_tokens": sum_present(
                [row.get("tool_api_prompt_tokens") for row in merged_rows]
            ),
            "tool_api_completion_tokens": sum_present(
                [row.get("tool_api_completion_tokens") for row in merged_rows]
            ),
            "tool_api_total_tokens": sum_present(
                [row.get("tool_api_total_tokens") for row in merged_rows]
            ),
            "gemini_tool_invocations": sum(
                row.get("gemini_tool_invocations", 0) or 0 for row in merged_rows
            ),
            "gemini_tool_api_calls": sum(
                row.get("gemini_tool_api_calls", 0) or 0 for row in merged_rows
            ),
            "gemini_tool_actual_api_requests": sum(
                row.get("gemini_tool_actual_api_requests", 0) or 0 for row in merged_rows
            ),
            "gemini_tool_cache_hits_or_no_api_call": sum(
                row.get("gemini_tool_cache_hits_or_no_api_call", 0) or 0
                for row in merged_rows
            ),
            "gemini_tool_prompt_tokens": sum_present(
                [row.get("gemini_tool_prompt_tokens") for row in merged_rows]
            ),
            "gemini_tool_output_tokens": sum_present(
                [row.get("gemini_tool_output_tokens") for row in merged_rows]
            ),
            "gemini_tool_thinking_tokens": sum_present(
                [row.get("gemini_tool_thinking_tokens") for row in merged_rows]
            ),
            "gemini_tool_total_tokens": sum_present(
                [row.get("gemini_tool_total_tokens") for row in merged_rows]
            ),
            "gemini_tool_estimated_cost_usd": total_gemini_tool_cost,
            "gpt_tool_invocations": sum(
                row.get("gpt_tool_invocations", 0) or 0 for row in merged_rows
            ),
            "gpt_tool_api_calls": sum(
                row.get("gpt_tool_api_calls", 0) or 0 for row in merged_rows
            ),
            "gpt_tool_actual_api_requests": sum(
                row.get("gpt_tool_actual_api_requests", 0) or 0 for row in merged_rows
            ),
            "gpt_tool_cache_hits_or_no_api_call": sum(
                row.get("gpt_tool_cache_hits_or_no_api_call", 0) or 0 for row in merged_rows
            ),
            "gpt_tool_prompt_tokens": sum_present(
                [row.get("gpt_tool_prompt_tokens") for row in merged_rows]
            ),
            "gpt_tool_output_tokens": sum_present(
                [row.get("gpt_tool_output_tokens") for row in merged_rows]
            ),
            "gpt_tool_total_tokens": sum_present(
                [row.get("gpt_tool_total_tokens") for row in merged_rows]
            ),
            "gpt_tool_estimated_cost_usd": total_gpt_tool_cost,
            "local_sage_tokens_complete_for_all_samples": qwen_sage_tokens_complete_for_all_samples,
            "qwen_sage_tokens_complete_for_all_samples": qwen_sage_tokens_complete_for_all_samples,
            "qwen_sage_prompt_tokens_total": qwen_sage_prompt_tokens_total,
            "qwen_sage_completion_tokens_total": qwen_sage_completion_tokens_total,
            "qwen_sage_total_tokens": qwen_sage_total_tokens,
            "qwen_sage_tokens_per_sample": (
                qwen_sage_total_tokens / total
                if qwen_sage_total_tokens is not None and total else None
            ),
            "total_tokens_all_samples": total_tokens_all_samples,
            "tokens_per_sample": (
                total_tokens_all_samples / total
                if total_tokens_all_samples is not None and total
                else None
            ),
            "logical_vlm_calls_total": sum(
                row.get("logical_vlm_calls_total", 0) or 0 for row in merged_rows
            ),
            "logical_vlm_calls_per_sample": (
                sum(row.get("logical_vlm_calls_total", 0) or 0 for row in merged_rows) / total
                if total else None
            ),
            "whisper_invocations_total": sum(
                row.get("whisper_invocations", 0) or 0 for row in merged_rows
            ),
            "gpu_inference_cost_usd": total_gpu_cost,
            "tool_api_cost_usd": total_tool_api_cost,
            "inference_cost_usd": inference_cost,
            "synthesis_metrics": synthesis,
            "amortized_synthesis_cost_usd_per_sample": (
                synthesis_cost / total if synthesis_cost is not None and total else None
            ),
            "total_cost_including_synthesis_usd": total_cost_with_synthesis,
            "cost_usd_per_sample_including_synthesis": (
                total_cost_with_synthesis / total
                if total_cost_with_synthesis is not None and total
                else None
            ),
        },
        "logging_notes": {
            "accuracy": "Computed by evaluate_responses.py semantic judge, then merged here. Null evaluation_reasoning cases are counted for audit but their released-evaluator is_correct labels are not repaired or overridden.",
            "tool_api_usage": "Exact only for OpenAI-compatible tool calls routed through tool_call_clients; OpenRouter returns usage/cost in non-streaming responses.",
            "gemini_tool_usage": (
                "Gemini tool invocations are routed through OpenRouter when not served by SAGE's "
                "persistent Gemini cache. gemini_tool_invocations counts all requested Gemini uses; "
                "gemini_tool_api_calls/gemini_tool_actual_api_requests count only intercepted "
                "OpenRouter requests. Tokens and cost are logged only for actual API requests. "
                "OpenRouter-reported usage.cost is preferred when present; otherwise estimated_cost_usd "
                "falls back to the configured Gemini 2.5 Flash token-rate estimate."
            ),
            "gpt_tool_usage": (
                "When USE_GPT_AS_TOOL=True, GPT-4o perform_reasoning calls are logged separately. "
                "Tokens and cost are recorded from the intercepted OpenAI-compatible response usage when exposed; "
                "no invented fallback dollar estimate is used if cost is not returned."
            ),
            "local_tokens": (
                "Collected passively from normal vLLM RequestOutput token IDs without enabling SAGE's broken return_ids=True path. The released inference path remains return_ids=False."
            ),
            "tokens_measurement_caveat": (
                "SAGE's return_ids=True path is deliberately not used because it crashes in this repository. "
                "Qwen token counts are read passively from the unchanged normal vLLM generation return objects."
            ),
            "vlm_calls": "Logical calls requested by SAGE; cached tool responses may not cause a new billed model request.",
            "whisper": "Counts transcribe HTTP requests made through sage.src.functions.utils.transcribe during this process run.",
            "synthesis": "Vanilla SAGE has no synthesis step; Dynamic SAGE should supply synthesis metrics via CLI or JSON.",
        },
        "run_info": run_info,
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(processor.output_file).stem
    evaluated_full_path = report_dir / f"{stem}_evaluated_full.json"
    summary_path = report_dir / f"{stem}_metrics_summary.json"

    with open(evaluated_full_path, "w", encoding="utf-8") as handle:
        json.dump(merged_rows, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    write_csv(report_dir / f"{stem}_per_sample.csv", merged_rows)
    write_csv(
        report_dir / f"{stem}_duration_accuracy.csv",
        summary["duration_wise_accuracy"],
    )
    write_csv(
        report_dir / f"{stem}_question_type_accuracy.csv",
        summary["question_type_accuracy"],
    )
    if processor.benchmark == "sage_bench":
        write_csv(report_dir / f"{stem}_difficulty_accuracy.csv", summary["difficulty_accuracy"])
        write_csv(report_dir / f"{stem}_modality_accuracy.csv", summary["modality_accuracy"])
    else:
        write_csv(report_dir / f"{stem}_domain_accuracy.csv", summary["domain_accuracy"])
        write_csv(report_dir / f"{stem}_category_accuracy.csv", summary["category_accuracy"])

    print("\n=== Final Table Metrics ===")
    print(f"Method: {processor.method_label}")
    print(f"Benchmark: {processor.benchmark}")
    print(f"Overall accuracy: {summary['overall_accuracy_percent']:.2f}% ({correct}/{total})")
    print("Duration-wise accuracy:")
    for row in summary["duration_wise_accuracy"]:
        print(
            f"  {row['duration_bucket']}: {row['accuracy_percent']:.2f}% "
            f"({row['correct']}/{row['total_questions']})"
        )
    print(f"Seconds/sample: {summary['cost_metrics']['seconds_per_sample']:.3f}")
    print(f"Turns/sample: {summary['cost_metrics']['average_turns_per_sample']:.3f}")
    print(f"Tool calls/sample: {summary['cost_metrics']['average_tool_calls_per_sample']:.3f}")
    print(f"Gemini tool invocations: {summary['cost_metrics']['gemini_tool_invocations']}")
    print(f"Actual Gemini/OpenRouter API requests: {summary['cost_metrics']['gemini_tool_actual_api_requests']}")
    print(f"Gemini cache hits/no API request: {summary['cost_metrics']['gemini_tool_cache_hits_or_no_api_call']}")
    print(f"GPT tool invocations: {summary['cost_metrics']['gpt_tool_invocations']}")
    print(f"Actual GPT API requests: {summary['cost_metrics']['gpt_tool_actual_api_requests']}")
    print(f"GPT tool API cost USD: {summary['cost_metrics']['gpt_tool_estimated_cost_usd']}")
    print(
        "Null evaluator reasoning cases (not repaired): "
        f"{summary['null_evaluation_reasoning_count']} "
        f"({summary['null_evaluation_reasoning_rate_percent']:.2f}% of all samples)"
    )
    print(
        "Qwen/SAGE-MM total tokens (passive vLLM logging): "
        f"{summary['cost_metrics']['qwen_sage_total_tokens']}"
    )
    print(
        "Empty-media handoff failures: "
        f"{summary['cost_metrics']['empty_media_handoff_failure_count']} "
        f"({summary['cost_metrics']['empty_media_handoff_failure_rate_percent']:.2f}% of all samples)"
    )
    print(f"Reports saved to: {report_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process SAGE/MINERVA videos and write table-ready metrics.")
    parser.add_argument("--benchmark", type=str, default="minerva_bench", choices=["minerva_bench", "sage_bench"])
    parser.add_argument("--model_name", type=str, default="sage:allenai/SAGE-MM-Qwen3-VL-8B-SFT_RL")
    parser.add_argument("--method_label", type=str, default="SAGE")
    parser.add_argument("--use_video", type=str, default="True")
    parser.add_argument("--use_gemini_as_tool", type=str, default="False")
    parser.add_argument("--num_sampled_frames", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--timeout_seconds", type=int, default=300)
    parser.add_argument("--tool_to_drop", type=str, default="None")
    parser.add_argument("--max_num_iterative_reasoner_calls", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--evaluate_after_run", type=str, default="True")
    parser.add_argument("--report_dir", type=str, default="evaluation_reports")
    parser.add_argument("--collect_local_token_ids", type=str, default="False")
    parser.add_argument("--gpu_hourly_cost_usd", type=float, default=None)
    parser.add_argument(
        "--clear_gemini_cache_before_run",
        type=str,
        default="False",
        help="Clear SAGE's persistent Gemini response cache once before inference. Use True for a clean cost-measurement run.",
    )
    parser.add_argument(
        "--synthesized_tool_names",
        type=str,
        default="",
        help="Comma-separated names of dynamically synthesized tools, for Dynamic SAGE runs.",
    )
    parser.add_argument("--synthesis_metrics_json", type=str, default=None)
    parser.add_argument("--synthesis_num_sampled_queries", type=int, default=None)
    parser.add_argument("--synthesis_wall_clock_seconds", type=float, default=None)
    parser.add_argument("--synthesis_prompt_tokens", type=int, default=None)
    parser.add_argument("--synthesis_completion_tokens", type=int, default=None)
    parser.add_argument("--synthesis_cost_usd", type=float, default=None)
    args = parser.parse_args()

    # Create a unique default reports folder for every run.
    # Keep an explicitly provided --report_dir unchanged.
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.report_dir == "evaluation_reports":
        args.report_dir = f"evaluation_reports_{run_stamp}"

    if parse_bool(args.collect_local_token_ids):
        raise SystemExit(
            "Do not use --collect_local_token_ids True for the faithful released-runtime run. "
            "The released SAGE return_ids=True path crashes with: too many values to unpack (expected 4). "
            "Run with --collect_local_token_ids False; Qwen tokens are collected passively from normal vLLM outputs, and GPT/Gemini tool tokens and costs are still logged."
        )

    synthesized_tool_names = [
        name.strip() for name in args.synthesized_tool_names.split(",") if name.strip()
    ]
    processor = Processor(
        model_name=args.model_name,
        method_label=args.method_label,
        num_sampled_frames=args.num_sampled_frames,
        gpu_idx=args.gpu_idx,
        benchmark=args.benchmark,
        timeout_seconds=args.timeout_seconds,
        tool_to_drop=args.tool_to_drop,
        max_num_iterative_reasoner_calls=args.max_num_iterative_reasoner_calls,
        use_gemini_as_tool=parse_bool(args.use_gemini_as_tool),
        use_video=parse_bool(args.use_video),
        collect_local_token_ids=parse_bool(args.collect_local_token_ids),
        gpu_hourly_cost_usd=args.gpu_hourly_cost_usd,
        synthesized_tool_names=synthesized_tool_names,
        clear_gemini_cache_before_run=parse_bool(args.clear_gemini_cache_before_run),
    )

    # (timestamp rename disabled — _skip_completed will resume from existing file)
    pass

    run_info = processor.process_all_videos(
        max_workers=args.workers,
        num_gpus=args.num_gpus,
        gpu_idx=args.gpu_idx,
        limit=args.limit,
    )
    print(f"Done processing for {args.method_label} on GPU {args.gpu_idx}")

    if parse_bool(args.evaluate_after_run):
        evaluate_and_write_reports(processor, run_info, args)


if __name__ == "__main__":
    main()