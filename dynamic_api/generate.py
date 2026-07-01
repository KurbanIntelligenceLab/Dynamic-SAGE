"""Offline VADAR-style API generator for SAGE.

Pipeline:
  1. Load a benchmark (sage_bench or minerva_bench) the same way
     `sage/eval/process-full.py` does.
  2. Stratified-sample N=50 questions for the holdout / API-generation pool.
  3. In batches of ~10, prompt a Signature Agent for new tool signatures.
  4. For each signature, prompt an Implementation Agent for the body,
     send it through `test_agent.validate`, retry up to 3x on failure.
  5. Write the surviving tools to `outputs/{benchmark}_synthesized.py`
     and the holdout IDs to `outputs/{benchmark}_holdout_ids.json`.

Usage:
    python -m sage.dynamic_api.generate \\
        --benchmark sage_bench \\
        --max_tools 8 \\
        --holdout 50

Required env vars:
    OPENAI_API_KEY      OpenRouter or OpenAI key
    OPENAI_BASE_URL     defaults to https://openrouter.ai/api/v1
    GENERATOR_MODEL     defaults to "openai/gpt-4o"
    VIDEO_DIR           same as the eval; defaults match process-full.py
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from prompts import (
    IMPLEMENTATION_AGENT_PROMPT,
    SIGNATURE_AGENT_PROMPT,
)
from sage.dynamic_api.test_agent import validate

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Existing-tool inventory. Hand-coded against the 4 tool files you
# pasted; the docstrings here MUST stay in sync with the real tools or
# the Signature Agent will be misled.
# ----------------------------------------------------------------------
EXISTING_TOOL_DOCS = """
def unified_web_search(query: str, search_type: str = "web", num_results: int = 3) -> dict:
    \"\"\"Unified function to perform web search or image search using a text query.
    Args:
        query: Search query (required).
        search_type: "web" or "image".
        num_results: 1..10.
    Returns:
        Dict with search results (a list under "results" or similar).
    \"\"\"

def parse_web_data(website_url: str, max_content_length: int = 5000) -> dict:
    \"\"\"Fetch + parse a webpage. Use only when title+snippet from search are insufficient.
    Args:
        website_url: URL.
        max_content_length: cap on returned content.
    Returns:
        Dict with "title", "content", "url".
    \"\"\"

def extract_parts_from_timestamp(
    video_path: str,
    timestamp_start: str,
    timestamp_end: str,
    extract_type: str = "frames",
) -> dict:
    \"\"\"Extract frames or a subclip from a video between two timestamps.
    Keep the duration under 5 minutes.
    Args:
        video_path: Path to the video.
        timestamp_start: HH:MM:SS.
        timestamp_end: HH:MM:SS.
        extract_type: "frames" or "subclips".
    Returns:
        Dict with "media_paths" (list of str).
    \"\"\"

def perform_reasoning(query: str, media_paths: list) -> dict:
    \"\"\"Reason over a query, optionally with images or a video as visual context.
    This is an LLM call (Gemini 2.5 Flash).
    Args:
        query: Question to answer.
        media_paths: List of image or video paths (may be empty for text-only).
    Returns:
        Dict with "answer" (str).
    \"\"\"

def identify_timestamps_visually(
    video_path: str,
    event: str,
    timestamp_start: str,
    timestamp_end: str,
) -> dict:
    \"\"\"Within a time window, find the precise timestamps of an event.
    Window must be under 10 minutes. You cannot scan the whole video.
    Args:
        video_path: Path to the video.
        event: Natural-language description of the event.
        timestamp_start: HH:MM:SS — start of search window.
        timestamp_end: HH:MM:SS — end of search window.
    Returns:
        Dict with "name" and "timestamps" = {"start", "end"}.
    \"\"\"

def verbal_transcript(video_path: str, timestamp_start: str, timestamp_end: str) -> dict:
    \"\"\"Transcribe the speech in a video range (Whisper-large-v3).
    Args:
        video_path: Path to the video.
        timestamp_start: HH:MM:SS.
        timestamp_end: HH:MM:SS.
    Returns:
        Dict with "transcript" (str).
    \"\"\"
""".strip()


# Full source code for the existing tools, pasted in for the Implementation
# Agent. These are exact copies of what you have on disk (the ones you
# pasted to me). The implementations are what the agent should
# mirror in style.
EXISTING_TOOL_SOURCES = """
# ---- sage/src/functions/tools/search.py ----
def unified_web_search(query: str, search_type: str = "web", num_results: int = 3) -> dict:
    \"\"\"... see EXISTING_TOOL_DOCS ...\"\"\"
    # Returns a dict with search results.

def parse_web_data(website_url: str, max_content_length: int = 5000) -> dict:
    \"\"\"... see EXISTING_TOOL_DOCS ...\"\"\"
    # Returns {"title": str, "content": str, "url": str}.

# ---- sage/src/functions/tools/extract.py ----
def extract_parts_from_timestamp(
    video_path: str,
    timestamp_start: str,
    timestamp_end: str,
    extract_type: str = "frames",
) -> dict:
    \"\"\"... see EXISTING_TOOL_DOCS ...\"\"\"
    # Returns {"media_paths": List[str]}.

# ---- sage/src/functions/tools/reason.py ----
def perform_reasoning(query: str, media_paths: list) -> dict:
    \"\"\"... see EXISTING_TOOL_DOCS ...\"\"\"
    # Returns {"answer": str}.

# ---- sage/src/functions/tools/temporal.py ----
def identify_timestamps_visually(
    video_path: str,
    event: str,
    timestamp_start: str,
    timestamp_end: str,
) -> dict:
    \"\"\"... see EXISTING_TOOL_DOCS ...\"\"\"
    # Returns {"name": str, "timestamps": {"start": "HH:MM:SS", "end": "HH:MM:SS"}}.

# ---- sage/src/functions/tools/audio.py ----
def verbal_transcript(video_path: str, timestamp_start: str, timestamp_end: str) -> dict:
    \"\"\"... see EXISTING_TOOL_DOCS ...\"\"\"
    # Returns {"transcript": str}.
""".strip()


STUBBABLE_TOOLS = [
    "unified_web_search",
    "parse_web_data",
    "extract_parts_from_timestamp",
    "perform_reasoning",
    "identify_timestamps_visually",
    "verbal_transcript",
]


# ----------------------------------------------------------------------
# LLM client (OpenAI-compatible, via OpenRouter by default)
# ----------------------------------------------------------------------

# Per-model USD per 1M tokens (input, output). Update if your model isn't here;
# unknown models default to 0 and are flagged in the log.
PRICE_TABLE_USD_PER_1M = {
    "openai/gpt-4o":            (2.50, 10.00),
    "openai/gpt-4o-mini":       (0.15,  0.60),
    "openai/gpt-4-turbo":       (10.00, 30.00),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-sonnet-4": (3.00, 15.00),
    "google/gemini-2.5-flash":  (0.30,  2.50),
    "google/gemini-2.5-pro":    (1.25, 10.00),
}


# Module-level cost ledger. Each entry: {role, attempt, prompt_tokens, completion_tokens, cost_usd, ts}
COST_LEDGER: List[Dict[str, Any]] = []


def _cost_for(model: str, in_toks: int, out_toks: int) -> Tuple[float, bool]:
    """Return (usd_cost, price_was_known)."""
    if model in PRICE_TABLE_USD_PER_1M:
        in_rate, out_rate = PRICE_TABLE_USD_PER_1M[model]
        cost = (in_toks * in_rate + out_toks * out_rate) / 1_000_000
        return cost, True
    return 0.0, False


def _llm_call(prompt: str, temperature: float = 0.7, max_tokens: int = 4000,
              role: str = "unknown") -> str:
    """Call the configured generator LLM via OpenAI-compatible API.
    Side effect: appends a row to COST_LEDGER with token + cost info."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package required. Install with: pip install openai"
        ) from e

    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    model = os.environ.get("GENERATOR_MODEL", "openai/gpt-4o")

    client = OpenAI(api_key=api_key, base_url=base_url)
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Record usage. OpenAI-compatible responses always include .usage.
            usage = getattr(resp, "usage", None)
            in_toks = getattr(usage, "prompt_tokens", 0) if usage else 0
            out_toks = getattr(usage, "completion_tokens", 0) if usage else 0
            cost, known = _cost_for(model, in_toks, out_toks)
            COST_LEDGER.append({
                "role": role,
                "attempt": attempt,
                "model": model,
                "prompt_tokens": in_toks,
                "completion_tokens": out_toks,
                "cost_usd": cost,
                "price_known": known,
                "ts": time.time(),
            })
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"[llm] attempt {attempt + 1} failed: {e}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after 3 attempts: {last_err}")


def _cost_summary() -> Dict[str, Any]:
    """Aggregate the COST_LEDGER into a summary dict."""
    by_role: Dict[str, Dict[str, Any]] = {}
    for row in COST_LEDGER:
        r = row["role"]
        if r not in by_role:
            by_role[r] = {"n_calls": 0, "prompt_tokens": 0,
                          "completion_tokens": 0, "cost_usd": 0.0}
        by_role[r]["n_calls"] += 1
        by_role[r]["prompt_tokens"] += row["prompt_tokens"]
        by_role[r]["completion_tokens"] += row["completion_tokens"]
        by_role[r]["cost_usd"] += row["cost_usd"]

    total = {
        "n_calls": len(COST_LEDGER),
        "prompt_tokens": sum(r["prompt_tokens"] for r in COST_LEDGER),
        "completion_tokens": sum(r["completion_tokens"] for r in COST_LEDGER),
        "cost_usd": sum(r["cost_usd"] for r in COST_LEDGER),
    }
    return {"by_role": by_role, "total": total,
            "any_unknown_price": any(not r["price_known"] for r in COST_LEDGER)}


# ----------------------------------------------------------------------
# Benchmark loaders. Mirror load_sage_bench_videos / load_minerva_videos.
# ----------------------------------------------------------------------

def load_sage_bench_questions(video_dir: str) -> List[Dict[str, Any]]:
    """Load SAGE-Bench question metadata, filtered to videos that exist."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "datasets package required. Install with: pip install datasets"
        ) from e

    print(f"Loading allenai/SAGE-Bench test split ...")
    videos = load_dataset("allenai/SAGE-Bench", split="test")
    out = []
    skipped = 0
    for v in videos:
        video_path = os.path.join(video_dir, f"{v['video_id']}.mp4")
        if not os.path.exists(video_path):
            skipped += 1
            continue
        qid = hashlib.md5(f"{v['question']}|{video_path}".encode()).hexdigest()
        out.append(
            {
                "id": qid,
                "video_id": v["video_id"],
                "question": v["question"],
                "ques_type": str(v["ques_type"]).replace("-", "_"),
                "difficulty": v.get("difficulty", "unknown"),
                "modality": v.get("modality", "unknown"),
                "duration_seconds": float(v["duration_seconds"]),
            }
        )
    print(f"  loaded {len(out)} samples; skipped {skipped} missing-video")
    return out


def load_minerva_questions(video_dir: str = "data/minerva_videos") -> List[Dict[str, Any]]:
    """Load MINERVA question metadata, filtered to videos that exist."""
    dataset_path = "data/minerva_videos/minerva.json"
    print(f"Loading MINERVA from {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        videos = json.load(f)
    out = []
    skipped = 0
    for v in videos:
        video_path = os.path.join(video_dir, v["video_id"], f"{v['video_id']}.mp4")
        if not os.path.exists(video_path):
            skipped += 1
            continue
        qid = hashlib.md5(f"{v['question']}|{video_path}".encode()).hexdigest()
        # Build the textual question with options inline so the agent sees what
        # the orchestrator sees.
        letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        choices = "\n".join(
            f"({letter[i]}) {v[f'answer_choice_{i}']}" for i in range(5)
        )
        full_question = (
            v["question"]
            + "\nAnswer from the given options:\n"
            + choices
        )
        out.append(
            {
                "id": qid,
                "video_id": v["video_id"],
                "question": full_question,
                "ques_type": "mcq",
                "domain": v.get("domain", "unknown"),
                "category": v.get("category", "unknown"),
                "question_type": v.get("question_type", "unknown"),
                "duration_seconds": float(
                    v.get("duration", v.get("duration_seconds", 0))
                ),
            }
        )
    print(f"  loaded {len(out)} samples; skipped {skipped} missing-video")
    return out


# ----------------------------------------------------------------------
# Stratified sampling
# ----------------------------------------------------------------------

def _duration_bucket(seconds: float) -> str:
    if seconds < 60:
        return "0-60"
    if seconds < 180:
        return "60-180"
    if seconds < 300:
        return "180-300"
    if seconds < 600:
        return "300-600"
    if seconds < 1200:
        return "600-1200"
    if seconds < 2400:
        return "1200-2400"
    return "2400+"


def stratified_sample(
    rows: List[Dict[str, Any]],
    n: int,
    keys: List[str],
    seed: int = 17,
) -> List[Dict[str, Any]]:
    """Sample `n` items from `rows` stratified by tuple of `keys`.
    Falls back to uniform sampling if `n` exceeds the row count."""
    rng = random.Random(seed)
    if n >= len(rows):
        return list(rows)

    def stratum(r: Dict[str, Any]) -> Tuple:
        return tuple(
            _duration_bucket(r["duration_seconds"]) if k == "duration_bucket"
            else r.get(k, "unknown")
            for k in keys
        )

    by_stratum: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_stratum[stratum(r)].append(r)

    strata = list(by_stratum.keys())
    # Proportional allocation
    total = len(rows)
    allocations = {
        s: max(1, round(n * len(by_stratum[s]) / total)) for s in strata
    }
    # Trim/expand to exactly n
    while sum(allocations.values()) > n:
        biggest = max(allocations, key=lambda s: allocations[s])
        allocations[biggest] -= 1
    while sum(allocations.values()) < n:
        s = rng.choice(strata)
        if allocations[s] < len(by_stratum[s]):
            allocations[s] += 1

    out: List[Dict[str, Any]] = []
    for s, k in allocations.items():
        items = list(by_stratum[s])
        rng.shuffle(items)
        out.extend(items[:k])
    rng.shuffle(out)
    return out


# ----------------------------------------------------------------------
# Parse Signature Agent and Implementation Agent outputs
# ----------------------------------------------------------------------

PROPOSAL_RE = re.compile(
    r"<proposal>\s*(.*?)\s*</proposal>", re.DOTALL
)
DOCSTRING_RE = re.compile(r"<docstring>\s*(.*?)\s*</docstring>", re.DOTALL)
SIGNATURE_RE = re.compile(r"<signature>\s*(.*?)\s*</signature>", re.DOTALL)
IMPORTS_RE = re.compile(r"<imports>\s*(.*?)\s*</imports>", re.DOTALL)
IMPL_RE = re.compile(r"<implementation>\s*(.*?)\s*</implementation>", re.DOTALL)


def parse_proposals(llm_text: str) -> List[Dict[str, str]]:
    out = []
    for block in PROPOSAL_RE.findall(llm_text):
        doc_m = DOCSTRING_RE.search(block)
        sig_m = SIGNATURE_RE.search(block)
        if not doc_m or not sig_m:
            continue
        out.append({"docstring": doc_m.group(1).strip(), "signature": sig_m.group(1).strip()})
    return out


def parse_implementation(llm_text: str) -> Dict[str, str]:
    """Permissive parser: tries multiple output formats the LLM might use.

    Recognised, in priority order:
      1. <implementation>...</implementation>  (preferred — what we ask for)
      2. ```python ... ``` or ``` ... ``` fenced code block
      3. Bare ``def ...:`` block — extract its body
    Imports are recognised from <imports>...</imports> OR from any
    top-of-text `from X import Y` / `import X` lines.
    """
    imports_buf: List[str] = []
    body = ""

    # --- imports ---
    imports_m = IMPORTS_RE.search(llm_text)
    if imports_m:
        imports_buf.append(imports_m.group(1).strip())

    # --- body, preferred form ---
    imp_m = IMPL_RE.search(llm_text)
    if imp_m:
        body = imp_m.group(1).strip()

    # --- fallback: fenced code block ---
    if not body:
        fenced = re.search(
            r"```(?:python|py)?\s*\n(.*?)```",
            llm_text,
            re.DOTALL,
        )
        if fenced:
            code = fenced.group(1).strip()
            # Split off imports from the top
            kept_body_lines: List[str] = []
            in_imports_block = True
            for line in code.splitlines():
                stripped = line.strip()
                if in_imports_block and (
                    stripped.startswith("from ") or stripped.startswith("import ")
                ):
                    imports_buf.append(stripped)
                    continue
                if stripped and in_imports_block:
                    in_imports_block = False
                kept_body_lines.append(line)
            code = "\n".join(kept_body_lines).strip()
            # If the code is a full `def ...:` block, peel off the def line
            # and the docstring, keep just the body.
            if code.startswith("def "):
                body = _peel_def_to_body(code)
            else:
                body = code

    # --- last-ditch fallback: scan for a def block in the raw text ---
    if not body:
        def_m = re.search(
            r"^(def\s+[A-Za-z_][\w]*\s*\([^)]*\)\s*->?\s*\w*\s*:.*?)(?=^\S|\Z)",
            llm_text,
            re.MULTILINE | re.DOTALL,
        )
        if def_m:
            body = _peel_def_to_body(def_m.group(1).strip())

    # --- imports from raw text if we still have none ---
    if not imports_buf:
        # Collect any line that looks like an import, anywhere in the text
        # (above or outside our tags).
        for line in llm_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("from sage.src.functions.tools") or (
                stripped.startswith("import ") and "sage" in stripped
            ):
                imports_buf.append(stripped)

    return {
        "imports": "\n".join(dict.fromkeys(imports_buf)),  # dedupe, preserve order
        "body": body,
    }


def _peel_def_to_body(def_block: str) -> str:
    '''Given a full `def name(...):` block with docstring + body,
    return just the body lines, dedented. If parsing fails, return the
    block as-is.'''
    try:
        tree = ast.parse(def_block)
    except SyntaxError:
        return def_block
    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef)), None
    )
    if fn is None:
        return def_block
    # Strip leading docstring expr if present
    body_nodes = fn.body
    if (
        body_nodes
        and isinstance(body_nodes[0], ast.Expr)
        and isinstance(body_nodes[0].value, ast.Constant)
        and isinstance(body_nodes[0].value.value, str)
    ):
        body_nodes = body_nodes[1:]
    if not body_nodes:
        return ""
    # Reconstruct via ast.unparse (Python 3.9+) for clean output
    try:
        body_src = "\n".join(ast.unparse(n) for n in body_nodes)
    except Exception:
        # Fallback: slice from source by line numbers
        lines = def_block.splitlines()
        start = body_nodes[0].lineno - 1
        body_src = "\n".join(lines[start:])
        body_src = textwrap.dedent(body_src)
    return body_src.strip()


# ----------------------------------------------------------------------
# Assemble a full function from (docstring, signature, body)
# ----------------------------------------------------------------------

def assemble_function(docstring: str, signature: str, imports: str, body: str) -> str:
    """Build a syntactically valid function definition from parts.

    The body coming from the LLM frequently has mixed indentation because
    the model writes the first line right after <implementation> (no leading
    whitespace) and continuation lines inside it (4 or more spaces). We
    normalise by AST-parsing the body inside a synthetic `def` wrapper and
    re-emitting it via ast.unparse, which produces a canonically-indented
    body. If parsing fails, we fall back to a heuristic.
    """
    body_normalised = _normalise_body(body, signature)

    doc = docstring.strip()
    if doc.startswith('"""'):
        doc = doc[3:]
    if doc.endswith('"""'):
        doc = doc[:-3]
    doc = textwrap.dedent(doc).strip()
    doc_indented = textwrap.indent('"""' + doc + '\n"""', "    ")

    source = f"{imports}\n\n{signature}\n{doc_indented}\n{body_normalised}\n"
    return source.lstrip()


def _normalise_body(body: str, signature: str) -> str:
    """Return the body indented uniformly to 4 spaces.

    Strategy: wrap the body in a synthetic def, try to parse it. If that
    works, the body's intrinsic indentation was already consistent and we
    just textwrap.dedent + re-indent it. If not, try heuristics that
    handle the common LLM failure mode (first line at column 0,
    continuation lines at deeper indentation).
    """
    # Strip trailing whitespace per line but keep leading
    raw = body.rstrip("\n")

    # First try: assume the body's indentation is consistent and only
    # needs dedent+indent. Build a candidate function and ast.parse it.
    candidate = signature + "\n" + textwrap.indent(textwrap.dedent(raw), "    ")
    try:
        ast.parse(candidate)
        return textwrap.indent(textwrap.dedent(raw).strip(), "    ")
    except SyntaxError:
        pass

    # Heuristic fix: find the minimum non-zero indentation that appears in
    # the body and treat THAT as the body's "column 0". This handles the
    # case where the first line is unindented (because it came right after
    # `<implementation>`) but subsequent lines are indented.
    lines = raw.split("\n")
    non_empty_indents = [
        len(ln) - len(ln.lstrip(" "))
        for ln in lines
        if ln.strip() and ln.startswith(" ")
    ]
    if non_empty_indents:
        min_indent = min(non_empty_indents)
        fixed_lines = []
        for ln in lines:
            if not ln.strip():
                fixed_lines.append("")
                continue
            if ln.startswith(" "):
                # Strip the body's "base" indentation
                fixed_lines.append(ln[min_indent:] if len(ln) > min_indent else ln.lstrip(" "))
            else:
                # First line / unindented top-level statement — leave as-is
                fixed_lines.append(ln)
        fixed = "\n".join(fixed_lines).strip("\n")
        candidate = signature + "\n" + textwrap.indent(fixed, "    ")
        try:
            ast.parse(candidate)
            return textwrap.indent(fixed, "    ")
        except SyntaxError:
            pass

    # Last resort: AST-parse the body wrapped in a try block to coerce it
    # into valid syntax, then re-emit. Many bodies are unparseable on
    # their own (return-only, etc), so we wrap them.
    try:
        wrapped = signature + "\n    pass\n"
        # ast.parse will succeed on the wrapped form; we use ast to rebuild
        # but if even the heuristic fix didn't parse, give up and return
        # the dedent-only version. The validate step will reject it.
        ast.parse(wrapped)
    except SyntaxError:
        pass
    return textwrap.indent(textwrap.dedent(raw).strip(), "    ")


# ----------------------------------------------------------------------
# Composition-based dedup
# ----------------------------------------------------------------------

def _composition_fingerprint(body: str) -> Tuple[str, ...]:
    """Return a tuple describing which base SAGE tools the body calls
    and in what order. Two tools with the same fingerprint compose the
    same primitives the same way; their difference is only the natural-
    language query strings, which the orchestrator can supply itself.

    Example:
      def _foo(...):
          x = extract_parts_from_timestamp(...)
          y = perform_reasoning(query=..., media_paths=x[...])
          return {...}
    Fingerprint: ("extract_parts_from_timestamp", "perform_reasoning")
    """
    base_tool_names = {
        "unified_web_search",
        "parse_web_data",
        "extract_parts_from_timestamp",
        "perform_reasoning",
        "identify_timestamps_visually",
        "verbal_transcript",
    }
    calls: List[str] = []
    try:
        tree = ast.parse(body)
    except SyntaxError:
        # Fall back to a textual scan if the body isn't standalone-parseable
        # (e.g. just statements, no enclosing def). Look for `name(`.
        for name in base_tool_names:
            # Conservative: count occurrences but preserve order via first-find
            idx = body.find(name + "(")
            if idx != -1:
                calls.append((idx, name))
        calls.sort()
        return tuple(name for _, name in calls)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = ""
            if isinstance(node.func, ast.Name):
                target = node.func.id
            elif isinstance(node.func, ast.Attribute):
                target = node.func.attr
            if target in base_tool_names:
                calls.append(target)
    return tuple(calls)


def _is_pure_wrapper(fingerprint: Tuple[str, ...]) -> bool:
    """A tool that calls exactly one base tool and nothing else is a
    pure wrapper — it adds prompt overhead with no composition benefit.
    """
    return len(fingerprint) == 1


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def generate(
    benchmark: str,
    holdout: int,
    max_tools: int,
    batch_size: int,
    video_dir: str,
    seed: int,
) -> None:
    # 1. Load
    if benchmark == "sage_bench":
        rows = load_sage_bench_questions(video_dir)
        strata_keys = ["duration_bucket", "ques_type", "modality"]
    elif benchmark == "minerva_bench":
        rows = load_minerva_questions(video_dir)
        strata_keys = ["duration_bucket", "domain"]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # 2. Sample holdout
    holdout_rows = stratified_sample(rows, holdout, strata_keys, seed=seed)
    holdout_ids = [r["id"] for r in holdout_rows]
    print(f"Selected {len(holdout_rows)} holdout questions.")
    print(
        "Strata distribution: "
        + str(Counter([
            (_duration_bucket(r["duration_seconds"]), r.get(strata_keys[1], "?"))
            for r in holdout_rows
        ]))
    )

    holdout_ids_path = OUTPUTS / f"{benchmark}_holdout_ids.json"
    with open(holdout_ids_path, "w") as f:
        json.dump(holdout_ids, f, indent=2)
    print(f"Wrote holdout IDs → {holdout_ids_path}")

    # 3. Signature Agent in batches
    proposals_by_batch: List[List[Dict[str, str]]] = []
    # Track all signatures proposed so far across batches so we can show
    # them to subsequent batches as "already covered, don't re-propose".
    sigs_seen_so_far: List[str] = []

    for i in range(0, len(holdout_rows), batch_size):
        batch = holdout_rows[i : i + batch_size]
        q_text = "\n".join(
            f"{j+1}. {r['question']}" for j, r in enumerate(batch)
        )
        # Augment the existing-tools section with any signatures proposed
        # in earlier batches, so the Signature Agent doesn't re-propose
        # near-duplicates. We append them under a clear header so the
        # agent sees them as "already covered".
        tool_docs_with_history = EXISTING_TOOL_DOCS
        if sigs_seen_so_far:
            tool_docs_with_history += (
                "\n\n# Already proposed in earlier batches "
                "(DO NOT propose duplicates or near-duplicates of these):\n"
                + "\n".join(sigs_seen_so_far)
            )
        prompt = SIGNATURE_AGENT_PROMPT.format(
            existing_tool_docs=tool_docs_with_history,
            question_batch=q_text,
        )
        print(f"\n[signature] batch {i // batch_size + 1} "
              f"({len(batch)} questions; {len(sigs_seen_so_far)} seen)")
        raw = _llm_call(prompt, temperature=0.7, role="signature_agent")
        proposals = parse_proposals(raw)
        print(f"  → {len(proposals)} proposals")
        for p in proposals:
            sig_line = p["signature"].splitlines()[0]
            print(f"    - {sig_line}")
            sigs_seen_so_far.append(sig_line)
        proposals_by_batch.append(proposals)

    all_proposals: List[Dict[str, str]] = []
    seen_names = set()
    for proposals in proposals_by_batch:
        for p in proposals:
            # Extract function name from `def _name(`
            m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", p["signature"])
            if not m:
                continue
            name = m.group(1)
            if name in seen_names:
                continue
            seen_names.add(name)
            p["name"] = name
            all_proposals.append(p)
    print(f"\n{len(all_proposals)} unique proposals across all batches.")

    # 4. Implementation Agent
    accepted: List[Dict[str, Any]] = []
    accepted_fingerprints: set = set()
    rejected: List[Dict[str, Any]] = []
    skipped_dup: List[Dict[str, Any]] = []
    skipped_wrapper: List[Dict[str, Any]] = []
    sibling_sigs_str = "\n".join(p["signature"] for p in all_proposals)

    for prop in all_proposals:
        if len(accepted) >= max_tools:
            print(f"\nReached max_tools={max_tools}; stopping.")
            break
        print(f"\n[impl] {prop['name']}")
        last_err = ""
        function_src = ""
        for attempt in range(3):
            extra = ""
            if attempt > 0 and last_err:
                extra = (
                    f"\n\nPrevious attempt FAILED validation:\n{last_err}\n"
                    "Fix the issue and try again."
                )
            prompt = (
                IMPLEMENTATION_AGENT_PROMPT.format(
                    existing_tool_sources=EXISTING_TOOL_SOURCES,
                    sibling_signatures=sibling_sigs_str,
                    docstring=prop["docstring"],
                    signature=prop["signature"],
                )
                + extra
            )
            raw = _llm_call(prompt, temperature=0.3, role="implementation_agent")
            parts = parse_implementation(raw)
            if not parts["body"]:
                last_err = "No implementation body could be parsed from output."
                print(f"  attempt {attempt + 1}: missing impl block")
                debug_path = OUTPUTS / f"{benchmark}_raw_outputs.txt"
                with open(debug_path, "a") as df:
                    df.write(f"\n\n===== {prop['name']} attempt {attempt + 1} =====\n")
                    df.write(raw)
                continue
            function_src = assemble_function(
                docstring=prop["docstring"],
                signature=prop["signature"],
                imports=parts["imports"],
                body=parts["body"],
            )
            ok, err = validate(function_src, stubbed_tools=STUBBABLE_TOOLS)
            if ok:
                # Composition-based dedup.
                fp = _composition_fingerprint(parts["body"])
                if _is_pure_wrapper(fp):
                    print(f"  attempt {attempt + 1}: PASS but is a pure wrapper of "
                          f"{fp[0]} — skipped (no composition benefit)")
                    skipped_wrapper.append({
                        "name": prop["name"],
                        "fingerprint": list(fp),
                    })
                    break
                if fp in accepted_fingerprints:
                    print(f"  attempt {attempt + 1}: PASS but duplicates "
                          f"composition {fp} — skipped")
                    skipped_dup.append({
                        "name": prop["name"],
                        "fingerprint": list(fp),
                    })
                    break
                print(f"  attempt {attempt + 1}: PASS — composition={fp}")
                accepted_fingerprints.add(fp)
                accepted.append({
                    "name": prop["name"],
                    "signature": prop["signature"],
                    "docstring": prop["docstring"],
                    "imports": parts["imports"],
                    "body": parts["body"],
                    "full_source": function_src,
                    "fingerprint": list(fp),
                })
                break
            else:
                last_err = err
                print(f"  attempt {attempt + 1}: FAIL — {err[:200]}")
                debug_path = OUTPUTS / f"{benchmark}_assembled_failures.txt"
                with open(debug_path, "a") as df:
                    df.write(f"\n\n===== {prop['name']} attempt {attempt + 1} =====\n")
                    df.write(f"ERROR: {err}\n")
                    df.write(f"--- RAW LLM OUTPUT ---\n{raw}\n")
                    df.write(f"--- PARSED IMPORTS ---\n{parts['imports']}\n")
                    df.write(f"--- PARSED BODY ---\n{parts['body']}\n")
                    df.write(f"--- ASSEMBLED SOURCE ---\n{function_src}\n")
                    df.write(f"--- END ---\n")
        else:
            rejected.append({"name": prop["name"], "last_err": last_err})

    print(f"\n{len(accepted)} tools accepted, "
          f"{len(rejected)} rejected by validator, "
          f"{len(skipped_dup)} skipped as duplicates, "
          f"{len(skipped_wrapper)} skipped as pure wrappers.")

    # 5. Write outputs
    synth_path = OUTPUTS / f"{benchmark}_synthesized.py"
    log_path = OUTPUTS / f"{benchmark}_generation_log.json"

    header = textwrap.dedent(f'''\
        """Auto-generated by sage/dynamic_api/generate.py.

        Benchmark: {benchmark}
        Tools: {len(accepted)}
        Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

        Drop this file into sage/src/functions/tools/synthesized.py to activate.
        """
        from typing import List, Dict, Any
        from sage.src.functions.tools.search import unified_web_search, parse_web_data
        from sage.src.functions.tools.extract import extract_parts_from_timestamp
        from sage.src.functions.tools.reason import perform_reasoning
        from sage.src.functions.tools.temporal import identify_timestamps_visually
        from sage.src.functions.tools.audio import verbal_transcript
        ''')

    body_blocks = []
    for tool in accepted:
        # Strip the agent's own imports (we have a unified set at the top)
        # but keep its full function source minus imports.
        src = tool["full_source"]
        # Drop lines that are imports
        kept = []
        for ln in src.splitlines():
            stripped = ln.strip()
            if stripped.startswith("from ") or stripped.startswith("import "):
                continue
            kept.append(ln)
        body_blocks.append("\n".join(kept).strip("\n"))

    full = header + "\n\n" + "\n\n\n".join(body_blocks) + "\n"
    with open(synth_path, "w") as f:
        f.write(full)
    print(f"\nWrote synthesized tools → {synth_path}")

    cost = _cost_summary()
    with open(log_path, "w") as f:
        json.dump(
            {
                "benchmark": benchmark,
                "n_proposals": len(all_proposals),
                "n_accepted": len(accepted),
                "n_rejected": len(rejected),
                "n_skipped_duplicate": len(skipped_dup),
                "n_skipped_wrapper": len(skipped_wrapper),
                "accepted": [
                    {
                        "name": t["name"],
                        "signature": t["signature"],
                        "fingerprint": t["fingerprint"],
                    }
                    for t in accepted
                ],
                "rejected": rejected,
                "skipped_duplicate": skipped_dup,
                "skipped_wrapper": skipped_wrapper,
                "holdout_ids": holdout_ids,
                "config": {
                    "holdout": holdout,
                    "max_tools": max_tools,
                    "batch_size": batch_size,
                    "seed": seed,
                    "model": os.environ.get("GENERATOR_MODEL", "openai/gpt-4o"),
                },
                "cost": cost,
                "cost_ledger": COST_LEDGER,
            },
            f,
            indent=2,
        )
    print(f"Wrote generation log → {log_path}")
    print(f"\nGeneration cost summary:")
    print(f"  total LLM calls: {cost['total']['n_calls']}")
    print(f"  prompt tokens:   {cost['total']['prompt_tokens']:,}")
    print(f"  completion toks: {cost['total']['completion_tokens']:,}")
    print(f"  cost:            ${cost['total']['cost_usd']:.4f}")
    if cost["any_unknown_price"]:
        print(f"  WARNING: model price not in table; cost may be undercounted. "
              f"Add to PRICE_TABLE_USD_PER_1M in generate.py.")
    for role_name, stats in cost["by_role"].items():
        print(f"  - {role_name}: {stats['n_calls']} calls, "
              f"${stats['cost_usd']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["sage_bench", "minerva_bench"], required=True)
    p.add_argument("--holdout", type=int, default=50, help="N questions to sample for API generation; these will be excluded from final eval.")
    p.add_argument("--max_tools", type=int, default=8, help="Cap on accepted tools.")
    p.add_argument("--batch_size", type=int, default=10, help="Questions per Signature Agent batch.")
    p.add_argument("--video_dir", default=None, help="Override video dir. Defaults to $VIDEO_DIR or 'converted_videos' (sage_bench) / 'data/minerva_videos' (minerva).")
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    if args.video_dir:
        video_dir = args.video_dir
    elif args.benchmark == "sage_bench":
        video_dir = os.environ.get("VIDEO_DIR", "converted_videos")
    else:
        video_dir = os.environ.get("VIDEO_DIR", "data/minerva_videos")

    generate(
        benchmark=args.benchmark,
        holdout=args.holdout,
        max_tools=args.max_tools,
        batch_size=args.batch_size,
        video_dir=video_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
