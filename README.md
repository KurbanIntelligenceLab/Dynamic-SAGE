# Dynamic-SAGE

[![arXiv](https://img.shields.io/badge/arXiv-2607.01469-b31b1b.svg)](https://arxiv.org/abs/2607.01469)

Dynamic-SAGE is an agentic **video question-answering** system that can **synthesize its own tools**. On top of a fixed set of atomic capabilities — frame extraction, speech transcription, visual reasoning, temporal grounding, and web search — it automatically proposes, implements, validates, and installs higher-level *composite* tools that chain those primitives together.

This repository accompanies the paper *A Cost-Aware, Paired Protocol for Auditing Dynamic Tool Synthesis in Agentic Video Question Answering*, and provides the full pipeline for generating, validating, and evaluating dynamically synthesized tools.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [1. Synthesize composite tools](#1-synthesize-composite-tools)
  - [2. Run evaluation](#2-run-evaluation)
  - [3. Try the demo](#3-try-the-demo)
- [The Tool-Synthesis Pipeline](#the-tool-synthesis-pipeline)
- [Atomic Tools](#atomic-tools)
- [Evaluation Notes](#evaluation-notes)
- [License](#license)

---

## How It Works

An orchestrator LLM answers a question about a video by iteratively calling tools. Each atomic tool does one thing — extract frames from a time window, transcribe speech, reason over images, locate an event, or search the web. In practice the orchestrator calls the *same combinations* of these tools over and over.

Dynamic-SAGE observes those recurring combinations and, in an offline pass, generates **composite tools**: single functions that bundle a common chain of atomic calls (e.g. "extract frames *and* transcribe speech for a time window"). Every candidate tool passes through a strict validator before it is allowed into the library, so the orchestrator gets fewer, higher-level tools without sacrificing correctness.

```
                        ┌──────────────────────────────┐
   video + question ──▶ │        Orchestrator (LLM)     │ ──▶ answer
                        └──────────────┬───────────────┘
                                       │ tool calls
                 ┌─────────────────────┼─────────────────────┐
                 ▼                     ▼                     ▼
           atomic tools          composite tools        web search
      (extract / reason /   (synthesized offline by   (search / parse)
       transcribe / ...)     the dynamic_api pipeline)
```

---

## Repository Layout

| Path | Purpose |
|------|---------|
| [main.py](main.py) | The `SAGE` orchestrator class: loads tools and runs the iterative reasoning loop. |
| [dynamic_api/](dynamic_api/) | **The tool-synthesis pipeline** (the core contribution). |
| ├ [generate.py](dynamic_api/generate.py) | Samples benchmark questions, drives the Signature and Implementation agents, validates, and writes the surviving tools. |
| ├ [test_agent.py](dynamic_api/test_agent.py) | The validation gate: static AST checks plus a stubbed dry-run that mirrors real tool contracts without spending API credits. |
| ├ [prompts.py](dynamic_api/prompts.py) | Prompts for the Signature and Implementation agents. |
| └ [outputs/](dynamic_api/outputs/) | Generated artifacts: synthesized tools, holdout IDs, and a generation log. |
| [src/functions/tools/](src/functions/tools/) | The atomic tools: `search`, `extract`, `reason`, `temporal`, `audio`. |
| [src/api/](src/api/) | Model backends (Gemini, GPT, Qwen-VL, Molmo2) with caching layers. |
| [src/models/](src/models/) | Vendored Molmo2 and Qwen-VL model code, including vLLM integration. |
| [src/train/sft.py](src/train/sft.py) | Supervised fine-tuning entry point. |
| [eval/](eval/) | `process-dynamic.py` runs the benchmark; `evaluate_responses.py` scores the results. |
| [serve/](serve/) | A demo app, a transcription API, and example video clips. |
| [prompts/](prompts/) | Text prompts for the reasoner, VLM context, and baselines. |
| [bootstrap.sh](bootstrap.sh) | Provisions a fresh Linux/CUDA GPU machine (conda, PyTorch, vLLM, flash-attn). |

---

## Installation

Dynamic-SAGE targets a Linux machine with an NVIDIA GPU (CUDA 12.6). A bootstrap script provisions everything from scratch:

```bash
bash bootstrap.sh
conda activate sage
```

The script installs system packages (`ffmpeg`, `git-lfs`), Miniconda, a `sage` conda environment (Python 3.11), PyTorch 2.8 / vLLM 0.11 / flash-attn, and the data-download tooling (`yt-dlp`, `huggingface_hub`). If you have a persistent volume, set `HF_HOME` to it so model weights survive instance restarts:

```bash
export HF_HOME=/workspace/hf-cache
```

---

## Configuration

Dynamic-SAGE uses GPT-4o as the orchestrator, served through [OpenRouter](https://openrouter.ai). Export the following before running anything:

```bash
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="YOUR-OPENROUTER-KEY"
export USE_GPT_AS_TOOL=true
export GPT_MAX_FRAMES=64
export VIDEO_DIR="/path/to/your/videos"     # required by the orchestrator
```

Additional variables used during synthesis and evaluation:

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATOR_MODEL` | `openai/gpt-4o` | Model used to propose and implement composite tools. |
| `VIDEO_DIR` | — | Directory of source videos (required). |
| `EXCLUDE_IDS_FILE` | — | Path to the holdout-IDs file to skip during evaluation (see below). |

---

## Usage

### 1. Synthesize composite tools

Sample a batch of questions and generate composite tools from the recurring atomic-tool chains:

```bash
python -m sage.dynamic_api.generate \
    --benchmark sage_bench \
    --max_tools 8 \
    --holdout 50
```

This writes the accepted tools to `dynamic_api/outputs/{benchmark}_synthesized.py`, the sampled question IDs to `{benchmark}_holdout_ids.json`, and a summary to `{benchmark}_generation_log.json`. To activate the tools, drop the synthesized file into `src/functions/tools/synthesized.py`.

### 2. Run evaluation

```bash
export EXCLUDE_IDS_FILE="PATH-TO/dynamic_api/outputs/sage_bench_holdout_ids.json"
python -m sage.eval.process-dynamic
python -m sage.eval.evaluate_responses
```

### 3. Try the demo

The [serve/](serve/) directory contains a demo app and example clips (`serve/examples/*.mp4`) for running the orchestrator on a single video interactively.

---

## The Tool-Synthesis Pipeline

Generation runs in two agent stages, gated by a strict validator:

1. **Signature Agent** — given the inventory of existing tools and a batch of benchmark questions, proposes new composite-tool *signatures* (name, arguments, docstring).
2. **Implementation Agent** — given a signature, writes the function body by chaining existing atomic tools. Failures are fed back and retried up to three times.
3. **Validation** ([test_agent.py](dynamic_api/test_agent.py)) — each candidate must survive both static checks and a stubbed dry-run before it enters the library.

The validator is deliberately aggressive; it encodes what LLM-synthesized tools reliably get wrong:

- **No `question` parameter.** A composite tool gathers *evidence*; answering is the orchestrator's job. A tool that takes the user's question and returns a verdict forecloses reasoning and empirically underperforms.
- **`perform_reasoning` must use `mode="observe"`.** Otherwise the tool returns a finished verdict instead of observations, and downstream code silently reads an empty result.
- **No broad `try/except Exception`.** A catch-all handler swallows the validator's own contract errors and lets buggy tools pass as benign `{"error": ...}` dicts.
- **No dangerous imports** (`subprocess`, `os.system`, `pickle`, `socket`, …).
- **Contract-faithful stubs.** Instead of calling real tools (which hit Gemini or shell out to `ffmpeg`), the dry-run monkey-patches each atomic tool with a stub that validates argument types the same way the real one would — catching, for example, a transcript string passed where a media *path* is expected — without spending API credits.

The included run illustrates how selective this is: of **48 proposals**, **5 were accepted**, **6 rejected**, and **37 skipped as duplicates**.

---

## Atomic Tools

The synthesized tools are built from these primitives (see [src/functions/tools/](src/functions/tools/)):

| Tool | What it does |
|------|--------------|
| `unified_web_search` | Web or image search from a text query. |
| `parse_web_data` | Fetch and parse a webpage when search snippets are insufficient. |
| `extract_parts_from_timestamp` | Extract frames or a subclip between two timestamps. |
| `perform_reasoning` | Reason over a query with optional image/video context (Gemini 2.5 Flash). |
| `identify_timestamps_visually` | Locate the precise timestamps of an event within a time window. |
| `verbal_transcript` | Transcribe speech in a video range (Whisper-large-v3). |

---

## Evaluation Notes

- Each query is assigned a **unique ID at inference time**, so a run that stops midway can resume without re-evaluating every query.
- Because synthesis samples a batch of questions from the dataset, those questions **must be excluded from evaluation** to avoid leakage. The sampled IDs are written to the holdout file; point `EXCLUDE_IDS_FILE` at it before running the benchmark:

  ```bash
  export EXCLUDE_IDS_FILE="PATH-TO/dynamic_api/outputs/sage_bench_holdout_ids.json"
  ```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{mohamed2026costawarepairedprotocolauditing,
      title={A Cost-Aware, Paired Protocol for Auditing Dynamic Tool Synthesis in Agentic Video Question Answering}, 
      author={Aseel Mohamed and Rama AlHamidi and Mohamed Rayan Barhdadi and Rasul Khanbayov and Erchin Serpedin and Hasan Kurban},
      year={2026},
      eprint={2607.01469},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.01469}, 
}
```

## License

See [LICENSE](LICENSE).
