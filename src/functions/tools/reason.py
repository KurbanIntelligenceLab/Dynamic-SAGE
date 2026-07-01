from typing import List, Dict, Any
from sage.src.api.response import get_response
import os

USE_GPT_AS_TOOL = os.getenv("USE_GPT_AS_TOOL", "False").lower() == "true"

VISUAL_REASONING_SYSTEM_PROMPT = """
You are a visual evidence analysis tool for video question answering.

You are given extracted images or video media that have already been selected as relevant to the user's question. Your job is to inspect that provided media and answer the question directly.

You are NOT a planner and you are NOT a tool-selection agent.

Rules:
- Use only visible evidence from the provided media.
- Answer the user's question directly whenever the visible evidence reasonably supports an answer.
- For visual identification questions, give the best visually supported identification.
- Do not refuse merely because minor uncertainty remains.
- Do not recommend tools.
- Do not request additional frames, OCR, image recognition, video analysis, transcription, web search, or any other capability.
- Do not invent tool names.
- Do not output recommended_tools.
- Do not output tool_calls.

Return ONLY valid JSON in exactly one of these formats.

If answerable:
{
  "answerable": {
    "verdict": true,
    "reasoning": "Brief visual evidence supporting the answer."
  },
  "final_answer": "Concise answer grounded in the visible evidence."
}

Only if the supplied media genuinely cannot support an answer:
{
  "answerable": {
    "verdict": false,
    "reasoning": "Why the supplied media is insufficient."
  },
  "final_answer": null
}
""".strip()


VISUAL_OBSERVATION_SYSTEM_PROMPT = """
You are a visual evidence extraction tool for video question answering.

You are given extracted images (or video media) and a description of what to look for. Your job is to REPORT CONCRETE, VERIFIABLE OBSERVATIONS from the provided media. You are NOT answering a question and you are NOT making a decision.

Rules:
- Report only what is visibly supported by the provided media (and any transcript text included in the request).
- Be specific and verifiable: exact colors, exact counts, any on-screen text or numbers (transcribe them verbatim), specific names, positions, and actions. If a transcript is provided in the request, quote the exact relevant wording.
- State plainly what is NOT visible or is ambiguous, rather than guessing.
- Do NOT answer a question, and do NOT produce a single conclusion, verdict, or "final answer".
- Do NOT recommend tools, request more frames, or mention capabilities you lack (beyond noting that something is not visible).

Output: a plain-text, itemized list of concrete observations. Do NOT output JSON. Do NOT output a verdict or a final_answer.
""".strip()


def perform_reasoning(query: str, media_paths: List[str], mode: str = "answer") -> Dict[str, Any]:
    """
    Analyze supplied image/video evidence.

    Args:
        query: What to look for / the question or focus.
        media_paths: List of frame-image (or video) file paths to inspect.
        mode: "answer" (default) returns a direct answer under the "answer" key —
            use when the caller wants a conclusion (e.g. the orchestrator using
            this tool as its eyes). "observe" returns concrete, verifiable
            observations under the "observations" key WITHOUT answering any
            question or producing a verdict — use this inside composite tools that
            gather evidence for the orchestrator to reason over, so the tool does
            not foreclose the orchestrator's own reasoning.

    This tool must not plan or recommend further tools.
    """
    media_paths = media_paths or []
    observe = str(mode).lower() == "observe"

    for media_path in media_paths:
        if not os.path.exists(media_path):
            raise FileNotFoundError(
                f"Media file does not exist: {media_path}, "
                f"in the passed media list: {media_paths}"
            )

    if media_paths:
        is_video = any(
            media_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))
            for media_path in media_paths
        )
        media_type = "video" if is_video else "image"
    else:
        media_type = None

    answer = get_response(
        query,
        sys_prompt=VISUAL_OBSERVATION_SYSTEM_PROMPT if observe else VISUAL_REASONING_SYSTEM_PROMPT,
        media_urls=media_paths or None,
        media_type=media_type,
        temperature=0.0,
        model_name="gemini:gemini-2.5-flash" if not USE_GPT_AS_TOOL else "gpt:gpt-4o",
    )

    # Preserve semantic visual evidence, but remove internal request payloads
    # containing prompts and base64-encoded media already analyzed by the tool.
    if isinstance(answer, tuple):
        semantic_answer = answer[0]
    elif isinstance(answer, list) and len(answer) > 0:
        semantic_answer = answer[0]
    else:
        semantic_answer = answer

    if observe:
        return {
            "observations": semantic_answer,
            "evidence_source": "visual analysis of supplied media_paths",
        }
    return {
        "answer": semantic_answer,
        "evidence_source": "visual analysis of supplied media_paths",
    }
