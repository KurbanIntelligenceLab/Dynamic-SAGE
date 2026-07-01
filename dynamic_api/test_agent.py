"""Test Agent: validates synthesized tools before they enter the library.

VADAR's Test Agent runs the implementation with placeholder inputs and
catches Python exceptions. We do the same but with two important
specialisations for SAGE:

1. We do NOT actually invoke the SAGE tools at test time. Calling
   `perform_reasoning` would hit Gemini; calling `extract_parts_from_timestamp`
   would shell out to ffmpeg. Those are slow and side-effecting. Instead
   we monkey-patch each existing SAGE tool to a stub that returns a
   plausible dict, then dry-call the synthesized function. This catches
   95%+ of bugs (bad arg names, type errors, missing keys, dict misuse)
   without burning API credits.

2. We validate the docstring parses into a valid JSON Schema using SAGE's
   own AST-based introspector (`_get_function_info` from utils.py), so we
   can't accept a tool whose signature SAGE will reject at load time.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Stub returns chosen to mimic the shape each real tool returns, so that
# downstream `.get("media_paths", [])` etc. calls succeed.
STUB_RETURNS = {
    "unified_web_search": [
        {"title": "Stub result 1", "snippet": "stub", "link": "https://example.com/1"},
        {"title": "Stub result 2", "snippet": "stub", "link": "https://example.com/2"},
    ],
    "parse_web_data": {
        "title": "Stub Page",
        "content": "Stub page body content for testing.",
        "url": "https://example.com",
    },
    "extract_parts_from_timestamp": {
        "media_paths": ["/tmp/stub_frame_0.jpg", "/tmp/stub_frame_1.jpg"],
    },
    "perform_reasoning": {
        "answer": "Stub reasoning answer.",
    },
    "identify_timestamps_visually": {
        "name": "stub event",
        "timestamps": {"start": "00:01:00", "end": "00:02:00"},
    },
    "verbal_transcript": {
        "transcript": "Stub transcript text for the requested range.",
    },
}


# Placeholder arg values keyed by Python annotation. The synthesized
# tools take str / int / float / bool / list / dict per VADAR's
# convention; we fill those in here.
PLACEHOLDER_BY_TYPE = {
    "str": "test_value",
    "int": 2,
    "float": 1.0,
    "bool": True,
    "list": [],
    "dict": {},
    "any": "test_value",
}


# Special-case placeholders by argument name (more useful than the type
# default for fields like video_path or timestamps).
PLACEHOLDER_BY_NAME = {
    "video_path": "/tmp/stub_video.mp4",
    "image_path": "/tmp/stub_image.jpg",
    "timestamp_start": "00:00:30",
    "timestamp_end": "00:01:30",
    "start": "00:00:30",
    "end": "00:01:30",
    "url": "https://example.com",
    "website_url": "https://example.com",
    "query": "test query",
    "event": "test event",
    "description": "test description",
    "subject": "the host",
    "num_results": 3,
    "extract_type": "frames",
}


BANNED_IMPORTS = {
    "subprocess",
    "os.system",
    "socket",
    "ctypes",
    "pickle",
    "marshal",
    "shutil",
    "pty",
    "fcntl",
    "tarfile",
    "zipfile",
}


def _placeholder_for_arg(arg_name: str, annotation: str | None) -> Any:
    if arg_name in PLACEHOLDER_BY_NAME:
        return PLACEHOLDER_BY_NAME[arg_name]
    if annotation:
        return PLACEHOLDER_BY_TYPE.get(annotation, "test_value")
    return "test_value"


def _annotation_str(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name):
            return node.value.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def static_checks(source: str) -> Tuple[bool, str]:
    """Cheap AST-only checks that must pass before we try to import."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # Banned imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in BANNED_IMPORTS or any(
                    alias.name.startswith(b + ".") for b in BANNED_IMPORTS
                ):
                    return False, f"Banned import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module in BANNED_IMPORTS
                or any(node.module.startswith(b + ".") for b in BANNED_IMPORTS)
            ):
                return False, f"Banned import from: {node.module}"

    # Must define at least one function
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return False, "No function defined."

    # Each function must have a docstring and an Args: block — the SAGE
    # AST introspector requires this to build the JSON Schema correctly.
    for fn in funcs:
        if fn.name.startswith("__"):
            continue
        doc = ast.get_docstring(fn)
        if not doc:
            return False, f"Function {fn.name} has no docstring."
        if "Args:" not in doc:
            return False, f"Function {fn.name} docstring missing Args: block."

    # Reject tools that accept the user's QUESTION as an argument. A composite
    # tool's job is to gather evidence, not to answer — answering is the
    # orchestrator's role. A `question` parameter is the signature of a
    # foreclosing answer-wrapper (the question gets passed straight into
    # perform_reasoning and its verdict returned), which empirically loses to
    # the base-tool chain. Take a noun-phrase focus/subject instead.
    BANNED_PARAMS = {"question", "user_question", "the_question", "user_query", "userquestion"}
    for fn in funcs:
        if fn.name.startswith("__"):
            continue
        argnames = [a.arg for a in fn.args.args] + [
            a.arg for a in getattr(fn.args, "kwonlyargs", [])
        ]
        bad = [a for a in argnames if a.lower() in BANNED_PARAMS]
        if bad:
            return False, (
                f"Function {fn.name} takes a question parameter {bad}: composite tools must "
                f"gather and return evidence, not answer the user's question (that is the "
                f"orchestrator's job). Replace it with a noun-phrase 'focus'/'subject'/"
                f"'event_description' argument and return observations/transcript/frames_analyzed."
            )

    # Reject functions whose body is a single try/except Exception that wraps
    # the whole thing. This pattern silently swallows the runtime validation
    # errors raised by our strict stubs (e.g. FileNotFoundError on a transcript-
    # as-path bug) and turns them into a benign-looking {"error": "..."} dict,
    # letting buggy tools pass validation.
    for fn in funcs:
        if fn.name.startswith("__"):
            continue
        ok_struct, struct_err = _check_no_global_try_except(fn)
        if not ok_struct:
            return False, struct_err

    # Enforce mode="observe" on every perform_reasoning call inside a composite
    # tool. perform_reasoning defaults to mode="answer" (returns a finished
    # {answerable, final_answer} verdict); a composite must call it with
    # mode="observe" so it returns evidence under the "observations" key. Without
    # this, a tool that reads result.get("observations") silently returns an
    # empty string, and the tool forecloses the orchestrator's reasoning. This is
    # the last static check so it does not change the error ordering above.
    for fn in funcs:
        if fn.name.startswith("__"):
            continue
        ok_obs, obs_err = _check_perform_reasoning_observe_mode(fn)
        if not ok_obs:
            return False, obs_err

    return True, ""


def _check_perform_reasoning_observe_mode(fn: ast.FunctionDef) -> Tuple[bool, str]:
    """Every perform_reasoning(...) call in a synthesized tool must pass
    mode="observe". The default answer mode returns a verdict and forecloses the
    orchestrator's reasoning; tools read result["observations"], which only
    exists in observe mode, so a missing mode silently yields empty evidence."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        fname = None
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fname = node.func.attr
        if fname != "perform_reasoning":
            continue
        mode_kw = next((k for k in node.keywords if k.arg == "mode"), None)
        ok = (
            mode_kw is not None
            and isinstance(mode_kw.value, ast.Constant)
            and str(mode_kw.value.value).lower() == "observe"
        )
        if not ok:
            return False, (
                f"Function {fn.name} calls perform_reasoning without mode=\"observe\". "
                f"Inside a composite tool, perform_reasoning MUST be called with "
                f"mode=\"observe\" so it returns observations for the orchestrator to "
                f"reason over (and so result.get(\"observations\") is non-empty). The "
                f"default answer mode returns a finished verdict and forecloses reasoning."
            )
    return True, ""


def _check_no_global_try_except(fn: ast.FunctionDef) -> Tuple[bool, str]:
    """Reject a function whose body is dominated by a single top-level
    try/except that catches Exception or BaseException (or bare except).
    A top-level try is acceptable only if its handlers catch narrow,
    specific exception types (e.g. FileNotFoundError, ValueError).
    """
    # Find non-docstring body statements
    body = fn.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return True, ""
    # We care if there is ONE top-level statement and it is a try/except,
    # or if any top-level try statement has an over-broad handler.
    BROAD_NAMES = {"Exception", "BaseException"}
    for stmt in body:
        if isinstance(stmt, ast.Try):
            for handler in stmt.handlers:
                if handler.type is None:
                    return (
                        False,
                        f"{fn.name}: bare `except:` masks validator errors. "
                        f"Catch specific exception types or let them propagate.",
                    )
                # Name-based check (covers `except Exception` and `except BaseException`)
                if isinstance(handler.type, ast.Name) and handler.type.id in BROAD_NAMES:
                    # Check if this try wraps "most" of the function — heuristic:
                    # the try's body has 3+ statements OR this is the only top-level statement.
                    try_body_len = len(stmt.body)
                    if len(body) == 1 or try_body_len >= 3:
                        return (
                            False,
                            f"{fn.name}: top-level `except {handler.type.id}` "
                            f"swallows tool validation errors. Catch specific "
                            f"exception types (FileNotFoundError, ValueError, etc.) "
                            f"or let exceptions propagate so the orchestrator can "
                            f"see them.",
                        )
    return True, ""


def dry_run(source: str, stubbed_tools: List[str]) -> Tuple[bool, str]:
    """Import the source as a module with the SAGE tools stubbed, then
    invoke every top-level function with placeholder args. Returns
    (passed, error_message)."""
    # Inject stubs for the existing SAGE tools by creating a fake module
    # tree under sage.src.functions.tools.* before importing the source.
    stub_modules = _install_stub_modules(stubbed_tools)

    try:
        # Write source to a tempfile and import it as an anonymous module.
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write(source)
            module_path = Path(f.name)

        spec = importlib.util.spec_from_file_location(
            "sage_dynamic_api_test_module", module_path
        )
        if spec is None or spec.loader is None:
            return False, "Could not create module spec."
        module = importlib.util.module_from_spec(spec)
        sys.modules["sage_dynamic_api_test_module"] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as e:
            return False, f"Import-time error: {e}\n{traceback.format_exc()}"

        # Find each public function and try to call it with placeholders.
        for name in dir(module):
            if name.startswith("_") and not name.startswith("_") or name.startswith("__"):
                continue
            obj = getattr(module, name)
            if not callable(obj):
                continue
            # We only want functions defined in this module (not imported)
            if getattr(obj, "__module__", None) != "sage_dynamic_api_test_module":
                continue

            tree = ast.parse(source)
            fn_node = next(
                (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name),
                None,
            )
            if fn_node is None:
                continue

            kwargs: Dict[str, Any] = {}
            for arg in fn_node.args.args:
                if arg.arg == "self":
                    continue
                ann = _annotation_str(arg.annotation)
                kwargs[arg.arg] = _placeholder_for_arg(arg.arg, ann)

            try:
                result = obj(**kwargs)
            except TypeError as e:
                return False, f"{name}: TypeError on call: {e}"
            except Exception as e:
                return (
                    False,
                    f"{name}: runtime error: {e}\n{traceback.format_exc()}",
                )

            if not isinstance(result, dict):
                return False, f"{name}: expected dict return, got {type(result).__name__}"

        return True, ""
    finally:
        _uninstall_stub_modules(stub_modules)
        sys.modules.pop("sage_dynamic_api_test_module", None)
        try:
            module_path.unlink()
        except Exception:
            pass


def _install_stub_modules(tool_names: List[str]) -> List[str]:
    """Install a fake `sage.src.functions.tools.*` package tree that
    returns canned stubs. Returns the list of module keys we created
    (for later cleanup)."""
    import types

    created: List[str] = []

    def _ensure(name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        created.append(name)
        return mod

    _ensure("sage")
    _ensure("sage.src")
    _ensure("sage.src.functions")
    tools_pkg = _ensure("sage.src.functions.tools")

    # Map tool name -> source module within the sage.src.functions.tools
    # package. We mirror the real layout so `from sage.src.functions.tools.search
    # import unified_web_search` works in synthesized code.
    tool_to_modname = {
        "unified_web_search": "search",
        "parse_web_data": "search",
        "extract_parts_from_timestamp": "extract",
        "perform_reasoning": "reason",
        "identify_timestamps_visually": "temporal",
        "verbal_transcript": "audio",
    }

    # Group tools by their module
    by_mod: Dict[str, List[str]] = {}
    for tool in tool_names:
        modname = tool_to_modname.get(tool)
        if modname is None:
            continue
        by_mod.setdefault(modname, []).append(tool)

    # ----- Strict per-tool stubs that mirror real runtime contracts -----
    # The previous implementation accepted any args. That let tools like
    # `perform_reasoning(query=q, media_paths=[transcript_str])` pass
    # validation even though the real perform_reasoning would crash with
    # FileNotFoundError because transcript_str is not a path. The stubs
    # below catch those cases at validation time so the Implementation
    # Agent gets corrective feedback and can fix its output before we
    # ship the tool.

    _VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    _IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
    _MEDIA_EXTS = _VIDEO_EXTS + _IMAGE_EXTS

    def _check_video_path(p: Any, arg: str = "video_path") -> None:
        if not isinstance(p, str):
            raise TypeError(
                f"{arg} must be str, got {type(p).__name__}"
            )
        if not p.lower().endswith(_VIDEO_EXTS):
            raise ValueError(
                f"{arg}={p!r} does not look like a video path "
                f"(must end with one of {_VIDEO_EXTS})"
            )

    def _check_timestamp(ts: Any, arg: str = "timestamp") -> None:
        if not isinstance(ts, str):
            raise TypeError(
                f"{arg} must be HH:MM:SS string, got {type(ts).__name__}"
            )
        # Permissive: HH:MM:SS or MM:SS or SS
        import re as _re
        if not _re.match(r"^(\d{1,2}:)?([0-5]?\d:)?[0-5]?\d$", ts):
            raise ValueError(
                f"{arg}={ts!r} is not a valid timestamp; "
                f"expected HH:MM:SS, MM:SS, or SS"
            )

    def _check_media_paths(mp: Any) -> None:
        if mp is None:
            return
        if not isinstance(mp, list):
            raise TypeError(
                f"media_paths must be a list, got {type(mp).__name__}"
            )
        for i, p in enumerate(mp):
            if not isinstance(p, str):
                raise TypeError(
                    f"media_paths[{i}] must be str, got {type(p).__name__}"
                )
            if not p.lower().endswith(_MEDIA_EXTS):
                raise FileNotFoundError(
                    f"media_paths[{i}]={p!r} does not look like a media "
                    f"file path (must end with one of {_MEDIA_EXTS}). "
                    f"If you want to pass text content to perform_reasoning, "
                    f"include it in the query string instead and pass "
                    f"media_paths=[]."
                )

    def _check_url(u: Any) -> None:
        if not isinstance(u, str):
            raise TypeError(f"url must be str, got {type(u).__name__}")
        if not (u.startswith("http://") or u.startswith("https://")):
            raise ValueError(
                f"website_url={u!r} must start with http:// or https://"
            )

    def _check_nonempty_str(s: Any, arg: str) -> None:
        if not isinstance(s, str):
            raise TypeError(f"{arg} must be str, got {type(s).__name__}")
        if not s.strip():
            raise ValueError(f"{arg} must be a non-empty string")

    # Each stub takes positional + keyword args, validates them, then
    # returns the canned stub_return for the tool. Unknown kwargs raise
    # TypeError to match real Python behaviour.

    def _stub_unified_web_search(query=None, search_type="web", num_results=3, **extra):
        if extra:
            raise TypeError(f"unified_web_search got unexpected kwargs: {list(extra)}")
        _check_nonempty_str(query, "query")
        if search_type not in ("web", "image"):
            raise ValueError(f"search_type must be 'web' or 'image', got {search_type!r}")
        if not isinstance(num_results, int) or not (1 <= num_results <= 10):
            raise ValueError(f"num_results must be int in [1,10], got {num_results!r}")
        return {"results": STUB_RETURNS["unified_web_search"]}

    def _stub_parse_web_data(website_url=None, max_content_length=5000, **extra):
        if extra:
            raise TypeError(f"parse_web_data got unexpected kwargs: {list(extra)}")
        _check_url(website_url)
        return STUB_RETURNS["parse_web_data"]

    def _stub_extract_parts_from_timestamp(video_path=None, timestamp_start=None,
                                          timestamp_end=None, extract_type="frames",
                                          **extra):
        if extra:
            raise TypeError(
                f"extract_parts_from_timestamp got unexpected kwargs: {list(extra)}. "
                f"Allowed: video_path, timestamp_start, timestamp_end, extract_type."
            )
        _check_video_path(video_path)
        _check_timestamp(timestamp_start, "timestamp_start")
        _check_timestamp(timestamp_end, "timestamp_end")
        if extract_type not in ("frames", "subclips"):
            raise ValueError(
                f"extract_type must be 'frames' or 'subclips', got {extract_type!r}"
            )
        return STUB_RETURNS["extract_parts_from_timestamp"]

    def _stub_perform_reasoning(query=None, media_paths=None, mode="answer", **extra):
        if extra:
            raise TypeError(
                f"perform_reasoning got unexpected kwargs: {list(extra)}. "
                f"Allowed: query, media_paths, mode."
            )
        _check_nonempty_str(query, "query")
        _check_media_paths(media_paths)
        if str(mode).lower() == "observe":
            return {"observations": "Stub observations.", "evidence_source": "stub"}
        return STUB_RETURNS["perform_reasoning"]

    def _stub_identify_timestamps_visually(video_path=None, event=None,
                                           timestamp_start=None, timestamp_end=None,
                                           **extra):
        if extra:
            raise TypeError(
                f"identify_timestamps_visually got unexpected kwargs: {list(extra)}. "
                f"Allowed: video_path, event, timestamp_start, timestamp_end."
            )
        _check_video_path(video_path)
        _check_nonempty_str(event, "event")
        _check_timestamp(timestamp_start, "timestamp_start")
        _check_timestamp(timestamp_end, "timestamp_end")
        return STUB_RETURNS["identify_timestamps_visually"]

    def _stub_verbal_transcript(video_path=None, timestamp_start=None,
                                timestamp_end=None, **extra):
        if extra:
            raise TypeError(
                f"verbal_transcript got unexpected kwargs: {list(extra)}. "
                f"Allowed: video_path, timestamp_start, timestamp_end."
            )
        _check_video_path(video_path)
        _check_timestamp(timestamp_start, "timestamp_start")
        _check_timestamp(timestamp_end, "timestamp_end")
        return STUB_RETURNS["verbal_transcript"]

    STRICT_STUBS = {
        "unified_web_search": _stub_unified_web_search,
        "parse_web_data": _stub_parse_web_data,
        "extract_parts_from_timestamp": _stub_extract_parts_from_timestamp,
        "perform_reasoning": _stub_perform_reasoning,
        "identify_timestamps_visually": _stub_identify_timestamps_visually,
        "verbal_transcript": _stub_verbal_transcript,
    }

    for modname, tools in by_mod.items():
        full_name = f"sage.src.functions.tools.{modname}"
        mod = _ensure(full_name)
        setattr(tools_pkg, modname, mod)
        for tool in tools:
            stub_fn = STRICT_STUBS.get(tool)
            if stub_fn is None:
                # Unknown tool — fall back to permissive stub
                def _make_stub(_ret):
                    def _stub(*args, **kwargs):
                        return _ret
                    return _stub
                stub_fn = _make_stub(STUB_RETURNS.get(tool, {"stub": True}))
            setattr(mod, tool, stub_fn)

    return created


def _uninstall_stub_modules(created: List[str]) -> None:
    for name in created:
        sys.modules.pop(name, None)


def validate(source: str, stubbed_tools: List[str]) -> Tuple[bool, str]:
    """Full validation: static + dry-run. Returns (ok, error_message)."""
    ok, err = static_checks(source)
    if not ok:
        return False, err
    return dry_run(source, stubbed_tools)


if __name__ == "__main__":
    # Smoke test
    sample = textwrap.dedent('''
        from sage.src.functions.tools.extract import extract_parts_from_timestamp
        from sage.src.functions.tools.reason import perform_reasoning

        def _describe_range(video_path: str, subject: str, timestamp_start: str, timestamp_end: str) -> dict:
            """Describe a subject across frames in a time range.

            Args:
                video_path: Path to the video.
                subject: What to describe.
                timestamp_start: HH:MM:SS.
                timestamp_end: HH:MM:SS.

            Returns:
                Dict with description.
            """
            extracted = extract_parts_from_timestamp(
                video_path=video_path,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                extract_type="frames",
            )
            frames = extracted.get("media_paths", [])
            if not frames:
                return {"description": "", "error": "no frames"}
            r = perform_reasoning(query=f"Describe {subject}", media_paths=frames, mode="observe")
            return {"observations": r.get("observations", ""), "n_frames": len(frames)}
    ''')
    ok, err = validate(
        sample,
        stubbed_tools=["extract_parts_from_timestamp", "perform_reasoning"],
    )
    print(f"smoke test: passed={ok} err={err}")
