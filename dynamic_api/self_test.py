"""End-to-end self-test for the generator pipeline. Verifies:
  - Stratified sampling math
  - Proposal & implementation parsers
  - Function assembly
  - Test agent acceptance on a well-formed synthetic implementation
  - Test agent rejection on common failure modes

No real LLM calls. Run before live generation.
"""
import textwrap
from sage.dynamic_api.generate import (
    parse_proposals,
    parse_implementation,
    assemble_function,
    stratified_sample,
    _duration_bucket,
)
from sage.dynamic_api.test_agent import validate


def test_proposal_parser():
    sample = """
    Some preamble from the agent.

    <proposal>
    <docstring>
    \"\"\"Count distinct visual events of a description in a range.

    Args:
        video_path: Path to the video.
        description: What event to count.
        timestamp_start: HH:MM:SS.
        timestamp_end: HH:MM:SS.

    Returns:
        Dict with "count" (int).
    \"\"\"
    </docstring>
    <signature>def _count_visual_events(video_path: str, description: str, timestamp_start: str, timestamp_end: str) -> dict:</signature>
    </proposal>

    <proposal>
    <docstring>\"\"\"Describe object X.

    Args:
        x: thing.
    Returns:
        dict.
    \"\"\"</docstring>
    <signature>def _describe_x(x: str) -> dict:</signature>
    </proposal>
    """
    props = parse_proposals(sample)
    assert len(props) == 2, f"expected 2, got {len(props)}"
    assert "_count_visual_events" in props[0]["signature"]
    assert "_describe_x" in props[1]["signature"]
    print("OK test_proposal_parser")


def test_impl_parser():
    sample = """
    <imports>
    from sage.src.functions.tools.extract import extract_parts_from_timestamp
    from sage.src.functions.tools.reason import perform_reasoning
    </imports>
    <implementation>
    extracted = extract_parts_from_timestamp(
        video_path=video_path,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        extract_type="frames",
    )
    frames = extracted.get("media_paths", [])
    if not frames:
        return {"count": 0, "error": "no frames"}
    r = perform_reasoning(
        query=f"Count visual instances of: {description}. Return only an integer.",
        media_paths=frames,
    )
    return {"count": r.get("answer", "0")}
    </implementation>
    """
    parts = parse_implementation(sample)
    assert "extract_parts_from_timestamp" in parts["imports"]
    assert "extracted = extract_parts_from_timestamp" in parts["body"]
    print("OK test_impl_parser")


def test_assemble_and_validate_passes():
    docstring = '''"""Count distinct visual events.

    Args:
        video_path: Path to the video.
        description: What to count.
        timestamp_start: HH:MM:SS.
        timestamp_end: HH:MM:SS.

    Returns:
        Dict with "count".
    """'''
    signature = (
        "def _count_visual_events(video_path: str, description: str, "
        "timestamp_start: str, timestamp_end: str) -> dict:"
    )
    imports = textwrap.dedent("""
        from sage.src.functions.tools.extract import extract_parts_from_timestamp
        from sage.src.functions.tools.reason import perform_reasoning
    """).strip()
    body = textwrap.dedent("""
        extracted = extract_parts_from_timestamp(
            video_path=video_path,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            extract_type="frames",
        )
        frames = extracted.get("media_paths", [])
        if not frames:
            return {"count": 0, "error": "no frames"}
        r = perform_reasoning(
            query=f"Count visual instances of: {description}.",
            media_paths=frames,
            mode="observe",
        )
        return {"count": r.get("observations", "0")}
    """).strip()
    src = assemble_function(docstring, signature, imports, body)
    print("---- ASSEMBLED ----")
    print(src)
    print("-------------------")
    ok, err = validate(
        src,
        stubbed_tools=[
            "extract_parts_from_timestamp",
            "perform_reasoning",
        ],
    )
    assert ok, f"validation should have passed: {err}"
    print("OK test_assemble_and_validate_passes")


def test_validate_rejects_syntax_error():
    bad = "def broken(x: str) -> dict:\n    return {x: }"  # syntax error
    ok, err = validate(bad, stubbed_tools=[])
    assert not ok
    assert "Syntax" in err
    print("OK test_validate_rejects_syntax_error")


def test_validate_rejects_non_dict_return():
    bad = textwrap.dedent('''
        def _bad(x: str) -> str:
            """Return wrong type.

            Args:
                x: arg.

            Returns:
                A string.
            """
            return "not a dict"
    ''').strip()
    ok, err = validate(bad, stubbed_tools=[])
    assert not ok
    assert "dict" in err
    print("OK test_validate_rejects_non_dict_return")


def test_validate_rejects_missing_docstring():
    bad = textwrap.dedent('''
        def _nodoc(x: str) -> dict:
            return {"x": x}
    ''').strip()
    ok, err = validate(bad, stubbed_tools=[])
    assert not ok
    assert "docstring" in err.lower()
    print("OK test_validate_rejects_missing_docstring")


def test_validate_rejects_banned_import():
    bad = textwrap.dedent('''
        import subprocess

        def _danger(x: str) -> dict:
            """Bad.
            Args:
                x: arg.
            Returns:
                dict.
            """
            return {"x": x}
    ''').strip()
    ok, err = validate(bad, stubbed_tools=[])
    assert not ok
    assert "Banned" in err
    print("OK test_validate_rejects_banned_import")


def test_validate_rejects_bad_kwarg():
    # Calls extract_parts_from_timestamp with arg names it doesn't accept.
    # Strict stubs now reject unknown kwargs with TypeError, so this is
    # caught at validation time.
    bad_kwarg = textwrap.dedent('''
        from sage.src.functions.tools.extract import extract_parts_from_timestamp

        def _bad_kwarg(video_path: str, timestamp_start: str, timestamp_end: str) -> dict:
            """Does the wrong thing.
            Args:
                video_path: video.
                timestamp_start: start.
                timestamp_end: end.
            Returns:
                dict.
            """
            r = extract_parts_from_timestamp(
                video_path=video_path,
                start_ts=timestamp_start,  # WRONG kwarg name
                end_ts=timestamp_end,
            )
            return r
    ''').strip()
    ok, err = validate(bad_kwarg, stubbed_tools=["extract_parts_from_timestamp"])
    assert not ok, f"strict stubs should have rejected this: {err}"
    assert "unexpected kwargs" in err.lower() or "typeerror" in err.lower()
    print("OK test_validate_rejects_bad_kwarg")


def test_validate_rejects_string_as_media_path():
    # The buggy pattern from the previous SAGE-Bench generation: passing
    # a transcript STRING as an element of media_paths. perform_reasoning
    # would FileNotFoundError on it at runtime.
    buggy = textwrap.dedent('''
        from sage.src.functions.tools.audio import verbal_transcript
        from sage.src.functions.tools.reason import perform_reasoning

        def _analyze_verbal(video_path: str, timestamp_start: str, timestamp_end: str, query: str) -> dict:
            """Analyze.
            Args:
                video_path: video.
                timestamp_start: start.
                timestamp_end: end.
                query: query.
            Returns:
                dict.
            """
            t = verbal_transcript(video_path=video_path, timestamp_start=timestamp_start, timestamp_end=timestamp_end)
            r = perform_reasoning(query=query, media_paths=[t.get("transcript", "")], mode="observe")
            return {"answer": r.get("answer", "")}
    ''').strip()
    ok, err = validate(buggy, stubbed_tools=["verbal_transcript", "perform_reasoning"])
    assert not ok, f"strict stubs should have rejected transcript-as-path: {err}"
    assert "media_paths" in err.lower() or "media file" in err.lower()
    print("OK test_validate_rejects_string_as_media_path")


def test_validate_rejects_whole_function_try_except():
    # The pattern from run 2 of SAGE-Bench: wrap the entire function in
    # try/except Exception. This silently swallows the strict-stub errors
    # that catch real bugs, so it should be rejected outright.
    bad = textwrap.dedent('''
        from sage.src.functions.tools.audio import verbal_transcript
        from sage.src.functions.tools.reason import perform_reasoning

        def _bad(video_path: str, timestamp_start: str, timestamp_end: str) -> dict:
            """Wraps everything.

            Args:
                video_path: Path.
                timestamp_start: Start.
                timestamp_end: End.

            Returns:
                Dict.
            """
            try:
                t = verbal_transcript(video_path, timestamp_start, timestamp_end)
                txt = t.get("transcript", "")
                r = perform_reasoning(query="x", media_paths=[txt])
                return {"answer": r.get("answer", "")}
            except Exception as e:
                return {"answer": "", "error": str(e)}
    ''').strip()
    ok, err = validate(bad, stubbed_tools=["verbal_transcript", "perform_reasoning"])
    assert not ok, f"should reject whole-function try/except: {err}"
    assert "except Exception" in err or "swallows" in err
    print("OK test_validate_rejects_whole_function_try_except")


def test_validate_accepts_narrow_except():
    # A try/except KeyError or other specific exception is fine.
    good = textwrap.dedent('''
        from sage.src.functions.tools.extract import extract_parts_from_timestamp

        def _ok(video_path: str, timestamp_start: str, timestamp_end: str) -> dict:
            """Narrow except.

            Args:
                video_path: Path.
                timestamp_start: Start.
                timestamp_end: End.

            Returns:
                Dict.
            """
            extracted = extract_parts_from_timestamp(
                video_path=video_path,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
            )
            try:
                first = extracted["media_paths"][0]
            except KeyError:
                first = ""
            return {"media_paths": [first]}
    ''').strip()
    ok, err = validate(good, stubbed_tools=["extract_parts_from_timestamp"])
    assert ok, f"narrow except should be accepted: {err}"
    print("OK test_validate_accepts_narrow_except")


def test_stratified_sampling():
    rows = []
    for i in range(1000):
        rows.append({
            "id": str(i),
            "duration_seconds": (i * 7) % 3000,
            "ques_type": "mcq" if i % 2 == 0 else "open_ended",
            "modality": ["visual", "verbal", "both"][i % 3],
        })
    sample = stratified_sample(rows, 50, ["duration_bucket", "ques_type", "modality"])
    assert len(sample) == 50, f"got {len(sample)}"
    buckets = set(_duration_bucket(r["duration_seconds"]) for r in sample)
    # Should hit most buckets
    assert len(buckets) >= 4, f"too few buckets sampled: {buckets}"
    # IDs should be unique
    ids = [r["id"] for r in sample]
    assert len(set(ids)) == len(ids)
    print(f"OK test_stratified_sampling (buckets: {buckets})")


if __name__ == "__main__":
    test_proposal_parser()
    test_impl_parser()
    test_assemble_and_validate_passes()
    test_validate_rejects_syntax_error()
    test_validate_rejects_non_dict_return()
    test_validate_rejects_missing_docstring()
    test_validate_rejects_banned_import()
    test_validate_rejects_bad_kwarg()
    test_validate_rejects_string_as_media_path()
    test_validate_rejects_whole_function_try_except()
    test_validate_accepts_narrow_except()
    test_stratified_sampling()
    print("\nAll self-tests passed.")
