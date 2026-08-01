from __future__ import annotations

from pathlib import Path


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_WINDOWS_FORBIDDEN_CHARACTERS = set('<>:"/\\|?*')


def validate_video_id(value: str) -> str:
    """Return a safe single-directory task identifier."""
    identifier = str(value)
    if not identifier or identifier != identifier.strip():
        raise ValueError("video_id must not be empty or have outer whitespace")
    if identifier in {".", ".."}:
        raise ValueError("video_id must not be a relative path segment")
    if identifier.endswith((".", " ")):
        raise ValueError("video_id must not end with a dot or space")
    if any(
        character in _WINDOWS_FORBIDDEN_CHARACTERS or ord(character) < 32
        for character in identifier
    ):
        raise ValueError("video_id contains invalid filename characters")
    basename = identifier.split(".", 1)[0].upper()
    if basename in _WINDOWS_RESERVED_NAMES:
        raise ValueError("video_id is a reserved Windows filename")
    return identifier


def resolve_run_directory(output_root: str | Path, video_id: str) -> Path:
    """Resolve a task directory and enforce it is a direct child of its root."""
    root = Path(output_root).expanduser().resolve()
    identifier = validate_video_id(video_id)
    run_dir = (root / identifier).resolve()
    if run_dir.parent != root:
        raise ValueError("video_id resolves outside the output root")
    return run_dir
