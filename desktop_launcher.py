import sys
from pathlib import Path

from video_page_detector.desktop_app import main, run_packaged_self_test


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--self-test":
        result = run_packaged_self_test(Path(sys.argv[2]))
        raise SystemExit(0 if result["ok"] else 2)
    raise SystemExit(main())
