import tempfile
import unittest
from pathlib import Path

from video_page_detector.output_paths import (
    resolve_run_directory,
    validate_video_id,
)


class OutputPathTests(unittest.TestCase):
    def test_accepts_normal_task_name(self) -> None:
        self.assertEqual(validate_video_id("课堂-01"), "课堂-01")

    def test_rejects_escape_and_windows_reserved_names(self) -> None:
        for value in ("..", ".", "../escape", "CON", "nul.txt", "lesson."):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_video_id(value)

    def test_run_directory_is_direct_child_of_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = resolve_run_directory(root, "lesson")
            self.assertEqual(run_dir.parent, root.resolve())


if __name__ == "__main__":
    unittest.main()
