import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize import audit, load_datasets, load_scores

HEADERS = ["Group", "Title", "Word_Count", "Variance", "Samples_Count"]
ROWS = [
    ["Imagism", "A", 10, .1, 20],
    ["Imagism", "B", 12, .2, 20],
    ["Romanticism", "C", 14, .3, 20],
    ["Romanticism", "D", 16, .4, 20],
]


def write_csv(path, rows, headers=HEADERS):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


class TableChecks(unittest.TestCase):
    def test_retained_archive_preserves_counts_and_known_cross_model_matches(self):
        result = audit(ROOT / "data")
        self.assertEqual({x["title"] for x in result["exact_cross_model_score_matches"]},
                         {"The Tyger", "To a Skylark"})
        for model in result["models"].values():
            self.assertEqual([g["poems"] for g in model["groups"].values()], [5, 5])

    def test_duplicate_headers_and_ragged_rows_are_rejected(self):
        variants = [(HEADERS + ["Variance"], ROWS),
                    (HEADERS, [ROWS[0] + ["extra"]] + ROWS[1:]),
                    (HEADERS, [ROWS[0][:-1]] + ROWS[1:])]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            for headers, rows in variants:
                write_csv(path, rows, headers)
                with self.subTest(headers=headers, first=rows[0]), self.assertRaises(ValueError):
                    load_scores(path)

    def test_invalid_cells_include_file_and_row_in_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            for column, value in [(0, "Unknown"), (1, " "), (2, "bad"), (3, "nan"),
                                  (3, "-1"), (3, "not a score"), (4, "0")]:
                rows = [row.copy() for row in ROWS]
                rows[0][column] = value
                write_csv(path, rows)
                with self.subTest(column=column, value=value), self.assertRaisesRegex(ValueError, "bad.csv:2:"):
                    load_scores(path)

    def test_mismatched_poem_sets_are_not_silently_compared(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_csv(folder / "flux.csv", ROWS)
            other = [row.copy() for row in ROWS]
            other[0][1] = "Different poem"
            write_csv(folder / "stable_diffusion.csv", other)
            with self.assertRaisesRegex(ValueError, "same poem/group keys"):
                load_datasets(folder)

    def test_bad_cli_input_is_reported_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"),
                                     "--data-dir", folder], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy is optional")
class TestStatisticChecks(unittest.TestCase):
    def test_constant_groups_have_explicit_status_and_strict_json(self):
        from scipy import stats
        from compare_ttests import compare_groups
        for b in ([1., 1.], [2., 2.]):
            result = compare_groups([1., 1.], b, stats)
            for method in result.values():
                self.assertIsNone(method["t"])
                self.assertIsNone(method["p"])
                self.assertEqual(method["status"], "zero_within_group_variance")
            json.dumps(result, allow_nan=False)

    def test_retained_t_tests_match_saved_reference_values(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/compare_ttests.py")],
                                check=True, text=True, capture_output=True)
        current = json.loads(result.stdout)
        reference = json.loads((ROOT / "reports/ttest-comparison.json").read_text())
        for model, methods in reference["results"].items():
            self.assertEqual(current["group_counts"][model], {"Imagism": 5, "Romanticism": 5})
            for method, values in methods.items():
                for metric in ("t", "p"):
                    self.assertAlmostEqual(current["results"][model][method][metric], values[metric], places=12)


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is optional")
class GridChecks(unittest.TestCase):
    def test_source_directory_is_never_an_output_destination(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            original = folder / "original.png"
            Image.new("RGB", (3, 3), "red").save(original)
            before = original.read_bytes()
            for output in (original, folder / "new-grid.png"):
                result = subprocess.run([sys.executable, str(ROOT / "scripts/make_grid.py"),
                                         str(folder), str(output), "--columns", "1", "--rows", "1"],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("outside the source", result.stderr)
                self.assertEqual(original.read_bytes(), before)
            self.assertFalse((folder / "new-grid.png").exists())

    def test_grid_records_selected_source_hashes(self):
        from PIL import Image
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "source"
            source.mkdir()
            Image.new("RGB", (6, 4), "red").save(source / "a.png")
            Image.new("RGB", (4, 6), "blue").save(source / "b.png")
            output = folder / "grid.png"
            subprocess.run([sys.executable, str(ROOT / "scripts/make_grid.py"), str(source),
                            str(output), "--columns", "2", "--rows", "1", "--tile-size", "8"],
                           check=True, text=True, capture_output=True)
            with Image.open(output) as image:
                self.assertEqual(image.size, (16, 8))
            manifest = json.loads(output.with_suffix(".png.json").read_text())
            self.assertEqual([x["filename"] for x in manifest["sources"]], ["a.png", "b.png"])
            for row in manifest["sources"]:
                self.assertEqual(row["sha256"], hashlib.sha256((source / row["filename"]).read_bytes()).hexdigest())

    def test_banner_tiles_match_saved_banner_from_any_directory(self):
        from PIL import Image, ImageChops
        from make_banner import GAP, LABEL_H, PAD, TILE
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "banner.png"
            subprocess.run([sys.executable, str(ROOT / "scripts/make_banner.py"), "--output", str(output)],
                           check=True, text=True, capture_output=True, cwd=folder)
            with Image.open(output) as rebuilt, Image.open(ROOT / "assets/banner.png") as saved:
                self.assertEqual(rebuilt.size, saved.size)
                for top in (PAD + LABEL_H, PAD + 2 * LABEL_H + TILE + GAP):
                    box = (0, top, saved.width, top + TILE)
                    diff = ImageChops.difference(rebuilt.crop(box), saved.crop(box))
                    self.assertLessEqual(max(high for _, high in diff.getextrema()), 2)


if __name__ == "__main__":
    unittest.main()
