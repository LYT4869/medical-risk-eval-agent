from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.benchmark_routing_runtime import (
    deployment_footprint,
    maximum_rss_kib,
    percentile,
    recursive_file_size,
    validate_counts,
)


class RoutingBenchmarkTest(unittest.TestCase):
    def test_total_footprint_includes_image_and_artifact(self):
        report = deployment_footprint(
            container_image_bytes=300,
            artifact_bytes=130,
            baseline_deployment_bytes=500)

        self.assertEqual(report["total_deployment_bytes"], 430)
        self.assertAlmostEqual(report["relative_change"], -0.14)

    def test_percentiles_use_observed_samples(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.95), 4.0)
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.50), 2.0)

    def test_counts_and_image_size_must_be_explicitly_valid(self):
        for warmup, iterations in ((-1, 1), (0, 0), (1, -1)):
            with self.subTest(warmup=warmup, iterations=iterations), \
                    self.assertRaises(ValueError):
                validate_counts(warmup, iterations)
        with self.assertRaises(ValueError):
            deployment_footprint(
                container_image_bytes=0, artifact_bytes=1,
                baseline_deployment_bytes=None)

    def test_recursive_size_excludes_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "one").write_bytes(b"123")
            nested = root / "nested"
            nested.mkdir()
            (nested / "two").write_bytes(b"4567")
            (root / "link").symlink_to(root / "one")

            self.assertEqual(recursive_file_size(root), 7)

    def test_linux_rss_is_already_reported_in_kib(self):
        self.assertEqual(maximum_rss_kib(2048, "linux"), 2048)
        self.assertEqual(maximum_rss_kib(2048, "darwin"), 2)

    def test_footprint_report_does_not_contain_queries(self):
        rendered = json.dumps(deployment_footprint(
            container_image_bytes=300, artifact_bytes=130,
            baseline_deployment_bytes=None), ensure_ascii=False)
        self.assertNotIn("解释刚才的预测", rendered)
        self.assertIsNone(json.loads(rendered)["relative_change"])


if __name__ == "__main__":
    unittest.main()
