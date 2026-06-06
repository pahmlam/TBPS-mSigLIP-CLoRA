import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from msiglip.data.bases import inject_false_negative_labels  # noqa: E402


def make_dataset(num_pids=3, samples_per_pid=4):
    dataset = []
    image_id = 0
    for pid in range(num_pids):
        for sample_idx in range(samples_per_pid):
            dataset.append((pid, image_id, f"img_{pid}_{sample_idx}.jpg", f"caption {pid} {sample_idx}"))
            image_id += 1
    return dataset


class FalseNegativeInjectionTest(unittest.TestCase):
    def test_zero_rate_does_not_change_dataset(self):
        dataset = make_dataset()
        original = list(dataset)

        mutated, changed = inject_false_negative_labels(dataset, 0.0)

        self.assertEqual(mutated, original)
        self.assertEqual(int(changed.sum()), 0)

    def test_positive_rate_creates_fake_pids_outside_original_range(self):
        dataset = make_dataset()
        original_max_pid = max(item[0] for item in dataset)

        mutated, changed = inject_false_negative_labels(dataset, 0.5)
        mutated_pids = [item[0] for item in mutated]
        changed_pids = [pid for pid, is_changed in zip(mutated_pids, changed) if is_changed]

        self.assertGreater(int(changed.sum()), 0)
        self.assertTrue(all(pid > original_max_pid for pid in changed_pids))

    def test_saved_mapping_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_file = str(Path(tmpdir) / "fn_mapping.npy")

            dataset_a = make_dataset()
            mutated_a, changed_a = inject_false_negative_labels(dataset_a, 0.5, mapping_file)
            pids_a = [item[0] for item in mutated_a]

            dataset_b = make_dataset()
            mutated_b, changed_b = inject_false_negative_labels(dataset_b, 0.5, mapping_file)
            pids_b = [item[0] for item in mutated_b]

            self.assertEqual(pids_a, pids_b)
            self.assertEqual(changed_a.tolist(), changed_b.tolist())

    def test_unpassed_eval_split_is_not_mutated(self):
        train = make_dataset()
        val = make_dataset(num_pids=2, samples_per_pid=2)
        val_snapshot = list(val)

        inject_false_negative_labels(train, 0.5)

        self.assertEqual(val, val_snapshot)


if __name__ == "__main__":
    unittest.main()
