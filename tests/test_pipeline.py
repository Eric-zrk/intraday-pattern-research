"""Checks for future-data leakage, alignment and reload consistency."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from intraday_patterns.data import FEATURES, build_dataset, grid, load_days, sample, time_masks
from intraday_patterns.demo import generate
from intraday_patterns.model import PatternPredictor


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "bars.csv"
        generate(cls.path, 30)
        cls.days, _, _ = load_days(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_shapes_and_chronological_masks(self):
        d = build_dataset(self.days)
        self.assertEqual(d["seq"].shape, (10, 240, 6))
        self.assertEqual(d["cross"].shape, (10, 25))
        self.assertEqual(len(FEATURES), 25)
        self.assertTrue(np.isfinite(d["cross"]).all())
        masks = time_masks(d["dates"], d["dates"][4], d["dates"][6])
        self.assertEqual([int(m.sum()) for m in masks], [5, 2, 3])

    def test_later_bars_cannot_change_current_features(self):
        before = build_dataset(self.days)
        changed = {k: v.copy() for k, v in self.days.items()}
        day = sorted(changed)[20]
        changed[day][10:, :4] *= 1.5
        changed[day][10:, 4:] *= 5
        after = build_dataset(changed)
        np.testing.assert_array_equal(before["seq"][0], after["seq"][0])
        np.testing.assert_array_equal(before["cross"][0], after["cross"][0])
        self.assertFalse(np.allclose(before["shapes"][0], after["shapes"][0]))

    def test_missing_day_cannot_be_silently_bridged(self):
        changed = dict(self.days)
        changed[sorted(changed)[10]] = None
        with self.assertRaises(ValueError):
            build_dataset(changed)

    def test_duplicate_times_rejected(self):
        df = pd.read_csv(self.path)
        bad = Path(self.temp.name) / "duplicate.csv"
        pd.concat([df, df.iloc[:1]]).to_csv(bad, index=False)
        with self.assertRaises(ValueError):
            load_days(bad)

    def test_end_labeled_bars(self):
        df = pd.read_csv(self.path)
        df["trade_time"] = pd.to_datetime(df.trade_time) + pd.Timedelta(minutes=1)
        path = Path(self.temp.name) / "end.csv"
        df.to_csv(path, index=False)
        days, _, _ = load_days(path, "end")
        for k in days:
            np.testing.assert_allclose(days[k], self.days[k])

    def test_inference_feature_consistency(self):
        keys = sorted(self.days)
        seq, cross = sample([self.days[k] for k in keys[:20]], self.days[keys[20]][:10])
        data = build_dataset(self.days)
        np.testing.assert_array_equal(seq, data["seq"][0])
        np.testing.assert_array_equal(cross, data["cross"][0])

    def test_checkpoint_roundtrip_all_model_modes(self):
        torch.set_num_threads(1)
        for mode in ["hybrid", "gru", "mlp"]:
            m = PatternPredictor(mode).eval()
            seq, cross = torch.zeros(1, 240, 6), torch.zeros(1, 25)
            with torch.no_grad():
                expected = m(seq, cross)
            path = Path(self.temp.name) / "weights.pt"
            torch.save(m.state_dict(), path)
            other = PatternPredictor(mode).eval()
            other.load_state_dict(torch.load(path, weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(expected, other(seq, cross))

    def test_sbd_assignment_is_independent_of_prediction_batch_size(self):
        from intraday_patterns.clustering import assign_clusters, prototypes
        centers = prototypes()
        batch = np.concatenate([centers, centers[[2, 0, 1]], centers[[0]]])
        expected = np.array([0, 1, 2, 2, 0, 1, 0])
        np.testing.assert_array_equal(assign_clusters(batch, centers), expected)
        singles = np.array([assign_clusters(x[None], centers)[0] for x in batch])
        np.testing.assert_array_equal(singles, expected)


if __name__ == "__main__":
    unittest.main()
