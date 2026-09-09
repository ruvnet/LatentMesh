import copy
import unittest
import numpy as np
from calibrate import evaluate, matrices
from qualify import qualify
from apply_calibration import apply

class CalibrationTests(unittest.TestCase):
    def rows(self):
        rng = np.random.default_rng(7)
        rows = []
        for tx in range(6):
            for _ in range(20):
                power = float(rng.uniform(-100, -40))
                distance = float(rng.uniform(1, 40))
                bounce = int(rng.integers(0, 4))
                rows.append(dict(site="fixture", tx=str(tx), rt_dbm=power,
                    measured_dbm=power + 20 + .2 * distance + bounce,
                    distance_m=distance, bounces=bounce, frequency_ghz=6.75))
        return rows

    def test_held_out_fit(self):
        result = evaluate(self.rows())
        self.assertLess(result["residual_rmse_db"], .1)
        self.assertTrue(result["calibration_gate"])
        self.assertFalse(result["production_qualified"])
        self.assertEqual(len(result["folds"]), 6)

    def test_invalid(self):
        for key, value in (("distance_m", -1), ("rt_dbm", float("nan")), ("tx", "")):
            rows = self.rows(); rows[0][key] = value
            with self.assertRaises(ValueError): matrices(rows)

    def test_single_tx_rejected(self):
        rows = self.rows()
        for row in rows: row["tx"] = "one"
        with self.assertRaises(ValueError): matrices(rows)

    def test_artifact_roundtrip_and_site_gate(self):
        import json
        rows = self.rows()
        model = json.loads(json.dumps(evaluate(rows)))
        result = apply(model, rows)
        self.assertLess(abs(result[0]["calibrated_dbm"] - rows[0]["measured_dbm"]), .1)
        rows[0]["site"] = "unseen"
        with self.assertRaises(ValueError): apply(model, rows)

class QualificationTests(unittest.TestCase):
    def document(self):
        rows = []
        for arm in ("conventional", "split"):
            for i in range(10000):
                rows.append(dict(trial_id=i, arm=arm, a_ns=1000, b_ns=1000,
                    fallback=False, block_error=False, utility=2 if arm == "split" else 1,
                    joules=1., seconds=1., bandwidth_hz=20e6))
        return dict(metadata=dict(evidence="fixture"), rows=rows)

    def test_fixture_never_promotes(self):
        result = qualify(self.document())
        self.assertTrue(all(result["gates"].values()))
        self.assertFalse(result["campaign_candidate"])
        self.assertFalse(result["production_qualified"])

    def test_stopping(self):
        d = self.document()
        for r in d["rows"]:
            if r["arm"] == "split": r["a_ns"] = 200000
        self.assertTrue(qualify(d)["stop"])

    def test_invalid_and_duplicate(self):
        d = self.document(); d["rows"][0]["joules"] = 0
        with self.assertRaises(ValueError): qualify(d)
        d = self.document(); d["rows"].append(copy.copy(d["rows"][0]))
        with self.assertRaises(ValueError): qualify(d)

if __name__ == "__main__": unittest.main()
