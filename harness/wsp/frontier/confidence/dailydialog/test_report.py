"""Regression checks for abstention and the joint native-logit acceptance criterion."""
import unittest
from unittest.mock import patch

import report


class GateTests(unittest.TestCase):
    @patch.object(report, "DRAWS", 250)
    def test_precision_recall_denominators_and_native_parity(self):
        # Each dialogue has one positive and nine negatives. One correct and one
        # false tag gives exactly 50% precision at 100% recall.
        d = {"truth": "1000000000"*8, "groups": [i for i in range(8) for _ in range(10)], "isClean": "1"*80,
             "clean": {"rows": 80, "positive": 8, "dialogues": 8},
             "predictions": {a: "1100000000"*8 for a in report.ARMS},
             "parity": {"frozenDrift": 0, "adaptedDrift": 0, "instrumentedDrift": 0, "frozenTagErrors": 0,
                        "adaptedMajorityErrors": 0, "maxFrozenEvidenceError": 0, "maxAdaptedEvidenceError": 0}}
        d["predictions"]["pca.majority"] = "0"*80
        p = report.panel(d, "clean")
        self.assertEqual(p["methods"]["frozen.scoreOnly"]["precision"], 50.)
        self.assertEqual(p["methods"]["frozen.scoreOnly"]["recall"], 100.)
        self.assertTrue(p["utilityGate"]["frozen.scoreOnly"])
        self.assertIsNone(p["methods"]["pca.majority"]["precision"])
        self.assertFalse(p["utilityGate"]["pca.majority"])
        self.assertEqual(p["paired"]["transfer.majority"]["local.majority"]["f05"]["paired95"], [0.,0.])
        self.assertFalse(p["transferBenefit"])
        d["parity"]["frozenDrift"] = .001
        again = report.panel(d, "clean")
        self.assertFalse(again["utilityGate"]["frozen.scoreOnly"])
        self.assertTrue(again["utilityGate"]["transfer.majority"])


if __name__ == "__main__":
    unittest.main()
