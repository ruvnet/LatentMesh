"""Check paired differences and the strict promotion gate against known joint decisions."""
import unittest

import numpy as np

from score_results import METHODS, population


class PairedDecision(unittest.TestCase):
    def test_identical_arms_never_promote(self):
        truth = "1100" * 32
        x = {"truth":truth, "predictions":dict.fromkeys(METHODS, truth)}
        p = population(x, np.ones(len(truth), dtype=bool), True)
        self.assertEqual(p["methods"]["transfer.majority"]["f05"], 100)
        self.assertEqual(p["paired"]["local.majority"]["ci95"], [0, 0])
        self.assertFalse(p["passes"])
        self.assertFalse(p["senderAttribution"])

    def test_strict_gain_and_native_parity(self):
        truth = "1100" * 32
        baseline = "1000" * 32
        x = {"truth":truth,"predictions":{k:truth if k.startswith("transfer.") else baseline for k in METHODS}}
        mask = np.ones(len(truth), dtype=bool)
        good = population(x, mask, True)
        self.assertTrue(good["replacesIncumbent"])
        self.assertEqual(good["paired"]["local.majority"]["disagreements"], 32)
        self.assertFalse(population(x, mask, False)["passes"])


if __name__ == "__main__":
    unittest.main()
