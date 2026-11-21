import unittest

import numpy as np

from dialogue_score import MODES, summarize


class FullSenderGate(unittest.TestCase):
    def test_same_predictions_do_not_advance(self):
        y=np.tile([0,1,0,1],32)
        flags="0101"*32
        data={a+"."+m:flags for a in ("transfer","local") for m in MODES}
        result=summarize(y,data,("transfer","local"))
        contrast=result["paired"]["local.majority"]
        self.assertEqual(contrast["ci95"],[0,0])
        self.assertFalse(contrast["positiveLower"])
        self.assertEqual(contrast["discordant"],0)

    def test_paired_disagreement_counts(self):
        y=np.tile([0,1,0,1],32)
        data={a+"."+m:("0101" if a=="transfer" else "0100")*32 for a in ("transfer","local") for m in MODES}
        diff=summarize(y,data,("transfer","local"))["paired"]["local.majority"]
        self.assertEqual((diff["newTrue"],diff["newFalse"],diff["lostTrue"],diff["removedFalse"]),(32,0,0,0))
        self.assertTrue(diff["positiveLower"])


if __name__=="__main__":
    unittest.main()
