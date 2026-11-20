import unittest

import numpy as np

from score import METHODS, panel
from stream import OUTPUTS, step


class ContextTest(unittest.TestCase):
    def test_joint_resampling_and_paired_turn_changes(self):
        observations = {"truth": "11000", "clean": "11110", "predictions": {k:"10010" for k in METHODS}}
        observations["predictions"]["context.transfer.majority"] = "11000"
        p = panel(observations,True,True)
        t = p["methods"]["context.transfer.majority"]
        self.assertEqual((t["tp"],t["fp"],t["precision"],t["recall"]),(2,0,100,100))
        pair = p["paired"]["context.transfer.majority/context.local.majority"]
        self.assertEqual((pair["newTrue"],pair["removedFalse"],pair["lostTrue"],pair["newFalse"]),(1,1,0,0))
        self.assertEqual(p["paired"]["context.transfer.f05/context.local.f05"]["paired95"],[0,0])
        observations["predictions"]["context.transfer.majority"] = "00000"
        z = panel(observations,True,True)
        self.assertIsNone(z["methods"]["context.transfer.majority"]["precision"])
        self.assertFalse(z["criteria"]["precision"])

    def test_state_belongs_to_each_conversation(self):
        class Stub:
            def run(self,outputs,feed):
                assert outputs == OUTPUTS
                return feed.get("ctx_previous")
        tokens = {"input_ids":np.zeros((2,3),dtype=np.int64)}
        self.assertIsNone(step(Stub(),tokens))
        two = np.arange(16).reshape(2,8)
        np.testing.assert_array_equal(step(Stub(),tokens,two),two)
        for bad in (two[:1],np.full((2,8),np.inf)):
            with self.assertRaises(ValueError):
                step(Stub(),tokens,bad)


if __name__ == "__main__":
    unittest.main()
