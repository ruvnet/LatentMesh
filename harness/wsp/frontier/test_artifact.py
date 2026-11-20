"""Exercise the structural invariants that make the portable readout noninterfering."""
import copy
import unittest

import onnx
from onnx import TensorProto, helper

from artifact import seal


class PreservationTest(unittest.TestCase):
    def setUp(self):
        info = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])
        original = helper.make_node("Add", ["input", "constant"], ["logits"], name="native")
        constant = helper.make_tensor("constant", TensorProto.FLOAT, [1], [1])
        graph = helper.make_graph([original], "native", [info("input")], [info("logits")], [constant])
        self.base = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        self.model = copy.deepcopy(self.base)
        for output in ("lmwsp_score", "lmwsp_route"):
            self.model.graph.node.append(helper.make_node("Cast", ["logits"], [output], name=output, to=TensorProto.FLOAT))
            self.model.graph.output.append(info(output))

    def test_preserves_native_computation(self):
        self.assertTrue(seal(self.base, self.model)["nativeComputationIdentical"])

    def test_changed_original_weight_is_rejected(self):
        self.model.graph.initializer[0].float_data[0] = 2
        with self.assertRaisesRegex(ValueError, "original computational graph"):
            seal(self.base, self.model)

    def test_original_output_cannot_be_replaced(self):
        self.model.graph.node[1].output[0] = "logits"
        with self.assertRaises((onnx.checker.ValidationError, ValueError)):
            seal(self.base, self.model)

    def test_unapproved_operator_is_rejected(self):
        self.model.graph.node[1].CopyFrom(helper.make_node("Neg", ["logits"], ["lmwsp_score"], name="lmwsp_score"))
        with self.assertRaisesRegex(ValueError, "unexpected operation"):
            seal(self.base, self.model)


if __name__ == "__main__":
    unittest.main()
