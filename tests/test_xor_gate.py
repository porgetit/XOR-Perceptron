import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from xor import XORGate


def test_xor_gate_outputs_expected_values():
    gate = XORGate()
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected = np.array([0, 1, 1, 0])
    np.testing.assert_array_equal(gate.predict(X), expected)


if __name__ == "__main__":
    test_xor_gate_outputs_expected_values()
    print("All tests passed.")