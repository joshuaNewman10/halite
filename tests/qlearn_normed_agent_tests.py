# coding=utf-8

import numpy as np
import unittest

from mock import Mock
from ml.agent.qlearn_normed import QLearnNormedAgent


class QLearnNormedAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = QLearnNormedAgent(
            model=Mock()
        )

    def test_normalize_predictions_none(self):
        predictions = [1.0, 0]
        normed_preds = self.agent._normalize_predictions(predictions)
        self.assertTrue(np.array_equal(predictions, normed_preds))

    def test_normalize_predictions_ok(self):
        predictions = [0, 0]
        normed_preds = self.agent._normalize_predictions(predictions)
        self.assertTrue(np.array_equal(normed_preds, [1.0, 0]))


if __name__ == '__main__':
    unittest.main()
