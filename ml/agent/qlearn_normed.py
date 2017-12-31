import numpy as np

from ml.agent.qlearn import QLearnAgent


class QLearnNormedAgent(QLearnAgent):
    name = 'qlearnnormed'

    STD_THRESHOLD = 0.02

    def get_commands(self, game_map, round_start_time):
        features = self.produce_features(game_map)
        predictions = self._model.predict(X=features)
        predictions = self._normalize_predictions(predictions)
        ship_planet_assignments = self.get_ship_planet_assignments(game_map, predictions)
        ship_commands = self.get_ship_commands(game_map, ship_planet_assignments, round_start_time)
        return ship_commands

    def _normalize_predictions(self, predictions):
        std = np.std(predictions, axis=0)

        if std <= self.STD_THRESHOLD:
            predictions = np.zeros(shape=len(predictions))
            predictions[0] = 1.0

        return predictions
