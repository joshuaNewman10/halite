import logging
from ml.agent.starter import StarterAgent


class QLearnAgent(StarterAgent):
    name = 'qlearn'

    def get_commands(self, game_map, round_start_time):
        features = self.produce_features(game_map)
        predictions = self._model.predict(X=features)
        logging.debug('Predictions %s', predictions)
        ship_planet_assignments = self.get_ship_planet_assignments(game_map, predictions)
        ship_commands = self.get_ship_commands(game_map, ship_planet_assignments, round_start_time)
        return ship_commands
