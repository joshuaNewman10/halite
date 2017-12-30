import hlt
import numpy as np
import time

from ml.config import PER_PLANET_FEATURES, PLANET_MAX_NUM


class GameRunner:
    def __init__(self, agent):
        self.agent = agent

        # Run prediction on random data to make sure that code path is executed at least once before the game starts
        random_input_data = np.random.rand(PLANET_MAX_NUM, PER_PLANET_FEATURES)
        predictions = self.agent._model.predict(random_input_data)

    def run(self):
        game = hlt.Game(self.agent.name)

        while True:
            game_map = game.update_map()
            start_time = time.time()
            action = self.agent.get_commands(game_map, start_time)
            game.send_command_queue(action)
