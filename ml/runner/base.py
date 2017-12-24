import hlt
import time


class GameRunner:
    def __init__(self, agent, data_dir):
        self._data_dir = data_dir
        self.agent = agent

    def run(self):
        game = hlt.Game(self.agent.name)

        while True:
            game_map = game.update_map()

            start_time = time.time()
            action = self.agent.get_commands(game_map, start_time)
            game.send_command_queue(action)
