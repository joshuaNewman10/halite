from abc import abstractmethod
from ml.util import angle, angle_dist


class Parser:
    def __init__(self, data_dir):
        self._data_dir = data_dir

    def parse(self, all_games_json_data, bot_to_imitate=None):
        """
        Parse the games to compute features. This method computes PER_PLANET_FEATURES features for each planet in each frame
        in each game the bot we're imitating played.

        :param all_games_json_data: list of json dictionaries describing games
        :param bot_to_imitate: name of the bot to imitate or None if we want to imitate the bot who won the most games
        :return: replays ready for training
        """
        print("Parsing replays...")
        training_data = []

        print("Bot to imitate: {}.".format(bot_to_imitate))
        for json_data in all_games_json_data:
            training_data += self.parse_game(json_data, bot_to_imitate)

        if not training_data:
            raise Exception("Didn't find any matching games. Try different bot.")

        print('Num samples %s' % (len(training_data)))
        return training_data

    @abstractmethod
    def parse_game(self, json_data, bot_to_imitate=None):
        raise NotImplementedError()

    def get_best_bot(self, all_games_json_data):
        players_games_count = {}
        for json_data in all_games_json_data:
            w = self.find_winner(json_data)
            p = json_data['player_names'][int(w)]
            if p not in players_games_count:
                players_games_count[p] = 0
            players_games_count[p] += 1

        bot_to_imitate = max(players_games_count, key=players_games_count.get)
        return bot_to_imitate

    def find_winner(self, data):
        for player, stats in data['stats'].items():
            if stats['rank'] == 1:
                return player
        return -1

    def find_target_planet(self, bot_id, current_frame, planets, move):
        """
        Find a planet which the ship tried to go to. We try to find it by looking at the angle that the ship moved
        with and the angle between the ship and the planet.
        :param bot_id: id of bot to imitate
        :param current_frame: current frame
        :param planets: planets replays
        :param move: current move to analyze
        :return: id of the planet that ship was moving towards
        """

        if move['type'] == 'dock':
            # If the move was to dock, we know the planet we wanted to move towards
            return move['planet_id']
        if move['type'] != 'thrust':
            # If the move was not "thrust" (i.e. it was "undock"), there is no angle to analyze
            return -1

        ship_angle = move['angle']
        ship_data = current_frame['ships'][bot_id][str(move['shipId'])]
        ship_x = ship_data['x']
        ship_y = ship_data['y']

        optimal_planet = -1
        optimal_angle = -1
        for planet_data in planets:
            planet_id = str(planet_data['id'])
            if planet_id not in current_frame['planets'] or current_frame['planets'][planet_id]['health'] <= 0:
                continue

            planet_x = planet_data['x']
            planet_y = planet_data['y']
            a = angle(planet_x - ship_x, planet_y - ship_y)
            # We try to find the planet with minimal angle distance
            if optimal_planet == -1 or angle_dist(ship_angle, a) < angle_dist(ship_angle, optimal_angle):
                optimal_planet = planet_id
                optimal_angle = a

        return optimal_planet
