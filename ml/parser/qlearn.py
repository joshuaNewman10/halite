import math

from ml.config import PLANET_MAX_NUM
from ml.parser.base import Parser
from ml.util import distance2, distance


class QlearnParser(Parser):
    name = 'qlearn'

    def _get_discrete_allocations(self, allocations):
        for planet_id, allocation in allocations.items():
            allocations[planet_id] = round(allocation, 2)

        return allocations

    def _get_frame_reward(self, frame, bot_to_imitate_id):
        planets = frame['planets']
        ships = frame['ships']

        return self._get_planet_reward(planets, bot_to_imitate_id) + self._get_ship_reward(ships, bot_to_imitate_id)

    def _get_planet_reward(self, planets, bot_to_imitate_id):
        planets = [planet for planet in planets.values() if planet['owner'] == bot_to_imitate_id]
        return len(planets)

    def _get_ship_reward(self, ships, bot_to_imitate_id):
        bot_ships = ships[bot_to_imitate_id]
        return len(bot_ships)

    def parse_game(self, json_data, bot_to_imitate=None):
        game_training_data = []
        observations = []

        frames = json_data['frames']
        moves = json_data['moves']
        width = json_data['width']
        height = json_data['height']

        if not bot_to_imitate:
            winner = self.find_winner(json_data)
            bot_to_imitate = json_data['player_names'][int(winner)]

        print("Bot to imitate: {}.".format(bot_to_imitate))
        # We train on all the games of the bot regardless whether it won or not.
        bot_to_imitate_id = str(json_data['player_names'].index(bot_to_imitate))

        # Ignore the last frame, no decision to be made there
        # Ignore the first
        for idx in range(0, len(frames) - 1):
            current_moves = moves[idx]
            current_frame = frames[idx]

            if bot_to_imitate_id not in current_frame['ships'] or len(
                    current_frame['ships'][bot_to_imitate_id]) == 0:
                continue

            planet_features = {}  # planet_id -> list of features per ship per planet
            current_planets = current_frame['planets']

            # find % allocation for all ships
            all_moving_ships = 0
            allocations = {}

            # for each planet we want to find how many ships are being moved towards it now
            for ship_id, ship_data in current_frame['ships'][bot_to_imitate_id].items():
                if ship_id in current_moves[bot_to_imitate_id][0]:
                    p = self.find_target_planet(bot_to_imitate_id, current_frame,
                                                json_data['planets'],
                                                current_moves[bot_to_imitate_id][0][ship_id],
                                                )
                    planet_id = int(p)
                    if planet_id < 0 or planet_id >= PLANET_MAX_NUM:
                        continue

                    if p not in allocations:
                        allocations[p] = 0
                    allocations[p] = allocations[p] + 1
                    all_moving_ships = all_moving_ships + 1

            if all_moving_ships == 0:
                continue

            # Compute what % of the ships should be sent to given planet
            for planet_id, allocated_ships in allocations.items():
                allocations[planet_id] = allocated_ships / all_moving_ships

            # Compute features
            for planet_id in range(PLANET_MAX_NUM):

                if str(planet_id) not in current_planets:
                    continue
                planet_data = current_planets[str(planet_id)]

                gravity = 0
                planet_base_data = json_data['planets'][planet_id]
                closest_friendly_ship_distance = 10000
                closest_enemy_ship_distance = 10000

                ownership = 0
                if str(planet_data['owner']) == bot_to_imitate_id:
                    ownership = 1
                elif planet_data['owner'] is not None:
                    ownership = -1

                average_distance = 0
                my_ships_health = 0

                for player_id, ships in current_frame['ships'].items():
                    for ship_id, ship_data in ships.items():
                        is_bot_to_imitate = 1 if player_id == bot_to_imitate_id else -1
                        dist2 = distance2(planet_base_data['x'], planet_base_data['y'], ship_data['x'],
                                          ship_data['y'])
                        dist = math.sqrt(dist2)
                        gravity = gravity + is_bot_to_imitate * ship_data['health'] / dist2
                        if is_bot_to_imitate == 1:
                            closest_friendly_ship_distance = min(closest_friendly_ship_distance, dist)
                            average_distance = average_distance + dist * ship_data['health']
                            my_ships_health = my_ships_health + ship_data['health']
                        else:
                            closest_enemy_ship_distance = min(closest_enemy_ship_distance, dist)

                distance_from_center = distance(planet_base_data['x'], planet_base_data['y'], width / 2, height / 2)
                average_distance = average_distance / my_ships_health

                is_active = 1.0 if planet_base_data['docking_spots'] > len(
                    planet_data['docked_ships']) or ownership != 1 else 0.0

                signed_current_production = planet_data['current_production'] * ownership

                # Features of the planet are inserted into the vector in the order described by FEATURE_NAMES
                planet_features[str(planet_id)] = [
                    planet_data['health'],
                    planet_base_data['docking_spots'] - len(planet_data['docked_ships']),
                    planet_data['remaining_production'],
                    signed_current_production,
                    gravity,
                    closest_friendly_ship_distance,
                    closest_enemy_ship_distance,
                    ownership,
                    distance_from_center,
                    average_distance,
                    is_active]

            reward = self._get_frame_reward(current_frame, bot_to_imitate_id)
            allocations = self._get_discrete_allocations(allocations)

            current_observation = planet_features

            # skip over first frame
            if not observations:
                observations.append(current_observation)
                continue

            previous_observation = observations[-1]

            frame_data = dict(
                next_observation=current_observation,
                observation=previous_observation,
                reward=reward,
                allocations=allocations
            )

            game_training_data.append(frame_data)

        return game_training_data
