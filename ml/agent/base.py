import hlt
import random
import time

from collections import namedtuple

Assignment = namedtuple('assignment', ('ship', 'planet'))


class Agent:
    name = 'agent'

    def __init__(self, max_allowed_time_seconds=1.2, max_ship_corrections=180):
        self.max_allowed_time_seconds = max_allowed_time_seconds
        self.max_ship_corrections = max_ship_corrections

    def get_commands(self, game_map, round_start_time):
        ship_planet_assignments = self.get_ship_planet_assignments(game_map)
        ship_commands = self.get_ship_commands(game_map, ship_planet_assignments, round_start_time)
        return ship_commands

    def get_ship_planet_assignments(self, game_map, _):
        assignments = []

        undocked_ships = self.get_undocked_ships(game_map)
        planets = self.get_planets(game_map)

        for ship in undocked_ships:
            planet = random.choice(planets)
            assignment = Assignment(ship=ship, planet=planet)
            assignments.append(assignment)

        return assignments

    def get_ship_commands(self, game_map, ship_planet_assignments, round_start_time):
        commands = []

        for assignment in ship_planet_assignments:
            ship = assignment.ship
            planet = assignment.planet

            commands.append(self.get_ship_command(ship, planet, game_map, round_start_time))

    def get_ship_command(self, ship, planet, game_map, round_start_time):
        speed = hlt.constants.MAX_SPEED

        if self.is_friendly_planet(planet, game_map):
            if ship.can_dock(planet):
                command = ship.dock(planet)
                return command

            closest_planet_point = ship.closest_point_to(planet)
            command = self.navigate(game_map, round_start_time, ship, closest_planet_point, speed)
            return command

        # hostile planet
        docked_ships = planet.all_docked_ships()
        weakest_ship = self.get_weakest_ship(docked_ships)
        closest_ship_point = ship.closest_point_to(weakest_ship)
        command = self.navigate(game_map, round_start_time, ship, closest_ship_point, speed)
        return command

    def navigate(self, game_map, start_of_round, ship, destination, speed):
        navigation_command = None
        have_time = self.have_time(start_of_round)

        if have_time:
            navigation_command = ship.navigate(
                destination,
                game_map,
                speed=speed,
                max_corrections=self.max_ship_corrections
            )

        if not have_time or navigation_command is None:
            distance = ship.calculate_distance_between(destination)
            speed = speed if (distance >= speed) else distance
            angle = ship.calculate_angle_between(destination)
            navigation_command = ship.thrust(magnitude=speed, angle=angle)

        return navigation_command

    def get_planets(self, game_map):
        return game_map.all_planets()

    def get_undocked_ships(self, game_map):
        return [ship for ship in game_map.get_me().all_ships()
                if ship.docking_status == ship.DockingStatus.UNDOCKED]

    def is_friendly_planet(self, planet, game_map):
        return not planet.is_owned() or planet.owner == game_map.get_me()

    def get_weakest_ship(self, ships):
        weakest_ship = None
        for ship in ships:
            if weakest_ship is None or weakest_ship.health > ship.health:
                weakest_ship = ship

        return weakest_ship

    def have_time(self, start_of_round):
        current_time = time.time()

        return current_time - start_of_round < self.max_allowed_time_seconds
