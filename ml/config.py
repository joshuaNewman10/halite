# Max number of planets.
PLANET_MAX_NUM = 28

# These are the features we compute per planet
FEATURE_NAMES = [
    "health",
    "available_docking_spots",
    "remaining_production",
    "signed_current_production",
    "gravity",
    "closest_friendly_ship_distance",
    "closest_enemy_ship_distance",
    "ownership",
    "distance_from_center",
    "weighted_average_distance_from_friendly_ships",
    "is_active"]

# Number of initial features per planet we have
PER_PLANET_FEATURES = len(FEATURE_NAMES)


