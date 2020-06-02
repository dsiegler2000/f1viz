import logging

import pandas as pd
from functools import lru_cache


@lru_cache(maxsize=None)
def load_circuits():
    logging.info("Loading circuits for the first time")
    return pd.read_csv("data/circuits_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_constructor_results():
    logging.info("Loading constructor results for the first time (UNIMPLEMENTED)")
    return None


@lru_cache(maxsize=None)
def load_constructor_standings():
    return pd.read_csv("data/constructor_standings_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_constructors():
    return pd.read_csv("data/constructors_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_driver_standings():
    return pd.read_csv("data/driver_standings_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_drivers():
    return pd.read_csv("data/drivers_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_lap_times():
    logging.info("Loading lap times for the first time")
    return pd.read_csv("data/lap_times_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_pit_stops():
    logging.info("Loading pit stops for the first time")
    return pd.read_csv("data/pit_stops_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_qualifying():
    logging.info("Loading qualifying for the first time")
    return pd.read_csv("data/qualifying_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_races():
    logging.info("Loading races for the first time")
    return pd.read_csv("data/races_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_results():
    logging.info("Loading results for the first time")
    return pd.read_csv("data/results_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_seasons():
    logging.info("Loading seasons for the first time")
    return pd.read_csv("data/seasons_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_status():
    logging.info("Loading status for the first time")
    return pd.read_csv("data/status_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_constructor_colors():
    logging.info("Loading constructor colors for the first time")
    return pd.read_csv("data/static_data/constructor_colors.csv").set_index("constructorId")


@lru_cache(maxsize=None)
def load_fastest_lap_data():
    logging.info("Loading fastest lap data for the first time")
    return pd.read_csv("data/fastest_lap_data.csv", index_col=0)
