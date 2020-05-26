import pandas as pd
from functools import lru_cache


@lru_cache(maxsize=None)
def load_circuits():
    return pd.read_csv("data/circuits_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_constructor_results():
    pass


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
    return pd.read_csv("data/lap_times_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_pit_stops():
    return pd.read_csv("data/pit_stops_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_qualifying():
    return pd.read_csv("data/qualifying_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_races():
    return pd.read_csv("data/races_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_results():
    return pd.read_csv("data/results_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_seasons():
    return pd.read_csv("data/seasons_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_status():
    return pd.read_csv("data/status_augmented.csv", index_col=0)


@lru_cache(maxsize=None)
def load_constructor_colors():
    return pd.read_csv("data/static_data/constructor_colors.csv").set_index("constructorId")


@lru_cache(maxsize=None)
def load_constructor_mean_lap_times():
    return pd.read_csv("data/constructor_mean_lap_times.csv")


@lru_cache(maxsize=None)
def load_fastest_lap_data():
    return pd.read_csv("data/fastest_lap_data.csv", index_col=0)
