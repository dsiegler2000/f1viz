from mode import yearcircuitdriver

# Straight alias of yearcircuitdriver


def get_layout(year_id=-1, circuit_id=-1, driver_id=-1, constructor_id=-1, **kwargs):
    return yearcircuitdriver.get_layout(year_id=year_id, circuit_id=circuit_id, driver_id=driver_id)
