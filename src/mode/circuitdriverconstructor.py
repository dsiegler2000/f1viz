from mode import circuitdriver

# Straight alias of circuitdriver


def get_layout(circuit_id=-1, driver_id=-1, constructor_id=-1, **kwargs):
    return circuitdriver.get_layout(circuit_id=circuit_id, driver_id=driver_id)
