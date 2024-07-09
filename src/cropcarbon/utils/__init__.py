def get_BBOX(west: str, north,
             buffer=500):
    """Method to get extent in EPSG:3035
    from 2 coordinates in LAEA. Add buffer in meter
    to these coordinates to get bbox

    """

    # we now have lower-left corner
    return {
        'east': int(west + buffer),
        'south': int(north - buffer),
        'west': int(west),
        'north': int(north),
    }