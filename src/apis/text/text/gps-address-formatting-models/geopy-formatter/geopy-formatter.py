# format the input string as an clean adress using geopy

from typing import Dict

from geopy.geocoders import Nominatim


def predict(latitude: float, longitude: float) -> Dict[str, str]:
    """
    Format the input string as an clean adress using geopy

    Args:
        latitude (float): lattitude of the location
        longitude (float): longitude of the location
    Returns:
        Dict[str, str]: The formated address
    """

    geolocator = Nominatim(user_agent="gladia")

    location = geolocator.reverse(f"{latitude}, {longitude}")

    raw_prediction = location.raw

    return {"prediction": location.address, "raw_prediction": raw_prediction}
