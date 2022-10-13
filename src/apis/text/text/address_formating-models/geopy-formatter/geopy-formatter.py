# format the input string as an clean adress using geopy

from typing import Dict
from geopy.geocoders import Nominatim


def predict(address: str) -> Dict[str, str]:
    """
    Format the input string as an clean adress using geopy

    Args:
        address (str): The address to format
    Returns:
        Dict[str, str]: The formated address
    """

    geolocator = Nominatim(user_agent="gladia")

    location = geolocator.geocode(address, addressdetails=True))

    raw_prediction = location.raw

    return {'prediction': location.address, 'raw_prediction': raw_prediction}
