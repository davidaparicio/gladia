# format the input string as an clean adress using geopy

from typing import Dict

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

geolocator = Nominatim(user_agent="gladia")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2, max_retries=3)


def predict(address: str) -> Dict[str, str]:
    """
    Format the input string as an clean adress using geopy

    Args:
        address (str): The address to format
    Returns:
        Dict[str, str]: The formated address
    """

    location = geocode(address, addressdetails=True)

    return {"prediction": location.address, "raw_prediction": location.raw}
