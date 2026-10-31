import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("XC_API_KEY")

BASE_URL = "https://xeno-canto.org/api/3/recordings"


def search_species(scientific_name):
    """
    Search Xeno-canto for one species.
    """

    params = {
        "query": f'sp:"{scientific_name}"',
        "key": API_KEY
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        raise Exception(
            f"API Error: {response.status_code}\n{response.text}"
        )

    return response.json()