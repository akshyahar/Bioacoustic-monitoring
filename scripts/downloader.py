import requests
import os


def download_audio(download_url, folder, filename):
    """
    Downloads one audio file.
    """

    response = requests.get(download_url)

    if response.status_code != 200:
        print("Download failed!")
        return None

    file_path = os.path.join(folder, filename)

    with open(file_path, "wb") as file:
        file.write(response.content)

    return file_path