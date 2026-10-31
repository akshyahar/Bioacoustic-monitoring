from file_manager import create_species_folder
from downloader import download_audio
from metadata_manager import save_metadata
from api_client import search_species


species_data = search_species(
    "Eudynamys scolopaceus"
)

folder = create_species_folder(
    "birds",
    "Asian Koel"
)

record = species_data["recordings"][0]

metadata = []
metadata.append({

    "Recording_ID": record["id"],

    "Common_Name": record["en"],

    "Scientific_Name": record["gen"] + " " + record["sp"],

    "Country": record["cnt"],

    "Location": record["loc"],

    "Quality": record["q"],

    "Length": record["length"],

    "Recordist": record["rec"],

    "Date": record["date"],

    "Download_URL": record["file"]

})


download_url = record["file"]
print(download_url)


download_audio(
    download_url,
    folder,
    "Asian_Koel_001.mp3"
)

save_metadata(
    metadata,
    "birds_metadata.csv"
)

print("Download completed.")


