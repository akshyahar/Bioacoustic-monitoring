import os

from api_client import search_species
from file_manager import create_species_folder
from downloader import download_audio
from metadata_manager import save_metadata
from species import BIRDS


MAX_RECORDINGS_PER_SPECIES = 20

all_metadata = []

print("Starting Bird Dataset Collection")
print("=" * 60)


for common_name, scientific_name in BIRDS:

    print("\n" + "=" * 60)
    print(f"Species: {common_name}")
    print(f"Scientific name: {scientific_name}")

    # Search Xeno-canto
    species_data = search_species(scientific_name)

    recordings = species_data.get("recordings", [])

    print(f"Available recordings: {len(recordings)}")

    # Keep only high-quality recordings
    good_recordings = [
        recording
        for recording in recordings
        if recording.get("q") in ["A", "B"]
    ]

    # Prefer recordings from India
    indian_recordings = [
        recording
        for recording in good_recordings
        if recording.get("cnt", "").lower() == "india"
    ]

    other_recordings = [
        recording
        for recording in good_recordings
        if recording.get("cnt", "").lower() != "india"
    ]

    # Indian recordings first
    selected_recordings = (
        indian_recordings + other_recordings
    )

    # Limit recordings per species
    selected_recordings = selected_recordings[
        :MAX_RECORDINGS_PER_SPECIES
    ]

    print(
        f"High-quality recordings: "
        f"{len(good_recordings)}"
    )

    print(
        f"Indian recordings: "
        f"{len(indian_recordings)}"
    )

    print(
        f"Selected for download: "
        f"{len(selected_recordings)}"
    )

    # Create species folder
    folder = create_species_folder(
    "birds",
    common_name
    )

    # Download recordings
    for index, recording in enumerate(
        selected_recordings,
        start=1
    ):

        recording_id = recording.get("id")

        download_url = recording.get("file")

        if not download_url:
            print(
                f"Skipping recording "
                f"{recording_id}: no download URL"
            )
            continue

        filename = (
            f"{common_name.replace(' ', '_')}_"
            f"{index:03d}.mp3"
        )

        filepath = os.path.join(
            folder,
            filename
        )

        # Avoid downloading duplicates
        if os.path.exists(filepath):

            print(
                f"[EXISTS] {filename}"
            )

        else:

            print(
                f"[DOWNLOAD] "
                f"{index}/{len(selected_recordings)} "
                f"{filename}"
            )

            result = download_audio(
                download_url,
                folder,
                filename
            )

            if result is None:
                continue

        # Store metadata
        metadata = {
            "Recording_ID": recording_id,
            "Common_Name": common_name,
            "Scientific_Name": scientific_name,
            "Country": recording.get("cnt", ""),
            "Location": recording.get("loc", ""),
            "Quality": recording.get("q", ""),
            "Length": recording.get("length", ""),
            "Recordist": recording.get("rec", ""),
            "Date": recording.get("date", ""),
            "Download_URL": download_url
        }

        all_metadata.append(metadata)


# Save metadata
save_metadata(
    all_metadata,
    "birds_metadata.csv"
)

print("\n" + "=" * 60)
print("Bird dataset collection completed!")
print(
    f"Total recordings processed: "
    f"{len(all_metadata)}"
)