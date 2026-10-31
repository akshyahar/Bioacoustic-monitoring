import os


def create_species_folder(category, species):
    """
    Creates the folder for a species if it doesn't exist.
    Example:
    dataset/raw_audio/birds/Asian_Koel/
    """

    folder = os.path.join(
        "dataset",
        "raw_audio",
        category,
        species.replace(" ", "_")
    )

    os.makedirs(folder, exist_ok=True)

    return folder