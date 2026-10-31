import os
import pandas as pd


def save_metadata(records, filename):
    """
    Save recording metadata to a CSV file.
    """

    os.makedirs("dataset/metadata", exist_ok=True)

    df = pd.DataFrame(records)

    filepath = os.path.join(
        "dataset",
        "metadata",
        filename
    )

    df.to_csv(filepath, index=False)

    print(f"Metadata saved to {filepath}")