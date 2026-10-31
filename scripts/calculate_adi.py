import numpy as np
import librosa


def calculate_adi(
    audio_path,
    sr=22050,
    n_fft=2048,
    hop_length=512
):
    """
    Calculate Acoustic Diversity Index (ADI)
    using frequency-band occupancy.

    Returns ADI as a value between 0 and 1.
    """

    # Load audio
    y, sr = librosa.load(
        audio_path,
        sr=sr,
        mono=True
    )

    # Compute spectrogram
    spectrogram = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length
        )
    )

    # Convert to power
    power = spectrogram ** 2

    # Average energy across time
    band_energy = np.mean(
        power,
        axis=1
    )

    # Avoid division by zero
    total_energy = np.sum(
        band_energy
    )

    if total_energy == 0:
        return 0.0

    # Normalize energy
    proportions = (
        band_energy /
        total_energy
    )

    # Remove zero values
    proportions = proportions[
        proportions > 0
    ]

    # Shannon entropy across frequency bins
    adi = -np.sum(
        proportions *
        np.log(proportions)
    )

    # Normalize to 0–1
    max_adi = np.log(
        len(band_energy)
    )

    if max_adi > 0:

        adi_normalized = (
            adi /
            max_adi
        )

    else:

        adi_normalized = 0

    return float(
        adi_normalized
    )


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python scripts\\calculate_adi.py "
            "\"audio_file.wav\""
        )

        sys.exit()

    audio_path = sys.argv[1]

    adi = calculate_adi(
        audio_path
    )

    print()
    print("=" * 60)
    print("ACOUSTIC DIVERSITY INDEX")
    print("=" * 60)

    print()
    print(
        "Audio:",
        audio_path
    )

    print(
        "ADI:",
        round(adi, 4)
    )

    print(
        "ADI (%):",
        round(adi * 100, 2)
    )

    print()
    print("=" * 60)