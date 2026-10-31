import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


PROCESSED_DIR = "dataset/processed_audio"
MEL_DIR = "dataset/features/mels"

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


def create_mel_spectrogram(audio_path, output_path):

    # Load audio
    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Generate Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # Convert power to decibels
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Create output folder
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    # Create spectrogram image
    plt.figure(figsize=(6, 4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel"
    )

    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


def process_dataset():

    total = 0

    for root, dirs, files in os.walk(PROCESSED_DIR):

        for filename in files:

            if not filename.lower().endswith(".wav"):
                continue

            input_path = os.path.join(
                root,
                filename
            )

            # Preserve folder structure
            relative_path = os.path.relpath(
                input_path,
                PROCESSED_DIR
            )

            output_path = os.path.join(
                MEL_DIR,
                os.path.splitext(relative_path)[0] + ".png"
            )

            print(
                f"[{total + 1}] {input_path}"
            )

            try:

                create_mel_spectrogram(
                    input_path,
                    output_path
                )

                total += 1

            except Exception as e:

                print(
                    f"ERROR: {input_path}"
                )

                print(e)


    print("\n" + "=" * 60)
    print("Mel-spectrogram generation completed!")
    print("Total spectrograms:", total)
    print("Saved to:", MEL_DIR)


if __name__ == "__main__":
    process_dataset()