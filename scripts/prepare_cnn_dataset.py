import os
import shutil
import random


SOURCE_DIR = "dataset/features/mels"
OUTPUT_DIR = "dataset/features/cnn_dataset"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


def collect_files():

    files_by_class = {}

    for root, dirs, files in os.walk(SOURCE_DIR):

        for filename in files:

            if not filename.lower().endswith(".png"):
                continue

            filepath = os.path.join(root, filename)

            relative_path = os.path.relpath(
                filepath,
                SOURCE_DIR
            )

            parts = relative_path.split(os.sep)

            # Bird structure:
            # mels/birds/Asian_Koel/file.png
            #
            # Frog structure:
            # mels/frogs/SCIALT/file.png

            if len(parts) >= 3:

                category = parts[0]
                species = parts[1]

                class_name = species

            else:

                continue

            if class_name not in files_by_class:
                files_by_class[class_name] = []

            files_by_class[class_name].append(filepath)

    return files_by_class


def copy_files(files, class_name, split):

    destination = os.path.join(
        OUTPUT_DIR,
        split,
        class_name
    )

    os.makedirs(
        destination,
        exist_ok=True
    )

    for filepath in files:

        filename = os.path.basename(filepath)

        destination_file = os.path.join(
            destination,
            filename
        )

        shutil.copy2(
            filepath,
            destination_file
        )


print("Preparing CNN Dataset")
print("=" * 60)


if os.path.exists(OUTPUT_DIR):

    shutil.rmtree(OUTPUT_DIR)


files_by_class = collect_files()


total_files = 0


for class_name, files in sorted(files_by_class.items()):

    random.shuffle(files)

    total = len(files)

    train_end = int(total * TRAIN_RATIO)

    val_end = train_end + int(total * VAL_RATIO)

    train_files = files[:train_end]

    val_files = files[train_end:val_end]

    test_files = files[val_end:]


    copy_files(
        train_files,
        class_name,
        "train"
    )

    copy_files(
        val_files,
        class_name,
        "validation"
    )

    copy_files(
        test_files,
        class_name,
        "test"
    )


    print(
        f"{class_name}: "
        f"{len(train_files)} train | "
        f"{len(val_files)} validation | "
        f"{len(test_files)} test"
    )


    total_files += total


print("\n" + "=" * 60)

print("CNN dataset preparation completed!")

print("Total files:", total_files)

print("Dataset location:", OUTPUT_DIR)