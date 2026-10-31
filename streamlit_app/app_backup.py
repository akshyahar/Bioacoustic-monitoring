import os
import json
import tempfile

import joblib
import numpy as np
import librosa
import librosa.display
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/forest_health_mfcc_svm.joblib"

MODEL = joblib.load(MODEL_PATH)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Forest Health Monitor",
    page_icon="🌳",
    layout="wide"
)

st.title("Forest Health Monitoring System")

st.write(
    "Bioacoustic forest assessment using MFCC-SVM "
    "classification and acoustic diversity analysis."
)

st.info(
    "MFCC + RBF-SVM | "
    "5-fold CV: 74.57% ± 4.60%"
)


# ============================================================
# MFCC EXTRACTION
# ============================================================

def extract_mfcc(y, sr):

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    mean = np.mean(
        mfcc,
        axis=1
    )

    std = np.std(
        mfcc,
        axis=1
    )

    return np.concatenate([
        mean,
        std
    ]).reshape(1, -1)


# ============================================================
# ADI CALCULATION
# ============================================================

def calculate_adi(
    y,
    sr,
    n_fft=2048,
    hop_length=512
):

    spectrogram = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length
        )
    )

    power = spectrogram ** 2

    band_energy = np.mean(
        power,
        axis=1
    )

    total_energy = np.sum(
        band_energy
    )

    if total_energy <= 0:
        return 0.0

    proportions = (
        band_energy /
        total_energy
    )

    proportions = proportions[
        proportions > 0
    ]

    entropy = -np.sum(
        proportions *
        np.log(proportions)
    )

    max_entropy = np.log(
        len(band_energy)
    )

    if max_entropy <= 0:
        return 0.0

    return float(
        entropy / max_entropy
    )


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(audio_bytes):

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as temp:

        temp.write(audio_bytes)

        temp_path = temp.name

    try:

        y, sr = librosa.load(
            temp_path,
            sr=22050,
            mono=True
        )

    finally:

        os.remove(temp_path)

    return y, sr


# ============================================================
# SINGLE RECORDING
# ============================================================

st.header("Individual Recording Analysis")

single_file = st.file_uploader(
    "Upload an audio recording",
    type=["wav", "mp3"],
    key="single"
)


if single_file is not None:

    audio_bytes = single_file.read()

    y, sr = load_audio(
        audio_bytes
    )

    features = extract_mfcc(
        y,
        sr
    )

    prediction = MODEL.predict(
        features
    )[0]

    probabilities = MODEL.predict_proba(
        features
    )[0]

    classes = MODEL.classes_

    confidence = np.max(
        probabilities
    )

    adi = calculate_adi(
        y,
        sr
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.subheader("Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Detected Species",
            prediction
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Acoustic Diversity Index",
            f"{adi:.4f}"
        )


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    st.subheader(
        "Species Classification Probabilities"
    )

    probability_data = {

        class_name:
        float(probability * 100)

        for class_name, probability
        in zip(
            classes,
            probabilities
        )
    }

    st.bar_chart(
        probability_data
    )


    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    st.subheader(
        "MFCC Representation"
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    img = librosa.display.specshow(
        mfcc,
        x_axis="time",
        sr=sr,
        ax=ax
    )

    ax.set_title(
        "MFCC Features"
    )

    fig.colorbar(
        img,
        ax=ax
    )

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# MULTI-RECORDING SURVEY
# ============================================================

st.header("Forest Acoustic Survey")

survey_files = st.file_uploader(
    "Upload multiple forest recordings",
    type=["wav", "mp3"],
    accept_multiple_files=True,
    key="survey"
)


if survey_files:

    st.write(
        f"Recordings uploaded: "
        f"**{len(survey_files)}**"
    )


    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    species_counts = {
        class_name: 0
        for class_name in MODEL.classes_
    }

    confidence_values = []

    adi_values = []

    results = []


    # --------------------------------------------------------
    # PROCESS EACH RECORDING
    # --------------------------------------------------------

    progress = st.progress(0)

    for index, uploaded in enumerate(
        survey_files
    ):

        try:

            audio_bytes = uploaded.read()

            y, sr = load_audio(
                audio_bytes
            )

            features = extract_mfcc(
                y,
                sr
            )

            prediction = MODEL.predict(
                features
            )[0]

            probabilities = MODEL.predict_proba(
                features
            )[0]

            confidence = np.max(
                probabilities
            )

            adi = calculate_adi(
                y,
                sr
            )

            species_counts[
                prediction
            ] += 1

            confidence_values.append(
                confidence
            )

            adi_values.append(
                adi
            )

            results.append({

                "file":
                    uploaded.name,

                "species":
                    prediction,

                "confidence":
                    round(
                        confidence * 100,
                        2
                    ),

                "ADI":
                    round(
                        adi,
                        4
                    )
            })

        except Exception as e:

            st.error(
                f"Error processing "
                f"{uploaded.name}: {e}"
            )

        progress.progress(
            (index + 1) /
            len(survey_files)
        )


    # ========================================================
    # ECOLOGICAL METRICS
    # ========================================================

    total_recordings = len(
        results
    )

    counts = np.array(
        list(
            species_counts.values()
        ),
        dtype=float
    )


    # --------------------------------------------------------
    # SPECIES RICHNESS
    # --------------------------------------------------------

    richness = int(
        np.sum(
            counts > 0
        )
    )

    total_species = len(
        species_counts
    )


    # --------------------------------------------------------
    # SHANNON DIVERSITY
    # --------------------------------------------------------

    if total_recordings > 0:

        proportions = (
            counts /
            total_recordings
        )

        nonzero = proportions[
            proportions > 0
        ]

        shannon = -np.sum(
            nonzero *
            np.log(nonzero)
        )

    else:

        shannon = 0


    # --------------------------------------------------------
    # NORMALIZED SHANNON
    # --------------------------------------------------------

    max_shannon = np.log(
        total_species
    )

    if max_shannon > 0:

        normalized_shannon = (
            shannon /
            max_shannon
        ) * 100

    else:

        normalized_shannon = 0


    # --------------------------------------------------------
    # EVENNESS
    # --------------------------------------------------------

    if richness > 1:

        evenness = (
            shannon /
            np.log(richness)
        ) * 100

    else:

        evenness = 0


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if confidence_values:

        average_confidence = (
            np.mean(
                confidence_values
            ) * 100
        )

    else:

        average_confidence = 0


    # --------------------------------------------------------
    # ADI
    # --------------------------------------------------------

    if adi_values:

        average_adi = np.mean(
            adi_values
        )

    else:

        average_adi = 0


    # ========================================================
    # PROTOTYPE FOREST HEALTH INDEX
    # ========================================================

    health_score = (

        0.30 *
        normalized_shannon

        +

        0.20 *
        evenness

        +

        0.20 *
        average_adi * 100

        +

        0.30 *
        average_confidence
    )

    health_score = min(
        100,
        health_score
    )


    # ========================================================
    # STATUS
    # ========================================================

    if health_score >= 80:

        status = "Healthy"

    elif health_score >= 60:

        status = "Moderately Healthy"

    elif health_score >= 40:

        status = "Vulnerable"

    else:

        status = "Degraded"


    # ========================================================
    # DASHBOARD
    # ========================================================

    st.subheader(
        "Forest Health Assessment"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Recordings",
            total_recordings
        )

    with col2:

        st.metric(
            "Species Richness",
            f"{richness}/{total_species}"
        )

    with col3:

        st.metric(
            "Shannon Diversity",
            f"{shannon:.2f}"
        )

    with col4:

        st.metric(
            "Average ADI",
            f"{average_adi:.3f}"
        )

    with col5:

        st.metric(
            "Health Score",
            f"{health_score:.2f}/100"
        )


    st.subheader(
        f"Forest Status: {status}"
    )


    # ========================================================
    # SPECIES DISTRIBUTION
    # ========================================================

    st.subheader(
        "Species Distribution"
    )

    st.bar_chart(
        species_counts
    )


    # ========================================================
    # METRICS
    # ========================================================

    st.subheader(
        "Acoustic & Ecological Metrics"
    )

    st.write(
        f"Species richness: "
        f"{richness}/{total_species}"
    )

    st.write(
        f"Normalized Shannon diversity: "
        f"{normalized_shannon:.2f}%"
    )

    st.write(
        f"Species evenness: "
        f"{evenness:.2f}%"
    )

    st.write(
        f"Average ADI: "
        f"{average_adi:.4f}"
    )

    st.write(
        f"Average SVM confidence: "
        f"{average_confidence:.2f}%"
    )


    # ========================================================
    # RECORDING TABLE
    # ========================================================

    st.subheader(
        "Recording-Level Results"
    )

    st.dataframe(
        results,
        use_container_width=True
    )


    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "classifier":
            "MFCC + RBF-SVM",

        "cv_accuracy":
            74.57,

        "cv_standard_deviation":
            4.60,

        "recordings":
            total_recordings,

        "species_richness":
            richness,

        "total_species":
            total_species,

        "shannon_diversity":
            round(
                float(shannon),
                4
            ),

        "normalized_shannon":
            round(
                float(normalized_shannon),
                2
            ),

        "species_evenness":
            round(
                float(evenness),
                2
            ),

        "average_adi":
            round(
                float(average_adi),
                4
            ),

        "average_confidence":
            round(
                float(average_confidence),
                2
            ),

        "forest_health_score":
            round(
                float(health_score),
                2
            ),

        "forest_status":
            status,

        "species_distribution":
            species_counts,

        "recording_results":
            results
    }


    st.download_button(
        "Download Forest Health Report",
        data=json.dumps(
            report,
            indent=4
        ),
        file_name="forest_health_report.json",
        mime="application/json"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Forest Health Monitoring System | "
    "MFCC + SVM + Acoustic Diversity Analysis"
)

st.caption(
    "The Forest Health Index is a proposed prototype "
    "metric and is not a validated ecological measurement."
)