import os
import json
import tempfile
import sqlite3
import hashlib
import hmac
import secrets
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import librosa
import librosa.display
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "forest_health_mfcc_svm.joblib"
BIRD_METADATA_PATH = BASE_DIR / "dataset" / "metadata" / "birds_metadata.csv"
FROG_METADATA_PATH = BASE_DIR / "dataset" / "metadata" / "frogs_metadata.csv"

st.set_page_config(
    page_title="Forest Health Monitor",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f5f8f5; }
    [data-testid="stSidebar"] { background-color: #eaf1eb; }
    .main-title { color: #174b35; font-size: 2.15rem; font-weight: 750; }
    .sub-title { color: #557365; font-size: 1rem; }
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #dce8df;
        padding: 14px 16px; border-radius: 12px;
    }
    .section-note {
        background: #edf5ef; padding: 12px 15px;
        border-left: 4px solid #397653; border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run the model training script before starting the dashboard."
        )
    return joblib.load(MODEL_PATH)


try:
    MODEL = load_model()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()


# ============================================================
# AUDIO AND FEATURE FUNCTIONS
# ============================================================

def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.concatenate([mean, std]).reshape(1, -1)


def calculate_adi(y, sr, n_fft=2048, hop_length=512):
    """Project-specific normalized spectral-entropy measure."""
    spectrogram = np.abs(
        librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    )
    power = spectrogram ** 2
    band_energy = np.mean(power, axis=1)
    total_energy = np.sum(band_energy)

    if total_energy <= 0:
        return 0.0

    proportions = band_energy / total_energy
    proportions = proportions[proportions > 0]
    entropy = -np.sum(proportions * np.log(proportions))
    max_entropy = np.log(len(band_energy))

    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def load_audio(audio_bytes):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as temp:
            temp.write(audio_bytes)
            temp_path = temp.name
        y, sr = librosa.load(temp_path, sr=22050, mono=True)
        if len(y) == 0:
            raise ValueError("The uploaded audio file is empty.")
        return y, sr
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def analyze_audio(audio_bytes, filename):
    y, sr = load_audio(audio_bytes)
    features = extract_mfcc(y, sr)
    prediction = MODEL.predict(features)[0]
    probabilities = MODEL.predict_proba(features)[0]
    confidence = float(np.max(probabilities))
    adi = calculate_adi(y, sr)
    return {
        "file": filename,
        "species": str(prediction),
        "confidence": confidence * 100,
        "ADI": adi,
        "probabilities": {
            str(name): float(prob * 100)
            for name, prob in zip(MODEL.classes_, probabilities)
        },
        "audio": y,
        "sample_rate": sr
    }


# ============================================================
# METADATA
# ============================================================

@st.cache_data
def read_metadata(path_string):
    path = Path(path_string)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, UnicodeDecodeError, OSError):
        return pd.DataFrame()


bird_meta = read_metadata(str(BIRD_METADATA_PATH))
frog_meta = read_metadata(str(FROG_METADATA_PATH))



# ============================================================
# AUTHENTICATION AND ROLE-BASED ACCESS
# ============================================================

DB_DIR = BASE_DIR / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
USER_DB_PATH = DB_DIR / "users.db"
PASSWORD_ITERATIONS = 310_000

def get_db_connection():
    conn = sqlite3.connect(USER_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_user_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('Admin', 'Researcher', 'Viewer')),
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def hash_password(password, salt_hex=None):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return salt.hex(), digest.hex()

def verify_password(password, salt_hex, stored_hash):
    _, candidate = hash_password(password, salt_hex)
    return hmac.compare_digest(candidate, stored_hash)

def user_count():
    with get_db_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

def create_user(username, password, role):
    salt, password_hash = hash_password(password)
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO users (username, salt, password_hash, role) VALUES (?, ?, ?, ?)",
            (username.strip(), salt, password_hash, role)
        )
        conn.commit()

def authenticate_user(username, password):
    with get_db_connection() as conn:
        user = conn.execute(
            "SELECT username, salt, password_hash, role FROM users WHERE username = ?",
            (username.strip(),)
        ).fetchone()
    if user and verify_password(password, user["salt"], user["password_hash"]):
        return {"username": user["username"], "role": user["role"]}
    return None

initialize_user_db()

if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None

if user_count() == 0:
    st.title("Set Up Forest Monitor")
    st.info("Create the first administrator account. This account will manage all other users.")
    with st.form("initial_admin_form"):
        new_admin = st.text_input("Admin username")
        admin_password = st.text_input("Create password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        setup_submitted = st.form_submit_button("Create administrator")
    if setup_submitted:
        if len(new_admin.strip()) < 3:
            st.error("Username must contain at least 3 characters.")
        elif len(admin_password) < 10:
            st.error("Use a password with at least 10 characters.")
        elif admin_password != confirm_password:
            st.error("The passwords do not match.")
        else:
            create_user(new_admin, admin_password, "Admin")
            st.success("Administrator account created. Please log in.")
            st.rerun()
    st.stop()

if st.session_state.authenticated_user is None:
    st.title("Forest Health Monitoring System")
    st.subheader("Secure Login")
    with st.form("login_form"):
        login_username = st.text_input("Username")
        login_password = st.text_input("Password", type="password")
        login_submitted = st.form_submit_button("Log in")
    if login_submitted:
        account = authenticate_user(login_username, login_password)
        if account:
            st.session_state.authenticated_user = account
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()

CURRENT_USER = st.session_state.authenticated_user
CURRENT_ROLE = CURRENT_USER["role"]

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("## 🌿 Forest Monitor")
st.sidebar.caption("Bioacoustic intelligence platform")
st.sidebar.write(f"Signed in: **{CURRENT_USER['username']}**")
st.sidebar.caption(f"Role: {CURRENT_ROLE}")

if CURRENT_ROLE == "Admin":
    available_pages = [
        "Overview", "Individual Analysis", "Forest Acoustic Survey",
        "Species & Diversity", "Geographic Monitoring", "Seasonal Activity",
        "User Management"
    ]
elif CURRENT_ROLE == "Researcher":
    available_pages = [
        "Overview", "Individual Analysis", "Forest Acoustic Survey",
        "Species & Diversity", "Geographic Monitoring", "Seasonal Activity"
    ]
else:
    available_pages = [
        "Overview", "Species & Diversity", "Geographic Monitoring",
        "Seasonal Activity"
    ]

page = st.sidebar.radio("Navigation", available_pages)

if st.sidebar.button("Log out"):
    st.session_state.authenticated_user = None
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("Classifier: MFCC + RBF-SVM")
st.sidebar.caption("5-fold CV: 74.57% ± 4.60%")


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">Forest Health Monitoring System</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Bioacoustic species classification and '
    'acoustic diversity assessment</div>',
    unsafe_allow_html=True
)
st.write("")


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    st.subheader("Project Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed recordings", "118")
    c2.metric("Species classes", "6")
    c3.metric("Test accuracy", "70.83%")
    c4.metric("5-fold CV accuracy", "74.57%")

    st.markdown("### System capabilities")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            **AI-based acoustic classification**
            - MFCC feature extraction (40 coefficients)
            - Mean and standard deviation feature aggregation
            - RBF-SVM species classification
            - Prediction confidence and class probabilities
            """
        )
    with right:
        st.markdown(
            """
            **Acoustic and ecological analysis**
            - Project-specific spectral diversity measure
            - Species richness and Shannon diversity
            - Species evenness
            - Proposed Forest Health Index (FHI)
            """
        )

    st.markdown("### Existing dataset metadata")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bird metadata records", len(bird_meta))
    m2.metric(
        "Bird location descriptions",
        int(bird_meta["Location"].nunique())
        if "Location" in bird_meta.columns else 0
    )
    m3.metric(
        "Bird recording dates",
        int(bird_meta["Date"].nunique())
        if "Date" in bird_meta.columns else 0
    )
    m4.metric("Frog metadata records", len(frog_meta))

    st.markdown(
        """
        <div class="section-note">
        <b>Interpretation note:</b> The FHI is a proposed project prototype
        score. It is not a validated ecological measurement. Model confidence
        measures classifier certainty, not forest health.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# INDIVIDUAL RECORDING ANALYSIS
# ============================================================

elif page == "Individual Analysis":
    st.subheader("Individual Recording Analysis")
    uploaded = st.file_uploader(
        "Upload one bird or frog recording",
        type=["wav", "mp3", "ogg", "flac"],
        key="individual_audio"
    )

    if uploaded is not None:
        with st.spinner("Analyzing audio..."):
            try:
                result = analyze_audio(uploaded.getvalue(), uploaded.name)
            except Exception as exc:
                st.error(f"Audio analysis failed: {exc}")
                st.stop()

        a, b, c = st.columns(3)
        a.metric("Predicted class", result["species"])
        b.metric("Prediction confidence", f'{result["confidence"]:.2f}%')
        c.metric("Spectral diversity measure", f'{result["ADI"]:.4f}')

        st.markdown("### Class probability distribution")
        probability_df = pd.DataFrame({
            "Species": list(result["probabilities"].keys()),
            "Probability (%)": list(result["probabilities"].values())
        }).set_index("Species")
        st.bar_chart(probability_df)

        st.markdown("### MFCC representation")
        mfcc = librosa.feature.mfcc(
            y=result["audio"], sr=result["sample_rate"], n_mfcc=40
        )
        fig, ax = plt.subplots(figsize=(11, 4))
        image = librosa.display.specshow(
            mfcc, x_axis="time", sr=result["sample_rate"], ax=ax
        )
        ax.set_title("40-coefficient MFCC representation")
        ax.set_xlabel("Time")
        ax.set_ylabel("MFCC coefficient")
        fig.colorbar(image, ax=ax, format="%+2.0f")
        st.pyplot(fig)
        plt.close(fig)

        st.audio(uploaded)


# ============================================================
# MULTI-RECORDING SURVEY
# ============================================================

elif page == "Forest Acoustic Survey":
    st.subheader("Multi-Recording Forest Acoustic Survey")
    survey_files = st.file_uploader(
        "Upload multiple forest recordings",
        type=["wav", "mp3", "ogg", "flac"],
        accept_multiple_files=True,
        key="survey_audio"
    )

    if survey_files:
        st.info(f"{len(survey_files)} recording(s) selected.")
        if st.button("Run acoustic survey", type="primary"):
            results = []
            progress = st.progress(0)
            status = st.empty()

            for index, uploaded in enumerate(survey_files):
                status.write(f"Processing {uploaded.name}...")
                try:
                    result = analyze_audio(uploaded.getvalue(), uploaded.name)
                    results.append({
                        "file": result["file"],
                        "species": result["species"],
                        "confidence": round(result["confidence"], 2),
                        "ADI": round(result["ADI"], 4)
                    })
                except Exception as exc:
                    st.warning(f"Could not process {uploaded.name}: {exc}")
                progress.progress((index + 1) / len(survey_files))

            status.empty()
            progress.empty()

            if not results:
                st.error("No recordings could be processed.")
                st.stop()

            result_df = pd.DataFrame(results)
            counts = result_df["species"].value_counts()
            total = len(result_df)
            richness = int(counts.size)
            total_classes = len(MODEL.classes_)

            proportions = counts.to_numpy(dtype=float) / total
            shannon = float(-np.sum(proportions * np.log(proportions)))
            normalized_shannon = (
                shannon / np.log(total_classes) * 100
                if total_classes > 1 else 0.0
            )
            evenness = (
                shannon / np.log(richness) * 100
                if richness > 1 else 0.0
            )
            average_confidence = float(result_df["confidence"].mean())
            average_adi = float(result_df["ADI"].mean())

            health_score = (
                0.30 * normalized_shannon
                + 0.20 * evenness
                + 0.20 * average_adi * 100
                + 0.30 * average_confidence
            )
            health_score = float(np.clip(health_score, 0, 100))

            if health_score >= 80:
                health_status = "Healthy"
            elif health_score >= 60:
                health_status = "Moderately Healthy"
            elif health_score >= 40:
                health_status = "Vulnerable"
            else:
                health_status = "Degraded"

            st.session_state["latest_survey"] = {
                "results": result_df,
                "species_counts": counts.to_dict(),
                "total": total,
                "richness": richness,
                "total_classes": total_classes,
                "shannon": shannon,
                "normalized_shannon": normalized_shannon,
                "evenness": evenness,
                "average_confidence": average_confidence,
                "average_adi": average_adi,
                "health_score": health_score,
                "health_status": health_status
            }

    survey = st.session_state.get("latest_survey")
    if survey:
        st.markdown("### Survey results")
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Processed recordings", survey["total"])
        x2.metric("Species richness", f'{survey["richness"]}/{survey["total_classes"]}')
        x3.metric("Average confidence", f'{survey["average_confidence"]:.2f}%')
        x4.metric("Proposed FHI", f'{survey["health_score"]:.2f}/100')

        st.subheader(f'Prototype status: {survey["health_status"]}')
        st.progress(survey["health_score"] / 100)

        left, right = st.columns(2)
        with left:
            st.markdown("**Species distribution**")
            st.bar_chart(pd.Series(survey["species_counts"], name="Recordings"))
        with right:
            st.markdown("**FHI component values**")
            components = pd.Series({
                "Normalized Shannon": survey["normalized_shannon"],
                "Evenness": survey["evenness"],
                "Spectral diversity": survey["average_adi"] * 100,
                "Classifier confidence": survey["average_confidence"]
            })
            st.bar_chart(components)

        st.markdown("### Recording-level predictions")
        st.dataframe(survey["results"], use_container_width=True)

        report = {
            "classifier": "MFCC + RBF-SVM",
            "test_accuracy_percent": 70.83,
            "five_fold_cv_accuracy_percent": 74.57,
            "five_fold_cv_standard_deviation": 4.60,
            "recordings": survey["total"],
            "species_richness": survey["richness"],
            "total_model_classes": survey["total_classes"],
            "shannon_diversity": round(survey["shannon"], 4),
            "normalized_shannon_percent": round(survey["normalized_shannon"], 2),
            "species_evenness_percent": round(survey["evenness"], 2),
            "average_spectral_diversity": round(survey["average_adi"], 4),
            "average_classifier_confidence_percent": round(survey["average_confidence"], 2),
            "proposed_fhi": round(survey["health_score"], 2),
            "prototype_status": survey["health_status"],
            "species_distribution": survey["species_counts"],
            "recording_results": survey["results"].to_dict(orient="records")
        }
        st.download_button(
            "Download survey report (JSON)",
            data=json.dumps(report, indent=4),
            file_name="forest_health_survey_report.json",
            mime="application/json"
        )

        st.caption(
            "The FHI is a dataset-level prototype indicator. It should not be "
            "interpreted as a validated measurement of actual forest condition."
        )


# ============================================================
# SPECIES AND DIVERSITY
# ============================================================

elif page == "Species & Diversity":
    st.subheader("Species and Acoustic Diversity")
    survey = st.session_state.get("latest_survey")

    if survey:
        st.markdown("#### Latest uploaded survey")
        st.bar_chart(pd.Series(survey["species_counts"], name="Recordings"))
        d1, d2, d3 = st.columns(3)
        d1.metric("Shannon diversity", f'{survey["shannon"]:.3f}')
        d2.metric("Normalized Shannon", f'{survey["normalized_shannon"]:.2f}%')
        d3.metric("Species evenness", f'{survey["evenness"]:.2f}%')
    else:
        st.info("Run a multi-recording survey to view its species and diversity results.")

    st.markdown("#### Dataset composition")
    if not bird_meta.empty and "Common_Name" in bird_meta.columns:
        bird_counts = bird_meta["Common_Name"].value_counts()
        st.bar_chart(bird_counts)
        st.caption("Bird metadata counts are metadata records, not model predictions.")
    else:
        st.warning("Bird species metadata is unavailable.")


# ============================================================
# GEOGRAPHIC MONITORING
# ============================================================

elif page == "Geographic Monitoring":
    st.subheader("Geographic Observation Monitoring")

    if bird_meta.empty or "Location" not in bird_meta.columns:
        st.warning("Bird location metadata is unavailable.")
    else:
        geo = bird_meta.copy()
        geo["Location"] = geo["Location"].astype(str).str.strip()
        geo = geo[geo["Location"].ne("") & geo["Location"].ne("nan")]

        if "Country" in geo.columns:
            countries = ["All"] + sorted(geo["Country"].dropna().astype(str).unique())
            selected_country = st.selectbox("Filter by country", countries)
            if selected_country != "All":
                geo = geo[geo["Country"].astype(str) == selected_country]

        st.metric("Metadata observations in selected view", len(geo))
        st.markdown("### Most frequently represented location descriptions")
        location_counts = geo["Location"].value_counts().head(20)
        st.bar_chart(location_counts)

        display_cols = [
            col for col in
            ["Common_Name", "Scientific_Name", "Country", "Location", "Date"]
            if col in geo.columns
        ]
        st.dataframe(geo[display_cols], use_container_width=True)

        st.markdown(
            """
            <div class="section-note">
            <b>Map data limitation:</b> The current bird metadata stores location
            descriptions, not latitude/longitude coordinates. This page therefore
            displays location records and frequency only. A geographic heatmap
            requires verified coordinates; locations must not be geocoded or
            plotted from guesses.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SEASONAL ACTIVITY
# ============================================================

elif page == "Seasonal Activity":
    st.subheader("Seasonal Species Observation Analysis")

    if bird_meta.empty or not {"Date", "Common_Name"}.issubset(bird_meta.columns):
        st.warning("Bird metadata does not contain the required date and species fields.")
    else:
        seasonal = bird_meta.copy()
        seasonal["Date"] = pd.to_datetime(seasonal["Date"], errors="coerce")
        seasonal = seasonal.dropna(subset=["Date", "Common_Name"])
        seasonal["Month"] = seasonal["Date"].dt.month
        seasonal["Month name"] = seasonal["Date"].dt.strftime("%b")

        if seasonal.empty:
            st.warning("No valid bird recording dates were found.")
        else:
            month_order = list(range(1, 13))
            month_names = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
            }
            pivot = seasonal.pivot_table(
                index="Common_Name",
                columns="Month",
                values="Date",
                aggfunc="count",
                fill_value=0
            ).reindex(columns=month_order, fill_value=0)
            pivot.columns = [month_names[m] for m in pivot.columns]

            st.markdown("### Monthly recording observations by species")
            fig, ax = plt.subplots(figsize=(12, 5))
            image = ax.imshow(pivot.values, aspect="auto", cmap="YlGn")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Recording month")
            ax.set_ylabel("Species")
            ax.set_title("Species observation frequency by month")
            fig.colorbar(image, ax=ax, label="Number of metadata records")
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Observation records")
            st.dataframe(
                seasonal[["Common_Name", "Date", "Location", "Country"]]
                if "Location" in seasonal.columns and "Country" in seasonal.columns
                else seasonal[["Common_Name", "Date"]],
                use_container_width=True
            )

            st.markdown(
                """
                <div class="section-note">
                <b>Scientific limitation:</b> This visualization shows when and
                where recordings were collected in the available dataset. It does
                not prove migration. Reliable migration inference needs repeated,
                standardized observations across multiple locations and seasons,
                with sampling effort accounted for.
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Frog recording activity")
    if not frog_meta.empty and "Date" in frog_meta.columns:
        frogs = frog_meta.copy()
        frogs["Date"] = pd.to_datetime(frogs["Date"], errors="coerce")
        st.dataframe(
            frogs[[c for c in ["Species", "Site", "Date", "Source"] if c in frogs.columns]],
            use_container_width=True
        )
        if "Site" in frogs.columns:
            st.caption(
                f'Frog metadata currently contains {frogs["Site"].nunique()} '
                "recording site(s); this is insufficient for geographic movement analysis."
            )
    else:
        st.info("Frog metadata is not available.")


elif page == "User Management":
    st.subheader("User Management")
    st.write("Create accounts and assign dashboard permissions.")

    with st.form("create_user_form"):
        username = st.text_input("New username")
        password = st.text_input("Temporary password", type="password")
        role = st.selectbox("Role", ["Researcher", "Viewer", "Admin"])
        create_submitted = st.form_submit_button("Create user")

    if create_submitted:
        if len(username.strip()) < 3:
            st.error("Username must contain at least 3 characters.")
        elif len(password) < 10:
            st.error("Password must contain at least 10 characters.")
        else:
            try:
                create_user(username, password, role)
                st.success(f"User '{username.strip()}' created with role '{role}'.")
            except sqlite3.IntegrityError:
                st.error("That username already exists.")

    st.markdown("### Registered users")
    with get_db_connection() as conn:
        users = conn.execute(
            "SELECT username, role, created_at FROM users ORDER BY created_at"
        ).fetchall()
    if users:
        st.dataframe(
            pd.DataFrame([dict(row) for row in users]),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Forest Health Monitoring System | MFCC + RBF-SVM | "
    "Project-specific spectral diversity analysis"
)
