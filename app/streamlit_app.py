"""Streamlit interface for TB chest X-ray prediction."""
from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_MODEL_ARTIFACT
from src.predict import load_artifacts, predict_from_pil_image


st.set_page_config(page_title="TB Classifier", page_icon="🩺", layout="centered")
st.title("Tuberculosis Chest X-ray Classifier")
st.write(
    "This app predicts whether an uploaded chest X-ray image is more likely "
    "to be **Normal** or **Tuberculosis** using a CNN trained from the project pipeline."
)


@st.cache_resource
def get_artifact(artifact_path: str):
    return load_artifacts(artifact_path)


artifact_path = st.text_input("Model artifact path", value=str(DEFAULT_MODEL_ARTIFACT))

if not Path(artifact_path).exists():
    st.warning("Model artifact not found. Train the model first with `python -m src.train`.")
else:
    artifact = get_artifact(artifact_path)

    uploaded_file = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Predict", type="primary"):
            result = predict_from_pil_image(image=image, artifact=artifact)
            st.subheader("Prediction")
            st.success(f"Class: {result['predicted_class']}")
            st.write(f"Confidence: `{result['confidence']:.2%}`")

            st.subheader("Class Probabilities")
            for class_name, prob in result["probabilities"].items():
                st.write(f"- {class_name}: {prob:.2%}")

st.caption("For research and educational purposes only. Not for clinical diagnosis.")
