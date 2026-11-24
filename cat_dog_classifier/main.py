"""Simple Streamlit application for Cat vs Dog image prediction."""

import streamlit as st

import prediction


st.set_page_config(page_title="Cat vs Dog Image Classification", layout="wide")


@st.cache_resource
def get_model():
    """Load the saved model once for this Streamlit process."""
    return prediction.load_model()


st.title("Cat vs Dog Image Classification")
st.write(
    "Upload a cat or dog image. The application will predict its class."
)

try:
    model = get_model()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    model = None

if model is not None:
    st.success("Model status: Ready")
else:
    st.warning("Model status: Not available")

st.header("Image Prediction")
uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
)

if uploaded_image is not None:
    image_column, prediction_column = st.columns([3, 2], gap="large")

    with image_column:
        st.image(uploaded_image, caption="Uploaded image", width=420)

    with prediction_column:
        with st.container(border=True):
            st.markdown("### Prediction box")
            predict_button = st.button("Predict")
            if predict_button and model is not None:
                try:
                    image_array = prediction.preprocess_image(uploaded_image, model)
                    label = prediction.predict(model, image_array)
                except (ValueError, OSError) as error:
                    st.error(str(error))
                else:
                    st.success(f"Prediction: **{label}**")
            else:
                st.caption("Click Predict to classify the uploaded image.")
