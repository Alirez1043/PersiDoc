import streamlit as st
from PIL import Image, ImageOps
import io

from static.styles import CUSTOM_CSS
from utils.image_processing import perform_ocr, adjust_orientation, adjust_skew

# Streamlit App UI
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("Image Orientation and OCR App")


# Step 1: Upload image and configuration
if "step" not in st.session_state:
    st.session_state.step = 1

if st.session_state.step == 1:
    col1, col2 = st.columns([2, 1])

    # Left Column: Image Upload
    with col1:
        uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            # Store original image data for processing
            st.session_state.image = uploaded_image.read()
            
            # Create a separate PIL image only for display
            display_image = Image.open(io.BytesIO(st.session_state.image))
            
            # Correct orientation using EXIF metadata (if available)
            display_image = ImageOps.exif_transpose(display_image)

            # Resize large images for display (optional)
            max_display_width = 800
            if display_image.width > max_display_width:
                scaling_factor = max_display_width / display_image.width
                new_width = max_display_width
                new_height = int(display_image.height * scaling_factor)
                display_image = display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Display the corrected image
            st.image(display_image, caption="Uploaded Image", use_container_width=True)

    # Right Column: Select Buttons
    with col2:
        orientation_model = st.selectbox(
            "Select Orientation Model:", ["Not Use", "ResNET"], index=0
        )
        skew_method = st.selectbox(
            "Select Skew Method:", ["Not Use", "Furier", "HoughBin", "HistScoreSkew", "IQRHLines"], index=0
        )

    # Process Button
    if st.button("Process"):
        if uploaded_image:
            st.session_state.step = 2
            st.session_state.orientation_model = orientation_model
            st.session_state.skew_method = skew_method
        else:
            st.warning("Please upload an image first.")


# Step 2: Display results
if st.session_state.step == 2:
    st.subheader("Processing Results")

    # Load uploaded image
    original_image = Image.open(io.BytesIO(st.session_state.image))

    # Apply orientation adjustment if selected
    if st.session_state.orientation_model != "Not Use":
        oriented_image, _, orientation_info = adjust_orientation(
            original_image, st.session_state.orientation_model
        )
        
        st.info(f"Orientation: {orientation_info}")
    else:
        oriented_image = original_image

    # Apply skew correction if selected
    if st.session_state.skew_method != "Not Use":
        final_image, ocr_result, skew_degree, inf_time, skew_info = adjust_skew(
            oriented_image, st.session_state.skew_method
        )
        st.info(f"Skew Correction: {skew_info}")
        st.info(f"Processing Time: {inf_time:.2f} seconds")
    else:
        final_image, ocr_result = oriented_image, perform_ocr(oriented_image)

    # Use tabs for Images and OCR Results
    tab1, tab2 = st.tabs(["Images", "OCR Results"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.image(oriented_image, caption="Original Image", use_container_width=True, output_format="auto", clamp=True)
        with col2:
            st.image(final_image, caption="Processed Image", use_container_width=True, output_format="auto", clamp=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.text_area("Original Image OCR Result:", perform_ocr(oriented_image), height=600)
        with col2:
            st.text_area("Processed Image OCR Result:", ocr_result, height=600)

    # Back Button to Step 1
    if st.button("Back to Upload"):
        st.session_state.step = 1
        st.rerun()   # Forces the app to rerun and reflect the changes immediately
