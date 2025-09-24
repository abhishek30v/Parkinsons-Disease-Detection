import random
import PIL
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np

# Page Configuration ->
st.set_page_config(
    page_title="Parkinson's Spiral Test",
    page_icon="🌀",
    layout="centered"
)

# --- Constants ---
IMG_HEIGHT = 256
IMG_WIDTH = 256

# --- Model Loading ---
@st.cache_resource
def load_keras_model():
    """Loads the pre-trained Parkinson's detection model."""
    try:
        model = tf.keras.models.load_model('parkinsons_detection_model.h5') 
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_keras_model()

# --- Image Preprocessing Function because this preprocessing is used in training the model->
def preprocess_image(input_image, augment=False, num_augmented=3):
    """
    Preprocess uploaded image similar to dataset preprocessing.
    
    Preprocessing:
        - Auto-orient (fix EXIF rotation)
        - Resize to 640x640 (stretch)
        - Auto-adjust contrast (adaptive equalization approx)
        - Grayscale
    
    Augmentations (if augment=True):
        - Horizontal flip
        - Random 90° rotations
        - Random crop/zoom (0–20%)
        - Random rotation (-15° to +15°)
        - Shear ±10° (approx using affine transform)
        - Grayscale (15% of images)
        - Random hue/saturation/brightness/exposure changes
        - Gaussian blur (0–2.5 px)
        - Random noise (0–0.1%)
    
    Args:
        input_image: PIL.Image or path
        augment: bool → whether to create augmented versions
        num_augmented: number of augmented images to return
    
    Returns:
        If augment=False → single np.array (1,256,256,1)
        If augment=True  → list of np.arrays (each shaped (1,256,256,1))
    """

    # Load image
    if isinstance(input_image, str):
        img = Image.open(input_image)
    else:
        img = input_image

    # --- Preprocessing ---
    img = ImageOps.exif_transpose(img)              # Auto-orient
    img = img.resize((640, 640))                    # Stretch resize
    img = ImageOps.autocontrast(img)                # Auto-contrast
    img = img.convert("L")                          # Grayscale

    def to_array(im):
        im = im.resize((IMG_WIDTH, IMG_HEIGHT))     # Resize to model input
        arr = np.array(im).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)           # batch dim
        arr = np.expand_dims(arr, axis=-1)          # channel dim
        return arr

    if not augment:
        return to_array(img)

    # --- Augmentation ---
    augmented_images = []
    for _ in range(num_augmented):
        aug = img.copy()

        # Flip horizontal
        if random.random() < 0.5:
            aug = aug.transpose(Image.FLIP_LEFT_RIGHT)

        # Random 90° rotations
        k = random.choice([0, 1, 2, 3])
        aug = aug.rotate(90 * k)

        # Random rotation (-15 to +15)
        angle = random.uniform(-15, 15)
        aug = aug.rotate(angle)

        # Random crop/zoom
        zoom = random.uniform(0, 0.2)
        crop_size = int(640 * (1 - zoom))
        if crop_size < 640:
            left = random.randint(0, 640 - crop_size)
            top = random.randint(0, 640 - crop_size)
            aug = aug.crop((left, top, left + crop_size, top + crop_size))
            aug = aug.resize((640, 640))

        # Shear approx (affine transform not directly in PIL, skip or simulate by rotate+skew if needed)

        # Random grayscale (15% chance)
        if random.random() < 0.15:
            aug = aug.convert("L")

        # Random color adjustments
        if random.random() < 0.5:
            aug = ImageEnhance.Color(aug).enhance(random.uniform(0.85, 1.15))   # saturation
        if random.random() < 0.5:
            aug = ImageEnhance.Brightness(aug).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.5:
            aug = ImageEnhance.Contrast(aug).enhance(random.uniform(0.9, 1.1))

        # Random blur
        if random.random() < 0.5:
            aug = aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2.5)))

        # Random noise
        arr = np.array(aug).astype(np.float32)
        if random.random() < 0.5:
            noise = np.random.normal(0, 10, arr.shape)   
            arr = np.clip(arr + noise, 0, 255)
            aug = Image.fromarray(arr.astype(np.uint8))

        augmented_images.append(to_array(aug))

    return augmented_images

def array_to_image(img_array):
    arr = np.squeeze(img_array)  # (256,256)
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    return img

# --- Streamlit App UI ---
st.title("Parkinson's Disease Detection via Spiral Drawings") 

st.markdown("""
This application uses a Convolutional Neural Network (CNN) to analyze a hand-drawn spiral and predict the likelihood of Parkinson's disease.

**How to Use:**
1. Upload an image of a spiral drawing (`.png`, `.jpg`, or `.jpeg`).
2. The model will analyze the drawing and display its prediction.

**Disclaimer:** This tool is for demonstration purposes only and is not a substitute for a professional medical diagnosis.
""")

# File uploader
# File uploader
temp_image = st.file_uploader("Upload a spiral drawing...", type=["png", "jpg", "jpeg"])

if temp_image is not None and model is not None:
    # Convert to PIL image first
    image = Image.open(temp_image)
    st.image(image, caption='Uploaded Drawing', use_container_width=True)

    # Predict when button is clicked
    if st.button('Analyze Drawing'):
        with st.spinner('The model is analyzing the image...'):
            processed_image = preprocess_image(image) 
            st.image(array_to_image(processed_image))
            prediction = model.predict(processed_image) 
            confidence = prediction[0][0]

        st.subheader("Prediction Result")
        
        if confidence > 0.5:
            st.warning(f"**Prediction: Parkinson's Detected** (Confidence: {confidence:.2%})") 
            st.info("The model suggests characteristics consistent with Parkinson's disease. Please consult a healthcare professional for an accurate diagnosis.") 
        else:
            st.success(f"**Prediction: Healthy** (Confidence: {(1 - confidence):.2%})") 
            st.info("The model suggests characteristics consistent with a healthy drawing.") 

elif model is None:
    st.error("The model file could not be loaded. Please ensure 'parkinsons_detection_model.h5' is in the same directory as this script.")
