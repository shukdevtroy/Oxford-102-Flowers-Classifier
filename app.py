"""
Oxford 102 Flowers classifier — Gradio app with flower gallery
"""

import json
import os
import requests
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
from io import BytesIO

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
IMG_SIZE = 227
RESCALE = 1.0 / 255.0
TOP_K = 5

MODEL_PATH = "alexnet_flowers102.keras"
CLASS_NAMES_PATH = "class_names.json"
CAT_TO_NAME_PATH = "cat_to_name.json"
GDRIVE_MODEL_FILE_ID = os.environ.get("GDRIVE_MODEL_FILE_ID", "")

# ---------------------------------------------------------------------
# Download model if needed
# ---------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    if not GDRIVE_MODEL_FILE_ID:
        raise RuntimeError("Model not found and GDRIVE_MODEL_FILE_ID not set.")
    print(f"Downloading model...")
    gdown.download(id=GDRIVE_MODEL_FILE_ID, output=MODEL_PATH, quiet=False)

# ---------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)

if os.path.exists(CAT_TO_NAME_PATH):
    with open(CAT_TO_NAME_PATH) as f:
        cat_to_name = json.load(f)
else:
    cat_to_name = {}

name_to_id = {v: k for k, v in cat_to_name.items()}
sorted_flower_names = sorted(cat_to_name.values())


def get_flower_image_url(flower_name):
    """Get the GitHub URL for a flower image"""
    encoded_name = flower_name.replace(' ', '%20')
    
    # Try different image name patterns
    for img_name in ['image_00001.jpg', 'image_00001.jpeg', '1.jpg', 'image.jpg']:
        github_url = f"https://raw.githubusercontent.com/shukdevtroy/Oxford-102-Flowers-Classifier/main/flowers/{encoded_name}/{img_name}"
        try:
            response = requests.head(github_url, timeout=3)
            if response.status_code == 200:
                return github_url
        except:
            continue
    return None


def get_flower_image(flower_name):
    """Download and return the flower image"""
    url = get_flower_image_url(flower_name)
    if not url:
        return None
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except:
        pass
    return None


def label_to_flower_name(folder_label):
    return cat_to_name.get(folder_label, folder_label)


def preprocess(pil_image):
    image = np.array(pil_image.convert("RGB"))
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)


def predict(image):
    if image is None:
        return {}
    processed = preprocess(image)
    probs = model.predict(processed, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:TOP_K]
    results = {
        label_to_flower_name(class_names[i]): float(probs[i])
        for i in top_indices
    }
    return results


def show_flower_image(flower_name):
    """Display flower image when clicked"""
    img = get_flower_image(flower_name)
    if img:
        return img, f"🌺 {flower_name}"
    else:
        # Create a placeholder
        placeholder = Image.new('RGB', (500, 500), color='#f0f0f0')
        return placeholder, f"❌ Image not found: {flower_name}"


# ---------------------------------------------------------------------
# Create the UI
# ---------------------------------------------------------------------
with gr.Blocks(title="🌸 Oxford 102 Flowers Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌸 Oxford 102 Flowers Classifier")
    
    with gr.Tab("📷 Classifier"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload a flower photo")
                predict_btn = gr.Button("🔍 Identify Flower", variant="primary")
            with gr.Column():
                output_labels = gr.Label(num_top_classes=TOP_K, label="Predictions")
        predict_btn.click(fn=predict, inputs=input_image, outputs=output_labels)
    
    with gr.Tab("🌸 Flower Gallery"):
        gr.Markdown("## Click on any flower name to see its image")
        
        # Create a row with the gallery
        with gr.Row():
            with gr.Column(scale=1):
                # Create accordion with flower names
                with gr.Accordion("🌺 Flower List", open=True):
                    # Create buttons for each flower
                    flower_buttons = []
                    current_letter = ''
                    
                    for flower_name in sorted_flower_names:
                        first_letter = flower_name[0].upper()
                        if first_letter != current_letter:
                            current_letter = first_letter
                            gr.Markdown(f"### {current_letter}")
                        
                        btn = gr.Button(
                            f"🌺 {flower_name.title()}",
                            variant="secondary",
                            size="sm"
                        )
                        flower_buttons.append((btn, flower_name))
            
            with gr.Column(scale=2):
                # Display the selected flower image
                flower_image = gr.Image(label="Selected Flower", height=400)
                flower_name_display = gr.Textbox(label="Flower Name", interactive=False)
        
        # Connect each button to the display function
        for btn, flower_name in flower_buttons:
            btn.click(
                fn=show_flower_image,
                inputs=gr.State(flower_name),
                outputs=[flower_image, flower_name_display]
            )

# Launch
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
