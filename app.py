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
import re

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

# Cache for image URLs
image_cache = {}


def get_flower_image_url(flower_name):
    """Get the GitHub URL for a flower image with multiple attempts"""
    
    # Check cache first
    if flower_name in image_cache:
        return image_cache[flower_name]
    
    # Clean the flower name for URL
    encoded_name = flower_name.replace(' ', '%20')
    
    # Try different methods to find the image
    urls_to_try = []
    
    # Method 1: Try common patterns with GitHub raw URL
    for img_name in ['image_00001.jpg', 'image_00001.jpeg', '1.jpg', 'image.jpg', f'{flower_name}.jpg']:
        urls_to_try.append(f"https://raw.githubusercontent.com/shukdevtroy/Oxford-102-Flowers-Classifier/main/flowers/{encoded_name}/{img_name}")
    
    # Method 2: Try using the GitHub API to list files in the folder
    api_url = f"https://api.github.com/repos/shukdevtroy/Oxford-102-Flowers-Classifier/contents/flowers/{encoded_name}"
    
    try:
        print(f"Trying GitHub API for: {flower_name}")
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            contents = response.json()
            for item in contents:
                if item['name'].endswith(('.jpg', '.jpeg', '.png')):
                    url = item['download_url']
                    image_cache[flower_name] = url
                    print(f"Found image via API: {url}")
                    return url
    except Exception as e:
        print(f"GitHub API error for {flower_name}: {e}")
    
    # Method 3: Try direct URL patterns
    for url in urls_to_try:
        try:
            print(f"Trying URL: {url}")
            response = requests.head(url, timeout=3)
            if response.status_code == 200:
                image_cache[flower_name] = url
                print(f"Found image: {url}")
                return url
        except Exception as e:
            print(f"Error checking URL {url}: {e}")
            continue
    
    # Method 4: Try using the GitHub blob URL format
    github_blob_url = f"https://github.com/shukdevtroy/Oxford-102-Flowers-Classifier/blob/main/flowers/{encoded_name}/image_00001.jpg?raw=true"
    try:
        response = requests.head(github_blob_url, timeout=3)
        if response.status_code == 200:
            image_cache[flower_name] = github_blob_url
            return github_blob_url
    except:
        pass
    
    # Method 5: Try searching the repo for the image
    search_url = f"https://api.github.com/search/code?q=repo:shukdevtroy/Oxford-102-Flowers-Classifier+path:flowers/{encoded_name}+extension:jpg"
    try:
        response = requests.get(search_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('items'):
                # Get the first item's download URL
                download_url = data['items'][0].get('download_url')
                if download_url:
                    image_cache[flower_name] = download_url
                    print(f"Found image via search: {download_url}")
                    return download_url
    except Exception as e:
        print(f"Search error: {e}")
    
    # If nothing found, cache None
    image_cache[flower_name] = None
    print(f"❌ No image found for: {flower_name}")
    return None


def get_flower_image(flower_name):
    """Download and return the flower image"""
    url = get_flower_image_url(flower_name)
    if not url:
        print(f"No URL found for: {flower_name}")
        return None
    
    try:
        print(f"Downloading image from: {url}")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            print(f"✅ Successfully downloaded image for: {flower_name}")
            return img
        else:
            print(f"❌ Failed to download: {response.status_code}")
    except Exception as e:
        print(f"❌ Error downloading image for {flower_name}: {e}")
    
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
    print(f"\n🔄 Showing image for: {flower_name}")
    
    # Try to get the image
    img = get_flower_image(flower_name)
    
    if img:
        return img, f"✅ {flower_name.title()}"
    else:
        # Create a placeholder with error message
        placeholder = Image.new('RGB', (500, 500), color='#f0f0f0')
        # Add text to placeholder (using PIL)
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(placeholder)
        try:
            # Try to use a default font
            font = ImageFont.load_default()
            text = f"Image not found:\n{flower_name}"
            # Center the text
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (500 - text_width) // 2
            y = (500 - text_height) // 2
            draw.text((x, y), text, fill='#666666', font=font)
        except:
            pass
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
                with gr.Accordion("🌺 Flower List (102 species)", open=True):
                    flower_buttons = []
                    current_letter = ''
                    
                    # Create buttons in a grid layout
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
                gr.Markdown("### Selected Flower")
                flower_image = gr.Image(label="", height=450, interactive=False)
                flower_name_display = gr.Textbox(label="Flower Name", interactive=False, value="Click a flower above to view")
        
        # Connect each button to the display function
        for btn, flower_name in flower_buttons:
            btn.click(
                fn=show_flower_image,
                inputs=gr.State(flower_name),
                outputs=[flower_image, flower_name_display]
            )
        
        # Add a note about the dataset
        gr.Markdown("""
        ---
        📚 **Dataset**: Oxford 102 Flowers dataset contains 102 flower categories commonly found in the United Kingdom.
        
        🔍 **Troubleshooting**: If you see "Image not found", the image URL might need to be updated. 
        Check that the images exist at: `flowers/flower_name/image_00001.jpg` in the GitHub repository.
        """)

# Launch
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
