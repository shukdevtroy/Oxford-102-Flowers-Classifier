"""
Oxford 102 Flowers classifier — Gradio app for Hugging Face Spaces with flower accordion.

Expects these files to sit next to this script:
    - alexnet_flowers102.keras   (from MODEL_PATH, saved in Cell 15)
    - class_names.json           (from CLASS_NAMES_PATH, saved in Cell 15)
    - cat_to_name.json           (the Kaggle-provided id -> flower-name file)

The app includes:
1. A classifier that predicts flower species from uploaded images
2. An accordion with all 102 flower names that shows images in a popup
"""

import json
import os
import base64
from io import BytesIO
import requests

import gdown
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------
# Config — must match the training notebook
# ---------------------------------------------------------------------
IMG_SIZE = 227
RESCALE = 1.0 / 255.0
TOP_K = 5

MODEL_PATH = "alexnet_flowers102.keras"
CLASS_NAMES_PATH = "class_names.json"
CAT_TO_NAME_PATH = "cat_to_name.json"

# GitHub repository base URL for flower images
GITHUB_REPO_BASE = "https://github.com/shukdevtroy/Oxford-102-Flowers-Classifier/blob/main/flowers"

# Google Drive file ID for the model
GDRIVE_MODEL_FILE_ID = os.environ.get("GDRIVE_MODEL_FILE_ID", "")

# ---------------------------------------------------------------------
# Download the model from Google Drive if it isn't already present
# ---------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    if not GDRIVE_MODEL_FILE_ID:
        raise RuntimeError(
            "MODEL_PATH not found locally and GDRIVE_MODEL_FILE_ID is not set. "
            "Either commit the .keras file to the repo, or set the "
            "GDRIVE_MODEL_FILE_ID environment variable."
        )
    print(f"Downloading model from Google Drive (file id: {GDRIVE_MODEL_FILE_ID})...")
    gdown.download(id=GDRIVE_MODEL_FILE_ID, output=MODEL_PATH, quiet=False)

# ---------------------------------------------------------------------
# Load artifacts once, at startup
# ---------------------------------------------------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)  # sorted folder-label strings, e.g. ['1','10','100',...]

if os.path.exists(CAT_TO_NAME_PATH):
    with open(CAT_TO_NAME_PATH) as f:
        cat_to_name = json.load(f)
else:
    cat_to_name = {}
    print(f"Warning: {CAT_TO_NAME_PATH} not found — showing raw class ids instead of names.")

# Create reverse mapping for flower name -> folder ID
name_to_id = {v: k for k, v in cat_to_name.items()}

# Sort flower names alphabetically
sorted_flower_names = sorted(cat_to_name.values())


def get_flower_image_url(flower_name):
    """Get the GitHub URL for a flower image"""
    # Get the folder ID for this flower
    folder_id = name_to_id.get(flower_name)
    if not folder_id:
        return None
    
    # Construct the URL - assuming the image is named image_*.jpg
    # We need to find the actual image filename in the folder
    # Since each folder has only one image, we'll use a pattern
    encoded_name = flower_name.replace(' ', '%20')
    
    # GitHub doesn't support wildcards, so we'll try a common pattern
    # Let's try to fetch the folder contents via GitHub API
    api_url = f"https://api.github.com/repos/shukdevtroy/Oxford-102-Flowers-Classifier/contents/flowers/{encoded_name}"
    
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            contents = response.json()
            # Find the first image file
            for item in contents:
                if item['name'].endswith('.jpg') or item['name'].endswith('.jpeg'):
                    return item['download_url']
    except:
        pass
    
    # Fallback: try common patterns
    patterns = [
        f"{GITHUB_REPO_BASE}/{encoded_name}/image_00001.jpg",
        f"{GITHUB_REPO_BASE}/{encoded_name}/image_00001.jpg?raw=true",
    ]
    
    # Try each pattern
    for url in patterns:
        try:
            response = requests.head(url)
            if response.status_code == 200:
                return url
        except:
            continue
    
    # If we can't find the specific image, try a direct GitHub blob URL
    for ext in ['.jpg', '.jpeg']:
        github_url = f"{GITHUB_REPO_BASE}/{encoded_name}/image_00001{ext}?raw=true"
        try:
            response = requests.head(github_url)
            if response.status_code == 200:
                return github_url
        except:
            continue
    
    # If still not found, return None
    return None


def get_flower_image_html(flower_name):
    """Get HTML to display the flower image in a popup"""
    # Get the image URL
    image_url = get_flower_image_url(flower_name)
    
    if not image_url:
        return f"""
        <div style="font-family: Arial, sans-serif; text-align: center; padding: 20px;">
            <h3>🌺 {flower_name}</h3>
            <p style="color: #666;">Image not found for this flower.</p>
            <p style="color: #999; font-size: 12px;">Please check if the image exists in the repository.</p>
        </div>
        """
    
    return f"""
    <div style="font-family: Arial, sans-serif; text-align: center; padding: 20px;">
        <h3>🌺 {flower_name}</h3>
        <img src="{image_url}" 
             style="max-width: 500px; max-height: 500px; width: auto; height: auto; 
                    border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); 
                    margin: 10px auto; display: block;" 
             alt="{flower_name}" />
        <p style="color: #666; margin-top: 10px;">Click outside to close</p>
    </div>
    """


def create_flower_accordion():
    """Create an accordion with all 102 flower names"""
    accordion_html = """
    <style>
    .accordion {
        max-width: 800px;
        margin: 0 auto;
        font-family: Arial, sans-serif;
    }
    .accordion-item {
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 5px;
        overflow: hidden;
    }
    .accordion-header {
        background-color: #f5f5f5;
        padding: 12px 20px;
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: background-color 0.3s;
    }
    .accordion-header:hover {
        background-color: #e8e8e8;
    }
    .accordion-header .flower-name {
        font-weight: 500;
        color: #2c3e50;
        flex-grow: 1;
    }
    .accordion-header .view-link {
        color: #3498db;
        text-decoration: none;
        font-size: 14px;
        padding: 5px 12px;
        border: 1px solid #3498db;
        border-radius: 20px;
        transition: all 0.3s;
        margin-left: 10px;
        background-color: white;
    }
    .accordion-header .view-link:hover {
        background-color: #3498db;
        color: white;
    }
    .popup-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.7);
        z-index: 9999;
        justify-content: center;
        align-items: center;
        cursor: pointer;
    }
    .popup-content {
        background: white;
        border-radius: 15px;
        max-width: 600px;
        max-height: 90vh;
        overflow: auto;
        padding: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        position: relative;
        cursor: default;
    }
    .popup-close {
        position: absolute;
        top: 10px;
        right: 20px;
        font-size: 30px;
        cursor: pointer;
        color: #333;
    }
    .popup-close:hover {
        color: #e74c3c;
    }
    </style>
    
    <script>
    function showPopup(flowerName) {
        console.log('Showing popup for:', flowerName);
        
        // Get the image URL
        fetch(`/get_flower_image?flower_name=${encodeURIComponent(flowerName)}`)
            .then(response => response.json())
            .then(data => {
                const popup = document.getElementById('popup-overlay');
                const content = document.getElementById('popup-content');
                
                let html = `
                    <div style="font-family: Arial, sans-serif; text-align: center;">
                        <h3>🌺 ${flowerName}</h3>
                `;
                
                if (data.url) {
                    html += `
                        <img src="${data.url}" 
                             style="max-width: 500px; max-height: 500px; width: auto; height: auto; 
                                    border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); 
                                    margin: 10px auto; display: block;" 
                             alt="${flowerName}" />
                    `;
                } else {
                    html += `
                        <p style="color: #666; margin: 20px;">Image not found for this flower.</p>
                    `;
                }
                
                html += `
                        <p style="color: #999; font-size: 12px; margin-top: 10px;">Click outside to close</p>
                    </div>
                `;
                
                content.innerHTML = html;
                popup.style.display = 'flex';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error loading image. Please try again.');
            });
    }
    
    function closePopup() {
        document.getElementById('popup-overlay').style.display = 'none';
    }
    
    // Close popup when clicking outside
    document.addEventListener('DOMContentLoaded', function() {
        const popup = document.getElementById('popup-overlay');
        if (popup) {
            popup.addEventListener('click', function(e) {
                if (e.target === this) {
                    closePopup();
                }
            });
        }
    });
    </script>
    """
    
    # Add the popup overlay
    accordion_html += """
    <div id="popup-overlay" class="popup-overlay" onclick="closePopup()">
        <div id="popup-content" class="popup-content" onclick="event.stopPropagation()">
            <span class="popup-close" onclick="closePopup()">&times;</span>
            <!-- Content will be loaded dynamically -->
        </div>
    </div>
    """
    
    # Add the accordion items
    accordion_html += '<div class="accordion">'
    
    for flower_name in sorted_flower_names:
        # Clean flower name for display
        display_name = flower_name.title()  # Capitalize each word
        accordion_html += f"""
        <div class="accordion-item">
            <div class="accordion-header" onclick="toggleAccordion(this)">
                <span class="flower-name">🌺 {display_name}</span>
                <button class="view-link" onclick="event.stopPropagation(); showPopup('{flower_name}')">
                    View Image
                </button>
            </div>
        </div>
        """
    
    accordion_html += '</div>'
    
    # Add toggle accordion function
    accordion_html += """
    <script>
    function toggleAccordion(header) {
        // Simple toggle functionality - we're just using it for visual feedback
        header.style.backgroundColor = header.style.backgroundColor === 'rgb(232, 232, 232)' ? '#f5f5f5' : '#e8e8e8';
        // You can add expand/collapse functionality here if needed
    }
    </script>
    """
    
    return accordion_html


def get_flower_image_endpoint(flower_name):
    """Endpoint to get flower image URL"""
    url = get_flower_image_url(flower_name)
    return {"url": url}


def label_to_flower_name(folder_label):
    """folder_label is the raw class-folder string, e.g. '21'."""
    return cat_to_name.get(folder_label, folder_label)


# ---------------------------------------------------------------------
# Preprocessing — identical to preprocess_single_image() in the notebook
# ---------------------------------------------------------------------
def preprocess(pil_image):
    image = np.array(pil_image.convert("RGB"))
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) * RESCALE
    return tf.expand_dims(image, axis=0)


# ---------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------
with gr.Blocks(title="🌸 Oxford 102 Flowers Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌸 Oxford 102 Flowers Classifier
    Upload a flower photo to identify it, or explore the flower gallery below!
    """)
    
    with gr.Tab("📷 Classifier"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload a flower photo")
                predict_btn = gr.Button("🔍 Identify Flower", variant="primary")
            
            with gr.Column():
                output_labels = gr.Label(num_top_classes=TOP_K, label="Predictions")
        
        gr.Markdown("""
        ### How to use:
        1. Upload a photo of a flower
        2. Click "Identify Flower" to see predictions
        3. The model will show the top 5 most likely species
        """)
        
        predict_btn.click(
            fn=predict,
            inputs=input_image,
            outputs=output_labels
        )
    
    with gr.Tab("🌸 Flower Gallery"):
        gr.Markdown("""
        ## Explore All 102 Flower Species
        Click the "View Image" button next to any flower name to see its image in a popup.
        """)
        
        # Create the accordion
        accordion_html = create_flower_accordion()
        gr.HTML(accordion_html)
        
        # Add the endpoint for fetching flower images
        gr.Markdown("""
        ### 🌺 Flower Information
        This gallery contains all 102 flower species from the Oxford 102 Flowers dataset.
        Each entry shows the flower name with a link to view its image.
        """)

# Add the get_flower_image route
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
