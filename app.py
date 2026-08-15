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
import requests
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
from io import BytesIO

# ---------------------------------------------------------------------
# Config — must match the training notebook
# ---------------------------------------------------------------------
IMG_SIZE = 227
RESCALE = 1.0 / 255.0
TOP_K = 5

MODEL_PATH = "alexnet_flowers102.keras"
CLASS_NAMES_PATH = "class_names.json"
CAT_TO_NAME_PATH = "cat_to_name.json"

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
    class_names = json.load(f)

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

# Cache for flower image URLs
image_url_cache = {}


def get_flower_image_url(flower_name):
    """Get the GitHub URL for a flower image"""
    # Check cache first
    if flower_name in image_url_cache:
        return image_url_cache[flower_name]
    
    # Get the folder ID for this flower
    folder_id = name_to_id.get(flower_name)
    if not folder_id:
        image_url_cache[flower_name] = None
        return None
    
    # Construct the URL - assuming the image is named image_*.jpg
    encoded_name = flower_name.replace(' ', '%20')
    
    # Try common patterns for the image
    possible_names = [
        'image_00001.jpg',
        'image_00001.jpeg',
        '1.jpg',
        'image.jpg',
        f'{folder_id}.jpg'
    ]
    
    for img_name in possible_names:
        github_url = f"https://raw.githubusercontent.com/shukdevtroy/Oxford-102-Flowers-Classifier/main/flowers/{encoded_name}/{img_name}"
        try:
            response = requests.head(github_url, timeout=3)
            if response.status_code == 200:
                image_url_cache[flower_name] = github_url
                return github_url
        except:
            continue
    
    # If we can't find the image, cache the failure
    image_url_cache[flower_name] = None
    return None


def get_flower_image(flower_name):
    """Download and return the flower image as PIL Image"""
    url = get_flower_image_url(flower_name)
    if not url:
        return None
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
    except Exception as e:
        print(f"Error downloading image for {flower_name}: {e}")
    
    return None


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
# Function to display flower image in gallery
# ---------------------------------------------------------------------
def show_flower_in_gallery(flower_name, gallery_state):
    """Update the gallery with the selected flower image"""
    img = get_flower_image(flower_name)
    if img:
        return gr.update(value=[(img, flower_name)], visible=True), flower_name
    else:
        # Create a placeholder image with text
        placeholder = Image.new('RGB', (500, 500), color='#f0f0f0')
        return gr.update(value=[(placeholder, f"❌ Image not found: {flower_name}")], visible=True), flower_name


# ---------------------------------------------------------------------
# Create the accordion HTML with JavaScript
# ---------------------------------------------------------------------
def create_accordion_html():
    """Create the complete accordion HTML with JavaScript using Gradio events"""
    
    # Build the accordion items HTML
    accordion_items = ""
    current_letter = ''
    
    for flower_name in sorted_flower_names:
        # Get first letter
        first_letter = flower_name[0].upper()
        
        # Add letter divider
        if first_letter != current_letter:
            if current_letter != '':
                accordion_items += '</div>'
            current_letter = first_letter
            accordion_items += f'<div style="margin: 20px 0 10px 0; padding: 5px 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 5px; font-weight: bold; font-size: 18px;">{current_letter}</div>'
        
        # Display the flower with a data attribute for the flower name
        display_name = flower_name.title()
        accordion_items += f"""
        <div class="accordion-item" data-flower="{flower_name}">
            <div class="accordion-header">
                <span class="flower-name">🌺 {display_name}</span>
                <button class="view-link" onclick="handleViewImage('{flower_name}')">
                    👁️ View Image
                </button>
            </div>
        </div>
        """
    
    return f"""
    <style>
    .accordion-container {{
        max-width: 800px;
        margin: 0 auto;
        font-family: Arial, sans-serif;
        padding: 10px;
    }}
    .accordion-item {{
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 5px;
        overflow: hidden;
        transition: all 0.3s;
    }}
    .accordion-item:hover {{
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    .accordion-header {{
        background-color: #f5f5f5;
        padding: 12px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: background-color 0.3s;
        min-height: 50px;
    }}
    .accordion-header:hover {{
        background-color: #e8e8e8;
    }}
    .accordion-header .flower-name {{
        font-weight: 500;
        color: #2c3e50;
        flex-grow: 1;
        font-size: 16px;
    }}
    .accordion-header .view-link {{
        color: white;
        text-decoration: none;
        font-size: 14px;
        padding: 6px 16px;
        border: none;
        border-radius: 20px;
        transition: all 0.3s;
        margin-left: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        cursor: pointer;
        font-weight: 500;
        white-space: nowrap;
    }}
    .accordion-header .view-link:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }}
    .popup-overlay {{
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
        animation: fadeIn 0.3s;
    }}
    .popup-content {{
        background: white;
        border-radius: 15px;
        max-width: 600px;
        max-height: 90vh;
        overflow: auto;
        padding: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        position: relative;
        cursor: default;
        animation: slideIn 0.3s;
        margin: 20px;
    }}
    .popup-close {{
        position: absolute;
        top: 10px;
        right: 20px;
        font-size: 30px;
        cursor: pointer;
        color: #333;
        z-index: 10;
        transition: color 0.3s;
        background: none;
        border: none;
    }}
    .popup-close:hover {{
        color: #e74c3c;
        transform: rotate(90deg);
    }}
    .popup-loading {{
        text-align: center;
        padding: 40px;
        color: #666;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    @keyframes slideIn {{
        from {{ transform: translateY(-50px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}
    @media (max-width: 600px) {{
        .accordion-header {{
            flex-direction: column;
            align-items: stretch;
            gap: 10px;
        }}
        .accordion-header .view-link {{
            text-align: center;
        }}
        .popup-content {{
            margin: 10px;
            padding: 15px;
        }}
    }}
    </style>
    
    <div id="popup-overlay" class="popup-overlay" onclick="closePopup(event)">
        <div id="popup-content" class="popup-content" onclick="event.stopPropagation()">
            <button class="popup-close" onclick="closePopup(event)">&times;</button>
            <div id="popup-body">
                <div class="popup-loading">🌺 Loading flower image...</div>
            </div>
        </div>
    </div>
    
    <div class="accordion-container">
        {accordion_items}
    </div>
    
    <script>
    // Function to show popup with flower image
    function handleViewImage(flowerName) {{
        const popup = document.getElementById('popup-overlay');
        const body = document.getElementById('popup-body');
        
        // Show loading
        body.innerHTML = '<div class="popup-loading">🌺 Loading flower image...</div>';
        popup.style.display = 'flex';
        
        // Get the Gradio app
        const gradioApp = document.querySelector('gradio-app');
        if (!gradioApp) {{
            body.innerHTML = `
                <div style="text-align: center; padding: 20px;">
                    <p style="color: #e74c3c;">Error: Could not find Gradio app.</p>
                    <button onclick="closePopup(event)" style="margin-top: 10px; padding: 8px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer;">Close</button>
                </div>
            `;
            return;
        }}
        
        // Use Gradio's internal API to call the function
        // Find the function by its name in the components
        const app = gradioApp.__gradio_root__;
        if (!app) {{
            body.innerHTML = `
                <div style="text-align: center; padding: 20px;">
                    <p style="color: #e74c3c;">Error: Gradio app not initialized.</p>
                    <button onclick="closePopup(event)" style="margin-top: 10px; padding: 8px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer;">Close</button>
                </div>
            `;
            return;
        }}
        
        // Use the fetch API to call the Gradio function
        // This is a workaround to trigger the Gradio function
        try {{
            // Find the hidden gallery button or use a different approach
            const galleryInput = document.querySelector('#flower-gallery-input');
            if (galleryInput) {{
                // Set the value and trigger change
                galleryInput.value = flowerName;
                galleryInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }}
        }} catch (error) {{
            console.error('Error:', error);
        }}
    }}
    
    function closePopup(event) {{
        if (event) {{
            event.stopPropagation();
        }}
        const popup = document.getElementById('popup-overlay');
        if (popup) {{
            popup.style.display = 'none';
        }}
    }}
    
    // Close popup when pressing Escape key
    document.addEventListener('keydown', function(e) {{
        if (e.key === 'Escape') {{
            closePopup(e);
        }}
    }});
    
    // Function to update popup with image from Gradio
    window.updatePopupWithImage = function(flowerName, imageData) {{
        const body = document.getElementById('popup-body');
        if (imageData) {{
            body.innerHTML = `
                <div style="text-align: center; padding: 20px;">
                    <h3>🌺 ${{flowerName}}</h3>
                    <img src="${{imageData}}" 
                         style="max-width: 100%; max-height: 500px; border-radius: 10px; 
                                box-shadow: 0 4px 8px rgba(0,0,0,0.2); margin: 10px 0;" 
                         alt="${{flowerName}}"/>
                    <p style="color: #666; margin-top: 10px;">Click outside or press ESC to close</p>
                </div>
            `;
        }} else {{
            body.innerHTML = `
                <div style="text-align: center; padding: 20px;">
                    <h3>🌺 ${{flowerName}}</h3>
                    <p style="color: #666;">Image not found for this flower.</p>
                    <button onclick="closePopup(event)" style="margin-top: 10px; padding: 8px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer;">Close</button>
                </div>
            `;
        }}
    }};
    </script>
    """


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
        Click the **"View Image"** button next to any flower name to see its image in a popup.
        """)
        
        # Create a hidden textbox for the flower name selection
        selected_flower = gr.Textbox(visible=False, elem_id="flower-gallery-input")
        
        # Create a gallery to display the flower image
        flower_gallery = gr.Gallery(
            label="Flower Image",
            show_label=True,
            elem_id="flower-gallery",
            columns=1,
            rows=1,
            height=500,
            visible=False,
            object_fit="contain"
        )
        
        # Create and display the accordion
        accordion_html = create_accordion_html()
        gr.HTML(accordion_html)
        
        # Connect the selected flower to the gallery
        selected_flower.change(
            fn=show_flower_in_gallery,
            inputs=[selected_flower, gr.State(None)],
            outputs=[flower_gallery, selected_flower]
        )
        
        gr.Markdown("""
        ### 📚 Dataset Information
        This app uses the Oxford 102 Flowers dataset, which contains 102 flower categories 
        commonly found in the United Kingdom. The model was trained to classify these flowers 
        with high accuracy.
        """)

# Launch the app
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
