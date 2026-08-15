"""
🌸 Floral Vibes - Premium Oxford 102 Flowers Classifier
Fixed UI with Flower Directory scroll and About section
"""

import json
import os
import requests
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import gdown
from io import BytesIO
import base64
import re
from collections import defaultdict

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

# ---------------------------------------------------------------------
# Premium CSS Styling
# ---------------------------------------------------------------------
PREMIUM_CSS = """
<style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
        background: #faf8f5 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Premium Header */
    .premium-header {
        background: linear-gradient(135deg, #2d1b0e 0%, #4a2c1a 50%, #2d1b0e 100%);
        padding: 40px 50px 30px 50px;
        border-radius: 24px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 215, 170, 0.15);
    }
    
    .premium-header::before {
        content: '✦';
        position: absolute;
        right: 40px;
        top: 20px;
        font-size: 120px;
        opacity: 0.05;
        color: #f5d6b3;
        font-family: serif;
    }
    
    .premium-header h1 {
        font-family: 'Playfair Display', serif !important;
        font-size: 3.2rem !important;
        font-weight: 700 !important;
        color: #f5d6b3 !important;
        margin: 0 !important;
        letter-spacing: 2px !important;
        text-shadow: 0 2px 20px rgba(0,0,0,0.3);
    }
    
    .premium-header .subtitle {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 300 !important;
        color: rgba(245, 214, 179, 0.7) !important;
        margin-top: 8px !important;
        letter-spacing: 4px !important;
        text-transform: uppercase;
    }
    
    .premium-stats {
        display: flex;
        gap: 40px;
        margin-top: 20px;
        padding-top: 20px;
        border-top: 1px solid rgba(245, 214, 179, 0.1);
        flex-wrap: wrap;
    }
    
    .premium-stats .stat-item {
        text-align: center;
    }
    
    .premium-stats .stat-number {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #f5d6b3 !important;
        display: block;
    }
    
    .premium-stats .stat-label {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: rgba(245, 214, 179, 0.5);
    }
    
    /* Premium Tabs */
    .tabs {
        border: none !important;
        background: transparent !important;
    }
    
    .tab-nav {
        background: transparent !important;
        border: none !important;
        gap: 8px !important;
        padding: 0 !important;
        margin-bottom: 25px !important;
    }
    
    .tab-nav button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 12px 28px !important;
        border-radius: 50px !important;
        border: 1px solid rgba(45, 27, 14, 0.1) !important;
        background: transparent !important;
        color: #4a3520 !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px !important;
    }
    
    .tab-nav button:hover {
        background: rgba(245, 214, 179, 0.2) !important;
        border-color: rgba(45, 27, 14, 0.2) !important;
        transform: translateY(-1px);
    }
    
    .tab-nav button.selected {
        background: linear-gradient(135deg, #2d1b0e, #4a2c1a) !important;
        color: #f5d6b3 !important;
        border-color: #2d1b0e !important;
        box-shadow: 0 8px 25px rgba(45, 27, 14, 0.2) !important;
    }
    
    /* Upload Area */
    .upload-area {
        border: 2px dashed rgba(45, 27, 14, 0.15) !important;
        border-radius: 20px !important;
        padding: 40px !important;
        background: white !important;
        transition: all 0.3s ease !important;
    }
    
    .upload-area:hover {
        border-color: #4a2c1a !important;
        box-shadow: 0 10px 40px rgba(45, 27, 14, 0.06) !important;
    }
    
    .upload-area label {
        font-family: 'Playfair Display', serif !important;
        color: #2d1b0e !important;
        font-size: 1.1rem !important;
    }
    
    /* Premium Buttons */
    .premium-btn-primary {
        background: linear-gradient(135deg, #2d1b0e, #4a2c1a) !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 14px 40px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        color: #f5d6b3 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(45, 27, 14, 0.2) !important;
        letter-spacing: 0.5px !important;
    }
    
    .premium-btn-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 35px rgba(45, 27, 14, 0.3) !important;
        color: #ffffff !important;
    }
    
    .premium-btn-secondary {
        background: transparent !important;
        border: 1px solid rgba(45, 27, 14, 0.15) !important;
        border-radius: 50px !important;
        padding: 10px 24px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: #4a3520 !important;
        transition: all 0.3s ease !important;
    }
    
    .premium-btn-secondary:hover {
        background: rgba(45, 27, 14, 0.05) !important;
        border-color: #4a2c1a !important;
        transform: translateY(-1px);
    }
    
    /* Flower Gallery - Scrollable Directory */
    .gallery-layout {
        display: flex;
        gap: 24px;
        height: 600px;
        align-items: stretch;
    }
    
    .gallery-directory {
        flex: 1;
        min-width: 300px;
        max-width: 420px;
        background: white;
        border-radius: 20px;
        padding: 16px 18px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.04);
        overflow-y: auto;
        height: 100%;
    }
    
    .gallery-directory::-webkit-scrollbar {
        width: 6px;
    }
    
    .gallery-directory::-webkit-scrollbar-track {
        background: #f0ebe5;
        border-radius: 10px;
    }
    
    .gallery-directory::-webkit-scrollbar-thumb {
        background: #4a2c1a;
        border-radius: 10px;
    }
    
    .gallery-viewer {
        flex: 2;
        min-width: 350px;
        background: white;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.04);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        overflow: hidden;
    }
    
    .flower-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 4px;
        width: 100%;
    }
    
    .flower-btn {
        background: transparent !important;
        border: 1px solid rgba(45, 27, 14, 0.08) !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
        color: #4a3520 !important;
        transition: all 0.3s ease !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        cursor: pointer !important;
        min-height: 38px;
        width: 100% !important;
    }
    
    .flower-btn:hover {
        background: rgba(245, 214, 179, 0.2) !important;
        border-color: #4a2c1a !important;
        transform: translateX(4px) !important;
        box-shadow: 0 4px 15px rgba(45, 27, 14, 0.06) !important;
    }
    
    .flower-btn .emoji {
        font-size: 1rem;
        flex-shrink: 0;
    }
    
    .flower-btn .flower-name {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .flower-section-header {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #2d1b0e !important;
        margin: 12px 0 6px 0 !important;
        padding-bottom: 6px !important;
        border-bottom: 2px solid #f5d6b3 !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    
    .flower-section-header .letter-badge {
        background: linear-gradient(135deg, #2d1b0e, #4a2c1a);
        color: #f5d6b3;
        border-radius: 50%;
        width: 26px;
        height: 26px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-family: 'Playfair Display', serif;
        flex-shrink: 0;
    }
    
    .flower-section-header .count-badge {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        color: #a0806a;
        font-weight: 400;
        margin-left: auto;
    }
    
    /* Display Area */
    .display-area {
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .display-area .flower-image-container {
        width: 100%;
        max-width: 480px;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 40px rgba(0,0,0,0.06);
        margin: 0 auto;
        flex-shrink: 0;
    }
    
    .display-area .flower-image-container img {
        width: 100%;
        height: auto;
        max-height: 400px;
        object-fit: contain;
        transition: transform 0.5s ease;
        background: #faf8f5;
    }
    
    .display-area .flower-image-container img:hover {
        transform: scale(1.02);
    }
    
    .display-area .flower-name-display {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #2d1b0e !important;
        margin-top: 14px !important;
        text-align: center;
    }
    
    .display-area .flower-subtitle {
        font-size: 0.8rem !important;
        color: #a0806a !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    
    .display-area .flower-status {
        font-size: 0.75rem !important;
        color: #a0806a !important;
        margin-top: 4px;
    }
    
    /* Predictions */
    .prediction-item {
        padding: 12px 20px !important;
        border-radius: 12px !important;
        margin-bottom: 8px !important;
        background: #faf8f5 !important;
        border-left: 4px solid #4a2c1a !important;
        transition: all 0.3s ease !important;
    }
    
    .prediction-item:hover {
        background: #f5f0ea !important;
        transform: translateX(4px);
    }
    
    /* Responsive */
    @media (max-width: 900px) {
        .gallery-layout {
            flex-direction: column;
            height: auto;
        }
        .gallery-directory {
            max-height: 400px;
            min-width: unset;
            max-width: unset;
        }
        .gallery-viewer {
            min-width: unset;
            height: 450px;
        }
    }
    
    @media (max-width: 768px) {
        .premium-header {
            padding: 25px 20px !important;
        }
        .premium-header h1 {
            font-size: 2rem !important;
        }
        .premium-stats {
            gap: 20px;
            flex-wrap: wrap;
        }
        .premium-stats .stat-number {
            font-size: 1.3rem !important;
        }
        .gallery-viewer {
            height: 350px;
            padding: 16px;
        }
        .display-area .flower-image-container {
            max-width: 100%;
        }
    }
    
    /* Custom Accordion */
    .accordion {
        border: none !important;
        background: transparent !important;
    }
    
    .accordion .label-wrap {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #2d1b0e !important;
        padding: 12px 0 !important;
        border-bottom: 2px solid #f5d6b3 !important;
    }
    
    /* Badge */
    .premium-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f5d6b3, #e8c4a0);
        color: #2d1b0e;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 50px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-left: 10px;
    }
    
    /* Gallery Directory Header */
    .directory-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 2px solid #f5d6b3;
    }
    
    .directory-header .title {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        color: #2d1b0e;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .directory-header .count {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #a0806a;
        background: #f0ebe5;
        padding: 4px 12px;
        border-radius: 50px;
    }
    
    /* About section styling */
    .about-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .about-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        color: #2d1b0e;
        text-align: center;
        margin-bottom: 8px;
    }
    
    .about-subtitle {
        text-align: center;
        color: #a0806a;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-size: 0.8rem;
        margin-bottom: 30px;
    }
    
    .about-card {
        background: white;
        border-radius: 20px;
        padding: 35px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.04);
        margin-bottom: 25px;
        line-height: 1.8;
        color: #4a3520;
        font-size: 1.05rem;
    }
    
    .about-card p {
        margin-bottom: 12px;
    }
    
    .about-card strong {
        color: #2d1b0e;
    }
    
    .about-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    
    .about-grid-item {
        background: white;
        border-radius: 16px;
        padding: 25px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        border: 1px solid rgba(45,27,14,0.06);
    }
    
    .about-grid-item .icon {
        font-size: 2.4rem;
        margin-bottom: 8px;
    }
    
    .about-grid-item .value {
        font-weight: 600;
        color: #2d1b0e;
        font-size: 1.2rem;
    }
    
    .about-grid-item .label {
        font-size: 0.8rem;
        color: #a0806a;
    }
    
    .about-tech {
        background: #faf8f5;
        border-radius: 16px;
        padding: 25px;
        margin-top: 25px;
        border-left: 4px solid #4a2c1a;
    }
    
    .about-tech-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        color: #2d1b0e;
        margin-bottom: 8px;
    }
    
    .about-tech-content {
        font-size: 0.9rem;
        color: #4a3520;
        line-height: 1.7;
    }
    
    .about-footer {
        text-align: center;
        margin-top: 25px;
        color: #a0806a;
        font-size: 0.85rem;
        letter-spacing: 1px;
    }
</style>
"""

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def get_flower_image_url(flower_name):
    """Get the GitHub URL for a flower image with multiple attempts"""
    if flower_name in image_cache:
        return image_cache[flower_name]
    
    encoded_name = flower_name.replace(' ', '%20')
    urls_to_try = []
    
    for img_name in ['image_00001.jpg', 'image_00001.jpeg', '1.jpg', 'image.jpg', f'{flower_name}.jpg']:
        urls_to_try.append(f"https://raw.githubusercontent.com/shukdevtroy/Oxford-102-Flowers-Classifier/main/flowers/{encoded_name}/{img_name}")
    
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
    
    img = get_flower_image(flower_name)
    
    if img:
        return img, f"🌸 {flower_name.title()}", f"✓ Image loaded successfully"
    else:
        placeholder = Image.new('RGB', (500, 500), color='#faf8f5')
        draw = ImageDraw.Draw(placeholder)
        try:
            font = ImageFont.load_default()
            text = f"🌸 {flower_name.title()}\n\nImage not found"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (500 - text_width) // 2
            y = (500 - text_height) // 2
            draw.text((x, y), text, fill='#a0806a', font=font)
        except:
            pass
        return placeholder, f"🌺 {flower_name.title()}", f"⚠️ Image not available"


def create_premium_header():
    """Create the premium header HTML"""
    return f"""
    <div class="premium-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <h1>🌸 Floral Vibes</h1>
                <div class="subtitle">✦ Oxford 102 Flowers Classifier ✦</div>
            </div>
            <div style="text-align: right;">
                <span style="color: rgba(245,214,179,0.4); font-size: 0.8rem; letter-spacing: 3px; text-transform: uppercase;">
                    AI-Powered Flower Recognition
                </span>
            </div>
        </div>
        <div class="premium-stats">
            <div class="stat-item">
                <span class="stat-number">102</span>
                <span class="stat-label">Flower Species</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">8K+</span>
                <span class="stat-label">Training Images</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">97%</span>
                <span class="stat-label">Top-5 Accuracy</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">✦</span>
                <span class="stat-label">AlexNet Architecture</span>
            </div>
        </div>
    </div>
    """


def create_flower_grid():
    """Create the flower grid HTML with alphabetical grouping"""
    grouped = defaultdict(list)
    for flower_name in sorted_flower_names:
        first_letter = flower_name[0].upper()
        grouped[first_letter].append(flower_name)
    
    html = '<div class="flower-grid">'
    
    for letter in sorted(grouped.keys()):
        flowers = grouped[letter]
        html += f'''
        <div class="flower-section-header">
            <span class="letter-badge">{letter}</span>
            <span>{letter}</span>
            <span class="count-badge">{len(flowers)} species</span>
        </div>
        '''
        
        for flower in flowers:
            html += f'''
            <button class="flower-btn" data-flower="{flower}">
                <span class="emoji">🌺</span>
                <span class="flower-name">{flower.title()}</span>
            </button>
            '''
    
    html += '</div>'
    return html


# ---------------------------------------------------------------------
# Create the Premium UI
# ---------------------------------------------------------------------
with gr.Blocks(
    title="🌸 Floral Vibes - Oxford 102 Flowers Classifier",
    theme=gr.themes.Soft(
        primary_hue="stone",
        secondary_hue="warm",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=PREMIUM_CSS,
    elem_id="floral-vibes-app"
) as demo:
    
    # Premium Header
    gr.HTML(create_premium_header())
    
    # Navigation Tabs
    with gr.Tabs(elem_classes="tabs"):
        
        # ============================================================
        # TAB 1: CLASSIFIER
        # ============================================================
        with gr.TabItem("📷 Flower Classifier", elem_classes="tab-item"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="font-family: 'Playfair Display', serif; font-size: 1.4rem; color: #2d1b0e; margin-bottom: 10px;">
                        Upload Your Flower
                        <span style="font-family: 'Inter', sans-serif; font-size: 0.8rem; font-weight: 400; color: #a0806a; margin-left: 10px;">
                            Drag & drop or click to browse
                        </span>
                    </div>
                    """)
                    
                    input_image = gr.Image(
                        type="pil",
                        label="",
                        elem_classes="upload-area",
                        height=350
                    )
                    
                    with gr.Row():
                        predict_btn = gr.Button(
                            "🔍 Identify Flower",
                            variant="primary",
                            elem_classes="premium-btn-primary",
                            scale=2
                        )
                        clear_btn = gr.Button(
                            "✕ Clear",
                            variant="secondary",
                            elem_classes="premium-btn-secondary",
                            scale=1
                        )
                    
                    gr.Markdown("""
                    <div style="font-size: 0.8rem; color: #a0806a; margin-top: 10px; text-align: center; letter-spacing: 1px;">
                        ✦ Powered by AlexNet trained on Oxford 102 dataset ✦
                    </div>
                    """)
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="font-family: 'Playfair Display', serif; font-size: 1.4rem; color: #2d1b0e; margin-bottom: 10px;">
                        🌸 Predictions
                        <span style="font-family: 'Inter', sans-serif; font-size: 0.8rem; font-weight: 400; color: #a0806a; margin-left: 10px;">
                            Top 5 most likely flowers
                        </span>
                    </div>
                    """)
                    
                    output_labels = gr.Label(
                        num_top_classes=TOP_K,
                        label="",
                        elem_classes="prediction-label"
                    )
                    
                    gr.Markdown("""
                    <div style="background: #faf8f5; border-radius: 12px; padding: 15px 20px; margin-top: 10px; border-left: 4px solid #4a2c1a;">
                        <div style="font-size: 0.8rem; color: #a0806a; letter-spacing: 1px;">
                            💡 The model analyzes flower features including petal shape, color patterns, 
                            and leaf structure to identify the species with high accuracy.
                        </div>
                    </div>
                    """)
            
            # Connect events
            predict_btn.click(
                fn=predict,
                inputs=input_image,
                outputs=output_labels
            )
            
            clear_btn.click(
                fn=lambda: (None, {}),
                inputs=[],
                outputs=[input_image, output_labels]
            )
        
        # ============================================================
        # TAB 2: FLOWER GALLERY
        # ============================================================
        with gr.TabItem("🌸 Flower Gallery", elem_classes="tab-item"):
            
            gr.Markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <span style="font-family: 'Playfair Display', serif; font-size: 1.8rem; color: #2d1b0e;">
                    Explore the Complete Collection
                </span>
                <br>
                <span style="font-family: 'Inter', sans-serif; font-size: 0.9rem; color: #a0806a; letter-spacing: 3px; text-transform: uppercase;">
                    All 102 flower species from the Oxford dataset
                </span>
            </div>
            """)
            
            # Gallery Layout - Directory on left, Viewer on right
            with gr.Row(equal_height=False):
                # Left: Flower Directory (scrollable)
                with gr.Column(scale=1, min_width=300):
                    with gr.Group(elem_classes="gallery-directory"):
                        gr.HTML("""
                        <div class="directory-header">
                            <div class="title">🌺 Flower Directory</div>
                            <div class="count">102 Species</div>
                        </div>
                        """)
                        
                        # Create flower buttons with letter grouping
                        flower_buttons = []
                        grouped_flowers = defaultdict(list)
                        
                        for flower_name in sorted_flower_names:
                            first_letter = flower_name[0].upper()
                            grouped_flowers[first_letter].append(flower_name)
                        
                        # Build HTML with proper grid
                        flower_grid_html = '<div class="flower-grid">'
                        
                        for letter in sorted(grouped_flowers.keys()):
                            flowers = grouped_flowers[letter]
                            flower_grid_html += f'''
                            <div class="flower-section-header">
                                <span class="letter-badge">{letter}</span>
                                <span>{letter}</span>
                                <span class="count-badge">{len(flowers)} species</span>
                            </div>
                            '''
                            
                            for flower in flowers:
                                # Create a button for each flower
                                btn = gr.Button(
                                    f"🌺 {flower.title()}",
                                    variant="secondary",
                                    size="sm",
                                    elem_classes="flower-btn"
                                )
                                flower_buttons.append((btn, flower))
                        
                        flower_grid_html += '</div>'
                        gr.HTML(flower_grid_html)
                
                # Right: Image Viewer (fixed position, doesn't scroll)
                with gr.Column(scale=2, min_width=400):
                    with gr.Group(elem_classes="gallery-viewer"):
                        with gr.Group(elem_classes="display-area"):
                            gr.Markdown("""
                            <div style="font-family: 'Playfair Display', serif; font-size: 1.1rem; color: #2d1b0e; margin-bottom: 12px; text-align: center;">
                                ✦ Selected Flower ✦
                            </div>
                            """)
                            
                            flower_image = gr.Image(
                                label="",
                                height=400,
                                interactive=False,
                                elem_classes="flower-image-container",
                                show_label=False
                            )
                            
                            flower_name_display = gr.Textbox(
                                label="",
                                value="🌸 Click a flower name to view",
                                interactive=False,
                                elem_classes="flower-name-display",
                                show_label=False
                            )
                            
                            flower_status = gr.Textbox(
                                label="",
                                value="",
                                interactive=False,
                                visible=False
                            )
            
            # Connect each button to the display function
            for btn, flower_name in flower_buttons:
                btn.click(
                    fn=show_flower_image,
                    inputs=gr.State(flower_name),
                    outputs=[flower_image, flower_name_display, flower_status]
                )
            
            # Footer note
            gr.Markdown("""
            <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid rgba(45,27,14,0.08);">
                <span style="font-size: 0.8rem; color: #a0806a; letter-spacing: 2px;">
                    📚 Dataset: Oxford 102 Flowers • 🤖 Model: AlexNet • 🎯 Top-5 Accuracy: 97%
                </span>
                <br>
                <span style="font-size: 0.7rem; color: #c4b5a6; letter-spacing: 1px;">
                    Images sourced from the Oxford 102 Flowers dataset • GitHub: shukdevtroy/Oxford-102-Flowers-Classifier
                </span>
            </div>
            """)
        
        # ============================================================
        # TAB 3: ABOUT
        # ============================================================
        with gr.TabItem("ℹ️ About", elem_classes="tab-item"):
            gr.HTML("""
            <div class="about-container">
                <div class="about-title">About Floral Vibes 🌸</div>
                <div class="about-subtitle">AI-POWERED FLOWER RECOGNITION SYSTEM</div>
                
                <div class="about-card">
                    <p><strong>🌸 Floral Vibes</strong> is an intelligent flower recognition system built with deep learning, designed to identify 102 different flower species from the Oxford 102 Flowers dataset.</p>
                    
                    <p><strong>🧠 Model Architecture:</strong> AlexNet — a powerful convolutional neural network architecture that revolutionized computer vision. The model has been trained specifically on flower images to achieve high accuracy in species identification.</p>
                    
                    <p><strong>📊 Dataset:</strong> The Oxford 102 Flowers dataset is a well-known benchmark in computer vision, containing 102 flower categories commonly found in the United Kingdom. It consists of 8,189 images with significant variations in pose, illumination, and background.</p>
                    
                    <p><strong>🎯 Accuracy:</strong> The model achieves approximately 97% Top-5 accuracy, making it highly reliable for flower identification tasks. This means the correct species is among the top 5 predictions 97% of the time.</p>
                    
                    <p><strong>💡 How it works:</strong> Simply upload a flower photo, and the AI analyzes visual features like petal shape, color patterns, leaf structure, and texture to identify the species. The system provides the top 5 most likely matches with confidence scores.</p>
                </div>
                
                <div class="about-grid">
                    <div class="about-grid-item">
                        <div class="icon">🏆</div>
                        <div class="value">97%</div>
                        <div class="label">Top-5 Accuracy</div>
                    </div>
                    <div class="about-grid-item">
                        <div class="icon">🌺</div>
                        <div class="value">102</div>
                        <div class="label">Flower Species</div>
                    </div>
                    <div class="about-grid-item">
                        <div class="icon">📸</div>
                        <div class="value">8K+</div>
                        <div class="label">Training Images</div>
                    </div>
                    <div class="about-grid-item">
                        <div class="icon">⚡</div>
                        <div class="value">AlexNet</div>
                        <div class="label">Deep Learning Model</div>
                    </div>
                </div>
                
                <div class="about-tech">
                    <div class="about-tech-title">📖 Technical Details</div>
                    <div class="about-tech-content">
                        <strong>Input Size:</strong> 227×227 pixels<br>
                        <strong>Architecture:</strong> AlexNet (8 layers with 5 convolutional and 3 fully connected layers)<br>
                        <strong>Training Data:</strong> Oxford 102 Flowers dataset with 8,189 images<br>
                        <strong>Framework:</strong> TensorFlow / Keras<br>
                        <strong>Deployment:</strong> Gradio Web Interface<br>
                        <strong>Features:</strong> Real-time prediction, Top-5 results, Flower gallery with 102 species
                    </div>
                </div>
                
                <div class="about-footer">
                    ✦ Built with ❤️ for flower lovers and AI enthusiasts ✦
                </div>
            </div>
            """)

# ---------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------
demo.queue()
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    share=False
)
