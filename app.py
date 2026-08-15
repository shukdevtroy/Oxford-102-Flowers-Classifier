"""
Oxford 102 Flowers Classifier — Premium UI Edition
Redesigned to match a modern floral-boutique aesthetic: warm cream
backgrounds, rose-pink accents, serif display headings, pill buttons,
soft-shadow cards and chip-style tags. Model / prediction / image-fetch
logic is unchanged from the original app.
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
    print("Downloading model...")
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

    github_blob_url = f"https://github.com/shukdevtroy/Oxford-102-Flowers-Classifier/blob/main/flowers/{encoded_name}/image_00001.jpg?raw=true"
    try:
        response = requests.head(github_blob_url, timeout=3)
        if response.status_code == 200:
            image_cache[flower_name] = github_blob_url
            return github_blob_url
    except Exception:
        pass

    search_url = f"https://api.github.com/search/code?q=repo:shukdevtroy/Oxford-102-Flowers-Classifier+path:flowers/{encoded_name}+extension:jpg"
    try:
        response = requests.get(search_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('items'):
                download_url = data['items'][0].get('download_url')
                if download_url:
                    image_cache[flower_name] = download_url
                    print(f"Found image via search: {download_url}")
                    return download_url
    except Exception as e:
        print(f"Search error: {e}")

    image_cache[flower_name] = None
    print(f"No image found for: {flower_name}")
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
            print(f"Successfully downloaded image for: {flower_name}")
            return img
        else:
            print(f"Failed to download: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image for {flower_name}: {e}")

    return None


def label_to_flower_name(folder_label):
    return cat_to_name.get(folder_label, folder_label)


def preprocess(pil_image):
    image = np.array(pil_image.convert("RGB"))
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)


def render_prediction_html(results):
    """Turn {name: prob} into premium gradient progress-bar rows."""
    if not results:
        return "<p class='fv-placeholder'>Predictions will appear here.</p>"

    rows = []
    for name, prob in results.items():
        pct = prob * 100
        rows.append(f"""
        <div class="pred-row">
            <div class="pred-info">
                <span class="pred-name">{name.title()}</span>
                <span class="pred-pct">{pct:.1f}%</span>
            </div>
            <div class="pred-bar-bg">
                <div class="pred-bar-fill" style="width:{pct:.1f}%"></div>
            </div>
        </div>
        """)
    return "<div class='fv-pred-list'>" + "".join(rows) + "</div>"


def predict(image):
    if image is None:
        return render_prediction_html({})
    processed = preprocess(image)
    probs = model.predict(processed, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:TOP_K]
    results = {
        label_to_flower_name(class_names[i]): float(probs[i])
        for i in top_indices
    }
    return render_prediction_html(results)


def name_badge(text, ok=True):
    icon = "🌸" if ok else "⚠️"
    return f'<span class="fv-name-badge">{icon} {text}</span>'


def show_flower_image(flower_name):
    """Display flower image when clicked"""
    if not flower_name:
        return None, name_badge("Select a flower to preview", ok=False)

    print(f"\nShowing image for: {flower_name}")
    img = get_flower_image(flower_name)

    if img:
        return img, name_badge(flower_name.title(), ok=True)
    else:
        placeholder = Image.new('RGB', (500, 500), color='#FFF0F5')
        draw = ImageDraw.Draw(placeholder)
        try:
            font = ImageFont.load_default()
            text = f"Image not found:\n{flower_name}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (500 - text_width) // 2
            y = (500 - text_height) // 2
            draw.text((x, y), text, fill='#B5406C', font=font)
        except Exception:
            pass
        return placeholder, name_badge(f"Image not found: {flower_name}", ok=False)


# ---------------------------------------------------------------------
# Premium theme CSS — cream + rose-pink boutique aesthetic
# ---------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Poppins:wght@300;400;500;600&display=swap');

:root {
    --fv-cream: #FDF6F0;
    --fv-cream-soft: #FFFBF7;
    --fv-pink: #F0357C;
    --fv-pink-dark: #D62A68;
    --fv-pink-light: #FBD9E6;
    --fv-pink-pale: #FFF0F5;
    --fv-dark: #201A1D;
    --fv-text-muted: #8a8386;
    --fv-shadow: 0 12px 34px rgba(240, 53, 124, 0.10);
}

.gradio-container {
    background: var(--fv-cream) !important;
    font-family: 'Poppins', sans-serif !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
}

footer { display: none !important; }

/* ---------- Hero header ---------- */
#fv-hero {
    background: linear-gradient(120deg, var(--fv-pink-pale) 0%, var(--fv-cream-soft) 65%);
    border-radius: 28px;
    padding: 40px 46px;
    margin-bottom: 26px;
    box-shadow: var(--fv-shadow);
    border: 1px solid rgba(240, 53, 124, 0.08);
}
#fv-hero .fv-logo {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 600;
    color: var(--fv-dark);
    margin-bottom: 20px;
    letter-spacing: 0.3px;
}
#fv-hero .fv-logo span { color: var(--fv-pink); }
#fv-hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 40px;
    line-height: 1.18;
    font-weight: 600;
    color: var(--fv-dark);
    margin: 0 0 14px 0;
}
#fv-hero h1 .accent { color: var(--fv-pink); font-style: italic; }
#fv-hero p {
    font-size: 15px;
    color: var(--fv-text-muted);
    max-width: 540px;
    margin: 0;
}
#fv-hero .fv-badge {
    display: inline-block;
    background: var(--fv-dark);
    color: #fff;
    padding: 9px 22px;
    border-radius: 999px;
    font-size: 12.5px;
    letter-spacing: 0.6px;
    margin-top: 20px;
}

/* ---------- Tabs as pill nav ---------- */
.tabs > .tab-nav {
    border: none !important;
    background: transparent !important;
    gap: 10px;
    margin-bottom: 22px !important;
}
.tabs > .tab-nav button {
    border-radius: 999px !important;
    border: 1px solid var(--fv-pink-light) !important;
    background: #fff !important;
    color: var(--fv-dark) !important;
    font-weight: 500 !important;
    padding: 10px 26px !important;
    font-size: 14px !important;
}
.tabs > .tab-nav button.selected {
    background: var(--fv-pink) !important;
    color: #fff !important;
    border-color: var(--fv-pink) !important;
}

/* ---------- Cards ---------- */
.fv-card {
    background: #fff !important;
    border-radius: 22px !important;
    padding: 22px !important;
    box-shadow: var(--fv-shadow);
    border: 1px solid rgba(240, 53, 124, 0.07);
}

.fv-section-title {
    font-family: 'Playfair Display', serif;
    font-size: 21px;
    color: var(--fv-dark);
    margin-bottom: 2px;
}
.fv-section-sub {
    color: var(--fv-text-muted);
    font-size: 13px;
    margin-bottom: 16px;
}

/* ---------- Buttons ---------- */
button.primary, .gr-button-primary {
    background: var(--fv-pink) !important;
    border: none !important;
    border-radius: 999px !important;
    color: #fff !important;
    font-weight: 500 !important;
    box-shadow: 0 10px 22px rgba(240, 53, 124, 0.28) !important;
}
button.primary:hover { background: var(--fv-pink-dark) !important; }

/* Chip buttons for the flower browse list */
.fv-chip button {
    border-radius: 999px !important;
    background: var(--fv-pink-pale) !important;
    border: 1px solid var(--fv-pink-light) !important;
    color: var(--fv-dark) !important;
    font-size: 12.5px !important;
    font-weight: 400 !important;
    padding: 6px 16px !important;
    min-width: unset !important;
    box-shadow: none !important;
}
.fv-chip button:hover {
    background: var(--fv-pink) !important;
    color: #fff !important;
    border-color: var(--fv-pink) !important;
}
#fv-chip-wrap {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    max-height: 380px;
    overflow-y: auto;
    padding: 6px 4px;
}
#fv-chip-wrap > * { width: auto !important; flex: 0 0 auto !important; }

/* ---------- Prediction bars ---------- */
.fv-placeholder { color: #b6adb0; font-size: 14px; }
.pred-row { margin-bottom: 16px; }
.pred-info {
    display: flex;
    justify-content: space-between;
    font-size: 14px;
    margin-bottom: 6px;
    color: var(--fv-dark);
}
.pred-name { font-weight: 500; }
.pred-pct { color: var(--fv-pink); font-weight: 600; }
.pred-bar-bg {
    background: var(--fv-pink-light);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.pred-bar-fill {
    background: linear-gradient(90deg, var(--fv-pink), var(--fv-pink-dark));
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}

/* ---------- Selected flower name badge ---------- */
.fv-name-badge {
    display: inline-block;
    background: var(--fv-pink-pale);
    border: 1px solid var(--fv-pink-light);
    color: var(--fv-dark);
    padding: 10px 22px;
    border-radius: 999px;
    font-family: 'Playfair Display', serif;
    font-size: 17px;
    margin-top: 12px;
}
"""

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
with gr.Blocks(title="🌸 Oxford 102 Flowers Classifier", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:

    gr.HTML("""
    <div id="fv-hero">
        <div class="fv-logo">🌸 Petal<span>ID</span></div>
        <h1>Discover the <span class="accent">Flower</span><br>in Every Photo</h1>
        <p>Upload a photo and the model instantly identifies it among 102 species —
        or browse the full gallery below to explore each one up close.</p>
        <span class="fv-badge">AI-POWERED · 102 SPECIES</span>
    </div>
    """)

    with gr.Tab("📷 Classifier"):
        with gr.Row():
            with gr.Column(scale=1, elem_classes="fv-card"):
                gr.HTML('<div class="fv-section-title">Upload a Photo</div><div class="fv-section-sub">JPG or PNG of any flower</div>')
                input_image = gr.Image(type="pil", label="")
                predict_btn = gr.Button("🔍 Identify Flower", variant="primary")
            with gr.Column(scale=1, elem_classes="fv-card"):
                gr.HTML('<div class="fv-section-title">Top Predictions</div><div class="fv-section-sub">Ranked by model confidence</div>')
                output_labels = gr.HTML(value="<p class='fv-placeholder'>Predictions will appear here.</p>")
        predict_btn.click(fn=predict, inputs=input_image, outputs=output_labels)

    with gr.Tab("🌺 Flower Gallery"):
        gr.HTML('<div class="fv-section-title">Explore All 102 Species</div><div class="fv-section-sub">Search or tap a name to preview its photo</div>')
        with gr.Row():
            with gr.Column(scale=1, elem_classes="fv-card"):
                search_box = gr.Dropdown(
                    choices=sorted_flower_names,
                    label="Search flowers",
                    filterable=True,
                )
                gr.Markdown("**Or browse the full list:**")
                with gr.Group(elem_id="fv-chip-wrap"):
                    flower_buttons = []
                    for flower_name in sorted_flower_names:
                        btn = gr.Button(flower_name.title(), size="sm", elem_classes="fv-chip")
                        flower_buttons.append((btn, flower_name))

            with gr.Column(scale=1, elem_classes="fv-card"):
                flower_image = gr.Image(label="", height=380, interactive=False)
                flower_name_display = gr.HTML(value=name_badge("Select a flower to preview", ok=False))

        for btn, flower_name in flower_buttons:
            btn.click(
                fn=show_flower_image,
                inputs=gr.State(flower_name),
                outputs=[flower_image, flower_name_display],
            )
        search_box.change(
            fn=show_flower_image,
            inputs=search_box,
            outputs=[flower_image, flower_name_display],
        )

        gr.Markdown("""
        ---
        📚 **Dataset**: Oxford 102 Flowers — 102 flower categories commonly found in the United Kingdom.
        """)

# Launch
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
