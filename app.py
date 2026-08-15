"""
Oxford 102 Flowers classifier — Gradio app for Hugging Face Spaces.

Expects these three files to sit next to this script (all produced by / used
in your training notebook):
    - alexnet_flowers102.keras   (from MODEL_PATH, saved in Cell 15)
    - class_names.json           (from CLASS_NAMES_PATH, saved in Cell 15)
    - cat_to_name.json           (the Kaggle-provided id -> flower-name file)

If cat_to_name.json isn't present, the app falls back to showing the raw
folder-label id (e.g. "21") instead of a flower name like "fire lily" —
still works, just less readable.
"""

import json
import os

import gdown
import gradio as gr
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------
# Config — must match the training notebook
# ---------------------------------------------------------------------
IMG_SIZE = 227
RESCALE = 1.0 / 255.0
TOP_K = 5

MODEL_PATH = "alexnet_flowers102.keras"
CLASS_NAMES_PATH = "class_names.json"
CAT_TO_NAME_PATH = "cat_to_name.json"

# Google Drive file ID for the model (from the shareable link).
# Set this as an env var in Render's dashboard rather than hardcoding it
# here, so you can swap models without editing code.
#   Link:    https://drive.google.com/file/d/XXXXXXXXXXXXXXXXXXXX/view
#   File ID: XXXXXXXXXXXXXXXXXXXX  (the part between /d/ and /view)
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
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a flower photo"),
    outputs=gr.Label(num_top_classes=TOP_K, label="Predictions"),
    title="🌸 Oxford 102 Flowers Classifier",
    description=(
        "Upload a photo of a flower and the model will predict which of "
        "102 flower species it is (trained on the Oxford 102 Flowers dataset)."
    ),
    examples=None,  # add example image paths here if you upload sample images too
    flagging_mode="never",
)

if __name__ == "__main__":
    # Render (and most PaaS hosts) inject the port to bind via $PORT and
    # require binding to 0.0.0.0, not localhost.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
