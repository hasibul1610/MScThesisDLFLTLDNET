import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import io
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# >>> NEW: try importing rembg
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

# ---- Model and class labels ----
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('inceptionv3_tomato.h5')
model = load_model()

class_indices = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
color_based = set(class_indices[:5])
texture_based = set(class_indices[5:9])

# ---- Severity Fuzzy Logic ----
disease_area = ctrl.Antecedent(np.arange(0, 101, 1), 'disease_area')
lesion_count = ctrl.Antecedent(np.arange(0, 101, 1), 'lesion_count')
severity_fuzzy = ctrl.Consequent(np.arange(0, 101, 1), 'severity')

disease_area['low'] = fuzz.trimf(disease_area.universe, [0, 0, 10])
disease_area['medium'] = fuzz.trimf(disease_area.universe, [10, 40, 60])
disease_area['high'] = fuzz.trimf(disease_area.universe, [40, 80, 100])

lesion_count['few'] = fuzz.trimf(lesion_count.universe, [0, 0, 5])
lesion_count['moderate'] = fuzz.trimf(lesion_count.universe, [5, 15, 30])
lesion_count['many'] = fuzz.trimf(lesion_count.universe, [25, 50, 100])

severity_fuzzy['primary'] = fuzz.trimf(severity_fuzzy.universe, [0, 0, 30])
severity_fuzzy['medium'] = fuzz.trimf(severity_fuzzy.universe, [25, 50, 70])
severity_fuzzy['extreme'] = fuzz.trimf(severity_fuzzy.universe, [50, 90, 100])

# rules = [
#     ctrl.Rule(disease_area['low'] & lesion_count['few'], severity_fuzzy['primary']),
#     ctrl.Rule(disease_area['low'] & lesion_count['moderate'], severity_fuzzy['medium']),
#     ctrl.Rule(disease_area['low'] & lesion_count['many'], severity_fuzzy['medium']),
#     ctrl.Rule(disease_area['medium'] & lesion_count['few'], severity_fuzzy['medium']),
#     ctrl.Rule(disease_area['medium'] & lesion_count['moderate'], severity_fuzzy['extreme']),
#     ctrl.Rule(disease_area['medium'] & lesion_count['many'], severity_fuzzy['extreme']),
#     ctrl.Rule(disease_area['high'] & lesion_count['few'], severity_fuzzy['extreme']),
#     ctrl.Rule(disease_area['high'] & lesion_count['moderate'], severity_fuzzy['extreme']),
#     ctrl.Rule(disease_area['high'] & lesion_count['many'], severity_fuzzy['extreme']),
#     ctrl.Rule(disease_area['high'], severity_fuzzy['extreme']),
#     ctrl.Rule(disease_area['low'] | disease_area['medium'] | disease_area['high'], severity_fuzzy['primary'])
# ]

rules = [
    ctrl.Rule(disease_area['low'] & lesion_count['few'], severity_fuzzy['primary']),
    ctrl.Rule(disease_area['low'] & lesion_count['moderate'], severity_fuzzy['medium']),
    ctrl.Rule(disease_area['low'] & lesion_count['many'], severity_fuzzy['medium']),
    ctrl.Rule(disease_area['medium'] & lesion_count['few'], severity_fuzzy['medium']),
    ctrl.Rule(disease_area['medium'] & lesion_count['moderate'], severity_fuzzy['extreme']),
    ctrl.Rule(disease_area['medium'] & lesion_count['many'], severity_fuzzy['extreme']),
    ctrl.Rule(disease_area['high'] & lesion_count['few'], severity_fuzzy['extreme']),
    ctrl.Rule(disease_area['high'] & lesion_count['moderate'], severity_fuzzy['extreme']),
    ctrl.Rule(disease_area['high'] & lesion_count['many'], severity_fuzzy['extreme'])
]
severity_ctrl = ctrl.ControlSystem(rules)

def fuzzy_severity_label(area, lesions, predicted_class=None):
    area = np.clip(area, 0, 100)
    lesions = np.clip(lesions, 0, 100)
    if area >= 50 or lesions >= 30:
        return "Extreme", 85.0
    if predicted_class in texture_based and (area >= 20 or lesions >= 25):
        return "Extreme", 90.0
    if predicted_class in texture_based and area >= 70:
        return "Extreme", 90.0
    sim = ctrl.ControlSystemSimulation(severity_ctrl)
    try:
        sim.input['disease_area'] = area
        sim.input['lesion_count'] = lesions
        sim.compute()
        score = sim.output['severity']
        label = 'Primary' if score < 30 else 'Medium' if score < 70 else 'Extreme'
        return label, score
    except Exception as e:
        return "Unknown", 0

# ---- Background removal helpers ----
@st.cache_resource
def get_rembg_session():
    if not REMBG_AVAILABLE:
        return None
    try:
        return new_session("isnet-general")
    except Exception:
        return new_session()

def remove_bg(pil_img, use_rembg=True):
    if not (use_rembg and REMBG_AVAILABLE):
        return pil_img.convert("RGB"), None

    session = get_rembg_session()
    np_img = np.array(pil_img.convert("RGB"))
    out = remove(np_img, session=session) 
    rgba = Image.fromarray(out).convert("RGBA")
    alpha = np.array(rgba.split()[-1])  
    return rgba, alpha

# ---- Preprocessing, Segmentation, Grad-CAM ----
def preprocess_image(img):
    img = img.convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="mixed9"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(class_channel, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

def apply_heatmap(heatmap, image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

def generate_segmentation_mask(img_cv_bgr, category, alpha_mask=None):
    """
    img_cv_bgr: OpenCV BGR (foreground already placed on black)
    alpha_mask: optional uint8 (0..255). If provided, we'll treat alpha<10 as BG.
    """
    img_resized = cv2.resize(img_cv_bgr, (224, 224))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    mask_rgb = np.zeros_like(img_resized)

    # BG mask from alpha (preferred), else near-black heuristic
    if alpha_mask is not None:
        alpha_small = cv2.resize(alpha_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask_bg = (alpha_small < 10).astype(np.uint8) * 255
    else:
        mask_bg = cv2.inRange(img_resized, np.array([0, 0, 0]), np.array([5, 5, 5]))

    if category == 'healthy':
        mask_rgb[:] = [0, 255, 0]
        mask_rgb[mask_bg > 0] = [0, 0, 255]
        return mask_rgb, 0

    elif category == 'color':
        # You can tune these if your dataset colors differ
        mask_disease = cv2.inRange(hsv, (0, 50, 20), (30, 255, 200))
        mask_healthy = cv2.inRange(hsv, (25, 40, 40), (90, 255, 255))
        mask_rgb[mask_healthy > 0] = [0, 255, 0]
        mask_rgb[mask_disease > 0] = [255, 0, 0]
        mask_rgb[mask_bg > 0] = [0, 0, 255]
        return mask_rgb, cv2.connectedComponents(mask_disease)[0] - 1

    else:
        edges = cv2.Canny(gray, 50, 150)
        _, thresh = cv2.threshold(
            gray, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        combined = cv2.bitwise_or(edges, thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)

        mask_rgb[combined > 0] = [255, 0, 0]
        mask_rgb[mask_bg > 0] = [0, 0, 255]
        mask_rgb[(combined == 0) & (mask_bg == 0)] = [0, 255, 0]
        return mask_rgb, cv2.connectedComponents(combined)[0] - 1

# ---- Streamlit Layout ----
st.set_page_config(page_title="Tomato Leaf Disease & Severity", layout="wide", initial_sidebar_state="expanded", page_icon="🍅")
st.title("🍅 DLFL-TLDNet")
st.write("Upload one or more tomato leaf images. The app will (optionally) remove background, classify, analyze severity, and visualize each result.")

use_bg_removal = st.sidebar.checkbox("Remove background with rembg (recommended)", value=True)
uploaded_files = st.file_uploader("Upload Image(s)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key="uploader")

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        orig_rgb_pil = Image.open(uploaded_file).convert('RGB')

        rgba_pil, alpha_mask = remove_bg(orig_rgb_pil, use_rembg=use_bg_removal)

        if alpha_mask is not None:
            fg_rgb = Image.new("RGB", rgba_pil.size, (0, 0, 0))
            fg_rgb.paste(rgba_pil.convert("RGB"), mask=rgba_pil.split()[-1])
            display_input = fg_rgb
        else:
            fg_rgb = orig_rgb_pil
            display_input = orig_rgb_pil

        img_resized = fg_rgb.resize((224, 224), resample=Image.BILINEAR)
        img_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        processed_img = preprocess_image(img_resized)

        preds = model.predict(processed_img)[0]
        pred_class = class_indices[np.argmax(preds)]
        confidence = np.max(preds) * 100

        category = 'color' if pred_class in color_based else ('texture' if pred_class in texture_based else 'healthy')

        mask, lesions = generate_segmentation_mask(img_cv, category, alpha_mask=alpha_mask)

        mask_full = cv2.resize(mask, (img_cv.shape[1], img_cv.shape[0]))
        disease = np.all(mask_full == [255, 0, 0], axis=-1)
        healthy = np.all(mask_full == [0, 255, 0], axis=-1)
        total = np.sum(disease | healthy)
        percent = (np.sum(disease) / total * 100) if total > 0 else 0
        severity, score = fuzzy_severity_label(percent, lesions, predicted_class=pred_class)

        overlay = cv2.addWeighted(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), 0.6, mask_full, 0.4, 0)

        heatmap = make_gradcam_heatmap(processed_img, model)
        grad_img = apply_heatmap(heatmap, img_cv)

        fig, axes = plt.subplots(1, 4, figsize=(16, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1]})
        axes[0].imshow(display_input)
        axes[0].set_title(f"{uploaded_file.name}\nClassified: {pred_class} ({confidence:.1f}%)", fontsize=9)
        axes[0].axis('off')

        axes[1].imshow(mask_full)
        axes[1].set_title("Segmentation\nRed=Disease | Green=Healthy | Blue=BG", fontsize=9)
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title(f"Area: {percent:.2f}% | Lesions: {lesions}\nSeverity: {severity} ({score:.2f})", fontsize=9)
        axes[2].axis('off')

        axes[3].imshow(cv2.cvtColor(grad_img, cv2.COLOR_BGR2RGB))
        axes[3].set_title("Grad-CAM Focus", fontsize=9)
        axes[3].axis('off')

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        with st.container():
            st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
            cols = st.columns(4, gap="large")
            cols[0].image(
                display_input,
                caption=(
                    f"File: {uploaded_file.name}\n"
                    f"Class: {pred_class}\n"
                    f"Confidence: {confidence:.1f}%\n"
                    f"BG Removed: {bool(alpha_mask is not None)}"
                ),
                use_container_width=True
            )
            cols[1].image(mask_full, caption="Segmentation (Red: Disease, Green: Healthy, Blue: BG)", use_container_width=True)
            cols[2].image(overlay, caption=f"Overlay | Area: {percent:.2f}%, Lesions: {lesions}, Severity: {severity} ({score:.2f})", use_container_width=True)
            cols[3].image(cv2.cvtColor(grad_img, cv2.COLOR_BGR2RGB), caption="Grad-CAM Focus", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.download_button(
                label="Download Result as Image",
                data=buf,
                file_name=f"{uploaded_file.name}_result.png",
                mime="image/png",
                key=f"download_{idx}"
            )
        st.write("")

    st.info("🔄 To upload more images, please refresh the page (press F5 or reload button).")
else:
    st.info("👆 Use the button above to upload one or more images.")
