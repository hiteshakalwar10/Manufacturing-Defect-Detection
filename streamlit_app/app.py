
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D, InputLayer
from tensorflow.keras.applications.efficientnet import preprocess_input
import json
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(page_title="Manufacturing Defect Detection", page_icon="🏭", layout="wide")

MODEL_PATH = "models/best_model.h5"
CLASS_INDICES_PATH = "models/class_indices.json"
IMG_SIZE = 224

@st.cache_resource
def load_defect_model():
    model = load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    class_names = sorted(class_indices, key=class_indices.get)

    base_model = model.layers[1]

    def get_last_conv_layer_name(base_model):
        for layer in reversed(base_model.layers):
            if isinstance(layer, Conv2D):
                return layer.name
        raise ValueError("No Conv2D layer found.")

    last_conv_layer_name = get_last_conv_layer_name(base_model)

    last_conv_output = base_model.get_layer(last_conv_layer_name).output
    base_output = base_model.output
    head_layers = [l for l in model.layers if l.name != base_model.name and not isinstance(l, InputLayer)]

    x = base_output
    for layer in head_layers:
        x = layer(x)

    grad_model = Model(inputs=base_model.input, outputs=[last_conv_output, x])

    return model, class_names, grad_model

model, class_names, grad_model = load_defect_model()

def make_gradcam_heatmap(img_array, pred_index=None):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_gradcam(img_rgb, heatmap, alpha=0.4, img_size=224):
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_resized, 1 - alpha, heatmap_colored_rgb, alpha, 0)
    return img_resized, heatmap_colored_rgb, overlay

def generate_pdf_report(pred_class, confidence, top_preds, img_path, gradcam_path, report_path):
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Manufacturing Defect Detection Report", styles["Title"]))
    elements.append(Paragraph("Automated surface defect classification using a fine-tuned EfficientNetB0 CNN, with Grad-CAM explainability.", styles["Normal"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Predicted Class: {pred_class}", styles["Heading2"]))
    elements.append(Paragraph(f"Confidence Score: {confidence*100:.2f}%", styles["Normal"]))
    elements.append(Spacer(1, 15))

    table_data = [["Class", "Probability (%)"]]
    for cls, prob in top_preds:
        cls_clean = cls.replace("_", " ").replace("-", " ").title()
        table_data.append([cls_clean, f"{prob*100:.2f}"])

    table = Table(table_data, colWidths=[250, 150])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER")
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Uploaded Image & Grad-CAM Visualization", styles["Heading2"]))
    elements.append(Paragraph("🔴🟡Red/Yellow = high model attention (defect area) | 🔵🟢Blue/Green = low attention (background)", styles["Normal"]))
    elements.append(Spacer(1, 10))

    img_table = Table([[
        RLImage(img_path, width=2.5*inch, height=2.5*inch),
        RLImage(gradcam_path, width=2.5*inch, height=2.5*inch)
    ]])
    elements.append(img_table)

    doc.build(elements)

st.title("🏭 Manufacturing Defect Detection")
st.write("Upload a product image to detect surface defects using a fine-tuned EfficientNetB0 model.")

uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("gradcam", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    img_path = os.path.join("outputs", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if st.button("Predict"):
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_array = preprocess_input(np.expand_dims(img_resized.astype(np.float32), axis=0))

        _, predictions_tensor = grad_model(img_array)
        predictions = predictions_tensor.numpy()[0]
        top_indices = np.argsort(predictions)[::-1][:3]
        top_preds = [(class_names[i], float(predictions[i])) for i in top_indices]
        pred_class = class_names[top_indices[0]]
        confidence = float(predictions[top_indices[0]])

        heatmap = make_gradcam_heatmap(img_array, pred_index=int(top_indices[0]))
        original, heatmap_colored, overlay = overlay_gradcam(img_rgb, heatmap, img_size=IMG_SIZE)

        gradcam_path = os.path.join("gradcam", "gradcam_overlay.png")
        cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction Card")
            st.success(f"Predicted Defect: {pred_class}")
            st.metric("Confidence Score", f"{confidence*100:.2f}%")
            st.write("Top 3 Predictions")
            for cls, prob in top_preds:
                st.write(f"{cls}: {prob*100:.2f}%")

        with col2:
            st.subheader("Original Image")
            st.image(original, width=350)

        st.subheader("Grad-CAM Visualization")
        st.caption("🔴🟡Red/Yellow = high model attention (defect area) | 🔵🟢Blue/Green = low attention (background)")
        gc1, gc2 = st.columns(2)
        with gc1:
            st.image(heatmap_colored, caption="Heatmap", width=350)
        with gc2:
            st.image(overlay, caption="Overlay", width=350)

        report_path = os.path.join("reports", "defect_detection_report.pdf")
        generate_pdf_report(pred_class, confidence, top_preds, img_path, gradcam_path, report_path)

        with open(report_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="defect_detection_report.pdf", mime="application/pdf")
