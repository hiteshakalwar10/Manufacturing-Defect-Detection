# Manufacturing Defect Detection

This project detects surface defects on steel using a CNN model. I built it as part of my deep learning portfolio to practice transfer learning, model explainability and deployment.

## What it does

The model classifies steel surface images into 6 defect categories:
- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

It's trained on the NEU Surface Defect Database using transfer learning on EfficientNetB0.

## Why I built this

Most defect detection demos just show a prediction and stop there. I wanted to go a step further and make the model explainable, so I added Grad-CAM to visualize which part of the image the model actually focused on before making a decision. This felt important because in a real manufacturing setting, you'd want to verify the model is looking at the actual defect and not something random in the background.

## How it works

1. Images go through preprocessing (resize to 224x224, normalization)
2. EfficientNetB0 (pretrained on ImageNet) extracts features
3. Training happens in two phases — first with the base frozen, then fine-tuning the last several layers with a lower learning rate
4. Grad-CAM generates a heatmap showing which regions influenced the prediction most
5. Final output includes the predicted class, confidence score, and a downloadable PDF report

## Tech stack

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit (for the web app)
- ReportLab (for PDF report generation)

## Try it

Live app: https://manufacturing-defect-detection-upp8eevlvjpkawgqpybymz.streamlit.app/

## Dataset

NEU Surface Defect Database (Kaggle)

## Limitations

This model is trained specifically on steel surface images from the NEU dataset. It won't generalize well to other materials (like fabric, plastic, or wood defects) without retraining on a different dataset.

## Notebook

The full training and evaluation notebook is included in this repo if you want to see the complete process, including the two-phase training and Grad-CAM implementation.
