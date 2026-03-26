---
title: Water Body Segmentation
emoji: 🤗
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 3.10.0
app_file: app.py
pinned: false
---

# UNET Water Body Segmentation - PyTorch

This project contains the code for training and deploying a UNET model for water body segmentation from satellite images. The model is trained on the [Satellite Images of Water Bodies](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies) from Kaggle. The model is trained using PyTorch and deployed using [Streamlit](https://streamlit.io/).

## 🚀 Getting Started

All the code for training the model and exporting to ONNX format is present in the [notebook](notebooks) folder. The [app.py](app.py) file contains the code for the interactive Streamlit application.

## Run Locally
- `pip install -r requirements.txt`
- `streamlit run app.py`
- Input images should ideally be satellite views; they will be resized to (256, 256) internally.

## 🖥️ Sample Inference

![Sample Inference](samples/sample1.png)

