# 🌍 UNET Water Body Segmentation - PyTorch

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://waterbodyanalysissatelliteimage.streamlit.app/)

This repository hosts a state-of-the-art **U-Net** deep learning model designed for the precise pixel-wise segmentation of water bodies from satellite imagery. It was built using **PyTorch** and deployed using a highly interactive **Streamlit** user interface.

👉 **[Try the Live Web App Here!](https://waterbodyanalysissatelliteimage.streamlit.app/)**

---

## 🧠 Model Architecture Deep Dive: How U-Net Works

The core of this project relies on **U-Net**, a fully convolutional network initially developed for medical image segmentation. Because satellite imagery is structurally similar (requiring dense, localized predictions rather than just single-image classification), U-Net is exceptionally powerful at identifying the boundaries of water features like rivers, lakes, and oceans.

The architecture gets its name from its symmetric "U" shape and is divided into three main sections:

### 1. The Encoder (Contracting Path)
The journey of an input image begins here. The satellite image is passed through a series of **convolution blocks**. Each block consists of two `3x3` Convolutions followed by a ReLU activation function and a `2x2` Max Pooling step. 
* **Purpose**: This acts as a feature extractor. As the image goes deeper, the spatial dimensions shrink, but the number of feature channels (depth) increases. It learns the "What" of the image (e.g., recognizing textures, colors, and the semantic meaning of water boundaries).

### 2. The Bottleneck
At the very bottom of the "U", the network reaches its highest level of abstraction. The image is compressed into a highly dense feature map representing the macro-level layout of the satellite image, without much focus on high-resolution spatial details.

### 3. The Decoder (Expansive Path) & Skip Connections
Once the network understands the context, it must map those features back to the original image dimensions to create a literal map. This is done via **Up-convolutions** (Transposed Convolutions).
* **The Magic of Skip Connections**: To avoid losing the fine-grained details during the compression process, U-Net uses "Skip Connections." It takes the high-resolution features from the Encoder path and directly concatenates them with the upsampled features in the Decoder.
* **Purpose**: The Decoder combines the *context* learned from the Bottleneck with the *location details* preserved by the Skip Connections. This tells the network exactly "Where" the water boundaries are.

Finally, a `1x1` Convolution maps the final layer down to a single channel (water vs. no-water). The network outputs a 2D mask, lighting up the exact pixels where a water body exists!

---

## 🏗️ The Training Process
The model was trained on the [Satellite Images of Water Bodies](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies) dataset from Kaggle. 

* **Loss Function**: The training optimizes a combination of **Binary Cross Entropy (BCE)** and **DICE Loss**. While BCE evaluates general pixel classification, DICE Loss mathematically calculates the overlap between our predicted water boundary and the real water boundary. This forces the model to trace exact, crisp edges around lakes and rivers.
* **Post-processing**: During inference via the ONNX runtime, the neural tensor is evaluated against a pixel threshold (usually >0.5 probability) to generate the sharp binary black/white masks you see in the application overlay.

---

## 🚀 Getting Started Locally

All the code required to run the interactive dashboard is contained in `app.py`. If you want to explore the ONNX transition or train the model from scratch, look inside the `notebooks/` directory.

### Running the App
Make sure your environment has Python installed, then simply:

1. Clone the repository to your local machine.
2. Install the necessary dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
3. Run the interactive Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

*(Alternatively, use the provided `Makefile` to instantly launch it with `make run`!)*

### Using Docker
This repository natively supports Docker deployments:
```bash
docker build -t waterbody-segmentation-app .
docker run -p 8501:8501 waterbody-segmentation-app
```

---
*Created and maintained by [Divy Sharma](https://github.com/divySharma369).*
