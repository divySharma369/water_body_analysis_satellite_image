import streamlit as st
import cv2
import numpy as np
import onnxruntime as rt
import os
from PIL import Image

# ---------------------------------------------------------
# UI Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Water Body Segmentation",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism and modern design
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #1f6feb 0%, #238636 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.4);
    }
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1rem;
    }
    h1, h2, h3 {
        color: #58a6ff;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Core Helper Functions (Adapted from functions.py)
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path='weights/model.onnx'):
    return rt.InferenceSession(model_path)

def resize_preserve_aspect_ratio(image, size):
    h, w = image.shape[:2]
    if h > w:
        return cv2.resize(image, (size * w // h, size))
    else:
        return cv2.resize(image, (size, size * h // w))

def predict(inp_image, session):
    inp_dim = inp_image.shape[:2]

    # Preprocess
    image = cv2.resize(inp_image, (256, 256))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    # Inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    pred_onx = session.run([output_name], {input_name: image.astype(np.float32)})[0]

    # Postprocess
    pred_onx = pred_onx > 0.5
    pred_onx = pred_onx * 255
    pred_onx = cv2.resize(pred_onx[0, 0].astype(np.uint8) , (inp_dim[1], inp_dim[0]))
    
    # Calculate water percentage
    water_pixels = np.count_nonzero(pred_onx)
    total_pixels = pred_onx.shape[0] * pred_onx.shape[1]
    water_percentage = (water_pixels / total_pixels) * 100

    # Expand dims for RGB visualization
    pred_onx = np.expand_dims(pred_onx, axis=2)
    pred_onx_rgb = np.concatenate((pred_onx, pred_onx, pred_onx), axis=2)
    
    # Mask coloring (Blue for water)
    colored_mask = np.zeros_like(pred_onx_rgb)
    colored_mask[:,:,0] = 0   # R
    colored_mask[:,:,1] = 100 # G
    colored_mask[:,:,2] = 255 # B
    colored_mask = np.where(pred_onx_rgb > 0, colored_mask, 0)
    
    # Create overlay
    overlay = cv2.addWeighted(inp_image, 0.7, colored_mask, 0.5, 0)

    return pred_onx_rgb, overlay, water_percentage

# ---------------------------------------------------------
# App Layout
# ---------------------------------------------------------
st.title("🌍 Water Body Segmentation")
st.markdown("### Powered by PyTorch & ONNX")

# Initialize Model
try:
    session = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load the model. Ensure `weights/model.onnx` exists. Error: {e}")
    model_loaded = False

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/PyTorch_logo_icon.svg/1200px-PyTorch_logo_icon.svg.png", width=100)
    st.header("Upload or Select")
    st.markdown("Choose a satellite image to detect water bodies.")
    
    upload_option = st.radio("Input Method", ["Use Examples", "Upload Image"])
    selected_image = None
    
    if upload_option == "Use Examples":
        examples = [f"examples/image{i}.png" for i in range(1, 6)]
        valid_examples = [x for x in examples if os.path.exists(x)]
        if valid_examples:
            example_choice = st.selectbox("Select an Example", valid_examples)
            selected_image = Image.open(example_choice).convert('RGB')
        else:
            st.warning("No examples found in `examples/` directory.")
            
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert('RGB')

# Main Body
if selected_image is not None and model_loaded:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Image")
        st.image(selected_image, use_column_width=True)
        img_array = np.array(selected_image)
        
    with col2:
        st.subheader("Segmentation Magic")
        if st.button("🌊 Run Segmentation"):
            with st.spinner('Analyzing satellite imagery...'):
                mask, overlay, water_pct = predict(img_array, session)
                
            st.image(overlay, use_column_width=True, caption="Water Overlay (Blue)")
            
            # Metrics
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Resolution", f"{img_array.shape[1]}x{img_array.shape[0]}")
            m2.metric("Water Area", f"{water_pct:.1f}%")
            m3.metric("Status", "Complete")
            
            # Show Raw Mask Toggle
            with st.expander("View Raw Mask"):
                st.image(mask, use_column_width=True, clamp=True)
                
    st.markdown('</div>', unsafe_allow_html=True)
elif not model_loaded:
    st.info("Please load the model to continue.")
else:
    st.info("Please select or upload an image from the sidebar.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #8b949e;'>Developed by Divy Sharma | PyTorch 🚀</div>", unsafe_allow_html=True)
