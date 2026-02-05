import streamlit as st
import torch
import torch.nn.functional as F
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
from model import MyModel

# --- 1. Load the Strong Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_trained_model():
    model = MyModel().to(device)
    # Load your 99.43% accuracy weights
    model.load_state_dict(torch.load("models/mymodel.pth", map_location=device))
    model.eval()
    return model

model = load_trained_model()

# --- 2. Page Configuration ---
st.set_page_config(page_title="Digit AI Pro", layout="centered")
st.title("🎯 Precision Digit Recognizer")
st.markdown("Draw any number from **0 to 9** below.")

# --- 3. Sidebar Instructions ---
st.sidebar.header("How it works")
st.sidebar.write("1. Draw in the center.")
st.sidebar.write("2. Use thick lines.")
st.sidebar.write("3. Watch the AI analyze your stroke.")

# --- 4. The Drawing Canvas ---
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,  # Matches the thickness the model likes
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# --- 5. Real-Time Prediction ---
if canvas_result.image_data is not None:
    # Convert canvas drawing to Grayscale
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    
    # Check if the user has actually drawn anything
    if img.getbbox(): 
        # CROP & CENTER (The most important step for 100% success)
        bbox = img.getbbox()
        img = img.crop(bbox)
        # Add 30px padding so the digit isn't touching the walls
        img = ImageOps.expand(img, border=30, fill=0)
        
        # Resize to 28x28 and Normalize
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Predict!
        if st.button("Identify Digit"):
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                digit = predicted.item()
                conf_score = confidence.item() * 100

                # Display Results
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Predicted Digit", value=digit)
                    st.write(f"Confidence: **{conf_score:.2f}%**")
                
                with col2:
                    # Show what the AI "sees" after centering/resizing
                    st.image(img, caption="AI View (Centered)", width=100)
                
                # Show probability bar chart
                st.bar_chart(probabilities.cpu().numpy()[0])
                
                if conf_score > 90:
                    st.balloons()
                    st.success(f"I am very sure this is a {digit}!")
    else:
        st.info("Draw something to start.")