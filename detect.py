import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

def process_image(image):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Perform object detection
    results = model(img_array)
    
    # Plot the results on the image
    res_plotted = results[0].plot()
    return Image.fromarray(res_plotted)

def main():
    st.title("YOLOv8 Object Detection")
    
    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Detect Objects'):
            processed_image = process_image(image)
            st.image(processed_image, caption='Processed Image', use_column_width=True)
    
    # Disable webcam functionality for deployment
    st.write("Note: Webcam functionality is disabled for this deployment.")

if __name__ == "__main__":
    main()
