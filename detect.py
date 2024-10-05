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
    
    # Check if the image has an alpha channel (4 channels)
    if img_array.shape[-1] == 4:
        # Convert RGBA to RGB
        img_array = img_array[:, :, :3]
    
    # Ensure the image is in RGB format
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[-1] == 3:  # RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Perform object detection
    results = model(img_array)
    
    # Plot the results on the image
    res_plotted = results[0].plot()
    return Image.fromarray(res_plotted)

def main():
    st.title("YOLOv8 Object Detection")
    
    # Sidebar for selecting input method
    input_method = st.sidebar.radio("Select Input Method", ["Webcam", "File Upload"])
    
    if input_method == "File Upload":
        # File uploader allows user to add their own image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button('Detect Objects'):
                processed_image = process_image(image)
                st.image(processed_image, caption='Processed Image', use_column_width=True)
    
    elif input_method == "Webcam":
        # Webcam input
        st.write("Webcam Feed")
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            processed_frame = process_image(frame)
            FRAME_WINDOW.image(processed_frame)
        else:
            st.write('Stopped')
            camera.release()

if __name__ == "__main__":
    main()
