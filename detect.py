import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import base64
from io import BytesIO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

def process_image(image):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Ensure the image is in RGB format
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif img_array.shape[-1] == 3:  # RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Perform object detection
    results = model(img_array)
    
    # Plot the results on the image
    res_plotted = results[0].plot()
    return Image.fromarray(res_plotted)

def main():
    st.title("YOLOv8 Object Detection")
    
    # JavaScript to capture webcam image
    js_code = """
    <script>
    const captureImage = () => {
        const canvas = document.createElement('canvas');
        const video = document.querySelector('video');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg');
        const data = {image: dataUrl};
        window.parent.postMessage({type: 'CAPTURED_IMAGE', data: JSON.stringify(data)}, '*');
    };
    
    navigator.mediaDevices.getUserMedia({video: true})
        .then((stream) => {
            const videoElement = document.createElement('video');
            videoElement.srcObject = stream;
            videoElement.autoplay = true;
            document.body.appendChild(videoElement);
            
            const captureButton = document.createElement('button');
            captureButton.textContent = 'Capture';
            captureButton.onclick = captureImage;
            document.body.appendChild(captureButton);
        })
        .catch((err) => {
            console.error("Error accessing webcam:", err);
        });
    </script>
    """
    
    st.components.v1.html(js_code, height=300)
    
    captured_image = st.empty()
    
    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None
    
    # Handle the captured image
    if captured_image_b64 := st.experimental_get_query_params().get('captured_image'):
        captured_image_b64 = captured_image_b64[0].split(',')[1]
        image_data = base64.b64decode(captured_image_b64)
        st.session_state.captured_image = Image.open(BytesIO(image_data))
    
    if st.session_state.captured_image:
        captured_image.image(st.session_state.captured_image, caption='Captured Image', use_column_width=True)
        
        if st.button('Detect Objects'):
            processed_image = process_image(st.session_state.captured_image)
            st.image(processed_image, caption='Processed Image', use_column_width=True)

if __name__ == "__main__":
    main()
