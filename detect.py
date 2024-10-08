import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import tempfile
import time

# Set page config
st.set_page_config(page_title="YOLO Object Detection", layout="wide")

# Custom CSS to improve aesthetics
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stRadio > label {
        background-color: #f1f1f1;
        padding: 10px;
        border-radius: 4px;
        cursor: pointer;
    }
    .stRadio > label:hover {
        background-color: #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # Load a pretrained YOLOv8 model

model = load_model()

def process_frame(frame):
    results = model(frame)
    img_array = np.array(frame)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for box, cls in zip(boxes, classes):
            cv2.rectangle(img_array, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img_array, result.names[cls], (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return Image.fromarray(img_array)

def main():
    st.title("YOLO Object Detection App")
    st.write("This app uses YOLOv8 to detect objects in images and videos.")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        input_option = st.radio("Choose input method:", ("Upload Image", "Upload Video", "Use Webcam"))

    with col2:
        if input_option == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                if st.button("Detect Objects"):
                    with st.spinner('Processing...'):
                        processed_image = process_frame(image)
                    st.success('Done!')
                    st.image(processed_image, caption="Processed Image", use_column_width=True)

        elif input_option == "Upload Video":
            uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                vf = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                stop_button = st.button("Stop Processing")
                while vf.isOpened() and not stop_button:
                    ret, frame = vf.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frame = process_frame(Image.fromarray(frame))
                    stframe.image(processed_frame, caption="Processed Video", use_column_width=True)
                    time.sleep(0.1)  # Add a small delay to make it more responsive
                vf.release()

        else:  # Use Webcam
            st.write("Click 'Start' to begin object detection using your webcam.")
            run = st.checkbox("Start")
            
            if run:
                try:
                    FRAME_WINDOW = st.empty()
                    camera = cv2.VideoCapture(0)
                    if not camera.isOpened():
                        raise IOError("Cannot open webcam")
                    
                    while run:
                        ret, frame = camera.read()
                        if not ret:
                            st.error("Failed to capture frame from webcam")
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed_frame = process_frame(Image.fromarray(frame))
                        FRAME_WINDOW.image(processed_frame, caption="Webcam Feed", use_column_width=True)
                        time.sleep(0.1)  # Add a small delay to make it more responsive
                    
                    camera.release()
                except Exception as e:
                    st.error(f"Error accessing webcam: {str(e)}")
                    st.error("The webcam cannot be accessed. This may be due to the server not being connected to a camera.")

if __name__ == "__main__":
    main()
