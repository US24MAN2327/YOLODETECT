import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # Load a pretrained YOLOv8 model

model = load_model()

def process_image(image):
    results = model(image)
    
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Plot the results on the image
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, cls in zip(boxes, classes):
            cv2.rectangle(img_array, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img_array, result.names[cls], (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return Image.fromarray(img_array)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()  # Placeholder for video frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        processed_frame = process_image(frame_pil)
        
        # Display the processed frame in the Streamlit app
        stframe.image(processed_frame, use_column_width=True)

    cap.release()

def main():
    st.title("YOLO Object Detection App")

    st.write("This app uses YOLOv8 to detect objects in images or videos.")

    upload_option = st.radio("Choose input method:", ("Upload Image", "Use Webcam", "Upload Video"))

    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Detect Objects"):
                processed_image = process_image(image)
                st.image(processed_image, caption="Processed Image", use_column_width=True)

    elif upload_option == "Upload Video":
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            st.video(uploaded_video)
            if st.button("Detect Objects in Video"):
                process_video(uploaded_video.name)

    else:  # Use Webcam
        st.write("Click 'Start' to begin object detection using your webcam.")
        run = st.checkbox("Start")
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            processed_frame = process_image(frame)
            FRAME_WINDOW.image(processed_frame)

        camera.release()

if __name__ == "__main__":
    main()
