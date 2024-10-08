import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import tempfile

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

    input_option = st.radio("Choose input method:", ("Upload Image", "Upload Video", "Use Webcam"))

    if input_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Detect Objects"):
                processed_image = process_frame(image)
                st.image(processed_image, caption="Processed Image", use_column_width=True)

    elif input_option == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            vf = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = process_frame(Image.fromarray(frame))
                stframe.image(processed_frame, caption="Processed Video", use_column_width=True)
            vf.release()

    else:  # Use Webcam
        st.write("Click 'Start' to begin object detection using your webcam.")
        run = st.checkbox("Start")
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = process_frame(Image.fromarray(frame))
            FRAME_WINDOW.image(processed_frame)
        camera.release()

if __name__ == "__main__":
    main()
