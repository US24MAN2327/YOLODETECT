import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # Load a pretrained YOLOv8 model

model = load_model()

class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        # Convert the frame to a numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")

        # Perform object detection
        results = self.model(img)

        # Plot the results on the image
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, cls in zip(boxes, classes):
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img, results[0].names[cls], (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img

def main():
    st.title("YOLO Object Detection App")

    st.write("This app uses YOLOv8 to detect objects in images.")

    upload_option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Detect Objects"):
                processed_image = process_image(image)
                st.image(processed_image, caption="Processed Image", use_column_width=True)

    else:  # Use Webcam
        st.write("Click 'Start' to begin object detection using your webcam.")
        webrtc_streamer(key="example", video_transformer_factory=YOLOVideoTransformer)

if __name__ == "__main__":
    main()
