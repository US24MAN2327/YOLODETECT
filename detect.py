import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load YOLO model
weights_path = 'yolov3.weights'  # Path to YOLOv3 weights
config_path = 'yolov3.cfg'  # Path to YOLOv3 config
labels_path = 'coco.names'  # Path to COCO class labels

# Load YOLO network and COCO labels
@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, labels, output_layers

net, labels, output_layers = load_model()

def process_image(image):
    (H, W) = image.shape[:2]

    # Create a 4D blob from the image and perform a forward pass to get detections
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Process each output layer
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Filter out weak predictions
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # Calculate top-left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in np.random.randint(0, 255, size=(3,))]
            label = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            # Draw bounding box and label
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def main():
    st.title("YOLO Object Detection App")

    st.write("This app uses YOLOv3 to detect objects in images.")

    upload_option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Detect Objects"):
                image_np = np.array(image.convert('RGB'))
                processed_image = process_image(image_np)
                st.image(processed_image, caption="Processed Image", use_column_width=True)

    else:  # Use Webcam
        st.write("Click 'Start' to begin object detection using your webcam.")
        run = st.checkbox("Start")
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = process_image(frame)
            FRAME_WINDOW.image(processed_frame)

        camera.release()

if __name__ == "__main__":
    main()
