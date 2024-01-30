import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the image detector model from TensorFlow Hub
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

def run_inference_for_single_image(model, image):
    input_tensor = tf.convert_to_tensor([image])
    output_dict = model(input_tensor)

    return output_dict

def draw_boxes(image, boxes, classes, scores, threshold):
    for i in range(boxes.shape[0]):
        if scores[i] > threshold:
            box = tuple(boxes[i].tolist())
            image = cv2.rectangle(image, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 255, 0), 2)
            label = f"{class_names[int(classes[i])]}: {int(100 * scores[i])}%"
            image = cv2.putText(image, label, (int(box[1]), int(box[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Load COCO class names
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

st.title("Object Detection with Streamlit using TensorFlow Hub")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

    # Run inference
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_dict = run_inference_for_single_image(detector, input_image)

    # Draw bounding boxes
    image_with_boxes = draw_boxes(image.copy(), output_dict['detection_boxes'][0], output_dict['detection_classes'][0], output_dict['detection_scores'][0], 0.5)

    # Display results
    st.image(image_with_boxes, channels="BGR", caption="Processed Image", use_column_width=True)
