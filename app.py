import streamlit as st
import cv2
import numpy as np
import joblib
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils import recognize_face  # get_embedding_from_face not needed here anymore

st.set_page_config(page_title="Real-Time Face Recognition", layout="wide")
st.title("Real-Time Student Face Recognition with Intruder Detection")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN and FaceNet models once
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load classifier and label encoder
clf = joblib.load('models/classifier.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

THRESHOLD = 0.7

frame_placeholder = st.empty()

if 'run' not in st.session_state:
    st.session_state.run = False

start_button = st.button("Start Camera")
stop_button = st.button("Stop Camera")

if start_button:
    st.session_state.run = True
if stop_button:
    st.session_state.run = False

cap = cv2.VideoCapture(0)

def process_frame(frame):
    boxes, _ = mtcnn.detect(frame)
    faces = mtcnn(frame)  # This returns list of face tensors aligned and cropped

    if boxes is not None and faces is not None:
        h, w, _ = frame.shape
        for box, face in zip(boxes, faces):
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if face is None:
                label, prob = "No Face", 0.0
            else:
                with torch.no_grad():
                    embedding = resnet(face.unsqueeze(0).to(device))
                embedding_np = embedding[0].cpu().numpy()
                label, prob = recognize_face(embedding_np, clf, label_encoder, threshold=THRESHOLD)

            print(f"Detected: {label} with confidence {prob:.4f}")

            color = (0, 255, 0) if label not in ["Intruder", "No Face"] else (0, 0, 255)
            text = f"{label}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        print("[DEBUG] No faces detected in frame.")
    return frame

while st.session_state.run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to grab frame")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = process_frame(frame)
    frame_placeholder.image(frame)

cap.release()
frame_placeholder.empty()
