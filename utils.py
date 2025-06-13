import os
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import joblib
from sklearn.preprocessing import LabelEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def recognize_face(embedding, clf, label_encoder, threshold=0.7):
    """Recognize face using the classifier and return label and probability."""
    probs = clf.predict_proba([embedding])[0]
    max_prob = np.max(probs)
    predicted_label = label_encoder.inverse_transform([np.argmax(probs)])[0]

    if max_prob < threshold:
        return "Intruder", max_prob

    return predicted_label, max_prob

def extract_embedding(img):
    face = mtcnn(img)
    if face is not None:
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0).to(device))
            return embedding.squeeze().cpu().numpy()
    return None

def detect_and_recognize(frame, classifier, label_encoder):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]

            embedding = extract_embedding(face)
            if embedding is not None:
                preds = classifier.predict_proba([embedding])[0]
                label_idx = np.argmax(preds)
                confidence = preds[label_idx]

                name = label_encoder.inverse_transform([label_idx])[0] if confidence > 0.8 else "Unknown"
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{name} ({confidence:.2f})', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

def add_new_student(student_name):
    save_path = os.path.join("data", "student_db", student_name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while count < 10:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]

                img_path = os.path.join(save_path, f"{count}.jpg")
                cv2.imwrite(img_path, face)
                count += 1

                if count >= 10:
                    break

    cap.release()
    cv2.destroyAllWindows()

def load_images_and_labels():
    base_path = "data/student_db"
    embeddings = []
    labels = []

    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        print("Created student database directory. Please add student images first.")
        return [], [], None

    for student in os.listdir(base_path):
        student_path = os.path.join(base_path, student)
        if not os.path.isdir(student_path):
            continue

        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            embedding = extract_embedding(img)
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(student)

    if not embeddings:
        print("No valid images found. Please add student images first.")
        return [], [], None

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    return embeddings, labels_encoded, label_encoder
