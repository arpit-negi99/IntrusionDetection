import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.preprocessing import LabelEncoder

# Initialize face detector and embedder once for training
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def load_images_and_labels(data_dir='data/student_db'):
    embeddings = []
    labels = []
    label_encoder = LabelEncoder()

    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            face = mtcnn(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if face is None:
                continue
            with torch.no_grad():
                embedding = resnet(face.unsqueeze(0))
            embeddings.append(embedding[0].numpy())
            labels.append(person_name)

    labels_encoded = label_encoder.fit_transform(labels)
    return np.array(embeddings), labels_encoded, label_encoder

def recognize_face(embedding, classifier, label_encoder, threshold=0.7):
    if embedding is None:
        return "Intruder", 0.0
    probs = classifier.predict_proba([embedding])[0]
    max_idx = np.argmax(probs)
    max_prob = probs[max_idx]
    if max_prob >= threshold:
        return label_encoder.inverse_transform([max_idx])[0], max_prob
    else:
        return "Intruder", max_prob
