import joblib
from sklearn.svm import SVC
from utils import load_images_and_labels

def train():
    print("Loading images and extracting embeddings...")
    embeddings, labels, label_encoder = load_images_and_labels()

    print(f"Training classifier on {len(embeddings)} samples...")
    clf = SVC(kernel='linear', probability=True)
    clf.fit(embeddings, labels)

    joblib.dump(clf, 'models/classifier.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    print("Training complete and models saved.")

if __name__ == "__main__":
    train()
