import torch
from transformers import AutoModel, AutoTokenizer
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import cv2

# Initialize MTCNN for face detection and alignment
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

# Initialize FaceNet model from Hugging Face
facenet_model = AutoModel.from_pretrained('timesler/facenet')
facenet_tokenizer = AutoTokenizer.from_pretrained('timesler/facenet')

def detect_and_align_faces(image):
    # Convert PIL image to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    boxes, _ = mtcnn.detect(image)
    aligned_faces = []
    if boxes is not None:
        for box in boxes:
            face = image.crop(box)
            aligned_faces.append(face)
    return aligned_faces

def get_face_embeddings(aligned_faces):
    embeddings = []
    for face in aligned_faces:
        face = face.resize((160, 160))
        face_tensor = mtcnn(face)
        if face_tensor is not None:
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = facenet_model(face_tensor)[0]
            embeddings.append(embedding.cpu().numpy())
    return np.array(embeddings)

def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    aligned_faces = detect_and_align_faces(image)
    embeddings = get_face_embeddings(aligned_faces)
    return embeddings

# Test the pipeline
image_path = '5235.jpg'
embeddings = process_image(image_path)
print(embeddings)
