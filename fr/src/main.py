from deepface import DeepFace

img1 = '/fr/Previous/dataset/5235/5235.jpg'
img2 = '/fr/Previous/dataset/5273/5273.jpg'
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

# result = DeepFace.verify(
#   img1_path = img1,
#   img2_path = img2,
# )

embedding_objs = DeepFace.represent(
  img_path = img1,
  model_name = models[2],
)

print(len(embedding_objs[0]["embedding"]))