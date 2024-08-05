from deepface import DeepFace

img1 = '/home/developer/Desktop/fr/Previous/dataset/5235/5235.jpg'
img2 = '/home/developer/Desktop/fr/Previous/dataset/5273/5273.jpg'

result = DeepFace.verify(
  img1_path = img1,
  img2_path = img2,
)

print(result)