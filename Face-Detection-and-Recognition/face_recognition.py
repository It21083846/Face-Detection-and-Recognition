from deepface import DeepFace
import json

result = DeepFace.find(img_path="anne.jpg", db_path="frames")
print(result)
