import face_recognition
import os
import cv2
from google.colab.patches import cv2_imshow

KNOWN_FACES_DIR = "knownFaces"
UNKNOWN_FACES_DIR = "unknownFaces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'

print("loading known faces...")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
  for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
    
    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
    encoding = face_recognition.face_encodings(image)

    if not len(encoding):
            print(filename, "can't be encoded")
            continue
    
    known_faces.append(encoding)
    known_names.append(name)

print("processing unknown faces...")
for filename in os.listdir(UNKNOWN_FACES_DIR):
  print(filename)
  
  image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
  locations = face_recognition.face_locations(image, model = MODEL)
  encodings = face_recognition.face_encodings(image, locations)
  
  if not len(encodings):
    print(filename, "can't be encoded")
    continue
  
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  for face_encoding, face_location in zip(encodings, locations):
    for faces in known_faces:
      results = face_recognition.compare_faces(faces, face_encoding, TOLERANCE)
           
      if True in results:
        match = known_names[results.index(True)]
      else:
        match = "Unknown face"

      top_left = (face_location[3], face_location[0])
      bottom_right = (face_location[1], face_location[2])

      color = [255, 0, 0]

      cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

      top_left = (face_location[3], face_location[2])
      bottom_right = (face_location[1], face_location[2] + 22)

      cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
      cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
  
  cv2_imshow(image)
  cv2.waitKey(10000)  



