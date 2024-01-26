import cv2
import dlib
import os

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Path to the folder containing images
folder_path = "pics"
output_path = "output"

all_landmarks = []

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    # Read the image
    image_path = os.path.join(folder_path, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Unable to read image: {image_path}")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = hog_face_detector(gray)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)

        # List to store landmarks for each face
        landmarks_for_face = []

        for n in range(68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            landmarks_for_face.append((x, y))

        all_landmarks.append(landmarks_for_face)

average_landmarks = []

#Find average landmarks
for i in range(68):
    x, y = 0, 0
    for j in all_landmarks:
        x+=j[i][0]
        y+=j[i][1]
    x//=len(all_landmarks)
    y//=len(all_landmarks)

    average_landmarks.append((x, y))

print(average_landmarks)