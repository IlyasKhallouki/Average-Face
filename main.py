import cv2
import dlib
import os
import numpy as np

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Path to the folder containing images
folder_path = "pics"
output_path = "output"

all_landmarks = []

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

def crop_and_resize(image, landmarks, image_name, target_size=(600, 600), left_eye=(180, 200), right_eye=(420, 200)):
    landmarks = np.array(landmarks)

    eyes_center = np.mean(landmarks[36:48], axis=0)

    left_eye = np.array(left_eye)
    right_eye = np.array(right_eye)

    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    eye_distance = np.linalg.norm(right_eye - left_eye)
    landmarks_distance = np.linalg.norm(landmarks[45] - landmarks[36])
    scale_factor = eye_distance / landmarks_distance if landmarks_distance != 0 else 1.0
    rotation_matrix = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale_factor)

    cropped_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    translation = (left_eye[0] - eyes_center[0], left_eye[1] - eyes_center[1])

    cropped_image = cv2.warpAffine(cropped_image, np.float32([[1, 0, translation[0]], [0, 1, translation[1]]]), (image.shape[1], image.shape[0]))

    x1 = max(int(left_eye[0] - target_size[0] / 2), 0)
    x2 = min(int(left_eye[0] + target_size[0] / 2), image.shape[1])
    y1 = max(int(left_eye[1] - target_size[1] / 2), 0)
    y2 = min(int(left_eye[1] + target_size[1] / 2), image.shape[0])

    if x1 >= x2 or y1 >= y2:
        print(f"Invalid cropping coordinates for image {image_name}")
        return
    
    cropped_image = cropped_image[y1:y2, x1:x2]
    print("Cropped image dimensions:", cropped_image.shape)

    local_output_path = os.path.join(output_path, image_name)
    cv2.imwrite(local_output_path, cropped_image)

    

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Unable to read image: {image_path}")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)

        landmarks_for_face = []

        for n in range(68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            landmarks_for_face.append((x, y))

        all_landmarks.append(landmarks_for_face)
        crop_and_resize(frame, landmarks_for_face, image_file)
