import cv2
import dlib
import os
import numpy as np

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

folder_path = "pics"
output_path = "output"

def crop_and_resize(image, landmarks, image_name, target_size=(800, 800), border_removal=True):
    landmarks = np.array(landmarks)

    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)

    eyes_center = np.mean([left_eye, right_eye], axis=0).astype('int')

    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    desired_face_width = target_size[0] * 0.25
    current_eye_dist = np.linalg.norm(right_eye - left_eye)
    scale_factor = desired_face_width / current_eye_dist

    eyes_center = tuple(map(float, eyes_center))

    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale_factor)

    rotated_and_scaled_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    new_eyes_center = np.matmul(rotation_matrix, np.array([eyes_center[0], eyes_center[1], 1]))

    translation_matrix = np.float32([[1, 0, target_size[0] / 2 - new_eyes_center[0]], 
                                     [0, 1, target_size[1] / 2 - new_eyes_center[1]]])
    translated_image = cv2.warpAffine(rotated_and_scaled_image, translation_matrix, target_size)

    if border_removal:
        mask = cv2.cvtColor(translated_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        translated_image = translated_image[y:y+h, x:x+w]

    output_file_path = os.path.join(output_path, image_name)
    cv2.imwrite(output_file_path, translated_image[:, 50:-49])
    
    print(f"Cropped image saved to {output_file_path}")
    return translated_image

def get_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)

    landmarks = []

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)

        landmarks_for_face = []

        for n in range(68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            landmarks_for_face.append((x, y))
        
        landmarks.append(landmarks_for_face)

    return landmarks

all_landmarks = []

image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Unable to read image: {image_path}")
        continue
    
    image_landmarks = get_landmarks(frame)

    cropped_frame = crop_and_resize(frame, image_landmarks[0], image_file)

    all_landmarks.extend(get_landmarks(cropped_frame))

average_landmarks = np.mean(np.array(all_landmarks), axis=0)