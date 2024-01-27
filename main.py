import cv2
import dlib
import os
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

folder_path = "pics"
output_path = "output"

def crop_to_face(image, face_rect):
    x, y, w, h = face_rect
    cropped_image = image[y-100:y+h+100, x-100:x+w+100]
    return cropped_image

def get_face_rect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    
    if len(faces) > 0:
        # Use the first detected face
        return (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
    else:
        # If no face is detected, return a default rectangle (entire image)
        return (0, 0, image.shape[1], image.shape[0])

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

def warp_image(image, src_points, dst_points, target_size=(800, 800)):
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)

    # Get the transformation matrix
    warp_matrix = cv2.getAffineTransform(src_points, dst_points)

    # Warp the image
    warped_triangle = cv2.warpAffine(image, warp_matrix, (target_size[0], target_size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return warped_triangle


all_landmarks = []
target_size=(800, 800)

boundary_points = np.array([(0, 0), (0, target_size[1] - 1), (target_size[0] - 1, 0), (target_size[0] - 1, target_size[1] - 1),
                            (target_size[0] // 2, 0), (target_size[0] // 2, target_size[1] - 1),
                            (0, target_size[1] // 2), (target_size[0] - 1, target_size[1] // 2)], dtype=np.int32)

image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Unable to read image: {image_path}")
        continue

    face_rect = get_face_rect(frame)

    cropped_frame = crop_to_face(frame, face_rect)

    local_output_path = os.path.join(output_path, image_file)
    cv2.imwrite(local_output_path, cropped_frame)
    print(f"Image {image_file} croped and saved")

    all_landmarks.extend(get_landmarks(cropped_frame))

average_landmarks = np.mean(np.array(all_landmarks), axis=0)

all_points = np.vstack((average_landmarks, boundary_points))

triangles = Delaunay(all_points)

triangle_indices = triangles.simplices

image_output = [f for f in os.listdir(output_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
warped_images = []

for image_name in image_output:
    image_path = os.path.join(output_path, image_name)
    frame = cv2.imread(image_path)

    image_landmarks = np.array(get_landmarks(frame))[0]
    image_all_points = np.vstack((image_landmarks, boundary_points))


    if frame is None:
        print(f"Unable to read image: {image_path}")
        continue

    warped_image = np.zeros_like(frame, dtype=np.float32)

    for triangle_index in triangle_indices:
        src_points = image_all_points[triangle_index]
        dst_points = all_points[triangle_index]

        warped_triangle = warp_image(frame, src_points, dst_points, target_size=frame.shape[:2])
        
        warped_triangle_resized = cv2.resize(warped_triangle, (frame.shape[1], frame.shape[0]))

        warped_image += warped_triangle_resized.astype(np.float32)

    warped_image /= len(triangle_indices)

    warped_images.append(warped_image.astype(np.uint8))

    local_output_path = os.path.join(output_path, image_name)
    cv2.imwrite(local_output_path, warped_image)

min_height = min(img.shape[0] for img in warped_images)

# Resize all images to have the same height
images_resized = [cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height)) for img in warped_images]

average_face = np.mean(images_resized, axis=0)
cv2.imwrite("output/average.jpeg", average_face)