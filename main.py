import cv2
import dlib
import os

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Path to the folder containing images
folder_path = "pics"
output_path, output_number = "output", 0

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

    # List to store facial landmarks for each face in the image
    all_landmarks = []

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)

        # List to store landmarks for each face
        landmarks_for_face = []

        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            landmarks_for_face.append((x, y))

        # Append the landmarks for the current face to the list
        all_landmarks.append(landmarks_for_face)

    # Visualize landmarks on the image (optional)
    for landmarks_for_face in all_landmarks:
        for (x, y) in landmarks_for_face:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

    # Display the image with landmarks (optional)
    cv2.imwrite(f'{output_path}/{output_number}.png', frame)
    output_number+=1
