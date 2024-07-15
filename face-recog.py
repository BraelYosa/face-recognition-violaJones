import cv2
from IPython.display import display
import ipywidgets as widgets
import numpy as np
import face_recognition
import os
import time

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize AdaBoost classifier for face detection
ada_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Function to load and encode known faces
def load_and_encode_faces(folder_path):
    known_face_encodings = []
    known_face_names = []
    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                    image_path = os.path.join(person_folder, filename)
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if len(face_encodings) > 0:  # Check if at least one face encoding is found
                        face_encoding = face_encodings[0]  # Assuming one face per image
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(person_name)  # Use the folder name as the person's name
                    else:
                        print(f"No face found in image {filename}")
    return known_face_encodings, known_face_names

# Load known faces
known_face_encodings, known_face_names = load_and_encode_faces('my_faces')

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Create a widget to display the video frames
image_widget = widgets.Image(format='jpeg')

display(image_widget)

# Resize frame for faster processing and then scale back for display
scale_factor = 0.25  # Process at 25% of the original size

try:
    while True:
        start_time = time.time()

        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        # Convert the frame to grayscale for Haar cascades
        gray_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar cascades (AdaBoost)
        faces_ada = ada_cascade.detectMultiScale(gray_small_frame, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

        # Detect faces using Haar cascades (original)
        faces_haar = face_cascade.detectMultiScale(gray_small_frame, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

        # Ensure both face detection results have the same shape
        if len(faces_ada) == 0:
            faces_ada = np.empty((0, 4), dtype=int)
        if len(faces_haar) == 0:
            faces_haar = np.empty((0, 4), dtype=int)

        # Combine both sets of faces detected
        faces = np.concatenate((faces_ada, faces_haar), axis=0)

        # Convert the frame to RGB (face_recognition uses RGB)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Scale coordinates back up for display on the original frame
            x *= int(1/scale_factor)
            y *= int(1/scale_factor)
            w *= int(1/scale_factor)
            h *= int(1/scale_factor)

            # Extract the face region from the frame
            face_frame = frame[y:y+h, x:x+w]

            # Convert the face region to RGB for face_recognition
            rgb_face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

            # Encode the face region
            face_encoding = face_recognition.face_encodings(rgb_face_frame)
            if len(face_encoding) > 0:
                face_encoding = face_encoding[0]

                # Compare with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # If a match was found, use the known face with the smallest distance
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), (255, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (x + 6, y + h - 6), font, 1.0, (255, 255, 255), 1)

        # Convert the frame to JPEG
        _, jpeg = cv2.imencode('.jpeg', frame)
        image_widget.value = jpeg.tobytes()

        # Add a delay to limit the frame rate to 10 FPS
        elapsed_time = time.time() - start_time
        if elapsed_time < 0.1:
            time.sleep(0.1 - elapsed_time)
except KeyboardInterrupt:
    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()
