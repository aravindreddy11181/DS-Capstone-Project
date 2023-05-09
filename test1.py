import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
emotion_model = load_model('model_file.h5')

# Open the default camera
camera = cv2.VideoCapture(0)

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml.txt')

# Define the emotion labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Process each frame from the camera
while True:
    # Read a frame from the camera
    ret, frame = camera.read()

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region from the grayscale frame
        face_img = gray[y:y+h, x:x+w]

        # Resize the face image to the input size of the model
        resized_face_img = cv2.resize(face_img, (48, 48))

        # Normalize the pixel values to the range [0, 1]
        normalized_face_img = resized_face_img / 255.0

        # Reshape the normalized image to match the input shape of the model
        reshaped_face_img = np.reshape(normalized_face_img, (1, 48, 48, 1))

        # Use the model to predict the emotion label for the face image
        predicted_emotion = emotion_model.predict(reshaped_face_img)

        # Determine the index of the highest probability emotion label
        predicted_emotion_index = np.argmax(predicted_emotion, axis=1)[0]

        # Draw a rectangle around the face region
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

        # Draw a filled rectangle behind the emotion label text
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)

        # Draw the emotion label text on top of the filled rectangle
        cv2.putText(frame, emotion_labels[predicted_emotion_index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the frame on the screen
    cv2.imshow("Emotion Detection", frame)

    # Check for user input to quit the program
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
