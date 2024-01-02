import tkinter as tk
import cv2
import numpy as np
from keras.models import model_from_json
import threading
from tkinter import filedialog

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into the new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

# Create a global variable to store the selected file path
selected_image_path = ""

# Function to process video stream
def process_video_stream():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        process_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to process a single image
def process_single_image():
    frame = cv2.imread(selected_image_path)
    frame = cv2.resize(frame, (1280, 720))
    process_frame(frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to process a frame (common logic for both video and image)
def process_frame(frame):
    face_detector = cv2.CascadeClassifier('Emotion_detection_with_CNN\Emotion_detection_with_CNN-main\haarcascades\haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)

# Function to open a file dialog and set the selected image path
def choose_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename()
    print(f"Selected Image Path: {selected_image_path}")

# Create the main window
window = tk.Tk()

# Set the window title
window.title("Emotion Detection Example")

# Function to process media based on the selected mode
def process_media(processing_mode):
    if processing_mode == "Video":
        threading.Thread(target=process_video_stream).start()
    elif processing_mode == "Image":
        choose_image()
        process_single_image()

# Create buttons
video_button = tk.Button(window, text="Process Video", command=lambda: process_media("Video"))
image_button = tk.Button(window, text="Process Image", command=lambda: process_media("Image"))

# Pack the buttons
video_button.pack()
image_button.pack()

# Start the Tkinter event loop
window.mainloop()
