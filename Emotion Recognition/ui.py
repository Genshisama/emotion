import tkinter as tk
import cv2
import numpy as np
from keras.models import model_from_json
import threading
from tkinter import filedialog
from customtkinter import *
import customtkinter

# Emotion labels dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load pre-trained emotion detection model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model.h5")
print("Loaded model from disk")

# Global variable to store selected file paths
selected_image_path = ""
selected_video_path = ""

# Function to process live video stream
def process_live_stream():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        width = 720
        height = int(frame.shape[0] * (width / frame.shape[1]))
        frame = cv2.resize(frame, (width, height))
        if not ret:
            break
        process_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to process video stream
def process_video_stream():
    cap = cv2.VideoCapture(selected_video_path)
    while True:
        ret, frame = cap.read()
        width = 720
        height = int(frame.shape[0] * (width / frame.shape[1]))
        frame = cv2.resize(frame, (width, height))
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
    width = 720
    height = int(frame.shape[0] * (width / frame.shape[1]))
    frame = cv2.resize(frame, (width, height))
    process_frame(frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to process a frame (common logic for both video and image)
def process_frame(frame):
    face_detector = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
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

# Function to open a file dialog and set the selected video path
def choose_video():
    global selected_video_path
    selected_video_path = filedialog.askopenfilename()
    print(f"Selected Video Path: {selected_video_path}")

# Function to open a file dialog and set the selected image path
def choose_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename()
    print(f"Selected Image Path: {selected_image_path}")

# Create the main window
window = CTk()
window.geometry("1280x720")
window.eval('tk::PlaceWindow %s center' % window.winfo_toplevel())
window.title("Emotion Detection Example")

frame = customtkinter.CTkFrame(master=window, fg_color="#F3F0F0", corner_radius=20)
frame.grid(row=0, column=0, pady=50, padx=50, sticky="nsew")
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

frame2 = customtkinter.CTkFrame(master=frame, fg_color="#A3A2A2", corner_radius=20)
frame2.grid(row=0, column=0, pady=180, padx=200, sticky="nsew")
frame.grid_rowconfigure(0, weight=2)
frame.grid_columnconfigure(0, weight=2)
frame2.grid_rowconfigure(0, weight=1)
frame2.grid_columnconfigure(0, weight=1)
frame2.place(relx=0.5, rely=0.4, anchor="center")

label = CTkLabel(master=frame2, text="EMOTION RECOGNITION", font=("Arial", 50, "bold"), text_color="#F3F8FF")
label.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")

def update_button_size(button):
    width = len(label.cget("text")) * 10  
    button.configure(width=width)  

def process_media(processing_mode):
    if processing_mode == "Live":
        threading.Thread(target=process_live_stream).start()
    elif processing_mode == "Video":
        choose_video()
        process_video_stream()
    elif processing_mode == "Image":
        choose_image()
        process_single_image()

live_button = CTkButton(master=frame, text="Live Video", font=("Arial", 20), corner_radius=32, fg_color="#64A2FF", hover_color="#5ECEFE", command=lambda: process_media("Live"))
video_button = CTkButton(master=frame, text="Process Video", font=("Arial", 20), corner_radius=32, fg_color="#64A2FF", hover_color="#5ECEFE", command=lambda: process_media("Video"))
image_button = CTkButton(master=frame, text="Process Image", font=("Arial", 20), corner_radius=32, fg_color="#64A2FF", hover_color="#5ECEFE", command=lambda: process_media("Image"))

live_button.grid(row=0, column=0, padx=10, pady=10)
video_button.grid(row=1, column=0, padx=10, pady=10)
image_button.grid(row=2, column=0, padx=10, pady=10)

live_button.place(relx=0.5, rely=0.6, anchor="center")
video_button.place(relx=0.5, rely=0.68, anchor="center")
image_button.place(relx=0.5, rely=0.76, anchor="center")

window.mainloop()
