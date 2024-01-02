import tkinter as tk
import subprocess

def onClick():
    print("clicked")
    subprocess.run(['python', 'Emotion_detection_with_CNN\Emotion_detection_with_CNN-main\TestEmotionDetector.py'])

window = tk.Tk()

window.title("Emotion Detection")

window.geometry("400x300")

button1 = tk.Button(window, text="Video",command=onClick)
button2 = tk.Button(window, text="image",command=onClick)
button1.pack()
button2.pack()

window.mainloop()
