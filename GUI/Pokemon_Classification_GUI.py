# Importing Libraries
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from pygame import mixer
from PIL import Image, ImageTk
from keras import models


# Function to Upload an Image to Classify
def upload_image():
    path = filedialog.askopenfilename()
    img = Image.open(path)
    img.thumbnail((400, 300))
    imgtk = ImageTk.PhotoImage(img)
    uploaded_image.configure(image=imgtk)
    uploaded_image.image = imgtk
    classify(path)


# Function to Classify the Uploaded Image
def classify(file_path):
    img_array = cv2.imread(file_path)
    rbg_img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    resized_img_array = cv2.resize(rbg_img_array, (64, 64))
    np_img_array = (np.array([resized_img_array])) / 255
    prediction = model.predict(np_img_array)[0]
    prediction = np.argmax(prediction)
    predicted_label.configure(text="It's "+classes[prediction])


# Making a List of Classes and Loading Model
classes = os.listdir("StartersSample")
model = models.load_model("StartersSample_Model.h5")

# Starting Background Music
mixer.init()
mixer.music.load("BGM.mp3")
mixer.music.play()

# GUI Initialization
window = tk.Tk()
window.title("Pokémon Classification Using CNN")
window_width = 800
window_height = 600
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
center_x = int(screen_width/2 - window_width/2)
center_y = int(screen_height/2 - window_height/2)
window.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
window.resizable(False, False)
window.iconbitmap("Icon.ico")
window.configure(background="black")

# Heading
heading = Label(window, text="Who's That Pokémon", padx=20, pady=10)
heading.configure(background="black", foreground="#FFC80A", font=('rockwell', 20, 'bold'))
heading.pack(side=TOP, pady=20)

# Upload Button
upload_button = Button(window, text="Upload a Pokémon", command=upload_image, padx=10, pady=5)
upload_button.configure(background="#3264C8", foreground="white", font=('rockwell', 10, 'bold'))
upload_button.pack(side=BOTTOM, pady=30)

# Predicted Label
predicted_label = Label(window, text="", padx=15, pady=5)
predicted_label.configure(background="black", foreground="#62FF32", font=('rockwell', 15, 'bold'))
predicted_label.pack(side=BOTTOM, pady=10)

# Uploaded Image
uploaded_image = Label(window)
uploaded_image.configure(background="black")
uploaded_image.pack(side=BOTTOM, expand=True)

window.mainloop()
