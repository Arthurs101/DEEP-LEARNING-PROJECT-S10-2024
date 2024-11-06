import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from InteractiveModel import Simpsins_CNN

class SimpsonRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simpson Character Recognition")
        self.model = Simpsins_CNN("model_weightsk.h5")
        
        # Widgets
        self.load_button = Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.image_label = Label(root)
        self.image_label.pack()

        self.prediction_label = Label(root, text="", font=("Arial", 16))
        self.prediction_label.pack()

        self.saliency_button = Button(root, text="Show Saliency Map", command=self.show_saliency_map, state='disabled')
        self.saliency_button.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")]) # Change type if we are only working with gifs
        if file_path:
            self.display_image(file_path)
            self.predict_character(file_path)
            self.image_path = file_path
            self.saliency_button['state'] = 'normal'

    def display_image(self, file_path):
        img = Image.open(file_path).resize((128, 128))
        img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img)
        self.image_label.image = img  # Keep a reference to avoid garbage collection

    def predict_character(self, file_path):
        prediction = self.model.input_image(file_path)
        self.prediction_label.config(text=f"Predicted Character: {prediction}") # Print the character prediction

    def show_saliency_map(self):
        if hasattr(self, 'image_path'):
            self.model.get_saliency_map(self.image_path)
            plt.show() # Print the saliency map
