import tkinter as tk
from tkinter import messagebox
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import torch.nn as nn

# Define the model architecture (same as used during training)
class ShapeCNN(nn.Module):
    def __init__(self):
        super(ShapeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 input channels for RGB
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 classes (Cross, Square, Triangle)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Load the trained model weights
model = ShapeCNN()  # Initialize the model
model.load_state_dict(torch.load("model/shape_cnn.pth"))  # Load the saved weights
model.eval()  # Set the model to evaluation mode

# Define classes (as you mentioned, Cross, Square, Triangle)
classes = ['Cross', 'Square', 'Triangle']

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Prediction function
def predict_image(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = model(image)
    _, predicted = torch.max(output, 1)  # Get the index of the predicted class
    return classes[predicted.item()]

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shape Predictor")

        self.canvas = tk.Canvas(self.root, width=256, height=256, bg="white")
        self.canvas.pack()

        self.image = Image.new("RGB", (256, 256), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        self.predict_button = tk.Button(self.root, text="Predict Shape", command=self.predict)
        self.predict_button.pack()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, width=5, fill="black", outline="black")
        self.draw.line([x1, y1, x2, y2], fill="black", width=5)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (256, 256), "white")
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        # Convert drawn image to the right format and preprocess
        image_resized = self.image.resize((64, 64))  # Resize to 64x64 as the model expects
        prediction = predict_image(image_resized)
        messagebox.showinfo("Prediction", f"Predicted Shape: {prediction}")

# Create and run the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()