import tkinter as tk
from PIL import Image, ImageDraw
import os

# Create dataset directory
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

CANVAS_SIZE = 300
BG_COLOR = "white"

class ShapeCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Shape")

        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=BG_COLOR)
        self.canvas.pack()

        # Create a blank image
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.image)

        # Buttons
        self.label = tk.StringVar()
        self.label.set("circle")
        tk.OptionMenu(root, self.label, "circle", "square", "triangle", "unknown").pack()
        tk.Button(root, text="Save", command=self.save_drawing).pack()
        tk.Button(root, text="Clear", command=self.clear_canvas).pack()

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        r = 8  # Brush size
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", width=0)
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.image)

    def save_drawing(self):
        label = self.label.get()
        folder = os.path.join(DATASET_DIR, label)
        os.makedirs(folder, exist_ok=True)
        count = len(os.listdir(folder))
        file_path = os.path.join(folder, f"{count}.png")
        self.image.save(file_path)
        print(f"Saved {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ShapeCollector(root)
    root.mainloop()
