import torch
from torchvision import transforms
from PIL import Image
import os

# Load the trained model
model = torch.load("model/shape_cnn.pth")
model.eval()  # Set the model to evaluation mode

# Define the same preprocessing transformation as during training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load and preprocess the input image
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')  # Ensure it's in RGB
        image = transform(image).unsqueeze(0)  # Add batch dimension
        output = model(image)
        _, predicted = torch.max(output, 1)  # Get the index of the predicted class
        
        # Assuming your classes are: 0 -> Circle, 1 -> Square, 2 -> Triangle, 3 -> 'I don't know'
        classes = ['Circle', 'Square', 'Triangle', 'I don\'t know']
        print(f"Predicted shape: {classes[predicted.item()]}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Get the image path from the user input
    image_path = input("Enter the image path: ").strip()
    if os.path.exists(image_path):
        predict_image(image_path)
    else:
        print("Invalid file path. Please try again.")