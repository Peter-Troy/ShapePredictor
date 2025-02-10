import torch  # Import PyTorch for deep learning
import torch.nn as nn  # Import PyTorch's neural network module
import torch.optim as optim  # Import optimization functions (Adam, SGD, etc.)
from torchvision import datasets, transforms  # Import datasets and transformations for image preprocessing
from torch.utils.data import DataLoader  # Import DataLoader for batch processing

# Define CNN Model
class ShapeCNN(nn.Module):
    def __init__(self):
        super(ShapeCNN, self).__init__()  # Initialize the parent class (nn.Module)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # First convolutional layer (RGB input, 32 output channels)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second convolutional layer (32 input channels, 64 output channels)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Fully connected layer (input: flattened feature map, output: 128 neurons)
        self.fc2 = nn.Linear(128, 4)  # Output layer (4 classes: Cross, Square, Triangle, "I don't know")
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer (reduces feature map size by half)
        self.relu = nn.ReLU()  # Activation function (ReLU)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Apply first convolution, ReLU activation, then max pooling
        x = self.pool(self.relu(self.conv2(x)))  # Apply second convolution, ReLU activation, then max pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = self.relu(self.fc1(x))  # Apply first fully connected layer with ReLU activation
        return self.fc2(x)  # Output layer (raw scores for each class)

# Data preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors (C, H, W format)
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to range [-1, 1] for better model performance
])

# Load dataset from specified directory
dataset = datasets.ImageFolder(root="dataset", transform=transform)  # Load images and apply transformations

# Create DataLoader for batch processing
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Load data in batches of 32, shuffle for randomness

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else fallback to CPU
model = ShapeCNN().to(device)  # Move model to selected device
criterion = nn.CrossEntropyLoss()  # Define loss function (CrossEntropy for multi-class classification)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use Adam optimizer with learning rate 0.001

# Training loop
num_epochs = 10  # Set number of training epochs
for epoch in range(num_epochs):  # Loop through epochs
    for images, labels in dataloader:  # Iterate over batches of data
        images, labels = images.to(device), labels.to(device)  # Move images and labels to GPU if available

        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)  # Forward pass through the model
        loss = criterion(outputs, labels)  # Compute loss between predictions and actual labels
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")  # Print loss for each epoch

# Save trained model weights
torch.save(model.state_dict(), "model/shape_cnn.pth")  # Save model parameters to a file
print("Model training complete!")  # Indicate completion of training