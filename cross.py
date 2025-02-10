import cv2
import numpy as np
import os

def generate_random_cross_image(output_dir, image_size=64, num_images=500):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_images):
        # Create a white background image
        image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255  # White background

        # Randomize the cross's properties
        center = (np.random.randint(20, image_size-20), np.random.randint(20, image_size-20))  # Random center
        cross_size = np.random.randint(10, image_size // 4)  # Random cross size
        color = (0, 0, 0)  # Black outline

        # Draw the horizontal and vertical lines to create a cross
        # Horizontal line
        x1, y1 = center[0] - cross_size, center[1]
        x2, y2 = center[0] + cross_size, center[1]
        cv2.line(image, (x1, y1), (x2, y2), color, 2)  # Thickness=2 for the outline

        # Vertical line
        x3, y3 = center[0], center[1] - cross_size
        x4, y4 = center[0], center[1] + cross_size
        cv2.line(image, (x3, y3), (x4, y4), color, 2)  # Thickness=2 for the outline

        # Save the image
        file_name = os.path.join(output_dir, f"cross_{i+1}.png")
        cv2.imwrite(file_name, image)

# Example usage
output_directory = "generated_random_crosses"
generate_random_cross_image(output_directory)
print(f"Generated 500 random cross images in '{output_directory}'")
