import cv2
import numpy as np
import os

def generate_random_square_image(output_dir, image_size=64, num_images=500):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_images):
        # Create a white background image
        image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255  # White background

        # Randomize the square's properties
        top_left = (np.random.randint(0, image_size-20), np.random.randint(0, image_size-20))  # Random top-left corner
        side_length = np.random.randint(10, image_size // 4)  # Random side length
        color = (0, 0, 0)  # Black outline
        bottom_right = (top_left[0] + side_length, top_left[1] + side_length)

        # Draw the hollow square (black outline) on the image
        cv2.rectangle(image, top_left, bottom_right, color, 2)  # Thickness=2 for the outline

        # Save the image
        file_name = os.path.join(output_dir, f"square_{i+1}.png")
        cv2.imwrite(file_name, image)

# Example usage
output_directory = "generated_random_squares"
generate_random_square_image(output_directory)
print(f"Generated 500 random square images in '{output_directory}'")
