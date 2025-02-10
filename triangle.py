import cv2
import numpy as np
import os

def generate_random_triangle_image(output_dir, image_size=64, num_images=500):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_images):
        # Create a white background image
        image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255  # White background

        # Randomize the triangle's properties
        size = np.random.randint(10, image_size // 3)  # Random size for the triangle
        x = np.random.randint(10, image_size - size - 10)  # Random x coordinate for the base
        y = np.random.randint(10, image_size - size - 10)  # Random y coordinate for the top point

        # Coordinates of the triangle's vertices
        point1 = (x, y)
        point2 = (x + size, y)
        point3 = (x + size // 2, y - size)  # Top vertex of the triangle

        # Black color for the triangle outline
        color = (0, 0, 0)

        # Draw the hollow triangle (black outline)
        points = np.array([point1, point2, point3], dtype=np.int32)
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)  # Draw the outline

        # Save the image
        file_name = os.path.join(output_dir, f"triangle_{i+1}.png")
        cv2.imwrite(file_name, image)

# Example usage
output_directory = "generated_random_triangles"
generate_random_triangle_image(output_directory)
print(f"Generated 500 random triangle images in '{output_directory}'")
