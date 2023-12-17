import numpy as np
import cv2
from scipy import ndimage


def rotate(images):
    showImages = False

    rotated_plates = []

    for idx, image in enumerate(images):
        # Ensure the image has non-zero dimensions
        if image.shape[0] != 0 and image.shape[1] != 0:
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Perform edge detection using Canny
            edges = cv2.Canny(gray_image, 50, 150)

            if showImages:
                # Display edges for visualization
                cv2.imshow(f'Edges of plate {idx}', edges)

            # Detect lines using Hough Transform
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

            # Calculate the rotation angle from detected lines
            angle = 0

            if lines is not None and len(lines) > 0:
                angle_rad = lines[0][0][1]  # Angle in radians
                angle = np.degrees(angle_rad)  # Convert radians to degrees

                # Adjust angle to match landscape orientation
                angle -= 90
                # Ensure angle is within -90 to 90 degrees range
                if angle >= 90:
                    angle -= 90
                elif angle <= -90:
                    angle += 90

            # Rotate the image based on the calculated angle
            rotated_img = ndimage.rotate(image, angle, reshape=False)
            rotated_plates.append(rotated_img)

    return rotated_plates

