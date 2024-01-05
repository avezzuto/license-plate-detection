import numpy as np
import cv2
from scipy import ndimage


def rotate(images):
    rotated_plates = []
    threshold = 100
    if len(images) > 1:
        threshold = 30

    for idx, image in enumerate(images):
        # Ensure the image has non-zero dimensions
        if image.shape[0] != 0 and image.shape[1] != 0:
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Perform edge detection using Canny
            edges = cv2.Canny(gray_image, 100, 150)

            # Detect lines using Hough Transform
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

            if lines is not None:
                angles = [np.degrees(line[0][1]) for line in lines]

                # Calculate the most common angle (mode)
                if angles:
                    angle = max(set(angles), key=angles.count)

                    # Adjust angle to match landscape orientation
                    angle -= 90
                    # Ensure angle is within -90 to 90 degrees range
                    if angle >= 90:
                        angle -= 90
                    elif angle <= -90:
                        angle += 90

                    # Rotate the image based on the calculated angle
                    image = ndimage.rotate(image, angle, reshape=False)

            rotated_plates.append(image)

    return rotated_plates

