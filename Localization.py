import cv2
import numpy as np


def plate_detection(image):
    """
    In this file, you need to define plate_detection function.
    To do:
        1. Localize the plates and crop the plates
        2. Adjust the cropped plate images
    Inputs:(One)
        1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
        type: Numpy array (imread by OpenCV package)
    Outputs:(One)
        1. plate_imgs: cropped and adjusted plate images
        type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
    Hints:
        1. You may need to define other functions, such as crop and adjust function
        2. You may need to define two ways for localizing plates(yellow or other colors)
    """

    # TODO: Replace the below lines with your code.

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define color range
    colorMin = np.array([16, 105, 120])  # Lower HSV values for yellow
    colorMax = np.array([25, 255, 255])  # Higher HSV values for yellow

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    mask = cv2.inRange(hsv_image, colorMin, colorMax)
    filtered = hsv_image.copy()
    filtered[mask == 0] = [0, 0, 0]

    structuring_element = np.array([[1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1]], np.uint8)

    structuring_element_closing = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], np.uint8)
    # Improve the mask using morphological dilation and erosion
    eroded = cv2.erode(filtered, structuring_element)
    dialated = cv2.dilate(eroded, structuring_element_closing)
    dialtedClosing = cv2.dilate(dialated, structuring_element_closing)
    # Return the improved mask
    eroded = cv2.erode(dialtedClosing, structuring_element)

    minx = 1000000
    miny = 1000000
    maxx = -1
    maxy = -1
    purple = eroded[0][0]
    for y in range(eroded.shape[0]):
        for x in range(eroded.shape[1]):
            if (eroded[y][x] != purple).all():
                if x < minx:
                    minx = x
                if y < miny:
                    miny = y
                if x > maxx:
                    maxx = x
                if y > maxy:
                    maxy = y
    # Crop the plate
    plate = image[miny:maxy + 1, minx:maxx + 1, :]

    plate_images = [plate, plate, plate]

    cv2.imshow('Filtered frame', eroded)
    cv2.imshow('First frame', image)
    if plate.size != 0:
        cv2.imshow('Cropped first frame', plate)
    cv2.waitKey(0)


    return plate_images

