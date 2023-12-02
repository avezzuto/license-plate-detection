import cv2
import numpy as np
import Rotation

# Function for splitting two plates
def split_license_plates(mask, image):
    height, width = mask.shape[:2]
    split_threshold = 120  # Number of black pixels to consider a split

    splits = []
    previous_color = None
    black_count = 0

    for x in range(width):
        column = mask[:, x]
        unique_colors = np.unique(column)

        if len(unique_colors) == 1 and unique_colors[0] == 0:  # Assuming black background
            black_count += 1
            if black_count >= split_threshold and previous_color is not None:
                splits.append(x)
                black_count = 0
                previous_color = None
        else:
            black_count = 0
            previous_color = unique_colors[0]

    # Split the image based on identified positions
    mask_images = []
    plate_images = []
    if len(splits) > 0:
        splits = [0] + splits + [width]
        for i in range(len(splits) - 1):
            start_x = splits[i]
            end_x = splits[i + 1]
            sub_mask = mask[:, start_x:end_x]  # Extract sub-mask from the original mask
            sub_plate = image[:, start_x:end_x, :]  # Extract sub-plate from the original image
            unique_colors = np.unique(sub_mask)
            if not (len(unique_colors) == 1 and unique_colors[0] == 0):
                # Append sub-mask and sub-plate as images to their respective lists
                mask_images.append(sub_mask)
                plate_images.append(sub_plate)
    return mask_images, plate_images

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
    dilated = cv2.dilate(eroded, structuring_element_closing)
    dilatedClosing = cv2.dilate(dilated, structuring_element_closing)
    eroded = cv2.erode(dilatedClosing, structuring_element)

    split_mask, split_image = split_license_plates(eroded, image)

    plates = []
    for i, singleMask in enumerate(split_mask):
        minx = 1000000
        miny = 1000000
        maxx = -1
        maxy = -1
        purple = singleMask[0][0]
        for y in range(singleMask.shape[0]):
            for x in range(singleMask.shape[1]):
                if (singleMask[y][x] != purple).all():
                    if x < minx:
                        minx = x
                    if y < miny:
                        miny = y
                    if x > maxx:
                        maxx = x
                    if y > maxy:
                        maxy = y
        # Crop the plate
        plate = split_image[i][miny:maxy + 1, minx:maxx + 1, :]
        plates.append(plate)

    plate_images = Rotation.rotate(plates)

    cv2.imshow('Original frame', image)
    cv2.imshow('Masked frame', eroded)
    for idx, plate_img in enumerate(plate_images):
        cv2.imshow(f'Plate {idx}', plate_img)
    cv2.waitKey(0)

    return plate_images