import cv2
import numpy as np
import Rotation


def crop_plate(mask, image):
    non_black_indices = np.argwhere(np.any(mask != mask[0, 0], axis=-1))
    if non_black_indices.size > 0:
        min_y, min_x = non_black_indices.min(axis=0)
        max_y, max_x = non_black_indices.max(axis=0)

        # Crop the plate
        image = image[min_y:max_y + 1, min_x:max_x + 1, :]
    return image


def check_ratios(plates):
    checked_plates = []
    for plate in plates:
        if np.shape(plate)[0] > 0:
            ratio = np.shape(plate)[1]/np.shape(plate)[0]
            if 1.8 < ratio < 4.2:
                checked_plates.append(plate)
    return checked_plates


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

    kernel_size = 5
    blur = cv2.GaussianBlur(hsv_image, (kernel_size, kernel_size), kernel_size / 6)

    # Define color range
    colorMin = np.array([16, 90, 90])  # Lower HSV values for yellow
    colorMax = np.array([25, 255, 255])  # Higher HSV values for yellow

    # Segment only the selected color from the image and leave out all the rest (apply a mask)
    mask = cv2.inRange(blur, colorMin, colorMax)
    filtered = blur.copy()
    filtered[mask == 0] = [0, 0, 0]

    kernelErode = np.ones((5, 5), dtype=np.uint8)
    kernelDilate = np.ones((12, 12), dtype=np.uint8)
    # Improve the mask using morphological dilation and erosion
    eroded = cv2.erode(filtered, kernelErode)
    dilated = cv2.dilate(eroded, kernelDilate)
    dilatedClosing = cv2.dilate(dilated, kernelDilate)
    eroded = cv2.erode(dilatedClosing, kernelErode)

    h, s, v1 = cv2.split(eroded)

    contours = cv2.findContours(v1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    cropped_plates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        cropped_plates.append(image[y:y + height, x:x + width])

    rotated_plates = Rotation.rotate(cropped_plates)

    plate_images = check_ratios(rotated_plates)

    current_plates = []
    current_non_zero = []

    for idx, plate in enumerate(plate_images):
        if np.shape(plate)[0] > 0 and np.shape(plate)[1] > 0:
            hsv_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)

            kernel_size = 5
            blur = cv2.GaussianBlur(hsv_plate, (kernel_size, kernel_size), kernel_size / 6)

            # Define color range
            colorMin = np.array([16, 70, 70])  # Lower HSV values for yellow
            colorMax = np.array([25, 255, 255])  # Higher HSV values for yellow

            # Segment only the selected color from the image and leave out all the rest (apply a mask)
            mask = cv2.inRange(blur, colorMin, colorMax)
            filtered = plate.copy()
            filtered[mask == 0] = [0, 0, 0]

            cropped_plate = crop_plate(filtered, plate)

            current_plates.append(cropped_plate)
            current_non_zero.append(np.count_nonzero(cropped_plate))

    final_cropped_plates = []
    if len(current_non_zero) > 0:
        maximal = max(current_non_zero)
        if len(current_plates) > 1:
            for idx, plate in enumerate(current_plates):
                if abs(maximal - current_non_zero[idx]) < 5000 or current_non_zero[idx] < 1000:
                    final_cropped_plates.append(plate)
        else:
            final_cropped_plates.append(current_plates[0])

    return final_cropped_plates
