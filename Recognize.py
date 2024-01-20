import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


def crop(plate):
    center_x, center_y = plate.shape[1] / 2, plate.shape[0] / 2
    width, height = plate.shape[1] * 0.93, plate.shape[0] * 0.8
    left_x, right_x = center_x - width / 2, center_x + width / 2
    top_y, bottom_y = center_y - height / 2, center_y + height / 2
    img_cropped = plate[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped


def read(path):
	to_read = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
	chars = []
	for char_path in to_read:
		chars.append(cv2.imread(char_path))
	return chars


def create_dutch_license_plate_mapping():
	# Create a mapping of numbers to letters
	mapping = {0: "B", 1: "N", 2: "P", 3: "R", 4: "S", 5: "T", 6: "V", 7: "X", 8: "Z", 9: "D", 10: "F", 11: "G", 12: "H", 13: "J", 14: "K", 
			15: "L", 16: "M", 17:"B", 18:"R", 19:"P"}
	return mapping


def segment_and_recognize(plate_image, lenBool):
	"""
	In this file, you will define your own segment_and_recognize function.
	To do:
		1. Segment the plates character by character
		2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
		3. Recognize the character by comparing the distances
	Inputs:(One)
		1. plate_imgs: cropped plate images by Localization.plate_detection function
		type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
	Outputs:(One)
		1. recognized_plates: recognized plate characters
		type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
	Hints:
		You may need to define other functions.
	"""
	showImages = False
	catThree = False

	if(5 < plate_image.shape[0] < 30 and 5 < plate_image.shape[1] < 120 and lenBool):
		catThree = True

	if(catThree):
		cropped = crop(plate_image)
		grey = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
		thresh = cv2.adaptiveThreshold(grey, 255,
										cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
	

	else:
		# Resize the licence to a wanted size
		new_height = 70
		ratio = new_height / plate_image.shape[0]
		new_width = int(plate_image.shape[1] * ratio)
		resized = cv2.resize(plate_image, (new_width, new_height))

		cropped = crop(resized)
		grey = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

		blurred = cv2.GaussianBlur(grey, (5, 5), 0)
		thresh = cv2.adaptiveThreshold(blurred, 255,
										cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8)

		if 2000 < np.count_nonzero(thresh) < 3000:
			blurred = cv2.GaussianBlur(grey, (3, 3), 0)
			thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 3)


	after_morph = thresh
	if not catThree:
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
		opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
		closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
		after_morph = closing

	else:
		ratio = 70 / after_morph.shape[0]
		width = int(after_morph.shape[1] * ratio)
		resized = cv2.resize(after_morph, (width, 70))

		after_morph = crop(resized)

	if showImages:
		cv2.imshow("After morphology", after_morph)
		
	ROI_number = 0
	cropped_char = list()
	contours = cv2.findContours(after_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
	hyphen_positions = list()
	for contour in contours:
		x, y, width, height = cv2.boundingRect(contour)
		ROI = after_morph[y:y + height, x:x + width]
		ratio = height / width
		if ROI.shape[1] >= 7:
			if 4 > ratio > 1 and ROI.shape[0] > 30:
				cropped_char.append(ROI)
				ROI_number += 1
			# Classified as char
			elif (0.9 > ratio > 0.3
                  and (not hyphen_positions or hyphen_positions[-1] + 2 < ROI_number)
                  and ROI.shape[0] < 13
                  and ROI_number != 0):
				hyphen_positions.append(ROI_number)
				ROI_number += 1

		if ROI_number >= 8:
			break

	if showImages:
		for idx, char in enumerate(cropped_char):
			cv2.imshow(f'Cropped character {idx}', char)

	letters = read('dataset/SameSizeLetters')
	numbers = read('dataset/SameSizeNumbers')

	mapping = create_dutch_license_plate_mapping()
	plate = ""

	UseCharGrouping = True

	# Each of 3 sections of a dutch licence plate consist on only letters or only numbers
	groups = []
	pre_pos = 0
	if(len(hyphen_positions) >= 2 and hyphen_positions[0] > 0 and hyphen_positions[1] > 0):
		hyphen_positions[1] -= 1
	for pos in hyphen_positions:
		if(pos == 0):
			continue
		if(pos > len(cropped_char)):
			continue
		group = [pre_pos, pos]
		groups.append(group)
		pre_pos = pos
	groups.append([pre_pos, len(cropped_char)])

	letterFit = []
	numberFit = []

	for i, char in enumerate(cropped_char):
		
		# Calculate the aspect ratio
		aspect_ratio = char.shape[1] / char.shape[0]

		# Calculate the new width based on the desired height and aspect ratio
		new_height = numbers[0].shape[0]
		new_width = int(new_height * aspect_ratio)

		# Resize the image using the calculated width and height
		char = cv2.resize(char, (new_width, new_height))

		# Calculate the difference in width
		width_difference = numbers[0].shape[1] - new_width

		if width_difference > 0:
			# Create a new image with the new width and original height
			black_padding = np.zeros((new_height, width_difference), dtype=np.uint8)
			black_padding[:] = 0  # Set the color to black

			# Concatenate the original image and black padding horizontally
			char = np.concatenate((char, black_padding), axis=1)

		if showImages:
			cv2.imshow(f'Char {i}', char)

		# find the best number fit for the character
		minChar = ""
		minDiff = 1000000
		for idx, number in enumerate(numbers):
			number = number[:, :, 0]
			if char.shape == number.shape:
				xor = cv2.bitwise_xor(char, number)
				diff = np.count_nonzero(xor)
				#cv2.imshow(f'Difference with number {idx}', xor)
				if diff < minDiff:
					minDiff = diff
					minChar = idx
		numberFit.append((minChar, minDiff))

		# find the best letter fit for the character
		minChar = ""
		minDiff = 1000000
		for idx, letter in enumerate(letters):
			letter = letter[:, :, 0]
			if char.shape == letter.shape:
				xor = cv2.bitwise_xor(char, letter)
				diff = np.count_nonzero(xor)
				#cv2.imshow(f'Difference with letter {mapping[index]}', xor)
				if diff < minDiff:
					minDiff = diff
					minChar = mapping[idx]
		letterFit.append((minChar, minDiff))

	# Decisions made based on the group of letters/numbers
	for idx, group in enumerate(groups):
		sumNumDif = 0
		sumLetDif = 0
		for i in range(group[0], group[1]):
			sumNumDif += numberFit[i][1]
			sumLetDif += letterFit[i][1]
		if(sumNumDif <= sumLetDif):
			for i in range(group[0], group[1]):
				plate += str(numberFit[i][0])
		else:
			for i in range(group[0], group[1]):
				plate += str(letterFit[i][0])
		if(idx < 2):
			plate += "-"

	if(not UseCharGrouping):
		plate = ""
		for idx in range(len(cropped_char)):
			if(idx in hyphen_positions and idx != 0 and idx < 6):
				plate += "-"
			if(letterFit[idx][1] < numberFit[idx][1]):
				plate += str(letterFit[idx][0])
			else:
				plate += str(numberFit[idx][0])


	#print("product: " + plate)
	return plate

