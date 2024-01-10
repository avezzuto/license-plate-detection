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
	mapping = {0: "B", 1: "N", 2: "P", 3: "R", 4: "S", 5: "T", 6: "V", 7: "X", 8: "Z", 9: "D", 10: "F", 11: "G", 12: "H", 13: "J", 14: "K", 15: "L", 16: "M", 17:"B"}
	#mapping = {0: "B", 1: "D", 2: "F", 3: "G", 4: "H", 5: "J", 6: "K", 7: "L", 8: "M", 9: "N", 10: "P", 11: "R", 12: "S", 13: "T", 14: "V", 15: "X", 16: "Z"}
	return mapping


def segment_and_recognize(plate_image):
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
	showImages = True #False

	# Resize the licence to a wanted size
	new_height = 70
	ratio = new_height / plate_image.shape[0]
	new_width = int(plate_image.shape[1] * ratio)
	resized = cv2.resize(plate_image, (new_width, new_height))

	cropped = crop(resized)

	hsv_plate = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

	kernel_size = 5
	blur = cv2.GaussianBlur(hsv_plate, (kernel_size, kernel_size), kernel_size / 6)
	#cv2.imshow("blur", blur)
	cv2.imshow("plate", plate_image)

	# Define color range
	colorMin = np.array([0, 120, 120]) #16, 130, 130])  # Lower HSV values for yellow
	colorMax = np.array([50, 255, 255]) #25, 255, 255])  # Higher HSV values for yellow

	# Segment only the selected color from the image and leave out all the rest (apply a mask)
	mask = cv2.inRange(blur, colorMin, colorMax)
	cv2.imshow("mask", mask)
	filtered = blur.copy()
	filtered[mask == 0] = [0, 0, 0]
	filtered_resized = cv2.resize(filtered, (plate_image.shape[1], plate_image.shape[0]))
	result = cv2.bitwise_and(plate_image, filtered_resized)

	
	grey_mask = result[:, :, 2]
	equalised = cv2.equalizeHist(grey_mask)
	binarised = np.where(equalised > 0, 0, 255).astype(np.uint8)

	structuring_element = np.array([[1, 1, 1],
									[1, 1, 1],
									[1, 1, 1]], np.uint8)

	# Improve the mask using morphological dilation and erosion
	eroded = cv2.erode(binarised, structuring_element)
	dilated = cv2.dilate(eroded, structuring_element)
	dilatedClosing = cv2.dilate(dilated, structuring_element)
	img = cv2.erode(dilatedClosing, structuring_element)

	indices_to_start = []
	indices_to_end = []
	started = False
	for i in range(img.shape[1]):
		section = img[:, i]
		unique_colors = np.unique(section)
		if 255 in unique_colors:
			if not started:
				indices_to_start.append(i)
				started = True
		else:
			if started:
				indices_to_end.append(i)
				started = False

	# Add the last section if it ends with white pixels
	if started:
		indices_to_end.append(img.shape[1])

	chars = []
	hyphen_pos = []
	count = 0
	for start, end in zip(indices_to_start, indices_to_end):
		segment_mask = img[:, start:end]
		if np.count_nonzero(segment_mask) > 100:
			chars.append(segment_mask)
			count += 1
		elif np.count_nonzero(segment_mask) > 50:
			hyphen_pos.append(count)
	if showImages:
		cv2.imshow("Binarised mask", img)

	letters = read('dataset/SameSizeLetters')
	numbers = read('dataset/SameSizeNumbers')

	mapping = create_dutch_license_plate_mapping()
	plate = ""

	UseCharGrouping = True

	# Each of 3 sections of a dutch licence plate consist on only letters or only numbers
	groups = []
	pre_pos = 0
	for pos in hyphen_pos:
		if(pos == 0):
			continue
		group = [pre_pos, pos]
		groups.append(group)
		pre_pos = pos
	groups.append([pre_pos, len(chars)])

	letterFit = []
	numberFit = []

	for i, char in enumerate(chars):
		

		# Get rid of black pixels around the character
		positions = np.nonzero(char)
		del_lines_top = positions[0].min()
		del_lines_bottom = positions[0].max()
		del_lines_left = positions[1].min()
		del_lines_right = positions[1].max()
		char = char[del_lines_top:del_lines_bottom, del_lines_left:del_lines_right]

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
		for idx in range(len(chars)):
			if(idx in hyphen_pos and idx != 0 and idx < 6):
				plate += "-"
			if(letterFit[idx][1] < numberFit[idx][1]):
				plate += str(letterFit[idx][0])
			else:
				plate += str(numberFit[idx][0])


	#print("product: " + plate)
	return plate
