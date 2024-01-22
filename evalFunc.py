import cv2
import Recognize
import Rotation

def positionsList():

	# list of manual bounding boxes for the plates in format (frame number, left-upper corner, right-lower corner)
	list = ((24, (217, 326), (443, 382), "XS-NB-23"),
			(49, (192, 249), (348, 325), "98-THD-4"), 
			(74, (282, 236), (412, 266), "23-GSX-6"),
			
			(124, (191, 271), (484, 367), "72-VGX-6"), 
			(149, (219, 402), (408, 448), "99-SZG-5"), 
			(199, (209, 272), (403, 332), "14-NJK-9"), 

			(224, (276, 323), (522, 391), "24-LB-HT"), 
			(274, (332, 94), (480, 136), "68-NS-ND"), 
			(299, (246, 166), (403, 224), "97-FB-FP"),

			(324, (117, 178), (498, 271), "67-FP-SJ"), 
			(374, (80, 106), (562, 230), "96-LKH-9"), 
			(399, (114, 310), (617, 426), "56-JTT-5"),

			(449, (219, 212), (431, 270), "88-BB-TS"), 
			(474, (110, 316), (615, 433), "56-JTT-5"), 
			(524, (263, 324), (413, 381), "ZD-PB-67"), 

			(549, (252, 397), (429, 435), "92-GS-VH"), 
			(599, (264, 187), (429, 227), "02-BBG-7"), #red shade of licence plate - no bounding box found
			(624, (244, 213), (366, 252), "73-PV-HB"),
			(674, (275, 302), (384, 334), "93-PXS-9"), 

			(699, (309, 247), (418, 273), "RP-NL-93"), #bounding box very inacurate, only the middle section was recognized
			(724, (286, 207), (504, 271), "43-HNN-2"), 
			(774, (212, 277), (426, 363), "43-JS-RT"), 
			(824, (331, 246), (467, 277), "63-HK-HD"), 

			(849, (346, 338), (542, 392), "92-LHR-6"), 
			(874, (345, 248), (483, 284), "63-HK-HD"), 
			(924, (76, 320), (328, 433), "3-VFX-39"),

			(949, (242, 279), (385, 311), "01-XJ-ND"),
			(974, (267, 164), (425, 208), "96-ND-JB"), # the undetected red minivan
			(1024, (303, 325), (437, 359), "57-LF-JB"),

			(1049, (347, 213), (498, 256), "VD-020-P"),
			(1099, (176, 262), (375, 311), "5-SXB-74"),
			(1149, (345, 282), (534, 328), "23-GSR-5"),
			
			(1199, (218, 313), (426, 375), "27-LH-TB"),
			#(1249, (6, 333), (210, 380), "XH-PN-44"),
			(1274, (383, 354), (621, 415), "XH-PN-44"),

			(1299, (101, 281), (294, 327), "01-RHD-6"),
			#(1324, (400, 400), (400, 400), "01-RHD-6"), # no plate should be detected in this frame
			(1349, (220, 270), (422, 321), "69-GR-DH"),
			(1399, (288, 221), (485, 287), "64-LJ-GF"),

			(1424, (88, 202), (232, 241), "41-GFZ-8"),
			(1474, (154, 97), (338, 159), "28-RXT-9")

			# from here we get to the caregory 3 with multiple plates on one screen, commented out for recongnize evaluation
			#(1524, (235, 115), (440, 166), "89-LZ-GX"),
			#(1574, (15, 158), (89, 195), "00-00-00"), # frame with two plates but only one is being registered # all plates from now on are double
			#(1599, (84, 349), (602, 409), "89-NV-JP"),
			#(1624, (13, 167), (634, 379), "TT-BJ-42"),
			#(1674, (109, 178), (582, 319), "25-XV-LX"),
			#(1699, (76, 277), (608, 397), "72-FP-RV")
		)
	
	return list

finalAccuracy = 0
accCount = 0

def evalLocalization(image, mask, frame_num):
	mina = mask.shape[1]
	minb = mask.shape[0]
	maxa = 0 
	maxb = 0
	for y in range(mask.shape[0]):
		for x in range(mask.shape[1]):
			if (mask[y][x] != mask[0][0]).all():
				if(x < mina):
					mina = x
				if(y < minb):
					minb = y
				if(x > maxa):
					maxa = x
				if(y > maxb):
					maxb = y
	if(mina == mask.shape[1]):
		mina = 0
	if(minb == mask.shape[0]):
		minb = 0
	if(maxa == 0):
		maxa = mask.shape[1]
	if(maxb == 0):
		maxb = mask.shape[0]
	minP = (mina, minb)
	maxP = (maxa, maxb)

	list = positionsList()

	for element in list:
		if(frame_num == element[0]):
			image = cv2.rectangle(image, minP, maxP, (0,0,255), 1)
			image = cv2.rectangle(image, element[1], element[2], (255,0,0), 1)
			x1 = max(element[1][0], minP[0])
			x2 = max(element[1][1], minP[1])
			y1 = min(element[2][0], maxP[0])
			y2 = min(element[2][1], maxP[1])
			image = cv2.rectangle(image, (x1, x2), (y1, y2), (0,255,0), 1)
			intersection =  abs((y1 - x1) * (y2 - x2))
			box1 = abs((maxP[0] - minP[0]) * (maxP[1] - minP[1]))
			box2 = abs((element[2][0] - element[1][0]) * (element[2][1] - element[1][1]))
			if(intersection > box1 or intersection > box2):
				intersection = 0
			accuracy = intersection / (box1 + box2 - intersection + 1e-6) * 100
			global finalAccuracy
			global accCount
			finalAccuracy += accuracy
			accCount += 1

			print("Frame "+ str(element[0]) + " accuracy: " + str(accuracy) + " % " + " and overall accuracy: " + str(finalAccuracy / accCount) + " % ")
	
	return image

totalAccuracy = 0
textCount = 0

def evalRecognition(image, frame_num, plates):
	list = positionsList()

	frame_found = False
	for element in list:
		if(frame_num == element[0]):
			frame_found = True
			plate_image = image[element[1][1]:element[2][1], element[1][0]:element[2][0]]
			list = Rotation.rotate([plate_image])
			plate_image = list[0]
			plate_image = Recognize.crop(plate_image)

			recognized = Recognize.segment_and_recognize(plate_image)
			eval_plates = plates.copy()
			eval_plates.append(recognized)
	
			expected_text = element[3]
			bestAccuracy = 0.0
			
			if(expected_text in eval_plates):
				bestAccuracy = 100
				recognized = expected_text
			
			global totalAccuracy
			global textCount
			totalAccuracy += bestAccuracy
			textCount += 1
			
			print("Frame " + str(element[0]) + " : " + recognized + " expected " + expected_text + 
	  		 " accuracy: " + str(bestAccuracy) + "% and average accuracy: " + str(totalAccuracy / textCount) + "% ")

	if(not frame_found):
		return None
	
	
	# returns accuracy in percent

	return bestAccuracy

