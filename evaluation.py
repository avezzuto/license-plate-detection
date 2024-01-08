import pandas as pd
import argparse
import numpy as np


def get_args():
	# ground truth header: 'License plate', 'Timestamp', 'First frame', 'Last frame', 'Category'
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_path', type=str, default='dataset/Output.csv')
	parser.add_argument('--ground_truth_path', type=str, default='dataset/groundTruth.csv')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = get_args()
	student_results = pd.read_csv(args.file_path)
	ground_truth = pd.read_csv(args.ground_truth_path)
	totalInput = len(student_results['License plate'])
	totalPlates = len(ground_truth['License plate'])
	# firstFrames = ground_truth['First frame'].tolist()
	# lastFrames = ground_truth['Last frame'].tolist()
	result = np.zeros((totalPlates, 4))
	# 0: TP, 1: FP, 2: LTP

	# Find the last frame and number of plates for each category
	numCategories = len(ground_truth['Category'].unique())
	numPlates = np.zeros(numCategories)
	lastframe = np.zeros(numCategories)
	for i,x in enumerate(ground_truth['Category'].unique()):
		numPlates[i] = len(ground_truth[ground_truth['Category']==x])
		lastframe[i] = ground_truth[ground_truth['Category']==x]['Last frame'].tolist()[-1]
	# For each line in the input list
	for i in range(totalInput):
		licensePlate = student_results['License plate'][i]
		frameNo = student_results['Frame no.'][i]
		timeStamp = student_results['Timestamp(seconds)'][i]
		# Find the lines of solution where frameNo fits into the interval
		interval = ground_truth[(ground_truth['First frame'] <= frameNo) & (ground_truth['Last frame'] >= frameNo)]
		for j in range(len(interval)):
			index = interval.index[j]
			solutionPlate = ground_truth['License plate'][index]
			solutionTimeStamp = ground_truth['Timestamp'][index]
			if licensePlate == solutionPlate:
				if timeStamp <= solutionTimeStamp + 2:
					result[index, 0] += 1
				else:
					result[index, 2] += 1
				if j == 1:
					result[index-1, 1] -= 1
				elif j == 0:
					break
			else:
				result[index, 1] += 1

	# Initialize arrays to save the final results per category
	TP = np.zeros(numCategories)
	FP = np.zeros(numCategories)
	FN = np.zeros(numCategories)
	LTP = np.zeros(numCategories)

	print('---------------------------------------------------------')
	print('%20s'%'License plate', '%10s'%'Result')
	for i in range(totalPlates):
		cat = int(ground_truth['Category'][i]-1)
		if result[i, 0] + result[i, 2] + result[i, 1] == 0:
			finalResult = 'FN'
			FN[cat] += 1
		else:
			if result[i, 0] > 0:
				TP[cat] += 1
				if result[i, 1] == 0:
					finalResult = 'TP'
				else:
					finalResult = 'TP+FP'
					FP[cat] += 1
			elif result[i, 2] > 0:
				LTP [cat] += 1
				if result[i, 1] == 0:
					finalResult = 'LTP'
				else:
					finalResult = 'LTP+FP'
					FP[cat] += 1
			else:
				finalResult = 'FP'
				FP[cat] = FP[cat]+1
		print('%4d'%i,'%14s'%ground_truth['License plate'][i],'%10s'%finalResult)

	output = np.zeros((5, numCategories*2+2))
	for i in range(numCategories):
		output[0, 2*i] = TP[i]
		output[0, 2*i+1] = TP[i]/numPlates[i]*100
		output[1, 2*i] = FP[i]
		output[1, 2*i+1] = 0
		output[2, 2*i] = FN[i]
		output[2, 2*i+1] = FN[i]/numPlates[i]*100
		output[3, 2*i] = LTP[i]
		output[3, 2*i+1] = LTP[i]/numPlates[i]*100
		output[4, i] = (TP[i]+LTP[i])/(FP[i]+FN[i]+TP[i]+LTP[i])

	output[0, 2*numCategories] = np.sum(TP)
	output[0, 2*numCategories+1] = np.sum(TP)/totalPlates*100
	output[1, 2*numCategories] = np.sum(FP)
	output[1, 2*numCategories+1] = 0
	output[2, 2*numCategories] = np.sum(FN)
	output[2, 2*numCategories+1] = np.sum(FN)/totalPlates*100
	output[3, 2*numCategories] = np.sum(LTP)
	output[3, 2*numCategories+1] = np.sum(LTP)/totalPlates*100
	output[4, 2*numCategories] = (np.sum(TP)+np.sum(LTP))/(np.sum(FP)+np.sum(FN)+np.sum(TP)+np.sum(LTP))
	print('********************************************************************')
	print('RESULTS:')
	print('%29s'%' ','%14s'%'Category I','%14s'%'Category II','%14s'%'Category III','%14s'%'Category IV','%14s'%'Total')
	print('%29s'%'True positives(TP)', output[0,:])
	print('%29s'%'False positives(FP)', output[1,:])
	print('%29s'%'False negatives(FN)', output[2,:])
	print('%29s'%'Too late true positives(LTP)', output[3,:])
	print('----------------------------------------------------')
	print('%29s'%'Score', output[4,:])

	TP12 = output[0,0]+output[0,2]+output[3,0]+output[3,2]
	FP12 = output[1,0]+output[1,2]
	FN12 = output[2,0]+output[2,2]
	c12score = TP12/(TP12+FP12+FN12)
	print('%29s'%'Score of Category I & II:', c12score)

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
	#mask = cv2.rectangle(mask, minP, maxP, (0,0,255), 1 )

	# list of manual bounding boxes for the plates in format (frame number, left-upper corner, right-lower corner)
	list = ((24, (217, 326), (443, 382)), 
			(49, (192, 249), (348, 325)), 
			(74, (282, 236), (412, 266)),
			
			(124, (191, 271), (484, 367)), 
			(149, (219, 402), (408, 448)), 
			(199, (209, 272), (403, 332)), 

			(224, (276, 323), (522, 391)), 
			(274, (332, 94), (480, 136)), 
			(299, (246, 166), (403, 224)),

			(324, (117, 178), (498, 271)), 
			(374, (80, 106), (562, 230)), 
			(399, (114, 310), (617, 426)),

			(449, (219, 212), (431, 270)), 
			(474, (110, 316), (615, 433)), 
			(524, (263, 324), (413, 381)), 

			(549, (252, 397), (429, 435)), 
			(599, (264, 187), (429, 227)), #red shade of licence plate - no bounding box found
			(624, (244, 213), (366, 252)),
			(674, (275, 302), (384, 334)), 

			(699, (309, 247), (418, 273)), #bounding box very inacurate, only the middle section was recognized
			(724, (286, 207), (504, 271)), 
			(774, (212, 277), (426, 363)), 
			(824, (331, 246), (467, 277)), 

			(849, (346, 338), (542, 392)), 
			(874, (345, 248), (483, 284)), 
			(924, (76, 320), (328, 433)),

			(949, (242, 279), (385, 311)),
			(974, (267, 164), (425, 208)), # the undetected red minivan
			(1024, (303, 325), (437, 359)),

			(1049, (347, 213), (498, 256)),
			(1099, (176, 262), (375, 311)),
			(1149, (345, 282), (534, 328)),
			
			(1199, (218, 313), (426, 375)),
			(1249, (6, 333), (210, 380)),
			(1274, (383, 354), (621, 415)),

			(1299, (101, 281), (294, 327)),
			(1324, (400, 400), (400, 400)), # no plate should be detected in this frame
			(1349, (220, 270), (422, 321)),
			(1399, (288, 221), (485, 287)),

			(1424, (88, 202), (232, 241)),
			(1474, (154, 97), (338, 159)),
			(1524, (235, 115), (440, 166)),
			#(1574, (15, 158), (89, 195)), # frame with two plates but only one is being registered # all plates from now on are double
			(1599, (84, 349), (602, 409)),
			(1624, (13, 167), (634, 379)),
			(1674, (109, 178), (582, 319)),
			(1699, (76, 277), (608, 397))
	)

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

			print("frame "+ str(element[0]) + " : " + str(accuracy) + " % " + " and overall accuracy: " + str(finalAccuracy / accCount) + " % ")
	#print("(" + str(frame_num) + ", " + str(minP) + ", " + str(maxP) + "),")
	return image
