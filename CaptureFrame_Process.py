import cv2
import numpy as np
import Localization
import Recognize
import evaluation


def hummingDistance(x, y):
	same = 0
	shorter = min(len(x), len(y))
	for i in range(shorter):
		if(x[i] == y[i]):
			same += 1
	return same / shorter * 100
	
def majorityPlate(plates):
    #find most common dash positions
    firstDash = []
    secondDash = []
    for plate in plates:
        second = False
        for i in range(len(plate)):
            if(plate[i] == "-" and i != 0):
                if(not second):
                    firstDash.append(i)
                    second = True
                else:
                    secondDash.append(i)
                    break
    if(secondDash == []):
        return None
    firstD = max(set(firstDash), key = firstDash.count)
    secondD = max(set(secondDash), key = secondDash.count)
    if(firstD < 1 or firstD > 3 or secondD < 4 or secondD > 6):
        return None
    #delete dashes
    platesNoDash = []
    for plate in plates:
        string = ""
        for char in plate:
            if(char != "-"):
                string += char
        platesNoDash.append(string)
    #find most common letters
    finalPlate = ""
    for i in range(6):
        letters = []
        for plate in platesNoDash:
            if(i < len(plate)):
                letters.append(plate[i])
        if(letters == []):
            continue
        char = max(set(letters), key = letters.count)
        finalPlate += char
    if(len(finalPlate) != 6):
        return None
    #insert dashes
    finalDashPlate = ""
    for char in finalPlate:
        finalDashPlate += char
        if(len(finalDashPlate) == firstD or len(finalDashPlate) == secondD):
            finalDashPlate += "-"
    
    return finalDashPlate
         

def CaptureFrame_Process(file_path, sample_frequency, save_path):
    """
    In this file, you will define your own CaptureFrame_Process funtion. In this function,
    you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
    To do:
        1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
        2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
        3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
    Inputs:(three)
        1. file_path: video path
        2. sample_frequency: second
        3. save_path: final .csv file path
    Output: None
    """
    showImages = False
    printToConsole = False

    cap = cv2.VideoCapture(file_path)

    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")

    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_no = int(0 * cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    prev_plates = [[]]
    prediction = [""]
    current_second = 0.0
    current_frame = 0
    current_plate = ""

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            if showImages:
                cv2.imshow('Original frame', frame)

            detections = Localization.plate_detection(frame)


            for idx, detection in enumerate(detections):
                if(idx >= len(prev_plates)):
                    prev_plates.append([])
                    prediction.append("")
                if np.shape(detection)[0] > 0 and np.shape(detection)[1] > 0:
                    if showImages:
                        cv2.imshow(f'Cropped plate {idx}', detection)
                plate = Recognize.segment_and_recognize(detection, len(detections) > 1)
                if plate != None and plate != "-":
                    prev_plates[idx].append(plate)
                    prediction[idx] = majorityPlate(prev_plates[idx])

                    if(prediction[idx] == None):
                        prediction[idx] = "Nothing_"
                    if(prediction[idx] == plate and current_plate != plate):
                        current_frame = frame_no
                        current_second = frame_no/cap.get(cv2.CAP_PROP_FPS)
                        current_plate = plate

                    # Scene change
                    # needs at least 20 frames to conclude a scene, for shorter scenes change here
                    if(len(prev_plates[idx]) >= 20):  
                        hM0 = hummingDistance(plate, prediction[idx])
                        hM1 = hummingDistance(prev_plates[idx][len(prev_plates[idx]) - 2], prediction[idx])
                        hM2 = hummingDistance(prev_plates[idx][len(prev_plates[idx]) - 3], prediction[idx])
                        percentile = 62
                        finalPrediction = majorityPlate(prev_plates[idx][:-3])
                        if(hM0 < percentile and hM1 < percentile and hM2 < percentile and finalPrediction != None):
                            a, b, c = prev_plates[idx][len(prev_plates[idx]) - 1], prev_plates[idx][len(prev_plates[idx]) - 2], prev_plates[idx][len(prev_plates[idx]) - 3]
                            prev_plates[idx] = [a, b, c]

                            if printToConsole:
                                print(f'{finalPrediction},{current_frame},{current_second}')
                            output.write(f'{finalPrediction},{current_frame},{current_second}\n')

            frame_no += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

            # Press Q on keyboard to  exit
            if showImages:
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    pass
