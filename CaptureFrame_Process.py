import cv2
import os
import pandas as pd
import Localization
import Recognize


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

    # TODO: Read frames from the video (saved at `file_path`) by making use of `sample_frequency`
    cap = cv2.VideoCapture(file_path)

    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    count = int(0 * cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, count)

    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            detections = Localization.plate_detection(frame)

            # Display the resulting frame
            """for detection in detections:
                cv2.imshow('Frame', detection)"""

            # TODO: Implement actual algorithms for Recognizing Characters
            output.write("XS-NB-23,34,1.822\n")
            # TODO: REMOVE THESE (below) and write the actual values in `output`

            count += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)

            # Press Q on keyboard to  exit
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
