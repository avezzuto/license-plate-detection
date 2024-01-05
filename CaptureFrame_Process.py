import cv2
import numpy as np
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
    showImages = False

    cap = cv2.VideoCapture(file_path)

    output = open(save_path, "w")
    output.write("License plate,Frame no.,Timestamp(seconds)\n")

    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_no = int(0 * cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    prev_plates = []

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            if showImages:
                cv2.imshow('Original frame', frame)

            detections = Localization.plate_detection(frame)

            for idx, detection in enumerate(detections):
                if np.shape(detection)[0] > 0 and np.shape(detection)[1] > 0:
                    if showImages:
                        cv2.imshow(f'Cropped plate {idx}', detection)
                plate = Recognize.segment_and_recognize(detection)
                if prev_plates is None or plate in prev_plates:
                    prev_plates = []
                    seconds = frame_no/cap.get(cv2.CAP_PROP_FPS)
                    print(f'{plate},{frame_no},{seconds}')
                    output.write(f'{plate},{frame_no},{seconds}\n')
                else:
                    prev_plates.append(plate)

            cv2.waitKey(0)

            frame_no += 5
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

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
