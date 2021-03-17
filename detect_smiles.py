import cv2 
import numpy as np 
import tensorflow as tf 
import imutils

# loading trained miniVGGNet
model = tf.keras.models.load_model("train/best_model/")



def play_video(video_path=None): 
    
    # pretrained cv2 face detector
    face_detector = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    
    if not video_path: 
        # get webcam
        camera = cv2.VideoCapture(0)
    else: 
        # read video
        camera = cv2.VideoCapture(video_path)


    while True: 

        (grabbed, frame) = camera.read()

        if video_path and not grabbed: # end of the video
            break 

        frame = imutils.resize(frame, width=300)
        frameClone = frame.copy() 

        rects = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (fX, fY, fW, fH) in rects:

            roi = frame[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = tf.keras.preprocessing.image.img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            smiling, notSmiling = model.predict(roi)[0]
            if smiling > notSmiling: 
                label = "Smiling" 
            else: 
                label = "not smiling"
            
            cv2.putText(frameClone, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2) 
            cv2.imshow("Face", frameClone)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release() 
    cv2.destroyAllWindows()


# test path 

test_path = "data/test_video.mp4"
play_video(test_path)