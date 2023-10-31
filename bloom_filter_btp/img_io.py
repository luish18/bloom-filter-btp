import cv2 as cv
import numpy as np

def show_img(img: cv.Mat) -> None:

    cv.imshow("Capture", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def capture_image(
    exit_key: str = "q", capture_key: str = "c"
) -> cv.Mat:
    cap = cv.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    done = False
    while not done:
        ret, frame = cap.read()
        frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        cv.imshow("Input", frame)

        c = cv.waitKey(1)
        if c == ord(exit_key):
            result = None
            done = True

        elif c == ord(capture_key):
            result = frame
            done = True

    cap.release()
    cv.destroyAllWindows()

    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    

    return result
