import cv2 as cv

def show_img(img: cv.Mat) -> None:

    cv.imshow("Capture", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
