from image_capture import capture
from image_capture import utils

def main():

    img = capture.capture_image()
    utils.show_img(img)

if __name__ == "__main__":
    main()
