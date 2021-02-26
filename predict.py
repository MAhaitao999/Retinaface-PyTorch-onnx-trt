import cv2

from nets.retinaface_inference import Retinaface


if __name__ == '__main__':
    retinaface = Retinaface()

    image = cv2.imread('img/street.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r_image = retinaface.detect_image(image)
    r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('after', r_image)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
