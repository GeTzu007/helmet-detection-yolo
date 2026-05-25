import cv2


def resize_image(image, size=(640, 640)):

    return cv2.resize(image, size)


def normalize_image(image):

    return image / 255.0