import numpy as np
from sys import argv
from os import environ


def rgb2srgb(image):
    return np.uint8(np.round(np.clip(np.where(
        image <= 0.00304,
        12.92 * image,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055
    ) * 255, 0, 255)))


def tonemapping(x):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return rgb2srgb(x * (a * x + b) / (x * (c * x + d) + e))


def imread(filename):
    return np.maximum(
        np.nan_to_num(cv.imread(filename, cv.IMREAD_UNCHANGED)[:, :, :3], nan=0.0, posinf=1e3, neginf=0), 0.0)


if __name__ == "__main__":
    environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2 as cv

    filename = argv[1]
    exp = 0 if len(argv) == 2 else float(argv[2])
    assert filename.endswith(".exr")
    image = imread(filename) * (2 ** exp)
    cv.imwrite(f"{filename[:-4]}.png", tonemapping(image))
