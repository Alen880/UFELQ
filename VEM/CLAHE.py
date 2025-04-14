import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def apply_CLAHE(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))

    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final_img


def process_images_in_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

            clahe_img = apply_CLAHE(original_img)

            output_path = os.path.join(output_folder, f"clahe_{filename}")
            cv2.imwrite(output_path, clahe_img)


if __name__ == "__main__":
    folder_path = ''
    output_folder = ''

    process_images_in_folder(folder_path, output_folder)
