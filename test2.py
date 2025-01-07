import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
import os

def detect_spots(image, min_intensity, max_intensity):
    spots = cv2.inRange(image, min_intensity, max_intensity)
    kernel = np.ones((3,3), np.uint8)
    spots = cv2.morphologyEx(spots, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = [contour for contour in contours if cv2.contourArea(contour) > 50]
    return significant_contours

def calculate_min_max_distances(dark_contour, medium_contours):
    M = cv2.moments(dark_contour)
    if M["m00"] == 0:
        return 0
    dark_center = np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]);
    distances = []
    for medium_contour in medium_contours:
        M = cv2.moments(medium_contour)
        if M["m00"] == 0:
            continue
        medium_center = np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]);
        distances.append(np.linalg.norm(dark_center - medium_center))
    return max(distances) - min(distances) if distances else 0

input_folder = r'D:\Capstone\Cancer\IMG\NILM'
output_folder = r'D:\Capstone\Cancer\IMG\NILM_M'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        file_path = os.path.join(input_folder, filename)
        original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        result_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        dark_spots = detect_spots(original_image, 50, 100)
        medium_spots = detect_spots(original_image, 100, 150)
        for contour in dark_spots:
            cv2.drawContours(result_image, [contour], -1, (0, 0, 255), 2)
            calculate_min_max_distances(contour, medium_spots)
        for contour in medium_spots:
            cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
        psnr_value = cv2.PSNR(cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY), original_image)
        ssim_value, _ = compare_ssim(original_image, cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY), full=True)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result_image)
        print(f"Processed {filename}: PSNR = {psnr_value:.2f}, SSIM = {ssim_value:.4f}")
