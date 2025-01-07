import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

def detect_spots(image, min_intensity, max_intensity):
    spots = cv2.inRange(image, min_intensity, max_intensity)
    kernel = np.ones((3,3), np.uint8)
    spots = cv2.morphologyEx(spots, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 50: 
            significant_contours.append(contour)
    return significant_contours

def calculate_min_max_distances(dark_contour, medium_contours):
    M = cv2.moments(dark_contour)
    if M["m00"] == 0:
        return 0
    dark_cx = int(M["m10"] / M["m00"])
    dark_cy = int(M["m01"] / M["m00"])
    dark_center = np.array([dark_cx, dark_cy])
    distances = []
    for medium_contour in medium_contours:
        M = cv2.moments(medium_contour)
        if M["m00"] == 0:
            continue
        medium_cx = int(M["m10"] / M["m00"])
        medium_cy = int(M["m01"] / M["m00"])
        medium_center = np.array([medium_cx, medium_cy])
        distance = np.linalg.norm(dark_center - medium_center)
        distances.append(distance)
    if not distances:
        return 0
    return max(distances) - min(distances)
original_image = cv2.imread(r'D:\Capstone\Cancer\IMG\HSIL\HSIL_6 (2).jpg', cv2.IMREAD_GRAYSCALE)
result_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
dark_spots = detect_spots(original_image, 50, 100)
medium_spots = detect_spots(original_image, 100, 150)
distance_differences = []
for contour in dark_spots:
    cv2.drawContours(result_image, [contour], -1, (0,0,255), 2)  # Red for dark spots
    diff = calculate_min_max_distances(contour, medium_spots)
    distance_differences.append(diff)
    
for contour in medium_spots:
    cv2.drawContours(result_image, [contour], -1, (0,255,0), 2)  # Green for medium spots
psnr_value = cv2.PSNR(cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY), original_image)
ssim_value, _ = compare_ssim(original_image, cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY), full=True)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title(f"Processed Image\nPSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}\n" 
         f"Dark Spots: {len(dark_spots)}, Medium Spots: {len(medium_spots)}\n")
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.tight_layout()
plt.show()
