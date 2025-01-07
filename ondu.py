import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
original_image = cv2.imread(r'D:\Capstone\Cancer\IMG\HSIL\HSIL_6 (2).jpg', cv2.IMREAD_GRAYSCALE)
_, thresholded_image = cv2.threshold(original_image, 120, 255, cv2.THRESH_BINARY)
filtered_image = cv2.medianBlur(thresholded_image, 5)
contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result_image = original_image.copy()
contour_count = 0
for contour in contours:
    if cv2.contourArea(contour) > 50: 
        cv2.drawContours(result_image, [contour], -1, (200), 2)  
        contour_count += 1
result_image = cv2.normalize(result_image, None, 0, 255, cv2.NORM_MINMAX)
psnr_value = cv2.PSNR(original_image, result_image)
ssim_value, _ = compare_ssim(original_image, result_image, full=True)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title(f"Processed Image\nPSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}\nContours Detected: {contour_count}")
plt.imshow(result_image, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()