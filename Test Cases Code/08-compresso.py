import cv2
import numpy as np

image_path = r"c:\Users\hp\Downloads\compresso.jpg"
image = cv2.imread(image_path)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
equalized = cv2.equalizeHist(gray)
binary = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


if contours:
    barcode_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(barcode_contour)

    barcode = gray[y:y+h, x:x+w]
    _, barcode_final = cv2.threshold(barcode, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    cv2.imshow("Enhanced Barcode", barcode_final)
    cv2.imwrite("enhanced_barcode.jpg", barcode_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No barcode found.")