#Code to detect the coordinates of ROI(done manually)
import cv2

img = cv2.imread(r"C:\Users\91854\OneDrive\Pictures\Screenshots\Screenshot 2025-11-28 234538.png")

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"X: {x}, Y: {y}")

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()