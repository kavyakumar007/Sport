import cv2
import numpy as np

# ------------ 0. LOAD IMAGE ------------
img = cv2.imread(r"C:\Users\91854\OneDrive\Pictures\Screenshots\Screenshot 2025-11-28 234538.png")
if img is None:
    print("Image not loaded!")
    exit()

print("Image shape:", img.shape)  #to avoid out of bound errors

# ------------ 1. CROP ROI -------------
ROI_Y1, ROI_Y2 = 320, 640
ROI_X1, ROI_X2 = 120, 520

roi = img[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
if roi is None or roi.size == 0:
    print("ROI empty, adjust coordinates!")
    exit()

# ------------ 2. PREPROCESS ------------
roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)

# ------------ 3. CREASE DETECTION (Hough) ------------
edges = cv2.Canny(roi_blur, 50, 150)

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=80,
    maxLineGap=10
)

best_line = None
max_length = 0

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1

        # keep only mostly vertical lines
        if abs(dx) < 20:
            length = np.hypot(dx, dy)
            if length > max_length:
                max_length = length
                best_line = (x1, y1, x2, y2)

if best_line is None:
    print("No crease line detected – tune Hough/Canny.")
    cv2.imshow("ROI", roi)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

x1, y1, x2, y2 = best_line

# draw crease on ROI and full image
cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
X1, Y1 = x1 + ROI_X1, y1 + ROI_Y1
X2, Y2 = x2 + ROI_X1, y2 + ROI_Y1
cv2.line(img, (X1, Y1), (X2, Y2), (0, 0, 255), 2)

# since line is vertical, x of crease is approximately constant
crease_x_roi = int((x1 + x2) / 2)
print("Crease line (ROI coords):", best_line, "crease_x_roi:", crease_x_roi)

# ------------ 4. FOOT CANDIDATE MASK (dark + bottom) ------------
# Otsu + inverse: dark = white
_, th = cv2.threshold(
    roi_blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

h, w = th.shape
foot_mask = np.zeros_like(th)
bottom_start = int(h * 0.55)          # keep only bottom part of ROI
foot_mask[bottom_start:h, :] = th[bottom_start:h, :]

kernel = np.ones((5, 5), np.uint8)
foot_clean = cv2.morphologyEx(foot_mask, cv2.MORPH_OPEN, kernel)
foot_clean = cv2.morphologyEx(foot_clean, cv2.MORPH_CLOSE, kernel)

# ------------ 5. SHADOW REDUCTION (HSV saturation) ------------
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

sat_thresh = 40   # tune if needed
_, sat_mask = cv2.threshold(S, sat_thresh, 255, cv2.THRESH_BINARY)

foot_candidate = cv2.bitwise_and(foot_clean, sat_mask)

# ------------ 6. FOOT CONTOUR SELECTION (use crease position) ------------
contours, _ = cv2.findContours(
    foot_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

foot_contour = None
best_score = -1

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 200:
        continue

    x, y, w_box, h_box = cv2.boundingRect(cnt)
    aspect_ratio = w_box / float(h_box + 1e-3)

    cx = x + w_box / 2.0
    cy = y + h_box / 2.0

    # 1) must be near crease horizontally (reject far-left shadow)
    if cx < (crease_x_roi - 20) or cx > (crease_x_roi + 100):
        continue

    # 2) must be low in the image (foot, not upper leg)
    if (y + h_box) < 0.6 * h:
        continue

    # 3) reasonable shape (not super long/flat)
    if not (0.4 < aspect_ratio < 3.0):
        continue

    dist_to_crease = abs(cx - crease_x_roi)
    score = area - 5 * dist_to_crease

    if score > best_score:
        best_score = score
        foot_contour = cnt

decision = "UNKNOWN"

if foot_contour is not None:
    # existing stuff...
    pts = foot_contour.reshape(-1, 2)
    # x-extents of the foot
    xs = pts[:, 0]
    xmin = int(np.min(xs))  # leftmost point of foot in ROI
    xmax = int(np.max(xs))  # rightmost point of foot in ROI
    cx = 0.5 * (xmin + xmax)

    margin = 3  # tolerance in pixels

    # Optional: draw leftmost/rightmost for debug
    heel_y = int(pts[np.argmin(xs), 1])
    toe_y = int(pts[np.argmax(xs), 1])
    cv2.circle(roi, (xmin, heel_y), 4, (0, 255, 255), -1)  # yellow: one side
    cv2.circle(roi, (xmax, toe_y), 4, (255, 0, 0), -1)  # blue: other side

    c = crease_x_roi

    # ---------- NEW DIRECTION-AWARE DECISION ----------
    if cx > c:
        # Foot mostly to the RIGHT of crease -> bowler coming from right, ball to LEFT
        # NO BALL only if *entire* foot is left of crease
        if xmax < c - margin:
            decision = "NO BALL"
        else:
            decision = "LEGAL"
    else:
        # Foot mostly to the LEFT of crease -> bowler coming from left, ball to RIGHT
        # NO BALL only if *entire* foot is right of crease
        if xmin > c + margin:
            decision = "NO BALL"
        else:
            decision = "LEGAL"
else:
    print("No valid foot contour found – tune filters.")

# put text on original image
cv2.putText(
    img, f"Decision: {decision}",
    (40, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.1,
    (0, 255, 0) if decision == "LEGAL" else (0, 0, 255),
    2
)

# ------------ 8. SHOW DEBUG WINDOWS ------------
cv2.imshow("ROI", roi)
cv2.imshow("Edges", edges)
cv2.imshow("Threshold dark", th)
cv2.imshow("Foot mask bottom", foot_mask)
cv2.imshow("Foot clean", foot_clean)
cv2.imshow("Sat mask", sat_mask)
cv2.imshow("Foot candidate", foot_candidate)
cv2.imshow("Original with crease & decision", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

