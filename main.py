"""
Cricket No-Ball Detector
========================
Detects whether a bowler's delivery is a NO BALL or LEGAL based on
front-foot position relative to the popping crease.

Usage:
    python noball_detector.py --image path/to/frame.png [--roi] [--debug]

Arguments:
    --image   Path to the input image/frame
    --roi     Launch the ROI picker (click to print coordinates, then close)
    --debug   Show all intermediate debug windows
"""

import cv2
import numpy as np
import argparse
import sys


# ──────────────────────────────────────────────────────────
# CONFIG  (edit these after running with --roi)
# ──────────────────────────────────────────────────────────
DEFAULT_ROI = (120, 320, 520, 640)   # x1, y1, x2, y2
FOOT_BOTTOM_FRACTION = 0.55          # keep only the bottom N% of ROI for foot
SAT_THRESH           = 40            # HSV saturation threshold (shadow filter)
MIN_CONTOUR_AREA     = 200           # pixels²
CREASE_MARGIN_PX     = 3             # tolerance around crease
MIN_LINE_LENGTH      = 80
MAX_LINE_GAP         = 10
CANNY_LOW, CANNY_HIGH = 50, 150
# ──────────────────────────────────────────────────────────


def pick_roi(image_path: str) -> None:
    """Interactive helper: click on image to print (x, y) coordinates."""
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"[ERROR] Cannot load image: {image_path}")

    print("[ROI PICKER] Click on image to get coordinates. Press any key to exit.")

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"  Clicked  X={x}, Y={y}")

    cv2.imshow("ROI Picker – press any key to close", img)
    cv2.setMouseCallback("ROI Picker – press any key to close", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_crease(roi_blur: np.ndarray, roi_gray: np.ndarray) -> tuple | None:
    """Return (x1, y1, x2, y2) of the best vertical crease line, or None."""
    edges = cv2.Canny(roi_blur, CANNY_LOW, CANNY_HIGH)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    best_line, max_len = None, 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if abs(dx) < 20:                          # mostly vertical
                length = np.hypot(dx, y2 - y1)
                if length > max_len:
                    max_len = length
                    best_line = (x1, y1, x2, y2)

    return best_line, edges


def build_foot_mask(roi: np.ndarray, roi_blur: np.ndarray) -> tuple:
    """Return (foot_candidate mask, sat_mask, th, foot_clean) for the given ROI."""
    h, w = roi_blur.shape

    # --- Otsu threshold (dark pixels = foreground) ---
    _, th = cv2.threshold(
        roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Keep only the bottom portion of the ROI
    foot_mask = np.zeros_like(th)
    bottom_start = int(h * FOOT_BOTTOM_FRACTION)
    foot_mask[bottom_start:, :] = th[bottom_start:, :]

    # Morphological clean-up
    kernel = np.ones((5, 5), np.uint8)
    foot_clean = cv2.morphologyEx(foot_mask, cv2.MORPH_OPEN, kernel)
    foot_clean = cv2.morphologyEx(foot_clean, cv2.MORPH_CLOSE, kernel)

    # Shadow suppression via HSV saturation
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)
    _, sat_mask = cv2.threshold(S, SAT_THRESH, 255, cv2.THRESH_BINARY)

    foot_candidate = cv2.bitwise_and(foot_clean, sat_mask)
    return foot_candidate, sat_mask, th, foot_clean


def select_foot_contour(
    foot_candidate: np.ndarray,
    crease_x: int,
    roi_h: int,
) -> np.ndarray | None:
    """Pick the contour most likely to be the front foot."""
    contours, _ = cv2.findContours(
        foot_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    foot_contour, best_score = None, -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = w_box / float(h_box + 1e-3)
        cx = x + w_box / 2.0

        # Must be near crease horizontally
        if not (crease_x - 20 <= cx <= crease_x + 100):
            continue

        # Must be low in the frame (foot, not knee/hip)
        if (y + h_box) < 0.6 * roi_h:
            continue

        # Reasonable bounding-box aspect ratio
        if not (0.4 < aspect_ratio < 3.0):
            continue

        score = area - 5 * abs(cx - crease_x)
        if score > best_score:
            best_score = score
            foot_contour = cnt

    return foot_contour


def classify_delivery(
    foot_contour: np.ndarray,
    crease_x: int,
) -> tuple[str, dict]:
    """
    Returns (decision, debug_info).
    decision: 'LEGAL' | 'NO BALL' | 'UNKNOWN'
    """
    if foot_contour is None:
        return "UNKNOWN", {}

    pts = foot_contour.reshape(-1, 2)
    xs = pts[:, 0]
    xmin, xmax = int(np.min(xs)), int(np.max(xs))
    cx = 0.5 * (xmin + xmax)

    c = crease_x
    m = CREASE_MARGIN_PX

    if cx > c:
        # Foot centre right of crease  → bowler coming from right side
        decision = "NO BALL" if xmax < c - m else "LEGAL"
    else:
        # Foot centre left of crease   → bowler coming from left side
        decision = "NO BALL" if xmin > c + m else "LEGAL"

    debug = {
        "xmin": xmin,
        "xmax": xmax,
        "foot_cx": cx,
        "crease_x": c,
    }
    return decision, debug


def annotate(
    img: np.ndarray,
    roi: np.ndarray,
    roi_offsets: tuple,
    crease_line: tuple,
    foot_contour: np.ndarray | None,
    decision: str,
    debug_info: dict,
) -> None:
    """Draw crease line, foot outline, and decision text on img and roi."""
    rx1, ry1, rx2, ry2 = roi_offsets
    x1, y1, x2, y2 = crease_line

    # Crease on ROI (green) and full image (red)
    cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(img, (x1 + rx1, y1 + ry1), (x2 + rx1, y2 + ry1), (0, 0, 255), 2)

    if foot_contour is not None:
        # Foot outline
        cv2.drawContours(roi, [foot_contour], -1, (255, 165, 0), 2)

        pts = foot_contour.reshape(-1, 2)
        xs = pts[:, 0]
        xmin, xmax = int(np.min(xs)), int(np.max(xs))

        heel_y = int(pts[np.argmin(xs), 1])
        toe_y  = int(pts[np.argmax(xs), 1])
        cv2.circle(roi, (xmin, heel_y), 5, (0, 255, 255), -1)   # yellow
        cv2.circle(roi, (xmax, toe_y),  5, (255,   0,   0), -1) # blue

    # Decision banner
    color = (0, 200, 0) if decision == "LEGAL" else (0, 0, 220)
    cv2.putText(
        img,
        f"Decision: {decision}",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        2,
        cv2.LINE_AA,
    )

    # Small debug info in corner
    info_lines = [
        f"Crease X : {debug_info.get('crease_x', '?')}",
        f"Foot xmin: {debug_info.get('xmin', '?')}",
        f"Foot xmax: {debug_info.get('xmax', '?')}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(
            img, line,
            (10, img.shape[0] - 20 - i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (200, 200, 200), 1, cv2.LINE_AA,
        )


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def run(image_path: str, roi_coords: tuple, show_debug: bool = False) -> str:
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"[ERROR] Cannot load image: {image_path}")

    print(f"[INFO] Image shape: {img.shape}")

    rx1, ry1, rx2, ry2 = roi_coords
    roi = img[ry1:ry2, rx1:rx2].copy()

    if roi.size == 0:
        sys.exit("[ERROR] ROI is empty – adjust coordinates.")

    # ── Preprocessing ──
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)

    # ── Crease detection ──
    crease_line, edges = detect_crease(roi_blur, roi_gray)
    if crease_line is None:
        print("[WARN] No crease line detected – tune Hough/Canny parameters.")
        if show_debug:
            cv2.imshow("ROI",   roi)
            cv2.imshow("Edges", edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return "UNKNOWN"

    crease_x = int((crease_line[0] + crease_line[2]) / 2)
    print(f"[INFO] Crease line (ROI): {crease_line}  crease_x={crease_x}")

    # ── Foot mask ──
    foot_candidate, sat_mask, th, foot_clean = build_foot_mask(roi, roi_blur)

    # ── Contour selection ──
    h_roi = roi.shape[0]
    foot_contour = select_foot_contour(foot_candidate, crease_x, h_roi)
    if foot_contour is None:
        print("[WARN] No valid foot contour found – tune filters.")

    # ── Classification ──
    decision, debug_info = classify_delivery(foot_contour, crease_x)
    print(f"[RESULT] Decision: {decision}  |  {debug_info}")

    # ── Annotate ──
    annotate(
        img, roi,
        (rx1, ry1, rx2, ry2),
        crease_line,
        foot_contour,
        decision,
        debug_info,
    )

    # ── Display ──
    cv2.imshow("No-Ball Detector – Final", img)
    cv2.imshow("ROI with crease & foot",   roi)

    if show_debug:
        cv2.imshow("Edges",            edges)
        cv2.imshow("Threshold (dark)", th)
        cv2.imshow("Foot clean",       foot_clean)
        cv2.imshow("Saturation mask",  sat_mask)
        cv2.imshow("Foot candidate",   foot_candidate)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return decision


# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cricket No-Ball Detector")
    parser.add_argument("--image", required=False, help="Path to input image")
    parser.add_argument(
        "--roi", action="store_true",
        help="Launch ROI picker (click to print coordinates)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show intermediate debug windows"
    )
    parser.add_argument(
        "--roi-coords", nargs=4, type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=list(DEFAULT_ROI),
        help="ROI bounding box: x1 y1 x2 y2"
    )
    args = parser.parse_args()

    if args.roi:
        if not args.image:
            sys.exit("[ERROR] --roi requires --image")
        pick_roi(args.image)
    elif args.image:
        roi = tuple(args.roi_coords)
        run(args.image, roi, show_debug=args.debug)
    else:
        parser.print_help()

