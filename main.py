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

