# 🏏 Cricket No-Ball Detector

> **AI-powered front-foot no-ball detection using computer vision.**  
> Automatically determines whether a bowler's delivery is **LEGAL** or a **NO BALL** by analysing the front-foot position relative to the popping crease — in real time from a single image or video frame.

---

## 📸 Demo

| Input Frame | Output |
|-------------|--------|
| Broadcast screenshot | `Decision: NO BALL` or `Decision: LEGAL` overlaid with crease line + foot outline |

---

## 🧠 How It Works

The pipeline runs entirely on classical computer vision — no neural network required:

```
Input Image
    │
    ▼
1. Crop ROI          ← Focus on the crease/foot area
    │
    ▼
2. Preprocess        ← Grayscale + Gaussian blur
    │
    ▼
3. Crease Detection  ← Canny edges + Probabilistic Hough Lines
    │                   (longest near-vertical line = popping crease)
    ▼
4. Foot Mask         ← Otsu threshold (dark = foot) + bottom-half filter
    │
    ▼
5. Shadow Removal    ← HSV saturation mask filters out shadow blobs
    │
    ▼
6. Contour Selection ← Pick contour closest to crease & in lower frame
    │
    ▼
7. Classification    ← Compare foot x-extents vs crease x-position
    │
    ▼
Decision: LEGAL / NO BALL
```

### Decision Logic

| Scenario | NO BALL condition |
|----------|-------------------|
| Foot centre **right** of crease | Entire foot is left of crease |
| Foot centre **left** of crease  | Entire foot is right of crease |

A configurable pixel margin (`CREASE_MARGIN_PX`, default 3 px) provides tolerance for boundary cases.

---

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- pip

### Steps

```bash
# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
opencv-python>=4.8.0
numpy>=1.24.0
```

---

## 🚀 Usage

### Step 1 — Find your ROI coordinates

Run the ROI picker and click on the image to print the coordinates of the crease area:

```bash
python noball_detector.py --image path/to/frame.png --roi
```

Update `DEFAULT_ROI` at the top of `noball_detector.py` with your four coordinates `(x1, y1, x2, y2)`.

### Step 2 — Run detection

```bash
# Basic usage (uses DEFAULT_ROI from config)
python noball_detector.py --image path/to/frame.png

# Override ROI from CLI
python noball_detector.py --image path/to/frame.png --roi-coords 120 320 520 640

# Show all debug windows
python noball_detector.py --image path/to/frame.png --debug
```

### Output windows

| Window | Description |
|--------|-------------|
| `No-Ball Detector – Final` | Full image with crease line (red) and decision text |
| `ROI with crease & foot` | Cropped ROI with crease (green) and foot outline (orange) |
| `Edges` *(debug)* | Canny edge map |
| `Foot candidate` *(debug)* | Binary mask used for contour search |
| `Saturation mask` *(debug)* | Shadow-rejection mask |

---

## ⚙️ Configuration

All tunable parameters are at the top of `noball_detector.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_ROI` | `(120, 320, 520, 640)` | `(x1, y1, x2, y2)` crop region |
| `FOOT_BOTTOM_FRACTION` | `0.55` | Keep only bottom N% of ROI for foot detection |
| `SAT_THRESH` | `40` | HSV saturation threshold for shadow rejection |
| `MIN_CONTOUR_AREA` | `200` | Minimum blob area (px²) to consider as foot |
| `CREASE_MARGIN_PX` | `3` | Pixel tolerance around the crease line |
| `MIN_LINE_LENGTH` | `80` | Hough: minimum crease line length |
| `CANNY_LOW/HIGH` | `50/150` | Canny edge detection thresholds |

---

## 📁 Project Structure

```
cricket-noball-detector/
│
├── noball_detector.py     # Main detection script
├── requirements.txt       # Python dependencies
├── README.md
└── samples/               # (optional) sample frames for testing
    ├── legal_delivery.png
    └── noball_delivery.png
```

---

## 🔮 Roadmap

- [ ] Video / live feed support (frame-by-frame processing)
- [ ] Automatic ROI calibration using pitch keypoint detection
- [ ] Deep learning–based foot segmentation for tighter masks
- [ ] GUI dashboard with confidence score
- [ ] Export results to CSV for match analytics

---

## 🤝 Contributing

Pull requests are welcome! If you have a sample image where detection fails, please open an issue with the frame and your ROI coordinates.

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/my-improvement`
3. Commit your changes: `git commit -m 'Add better shadow removal'`
4. Push and open a PR

---

## 🙏 Acknowledgements

- [OpenCV](https://opencv.org/) — computer vision backbone
- Inspired by the need for affordable, broadcast-independent no-ball detection in grassroots cricket
