# 🎨 AI Air Canvas

Draw in the air using just your webcam and hand gestures — powered by **MediaPipe** + **OpenCV**.

---

## ✨ Features

| Feature | Description |
|---|---|
| ✏ Air Drawing | Use your index finger as a brush |
| 🎨 Color Picker | Move finger to top bar to change color |
| ✌ Erase | Peace sign gesture (index + middle) erases |
| ✋ Clear | Open palm wipes the canvas |
| 🔵 Shape AI | Auto-detects circles, squares, triangles, stars, and more |
| 💾 Save | Press `S` to save your canvas as PNG |

---

## 🚀 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run
```bash
python air_canvas.py
```

---

## 🖐 Gesture Guide

```
Index finger only  →  Draw
Index + Middle     →  Erase  (peace sign ✌)
All fingers open   →  Clear canvas  (open palm ✋)
Finger in top bar  →  Pick color / clear
Press S            →  Save canvas as PNG
Press Q            →  Quit
```

---

## 🔵 Detected Shapes

After you lift your finger, the AI analyzes your stroke and labels:

- **Circle** — round, closed stroke
- **Ellipse** — oval shape
- **Triangle** — 3-sided polygon
- **Square** — equal-sided rectangle
- **Rectangle** — 4-sided polygon
- **Pentagon / Hexagon** — 5–6 sided shapes
- **Star** — spiky, concave shape
- **Arrow** — directional elongated shape
- **Line** — single straight stroke
- **Zigzag** — back-and-forth stroke
- **Free Form** — anything else

---

## 📁 Project Structure

```
air_canvas/
├── air_canvas.py       # Main app — webcam loop, gesture logic, UI
├── shape_recognizer.py # AI shape detection using contour geometry
├── requirements.txt    # Dependencies
└── README.md
```

---

## 🛠 How It Works

1. **MediaPipe Hands** detects 21 hand landmarks from the webcam feed in real-time.
2. Finger states (up/down) are computed by comparing tip vs. PIP joint positions.
3. Gesture rules map finger combinations to actions (draw / erase / clear).
4. Each stroke's (x, y) points are buffered; when you stop drawing, `shape_recognizer.py` renders them onto a binary image, runs contour analysis, measures **circularity**, **vertex count**, and **aspect ratio** to classify the shape.
5. The canvas is blended over the live camera feed for an AR effect.