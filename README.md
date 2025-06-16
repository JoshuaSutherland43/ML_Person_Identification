# 🧠 Real-Time Person Re-Identification System

This project is a **real-time person detection and re-identification system** built using **YOLOv8**, **DeepFace**, and **OpenCV**. It tracks individuals across frames and persists their identities even after they leave and re-enter the scene—based on clothing, facial features, and embedding similarities.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-success)
![DeepFace](https://img.shields.io/badge/Face-Recognition-DeepFace-yellow)

---

## 🎥 Demo



---

## 🚀 Features

- 🧍‍♂️ **Person Detection** using YOLOv8
- 🔁 **Re-identification** based on:
  - Visual clothing embeddings
  - Facial recognition via DeepFace
- 🧠 **Persistent ID Tracking** using saved embeddings
- 💾 **Automatic Snapshot Capture** every few seconds
- 🗂️ **ID Management** using JSON storage
- 🔄 **Multi-threaded** video processing for improved performance
- 📷 **Clothing-Aware Re-Identification**, even without clear facial data

---

## 🧰 Tech Stack

| Component     | Library/Tool        |
|---------------|---------------------|
| Detection     | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Face Matching | [DeepFace](https://github.com/serengil/deepface)               |
| Image/Video   | OpenCV + PIL         |
| Embedding Store | JSON / Local Storage |
| Threading     | Python `threading`, `queue`, `deque` |

---

## 🗃️ Directory Structure

```bash
📂 your_project/
├── main.py                 # Entry point
├── tracked_people/         # Snapshots of detected individuals
├── embeddings.json         # Persistent person ID mapping
├── requirements.txt
└── README.md
```

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/real-time-person-id.git
   cd real-time-person-id
   ```

2. **Install dependencies:**
   > Requires Python 3.10+
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

---

## 🧠 How It Works

1. **Detects persons** in each frame using YOLOv8.
2. **Takes a snapshot** of each individual every few seconds.
3. **Embeds** the image using DeepFace and/or clothing-based features.
4. **Matches** embeddings to existing IDs or creates new ones.
5. **Stores** the snapshot and ID reference in a JSON file.
6. **Reuses IDs** if the same person reappears—even with minor changes in position or lighting.

---

## 🔧 Configuration

You can configure:
- Snapshot interval (e.g., every 2 seconds)
- YOLO confidence threshold
- DeepFace backend (VGG-Face, Facenet, ArcFace, etc.)

---

## 📹 Sample Output

```text
[INFO] ID 002 matched with existing person (similarity: 0.91)
[INFO] ID 003 saved - new person detected.
[INFO] ID 001 re-identified after 15s absence.
```

---

## 🛠️ Requirements

- `opencv-python`
- `ultralytics`
- `deepface`
- `numpy`
- `Pillow`
- `logging`

---

## 📌 Future Enhancements

- ✅ Face + Clothing hybrid embeddings (done)
- ⏳ SQLite or NoSQL persistence for larger scale
- 🔍 Live camera streaming support
- 📈 Dashboard UI for visual tracking history

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Feel free to fork and improve the system.

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.

---

## 📬 Contact

For inquiries, ideas, or collaborations:  
📧 [joshua.jms11@gmail.com]  
🔗 [LinkedIn](https://linkedin.com/in/joshua-sutherland)
