Below is the full markdown code for the **Deepfake Detection Tool - HEAR Framework** as requested:

```markdown
# 🚀 Deepfake Detection Tool - HEAR Framework

A forensic deepfake detection tool implementing the **Hierarchical Ensemble with Adaptive Routing (HEAR)** framework to identify manipulated images, videos, and audio using TSIGN, CMAF, and EAD-SNM approaches.

---

## 🛡️ Overview

The HEAR system is a **state-of-the-art deepfake detection tool** designed for forensic applications.  
It combines three specialized detection approaches:

- **TSIGN** → Temporal-Spatial Inconsistency Graph Network (temporal analysis)
- **CMAF** → Cross-Modal Alignment Filter (audio-visual synchronization)
- **EAD-SNM** → Entropy-Aware Distribution Shift Neural Monitor (anomaly detection)

---

## 🏗️ Architecture

The system operates across **three tiers**:

1. **Tier 1: Parallel Processing**  
   Simultaneous analysis by all three detectors.

2. **Tier 2: Adaptive Fusion**  
   Dynamic weighting based on input characteristics.

3. **Tier 3: Meta-Classification**  
   Final ensemble decision with uncertainty quantification.

---

## 🚀 Quick Start

### 🔧 Installation

```bash
git clone https://github.com/your-username/Deepfake-Detection-HEAR.git
cd Deepfake-Detection-HEAR
pip install -r requirements.txt
```

#### 📥 Download Pre-trained Models
Refer to `datasets/README.md` and place model files in:

```bash
src/models/weights/
```

### 🎬 Usage

#### Launch Streamlit Interface

```bash
streamlit run src/app.py
```

Access the web interface at → [http://localhost:8501](http://localhost:8501)

#### 🧩 API Usage

```python
from src.models.hear_detector import HEARDetector

# Initialize detector
detector = HEARDetector()

# Analyze media file
result = detector.analyze('path/to/media/file.mp4')

print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 📊 Model Components

### TSIGN (Temporal-Spatial Inconsistency Graph Network)
- **Purpose**: Detects temporal inconsistencies in facial landmarks and micro-expressions.
- **Architecture**: Graph Neural Network (GNN) with MediaPipe integration.
- **Input**: Video frames with facial landmark extraction.
- **Output**: Temporal consistency scores and anomaly regions.

### CMAF (Cross-Modal Alignment Filter)
- **Purpose**: Analyzes audio-visual synchronization for manipulation detection.
- **Architecture**: Vision Transformer + Audio CNN with cross-attention.
- **Input**: Video frames and audio spectrograms.
- **Output**: Synchronization scores and misalignment detection.

### EAD-SNM (Entropy-Aware Distribution Shift Neural Monitor)
- **Purpose**: Detects statistical anomalies using ensemble methods.
- **Architecture**: Autoencoder + Isolation Forest + One-Class SVM.
- **Input**: Raw media features.
- **Output**: Anomaly scores and distribution deviation metrics.

---

## 🧪 Testing

### Run all tests

```bash
pytest tests/
```

### Run specific component tests

```bash
pytest tests/test_tsign.py
pytest tests/test_cmaf.py
pytest tests/test_ead_snm.py
pytest tests/test_hear.py
```

### Run with coverage

```bash
pytest --cov=src tests/
```

---

## 📈 Performance

| Metric              | Value                              |
|---------------------|------------------------------------|
| Accuracy            | 77–85% on cross-dataset eval       |
| Processing Time     | 15–25 sec per 30-sec video         |
| Supported Formats   | MP4, AVI (video), MP3, WAV (audio), JPG, PNG (images) |

---

## 🔧 Configuration

Place model weights in the following paths:
```bash
src/models/weights/tsign_model.h5
src/models/weights/cmaf_model.h5
src/models/weights/ead_snm_model.h5
```

---

## 📄 License

This project is licensed under the terms of the LICENSE file.
```

This markdown code is formatted to match the structure and content you provided, ensuring proper headings, code blocks, tables, and links for clarity and readability. You can copy and paste this into any markdown-supported environment (e.g., GitHub README, Jupyter Notebook, or a markdown editor) to render it as intended.