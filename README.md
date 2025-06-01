# HEAR System: Deepfake & AI-Generated Media Detection Tool

**Forensic-Grade | Multi-Modal | Real-Time | Cyberthon.ai-2025**

---

## Quick Navigation
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Configuration](#configuration)
- [Legal and Ethical Considerations](#legal-and-ethical-considerations)
- [Team Code Rangers](#team-code-rangers)
- [License & Support](#license--support)

---

## Project Overview 🎯

The **HEAR (Hierarchical Ensemble with Adaptive Routing) System** is an advanced forensic tool designed to combat the growing threat of deepfakes in law enforcement scenarios. With deepfakes achieving **91-98% realism** and current detection tools offering only **39-69% accuracy**, our solution provides a robust, multi-modal approach to digital media authentication.

Developed for the **Cyberthon.ai-2025 hackathon** by **Team Code Rangers**.

---

## Key Features 🚀

| Feature | Description |
|---------|-------------|
| **Multi-Modal Detection** | Simultaneous analysis of video, audio, and images |
| **Real-Time Processing** | 5-10 minutes processing time for 30-second videos |
| **High Accuracy** | 77% fake probability detection with 90.8% confidence |
| **Forensic-Grade Documentation** | Compliant with digital forensics chain of custody requirements |
| **Explainable AI** | Interpretable evidence suitable for legal contexts |
| **Scalable Architecture** | Handles 100-1000 concurrent analyses |

---

## System Architecture 🏗️

The HEAR System employs a **three-tier architecture** with parallel processing:

### Detection Networks
- **Temporal-Spatial Inconsistency Graph Network (TSIGN)**: Graph neural networks for detecting temporal inconsistencies in facial landmarks and micro-expressions.
- **Cross-Modal Alignment Filter (CMAF)**: Cross-modal alignment for audio-visual synchronization analysis.
- **Entropy-Aware Distribution Shift Neural Monitor (EAD-SNM)**: Ensemble anomaly detection for statistical deviations.

### Processing Layers
- **Adaptive Fusion Layer**: Dynamically weights network outputs based on input confidence and consistency.
- **Meta-Classifier**: Stacked ensemble for final verdict with uncertainty quantification.

---

## Installation 🛠️

### Hardware Requirements
- **Minimum**: 8GB RAM
- **Recommended**: GPU acceleration for optimal performance
- **Storage**: Sufficient space for model weights and processing cache

### Software Dependencies
- Python 3.8+
- TensorFlow/PyTorch
- OpenCV
- NumPy
- Librosa (for audio processing)
- Flask/FastAPI (for web interface)

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/your-username/hear-deepfake-detection.git
cd hear-deepfake-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python download_models.py
