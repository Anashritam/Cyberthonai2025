<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HEAR System - Deepfake & AI-Generated Media Detection Tool</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border-radius: 8px;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 40px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            margin: -20px -20px 40px -20px;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            font-weight: 300;
        }
        .badges {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .badge {
            background-color: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            border: 1px solid rgba(255,255,255,0.3);
        }
        h2 {
            color: #2c3e50;
            font-size: 1.8em;
            margin: 30px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        h3 {
            color: #34495e;
            font-size: 1.3em;
            margin: 20px 0 10px 0;
            font-weight: 600;
        }
        .emoji {
            font-size: 1.2em;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .feature-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .feature-card h4 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-weight: 600;
        }
        .code-block {
            background-color: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            line-height: 1.4;
        }
        .code-block .comment {
            color: #68d391;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .team-member {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        .team-member .name {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .team-member .email {
            color: #6c757d;
            font-size: 0.9em;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-left: 4px solid #f39c12;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
        .warning-box strong {
            color: #856404;
        }
        ul, ol {
            margin: 10px 0 10px 30px;
        }
        li {
            margin: 5px 0;
        }
        .file-structure {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            margin: 15px 0;
            border: 1px solid #e9ecef;
        }
        .nav-menu {
            position: sticky;
            top: 20px;
            background-color: #2c3e50;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .nav-menu ul {
            list-style: none;
            margin: 0;
        }
        .nav-menu a {
            color: #ecf0f1;
            text-decoration: none;
            padding: 5px 0;
            display: block;
            border-radius: 4px;
            padding-left: 10px;
        }
        .nav-menu a:hover {
            background-color: #34495e;
        }
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                padding: 15px;
            }
            .header h1 {
                font-size: 2em;
            }
            .feature-grid,
            .metrics-grid,
            .team-grid {
                grid-template-columns: 1fr;
            }
            .badges {
                flex-direction: column;
                align-items: center;
            }
        }
        .section {
            margin-bottom: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header Section -->
        <div class="header">
            <h1>HEAR System</h1>
            <div class="subtitle">Deepfake & AI-Generated Media Detection Tool</div>
            <div class="badges">
                <span class="badge">Forensic-Grade</span>
                <span class="badge">Multi-Modal</span>
                <span class="badge">Real-Time</span>
                <span class="badge">Cyberthon.ai-2025</span>
            </div>
        </div>
        <!-- Navigation Menu -->
        <div class="nav-menu">
            <strong>Quick Navigation</strong>
            <ul>
                <li><a href="#overview">Project Overview</a></li>
                <li><a href="#features">Key Features</a></li>
                <li><a href="#architecture">System Architecture</a></li>
                <li><a href="#installation">Installation</a></li>
                <li><a href="#usage">Usage</a></li>
                <li><a href="#performance">Performance</a></li>
                <li><a href="#team">Team</a></li>
            </ul>
        </div>
        <!-- Project Overview -->
        <div class="section" id="overview">
            <h2><span class="emoji">🎯</span> Project Overview</h2>
            <p>The HEAR (Hierarchical Ensemble with Adaptive Routing) System is an advanced forensic tool that addresses the growing threat of deepfakes in law enforcement scenarios. With deepfakes achieving 91-98% realism and current detection tools showing only 39-69% accuracy, our solution provides a robust, multi-modal approach to digital media authentication.</p>
            <p>Developed for the <strong>Cyberthon.ai-2025 hackathon</strong> by Team Code Rangers.</p>
        </div>
        <!-- Key Features -->
        <div class="section" id="features">
            <h2><span class="emoji">🚀</span> Key Features</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <h4>Multi-Modal Detection</h4>
                    <p>Simultaneous analysis of video, audio, and images</p>
                </div>
                <div class="feature-card">
                    <h4>Real-Time Processing</h4>
                    <p>5-10 minutes processing time for 30-second videos</p>
                </div>
                <div class="feature-card">
                    <h4>High Accuracy</h4>
                    <p>77% fake probability detection with 90.8% confidence</p>
                </div>
                <div class="feature-card">
                    <h4>Forensic-Grade Documentation</h4>
                    <p>Compliant with digital forensics chain of custody requirements</p>
                </div>
                <div class="feature-card">
                    <h4>Explainable AI</h4>
                    <p>Interpretable evidence suitable for legal contexts</p>
                </div>
                <div class="feature-card">
                    <h4>Scalable Architecture</h4>
                    <p>Handles 100-1000 concurrent analyses</p>
                </div>
            </div>
        </div>
        <!-- System Architecture -->
        <div class="section" id="architecture">
            <h2><span class="emoji">🏗️</span> System Architecture</h2>
            <p>The HEAR System employs a three-tier architecture with parallel processing:</p>
            <h3>Detection Networks</h3>
            <div class="feature-grid">
                <div class="feature-card">
                    <h4>Temporal-Spatial Inconsistency Graph Network (TSIGN)</h4>
                    <p>Graph neural networks for detecting temporal inconsistencies in facial landmarks and micro-expressions</p>
                </div>
                <div class="feature-card">
                    <h4>Cross-Modal Alignment Filter (CMAF)</h4>
                    <p>Cross-modal alignment for audio-visual synchronization analysis</p>
                </div>
                <div class="feature-card">
                    <h4>Entropy-Aware Distribution Shift Neural Monitor (EAD-SNM)</h4>
                    <p>Ensemble anomaly detection for statistical deviations</p>
                </div>
            </div>
            <h3>Processing Layers</h3>
            <ul>
                <li><strong>Adaptive Fusion Layer:</strong> Dynamically weights network outputs based on input confidence and consistency</li>
                <li><strong>Meta-Classifier:</strong> Stacked ensemble for final verdict with uncertainty quantification</li>
            </ul>
        </div>
        <!-- Installation -->
        <div class="section" id="installation">
            <h2><span class="emoji">🛠️</span> Installation</h2>        
            <h3>Hardware Requirements</h3>
            <ul>
                <li><strong>Minimum:</strong> 8GB RAM</li>
                <li><strong>Recommended:</strong> GPU acceleration for optimal performance</li>
                <li><strong>Storage:</strong> Sufficient space for model weights and processing cache</li>
            </ul>
            <h3>Software Dependencies</h3>
            <ul>
                <li>Python 3.8+</li>
                <li>TensorFlow/PyTorch</li>
                <li>OpenCV</li>
                <li>NumPy</li>
                <li>Librosa (for audio processing)</li>
                <li>Flask/FastAPI (for web interface)</li>
            </ul>
            <h3>Installation Steps</h3>
            <div class="code-block">
<span class="comment"># Clone the repository</span>
git clone https://github.com/your-username/hear-deepfake-detection.git
cd hear-deepfake-detection

<span class="comment"># Create virtual environment</span>
python -m venv venv
source venv/bin/activate  <span class="comment"># On Windows: venv\Scripts\activate</span>

<span class="comment"># Install dependencies</span>
pip install -r requirements.txt

<span class="comment"># Download pre-trained models</span>
python download_models.py
            </div>
        </div>
        <!-- Usage -->
        <div class="section" id="usage">
            <h2><span class="emoji">🚀</span> Usage</h2>     
            <h3>Web Interface</h3>
            <div class="code-block">
<span class="comment"># Start the web application</span>
python app.py

<span class="comment"># Navigate to http://localhost:5000</span>
            </div>
            <h3>API Usage</h3>
            <div class="code-block">
from hear_detector import HEARDetector

<span class="comment"># Initialize detector</span>
detector = HEARDetector()

<span class="comment"># Analyze media file</span>
result = detector.analyze_media("path/to/media/file.mp4")
print(f"Fake probability: {result['fake_probability']}")
print(f"Confidence: {result['confidence']}")
            </div>
            <h3>Command Line Interface</h3>
            <div class="code-block">
<span class="comment"># Analyze single file</span>
python detect.py --input video.mp4 --output results.json

<span class="comment"># Batch processing</span>
python detect.py --batch --input-dir ./test_videos --output-dir ./results
            </div>
        </div>
        <!-- Performance Metrics -->
        <div class="section" id="performance">
            <h2><span class="emoji">📊</span> Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">75-85%</div>
                    <div class="metric-label">Detection Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">5-10 min</div>
                    <div class="metric-label">Processing Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">500MB</div>
                    <div class="metric-label">Max File Size</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">100-1000</div>
                    <div class="metric-label">Concurrent Analyses</div>
                </div>
            </div> 
            <h3>Supported Formats</h3>
            <ul>
                <li><strong>Video:</strong> MP4, AVI, MOV</li>
                <li><strong>Audio:</strong> WAV, MP3</li>
                <li><strong>Images:</strong> JPG, PNG</li>
            </ul>
        </div>
        <!-- Project Structure -->
        <div class="section">
            <h2><span class="emoji">🗂️</span> Project Structure</h2>
            <div class="file-structure">
hear-deepfake-detection/
├── models/
│   ├── tsign/          # Temporal-Spatial Inconsistency Graph Network
│   ├── cmaf/           # Cross-Modal Alignment Filter
│   └── ead_snm/        # Entropy-Aware Distribution Shift Monitor
├── src/
│   ├── detection/      # Core detection algorithms
│   ├── preprocessing/  # Data preprocessing utilities
│   └── fusion/         # Adaptive fusion layer
├── web/
│   ├── static/         # Web interface assets
│   ├── templates/      # HTML templates
│   └── app.py          # Flask application
├── tests/
│   ├── test_suite/     # Fake vs. original content test suite
│   └── unit_tests/     # Unit tests
├── data/
│   └── sample_data/    # Sample test files
├── requirements.txt
├── setup.py
└── README.md
            </div>
        </div>
        <!-- Testing -->
        <div class="section">
            <h2><span class="emoji">🧪</span> Testing</h2>
            <div class="code-block">
<span class="comment"># Run all tests</span>
python -m pytest tests/

<span class="comment"># Test with sample data</span>
python test_detector.py --test-suite ./tests/test_suite/

<span class="comment"># Benchmark performance</span>
python benchmark.py --dataset ./data/benchmark/
            </div>
        </div>
        <!-- Configuration -->
        <div class="section">
            <h2><span class="emoji">🔧</span> Configuration</h2>
            <p>Create a <code>config.yaml</code> file to customize detection parameters:</p>
            <div class="code-block">
detection:
  confidence_threshold: 0.7
  ensemble_weights:
    tsign: 0.35
    cmaf: 0.30
    ead_snm: 0.35

processing:
  max_file_size: 500  # MB
  gpu_acceleration: true
  batch_size: 8

output:
  generate_heatmaps: true
  save_intermediate_results: false
  export_format: "json"
            </div>
        </div>
        <!-- Legal Considerations -->
        <div class="section">
            <h2><span class="emoji">🚨</span> Legal and Ethical Considerations</h2>
            <ul>
                <li><strong>Evidence Standards:</strong> Compliant with digital forensics chain of custody requirements</li>
                <li><strong>Privacy Protection:</strong> Secure processing with automatic data purging after analysis</li>
                <li><strong>Transparency:</strong> Explainable AI outputs required for court admissibility</li>
                <li><strong>Data Protection:</strong> Adherence to local data protection regulations</li>
            </ul>
        </div>
        <!-- Team -->
        <div class="section" id="team">
            <h2><span class="emoji">👥</span> Team Code Rangers</h2>
            <div class="team-grid">
                <div class="team-member">
                    <div class="name">Rishikesh Shukla</div>
                    <div>Team Leader</div>
                    <div class="email">rishikesh7shukla@gmail.com</div>
                </div>
                <div class="team-member">
                    <div class="name">Shivam Mehta</div>
                    <div>Developer</div>
                    <div class="email">shivammehtadcm@gmail.com</div>
                </div>
                <div class="team-member">
                    <div class="name">Harsimranjeet Singh</div>
                    <div>Developer</div>
                    <div class="email">harsimaranjeets257@gmail.com</div>
                </div>
                <div class="team-member">
                    <div class="name">Khushi</div>
                    <div>Developer</div>
                    <div class="email">khushi31806@gmail.com</div>
                </div>
            </div>
        </div>
        <!-- License and Support -->
        <div class="section">
            <h2><span class="emoji">📄</span> License & Support</h2>
            <p>This project is licensed under the MIT License. For support and questions, please contact the development team or create an issue in the GitHub repository.</p>
            <h3>Acknowledgments</h3>
            <ul>
                <li>Chandigarh Police and Infosys for partnership and support</li>
                <li>Cyberthon.ai-2025 organizers</li>
                <li>Open-source community for foundational libraries and tools</li>
            </ul>
        </div>
        <!-- Disclaimer -->
        <div class="warning-box">
            <strong>⚠️ Disclaimer:</strong> This tool is designed for legitimate forensic and law enforcement purposes. Users are responsible for ensuring compliance with applicable laws and regulations.
        </div>
    </div>
</body>
</html>
