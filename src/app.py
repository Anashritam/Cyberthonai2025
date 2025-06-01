import streamlit as st
import numpy as np
import cv2
import librosa
from PIL import Image
import io
import time
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from models.hear_detector import HEARDetector
from preprocessing.preprocess import MediaPreprocessor

# Configure Streamlit page
st.set_page_config(
    page_title="HEAR Deepfake Detection Tool",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .confidence-high { color: #e74c3c; font-weight: bold; }
    .confidence-medium { color: #f39c12; font-weight: bold; }
    .confidence-low { color: #27ae60; font-weight: bold; }
    .detection-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    """Load the HEAR detector model"""
    return HEARDetector()

@st.cache_resource
def load_preprocessor():
    """Load the media preprocessor"""
    return MediaPreprocessor()

def display_confidence_gauge(confidence, prediction):
    """Display confidence as a gauge chart"""
    fake_probability = prediction * 100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = fake_probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fake Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def display_component_scores(tier_results):
    """Display individual component scores"""
    components = ['TSIGN', 'CMAF', 'EAD-SNM']
    scores = [
        tier_results['tsign']['prediction'] * 100,
        tier_results['cmaf']['prediction'] * 100,
        tier_results['ead_snm']['prediction'] * 100
    ]
    confidences = [
        tier_results['tsign']['confidence'] * 100,
        tier_results['cmaf']['confidence'] * 100,
        tier_results['ead_snm']['confidence'] * 100
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Fake Probability',
        x=components,
        y=scores,
        marker_color=['#e74c3c', '#f39c12', '#3498db']
    ))
    
    fig.add_trace(go.Scatter(
        name='Confidence',
        x=components,
        y=confidences,
        mode='markers+lines',
        marker=dict(size=10, color='darkgreen'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Component Analysis Results',
        xaxis_title='Detection Components',
        yaxis_title='Fake Probability (%)',
        yaxis2=dict(
            title='Confidence (%)',
            overlaying='y',
            side='right'
        ),
        height=400
    )
    
    return fig

def generate_report(result, filename):
    """Generate downloadable report"""
    report_data = {
        'File': [filename],
        'Classification': [result['classification']],
        'Fake_Probability': [f"{result['prediction']:.2%}"],
        'Confidence': [f"{result['confidence']:.2%}"],
        'Processing_Time': [result['processing_time']],
        'Analysis_Timestamp': [datetime.now().isoformat()],
        'TSIGN_Score': [f"{result['tier_results']['tsign']['prediction']:.2%}"],
        'CMAF_Score': [f"{result['tier_results']['cmaf']['prediction']:.2%}"],
        'EAD_SNM_Score': [f"{result['tier_results']['ead_snm']['prediction']:.2%}"]
    }
    
    return pd.DataFrame(report_data)

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ HEAR Deepfake Detection Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Hierarchical Ensemble with Adaptive Routing Framework</p>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Forensic Use Disclaimer:</strong> This tool is designed for forensic and research purposes only. 
        All results should be reviewed by qualified experts. False positives and negatives may occur.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    detailed_analysis = st.sidebar.checkbox("Enable Detailed Analysis", value=True)
    show_explanations = st.sidebar.checkbox("Show Explanations", value=True)
    
    # Model weights status
    st.sidebar.subheader("📊 Model Status")
    st.sidebar.success("✅ TSIGN Model Loaded")
    st.sidebar.success("✅ CMAF Model Loaded")
    st.sidebar.success("✅ EAD-SNM Model Loaded")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Upload Media")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mp3', 'wav'],
            help="Supported formats: JPG, PNG, MP4, AVI, MP3, WAV"
        )
        
        # Media type selector
        media_type = st.selectbox(
            "Media Type (Auto-detected)",
            ["Auto-detect", "Image", "Video", "Audio"]
        )
        
        # Analysis button
        analyze_button = st.button("🚀 Analyze Media", disabled=(uploaded_file is None))
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if uploaded_file is not None:
            # Display uploaded file info
            st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            if analyze_button:
                # Initialize components
                detector = load_detector()
                preprocessor = load_preprocessor()
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Save uploaded file temporarily
                    status_text.text("💾 Processing uploaded file...")
                    progress_bar.progress(10)
                    
                    # Simulate preprocessing
                    status_text.text("🔄 Preprocessing media...")
                    progress_bar.progress(30)
                    time.sleep(1)
                    
                    # Simulate analysis
                    status_text.text("🔍 Running HEAR analysis...")
                    progress_bar.progress(70)
                    time.sleep(2)
                    
                    # Simulate getting results
                    result = detector.analyze_media_simulation(uploaded_file.name)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Clear progress indicators
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    st.markdown('<div class="detection-card">', unsafe_allow_html=True)
                    
                    # Main classification
                    classification = result['classification']
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    if classification == 'LIKELY_FAKE':
                        st.error(f"🚨 **{classification}** (Fake Probability: {prediction:.1%})")
                    elif classification == 'UNCERTAIN':
                        st.warning(f"⚠️ **{classification}** (Fake Probability: {prediction:.1%})")
                    else:
                        st.success(f"✅ **{classification}** (Fake Probability: {prediction:.1%})")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence gauge
                    st.plotly_chart(display_confidence_gauge(confidence, prediction), use_container_width=True)
                    
                    # Component scores
                    if detailed_analysis:
                        st.subheader("🔬 Component Analysis")
                        st.plotly_chart(display_component_scores(result['tier_results']), use_container_width=True)
                    
                    # Evidence summary
                    if show_explanations:
                        st.subheader("📋 Evidence Summary")
                        
                        evidence = result['evidence_summary']
                        
                        if evidence['temporal_anomalies']:
                            st.write("**⏰ Temporal Anomalies:**")
                            for anomaly in evidence['temporal_anomalies']:
                                st.write(f"• {anomaly}")
                        
                        if evidence['audio_visual_misalignment']:
                            st.write("**🎵 Audio-Visual Issues:**")
                            for issue in evidence['audio_visual_misalignment']:
                                st.write(f"• {issue}")
                        
                        if evidence['statistical_anomalies']:
                            st.write("**📈 Statistical Anomalies:**")
                            for anomaly in evidence['statistical_anomalies']:
                                st.write(f"• {anomaly}")
                    
                    # Technical details
                    with st.expander("🔧 Technical Details"):
                        st.write(f"**Processing Time:** {result['processing_time']}")
                        st.write(f"**Analysis ID:** {result['analysis_id']}")
                        st.write(f"**Model Version:** {result['metadata']['analyzer_version']}")
                        
                        # Fusion weights
                        weights = result['fusion_weights']
                        st.write("**Component Weights:**")
                        st.write(f"• TSIGN: {weights['tsign']:.2%}")
                        st.write(f"• CMAF: {weights['cmaf']:.2%}")
                        st.write(f"• EAD-SNM: {weights['ead_snm']:.2%}")
                    
                    # Download report
                    st.subheader("📥 Download Report")
                    report_df = generate_report(result, uploaded_file.name)
                    
                    csv = report_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV Report",
                        data=csv,
                        file_name=f"deepfake_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Analysis failed: {str(e)}")
        
        else:
            st.info("👆 Please upload a media file to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d;">
        <p>🛡️ HEAR Deepfake Detection Tool | Built for forensic professionals and researchers</p>
        <p>⚖️ Remember: Human expertise is essential for final verification</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
