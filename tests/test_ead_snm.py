"""
Unit tests for EAD-SNM detector
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.ead_snm_detector import EADSNMDetector


class TestEADSNMDetector:
    """Test cases for EAD-SNM detector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = EADSNMDetector()
    
    def test_initialization(self):
        """Test detector initialization"""
        assert self.detector.name == "EAD-SNM"
        assert self.detector.version == "1.0"
        assert self.detector is not None
    
    def test_analyze_video_simulation(self):
        """Test video analysis in simulation mode"""
        result = self.detector.analyze("dummy_video.mp4", "video")
        
        # Check result structure
        assert "prediction" in result
        assert "confidence" in result
        assert "anomaly_scores" in result
        assert "distribution_deviation" in result
        assert "processing_time" in result
        
        # Check value ranges
        assert 0 <= result["prediction"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert result["processing_time"] > 0
    
    def test_analyze_image_simulation(self):
        """Test image analysis in simulation mode"""
        result = self.detector.analyze("dummy_image.jpg", "image")
        
        assert "prediction" in result
        assert "confidence" in result
        assert "anomaly_scores" in result
        assert result["features_analyzed"] == 1
    
    def test_analyze_audio_simulation(self):
        """Test audio analysis in simulation mode"""
        result = self.detector.analyze("dummy_audio.mp3", "audio")
        
        assert "prediction" in result
        assert "confidence" in result
        assert "anomaly_scores" in result
    
    def test_ensemble_anomaly_detection(self):
        """Test ensemble anomaly detection"""
        # Create dummy features
        features = [np.random.rand(1024) for _ in range(10)]
        
        anomaly_scores = self.detector._ensemble_anomaly_detection(features)
        
        assert "reconstruction" in anomaly_scores
        assert "isolation_forest" in anomaly_scores
        assert "one_class_svm" in anomaly_scores
        
        for score in anomaly_scores.values():
            assert 0 <= score <= 1
    
    def test_empty_features(self):
        """Test handling of empty features"""
        anomaly_scores = self.detector._ensemble_anomaly_detection([])
        
        for score in anomaly_scores.values():
            assert score == 0.5
    
    def test_reconstruction_anomaly_score(self):
        """Test reconstruction-based anomaly scoring"""
        features = np.random.rand(10, 1024)
        score = self.detector._reconstruction_anomaly_score(features)
        
        assert 0 <= score <= 1
    
    def test_calculate_distribution_deviation(self):
        """Test distribution deviation calculation"""
        # Test with normal features
        features = [np.random.rand(100) for _ in range(10)]
        deviation = self.detector._calculate_distribution_deviation(features)
        assert deviation >= 0
        
        # Test with empty features
        deviation = self.detector._calculate_distribution_deviation([])
        assert deviation == 0.0
    
    def test_calculate_confidence(self):
        """Test confidence calculation"""
        anomaly_scores = {"reconstruction": 0.7, "isolation_forest": 0.6, "one_class_svm": 0.8}
        
        # Test with sufficient features
        confidence = self.detector._calculate_confidence(anomaly_scores, 25)
        assert 0.1 <= confidence <= 1.0
        
        # Test with insufficient features
        confidence = self.detector._calculate_confidence(anomaly_scores, 3)
        assert confidence < 0.7  # Should be lower
    
    def test_extract_frame_features(self):
        """Test frame feature extraction"""
        # Create dummy frame
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        features = self.detector._extract_frame_features(frame)
        
        if features is not None:
            assert len(features) == 1024  # Fixed feature size
            assert features.dtype == np.float64
    
    def test_train_models(self):
        """Test model training functionality"""
        # Create dummy training data
        training_data = [np.random.rand(1024) for _ in range(50)]
        
        # Should not raise exception
        self.detector.train_models(training_data)
    
    def test_unsupported_media_type(self):
        """Test handling of unsupported media types"""
        result = self.detector.analyze("dummy_file.txt", "text")
        
        assert result["prediction"] == 0.5
        assert result["confidence"] == 0.1
        assert "error" in result
    
    def test_create_autoencoder(self):
        """Test autoencoder creation"""
        autoencoder = self.detector._create_autoencoder()
        
        # Should create model or return None in simulation mode
        assert autoencoder is not None or not hasattr(self.detector, 'DEPENDENCIES_AVAILABLE')


if __name__ == "__main__":
    pytest.main([__file__])
