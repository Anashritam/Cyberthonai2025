"""
Unit tests for CMAF detector
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cmaf_detector import CMAFDetector


class TestCMAFDetector:
    """Test cases for CMAF detector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = CMAFDetector()
    
    def test_initialization(self):
        """Test detector initialization"""
        assert self.detector.name == "CMAF"
        assert self.detector.version == "1.0"
        assert self.detector is not None
    
    def test_phoneme_viseme_mapping(self):
        """Test phoneme-viseme mapping creation"""
        mapping = self.detector.phoneme_viseme_map
        
        assert isinstance(mapping, dict)
        assert "bilabial" in mapping
        assert "vowel_open" in mapping
        assert isinstance(mapping["bilabial"], list)
    
    def test_analyze_video_simulation(self):
        """Test video analysis in simulation mode"""
        result = self.detector.analyze("dummy_video.mp4", "video")
        
        # Check result structure
        assert "prediction" in result
        assert "confidence" in result
        assert "sync_anomalies" in result
        assert "audio_visual_consistency" in result
        assert "processing_time" in result
        
        # Check value ranges
        assert 0 <= result["prediction"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert result["processing_time"] > 0
    
    def test_analyze_audio_only(self):
        """Test audio-only analysis"""
        result = self.detector.analyze("dummy_audio.mp3", "audio")
        
        assert "prediction" in result
        assert "confidence" in result
        assert "audio_consistency" in result
        assert result["confidence"] == 0.5  # Lower confidence for audio-only
    
    def test_detect_sync_anomalies(self):
        """Test synchronization anomaly detection"""
        # Create dummy features
        visual_features = [np.random.rand(512) for _ in range(10)]
        audio_features = [np.random.rand(1024) for _ in range(10)]
        
        anomalies = self.detector._detect_sync_anomalies(visual_features, audio_features)
        
        assert isinstance(anomalies, list)
        for anomaly in anomalies:
            assert "timestamp" in anomaly
            assert "type" in anomaly
            assert "score" in anomaly
            assert "description" in anomaly
    
    def test_calculate_sync_score(self):
        """Test synchronization score calculation"""
        # Test with no anomalies
        sync_score = self.detector._calculate_sync_score([])
        assert sync_score == 0.9
        
        # Test with anomalies
        anomalies = [{"score": 0.8}, {"score": 0.6}]
        sync_score = self.detector._calculate_sync_score(anomalies)
        assert 0 <= sync_score <= 1
    
    def test_calculate_confidence(self):
        """Test confidence calculation"""
        anomalies = [{"score": 0.8}]
        confidence = self.detector._calculate_confidence(anomalies, 20)
        
        assert 0 <= confidence <= 1
    
    def test_analyze_audio_consistency(self):
        """Test audio consistency analysis"""
        # Test with dummy audio features
        audio_features = [np.random.rand(128) for _ in range(5)]
        consistency = self.detector._analyze_audio_consistency(audio_features)
        
        assert 0 <= consistency <= 1
        
        # Test with insufficient features
        consistency = self.detector._analyze_audio_consistency([])
        assert consistency == 0.5
    
    def test_align_temporal_features(self):
        """Test temporal feature alignment"""
        visual_features = [np.random.rand(512) for _ in range(5)]
        visual_timestamps = [i * 0.1 for i in range(5)]
        audio_features = [np.random.rand(1024) for _ in range(5)]
        audio_timestamps = [i * 0.1 for i in range(5)]
        
        aligned_visual, aligned_audio = self.detector._align_temporal_features(
            visual_features, visual_timestamps, audio_features, audio_timestamps
        )
        
        assert len(aligned_visual) == len(aligned_audio)
        assert len(aligned_visual) <= len(visual_features)
    
    def test_unsupported_media_type(self):
        """Test handling of unsupported media types"""
        result = self.detector.analyze("dummy_file.txt", "text")
        
        assert result["prediction"] == 0.5
        assert result["confidence"] == 0.1
        assert "error" in result
    
    def test_mismatched_feature_lengths(self):
        """Test handling of mismatched feature lengths"""
        visual_features = [np.random.rand(512) for _ in range(3)]
        audio_features = [np.random.rand(1024) for _ in range(5)]
        
        # Should handle gracefully
        anomalies = self.detector._detect_sync_anomalies(visual_features, audio_features)
        assert isinstance(anomalies, list)


if __name__ == "__main__":
    pytest.main([__file__])
