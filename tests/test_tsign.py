"""
Unit tests for TSIGN detector
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.tsign_detector import TSIGNDetector


class TestTSIGNDetector:
    """Test cases for TSIGN detector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = TSIGNDetector()
    
    def test_initialization(self):
        """Test detector initialization"""
        assert self.detector.name == "TSIGN"
        assert self.detector.version == "1.0"
        assert self.detector is not None
    
    def test_analyze_video_simulation(self):
        """Test video analysis in simulation mode"""
        # Create a dummy video path
        result = self.detector.analyze("dummy_video.mp4", "video")
        
        # Check result structure
        assert "prediction" in result
        assert "confidence" in result
        assert "processing_time" in result
        assert "frames_analyzed" in result
        
        # Check value ranges
        assert 0 <= result["prediction"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert result["processing_time"] > 0
    
    def test_analyze_image_simulation(self):
        """Test image analysis in simulation mode"""
        result = self.detector.analyze("dummy_image.jpg", "image")
        
        # Check result structure
        assert "prediction" in result
        assert "confidence" in result
        assert "spatial_consistency_score" in result
        
        # Check value ranges
        assert 0 <= result["prediction"] <= 1
        assert 0 <= result["confidence"] <= 1
    
    def test_unsupported_media_type(self):
        """Test handling of unsupported media types"""
        result = self.detector.analyze("dummy_file.txt", "text")
        
        assert result["prediction"] == 0.5
        assert result["confidence"] == 0.1
        assert "error" in result
    
    def test_detect_anomalous_regions(self):
        """Test anomalous region detection"""
        # Create dummy landmarks sequence
        landmarks_sequence = [np.random.rand(468 * 3) for _ in range(10)]
        
        anomalous_regions = self.detector._detect_anomalous_regions(landmarks_sequence)
        
        assert isinstance(anomalous_regions, list)
        for region in anomalous_regions:
            assert "region" in region
            assert "score" in region
            assert "description" in region
    
    def test_calculate_region_variance(self):
        """Test region variance calculation"""
        # Create dummy landmarks with varying patterns
        landmarks_sequence = [np.random.rand(468 * 3) for _ in range(5)]
        
        variance = self.detector._calculate_region_variance(landmarks_sequence, 0, 100)
        
        assert isinstance(variance, float)
        assert variance >= 0
    
    def test_analyze_spatial_consistency(self):
        """Test spatial consistency analysis"""
        # Create dummy landmarks
        landmarks = np.random.rand(468 * 3)
        
        consistency_score = self.detector._analyze_spatial_consistency(landmarks)
        
        assert 0 <= consistency_score <= 1
    
    def test_build_temporal_graph(self):
        """Test temporal graph construction"""
        landmarks_sequence = [np.random.rand(468 * 3) for _ in range(5)]
        
        graph = self.detector._build_temporal_graph(landmarks_sequence)
        
        assert "nodes" in graph
        assert "edge_index" in graph
        assert "num_frames" in graph
        assert graph["num_frames"] == 5
    
    def test_empty_landmarks_sequence(self):
        """Test handling of empty landmarks sequence"""
        result = self.detector._detect_anomalous_regions([])
        assert result == []
        
        variance = self.detector._calculate_region_variance([], 0, 100)
        assert variance == 0.0
    
    def test_error_handling(self):
        """Test error handling in analysis"""
        # Test with invalid file path
        result = self.detector.analyze("/invalid/path.mp4", "video")
        
        # Should return default result with error
        assert result["prediction"] == 0.5
        assert result["confidence"] == 0.1


if __name__ == "__main__":
    pytest.main([__file__])
