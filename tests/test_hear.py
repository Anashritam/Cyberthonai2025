"""
Unit tests for HEAR system integration
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.hear_detector import HEARDetector


class TestHEARDetector:
    """Test cases for HEAR system integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.detector = HEARDetector()
    
    def test_initialization(self):
        """Test HEAR detector initialization"""
        assert self.detector.version == "HEAR-v1.0"
        assert self.detector.tsign is not None
        assert self.detector.cmaf is not None
        assert self.detector.ead_snm is not None
    
    def test_analyze_media_simulation(self):
        """Test complete media analysis simulation"""
        result = self.detector.analyze_media_simulation("test_video.mp4")
        
        # Check main result structure
        assert "status" in result
        assert "analysis_id" in result
        assert "prediction" in result
        assert "confidence" in result
        assert "classification" in result
        assert "processing_time" in result
        
        # Check tier results
        assert "tier_results" in result
        tier_results = result["tier_results"]
        assert "tsign" in tier_results
        assert "cmaf" in tier_results
        assert "ead_snm" in tier_results
        
        # Check fusion weights
        assert "fusion_weights" in result
        weights = result["fusion_weights"]
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Should sum to ~1
        
        # Check evidence summary
        assert "evidence_summary" in result
        evidence = result["evidence_summary"]
        assert "temporal_anomalies" in evidence
        assert "audio_visual_misalignment" in evidence
        assert "statistical_anomalies" in evidence
    
    def test_detect_media_type(self):
        """Test media type detection"""
        assert self.detector._detect_media_type("test.mp4") == "video"
        assert self.detector._detect_media_type("test.jpg") == "image"
        assert self.detector._detect_media_type("test.mp3") == "audio"
        assert self.detector._detect_media_type("test.txt") == "unknown"
    
    def test_extract_metadata(self):
        """Test metadata extraction"""
        metadata = self.detector._extract_metadata("test.mp4", "video")
        
        assert "file_size_mb" in metadata
        assert "quality_score" in metadata
        assert "has_audio" in metadata
        assert "has_video" in metadata
        assert metadata["has_audio"] == True
        assert metadata["has_video"] == True
    
    def test_tier2_adaptive_fusion(self):
        """Test adaptive fusion logic"""
        # Create dummy tier 1 results
        tier1_results = {
            "tsign": {"prediction": 0.7, "confidence": 0.8},
            "cmaf": {"prediction": 0.6, "confidence": 0.9},
            "ead_snm": {"prediction": 0.8, "confidence": 0.7}
        }
        
        metadata = {"has_audio": True, "has_video": True}
        
        fusion_result = self.detector._tier2_adaptive_fusion(tier1_results, metadata)
        
        assert "weighted_prediction" in fusion_result
        assert "weights" in fusion_result
        assert "consensus_strength" in fusion_result
        
        # Check weights sum to 1
        weights = fusion_result["weights"]
        assert abs(sum(weights.values()) - 1.0) < 0.01
        
        # Check consensus strength
        assert 0 <= fusion_result["consensus_strength"] <= 1
    
    def test_tier3_meta_classifier(self):
        """Test meta-classifier logic"""
        fusion_result = {
            "weighted_prediction": 0.75,
            "consensus_strength": 0.85
        }
        
        metadata = {"quality_score": 0.8}
        
        final_result = self.detector._tier3_meta_classifier(fusion_result, metadata)
        
        assert "final_prediction" in final_result
        assert "final_confidence" in final_result
        
        # Check value ranges
        assert 0 <= final_result["final_prediction"] <= 1
        assert 0 <= final_result["final_confidence"] <= 1
    
    def test_get_classification(self):
        """Test classification mapping"""
        assert self.detector._get_classification(0.8) == "LIKELY_FAKE"
        assert self.detector._get_classification(0.5) == "UNCERTAIN"
        assert self.detector._get_classification(0.2) == "LIKELY_AUTHENTIC"
    
    def test_generate_evidence_summary(self):
        """Test evidence summary generation"""
        tier1_results = {
            "tsign": {"prediction": 0.7, "confidence": 0.8},
            "cmaf": {"prediction": 0.6, "confidence": 0.9},
            "ead_snm": {"prediction": 0.8, "confidence": 0.7}
        }
        
        evidence = self.detector._generate_evidence_summary(tier1_results)
        
        assert "temporal_anomalies" in evidence
        assert "audio_visual_misalignment" in evidence
        assert "statistical_anomalies" in evidence
        
        assert isinstance(evidence["temporal_anomalies"], list)
        assert isinstance(evidence["audio_visual_misalignment"], list)
        assert isinstance(evidence["statistical_anomalies"], list)
    
    def test_generate_chain_of_custody(self):
        """Test chain of custody generation"""
        chain = self.detector._generate_chain_of_custody("test_video.mp4", "test-id-123")
        
        assert "analysis_id" in chain
        assert "original_filename" in chain
        assert "analysis_timestamp" in chain
        assert "analyzer_version" in chain
        assert "processing_node" in chain
        assert "chain_verification" in chain
        
        assert chain["analysis_id"] == "test-id-123"
        assert chain["analyzer_version"] == "HEAR-v1.0"
    
    def test_weight_adjustment_audio_video(self):
        """Test weight adjustment for audio-video content"""
        tier1_results = {
            "tsign": {"prediction": 0.7, "confidence": 0.8},
            "cmaf": {"prediction": 0.6, "confidence": 0.9},
            "ead_snm": {"prediction": 0.8, "confidence": 0.7}
        }
        
        # Test with audio+video content
        metadata_av = {"has_audio": True, "has_video": True}
        fusion_av = self.detector._tier2_adaptive_fusion(tier1_results, metadata_av)
        
        # Test with video-only content
        metadata_v = {"has_audio": False, "has_video": True}
        fusion_v = self.detector._tier2_adaptive_fusion(tier1_results, metadata_v)
        
        # CMAF should get higher weight for audio+video content
        assert fusion_av["weights"]["cmaf"] >= fusion_v["weights"]["cmaf"]
    
    def test_consensus_strength_calculation(self):
        """Test consensus strength calculation"""
        # High consensus case
        tier1_high_consensus = {
            "tsign": {"prediction": 0.75, "confidence": 0.8},
            "cmaf": {"prediction": 0.73, "confidence": 0.9},
            "ead_snm": {"prediction": 0.77, "confidence": 0.7}
        }
        
        # Low consensus case
        tier1_low_consensus = {
            "tsign": {"prediction": 0.3, "confidence": 0.8},
            "cmaf": {"prediction": 0.8, "confidence": 0.9},
            "ead_snm": {"prediction": 0.6, "confidence": 0.7}
        }
        
        metadata = {"has_audio": True, "has_video": True}
        
        fusion_high = self.detector._tier2_adaptive_fusion(tier1_high_consensus, metadata)
        fusion_low = self.detector._tier2_adaptive_fusion(tier1_low_consensus, metadata)
        
        # High consensus should have higher consensus strength
        assert fusion_high["consensus_strength"] > fusion_low["consensus_strength"]


if __name__ == "__main__":
    pytest.main([__file__])
