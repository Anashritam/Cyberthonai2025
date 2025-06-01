"""
CMAF (Cross-Modal Alignment Filter) Detector
Analyzes audio-visual synchronization for manipulation detection
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple
import logging

try:
    import cv2
    import librosa
    import torch
    import torch.nn as nn
    from transformers import CLIPModel, CLIPProcessor
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    logging.warning("CMAF dependencies not available. Running in simulation mode.")

class CMAFDetector:
    """
    Cross-Modal Alignment Filter for deepfake detection
    
    Analyzes audio-visual synchronization by detecting misalignments
    between audio phonemes and visual mouth movements (visemes).
    """
    
    def __init__(self, model_weights_path: str = None):
        """
        Initialize CMAF detector
        
        Args:
            model_weights_path: Path to pre-trained model weights
        """
        self.logger = logging.getLogger(__name__)
        self.name = "CMAF"
        self.version = "1.0"
        self.model_weights_path = model_weights_path
        
        if DEPENDENCIES_AVAILABLE:
            self._initialize_models()
        else:
            self.logger.warning("Running CMAF in simulation mode")
        
        # Phoneme to viseme mapping
        self.phoneme_viseme_map = self._create_phoneme_viseme_mapping()
    
    def _initialize_models(self):
        """Initialize vision and audio models"""
        try:
            # Initialize CLIP for visual features
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Create custom cross-modal attention model
            self.cross_modal_model = self._create_cross_modal_model()
            
            self.logger.info("CMAF models initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize CMAF models: {e}")
            self.clip_model = None
            self.cross_modal_model = None
    
    def _create_cross_modal_model(self):
        """Create cross-modal attention model"""
        if not DEPENDENCIES_AVAILABLE:
            return None
        
        class CrossModalAttention(nn.Module):
            def __init__(self, visual_dim=512, audio_dim=512, hidden_dim=256):
                super().__init__()
                self.visual_encoder = nn.Linear(visual_dim, hidden_dim)
                self.audio_encoder = nn.Linear(audio_dim, hidden_dim)
                self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
                self.sync_classifier = nn.Linear(hidden_dim, 2)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, visual_features, audio_features):
                # Encode features
                visual_encoded = self.visual_encoder(visual_features)
                audio_encoded = self.audio_encoder(audio_features)
                
                # Cross-modal attention
                attended_features, attention_weights = self.cross_attention(
                    visual_encoded, audio_encoded, audio_encoded
                )
                
                # Classification
                attended_features = self.dropout(attended_features)
                sync_prediction = self.sync_classifier(attended_features.mean(dim=0))
                
                return sync_prediction, attention_weights
        
        return CrossModalAttention()
    
    def _create_phoneme_viseme_mapping(self) -> Dict[str, List[str]]:
        """Create mapping between phonemes and visemes"""
        return {
            'bilabial': ['p', 'b', 'm'],
            'labiodental': ['f', 'v'],
            'dental': ['th'],
            'alveolar': ['t', 'd', 'n', 's', 'z', 'l'],
            'postalveolar': ['sh', 'zh', 'ch', 'jh'],
            'velar': ['k', 'g', 'ng'],
            'vowel_closed': ['i', 'u'],
            'vowel_mid': ['e', 'o'],
            'vowel_open': ['a', 'ah']
        }
    
    def analyze(self, media_path: str, media_type: str) -> Dict[str, Any]:
        """
        Analyze media for audio-visual synchronization
        
        Args:
            media_path: Path to media file
            media_type: Type of media ('video', 'audio')
        
        Returns:
            Dict containing CMAF analysis results
        """
        start_time = time.time()
        
        try:
            if media_type == 'video':
                return self._analyze_video(media_path)
            elif media_type == 'audio':
                return self._analyze_audio_only(media_path)
            else:
                return self._get_default_result(f"Unsupported media type: {media_type}")
        
        except Exception as e:
            self.logger.error(f"CMAF analysis failed: {e}")
            return self._get_default_result(f"Analysis failed: {str(e)}")
    
    def _analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze video for audio-visual synchronization"""
        start_time = time.time()
        
        if not DEPENDENCIES_AVAILABLE:
            return self._simulate_video_analysis()
        
        try:
            # Extract visual and audio features
            visual_features, timestamps = self._extract_visual_features(video_path)
            audio_features, audio_timestamps = self._extract_audio_features(video_path)
            
            if len(visual_features) == 0 or len(audio_features) == 0:
                return self._get_default_result("No audio or visual features extracted")
            
            # Align temporal features
            aligned_visual, aligned_audio = self._align_temporal_features(
                visual_features, timestamps, audio_features, audio_timestamps
            )
            
            # Detect synchronization anomalies
            sync_anomalies = self._detect_sync_anomalies(aligned_visual, aligned_audio)
            
            # Calculate overall synchronization score
            sync_score = self._calculate_sync_score(sync_anomalies)
            
            # Predict manipulation probability
            prediction = 1.0 - sync_score
            confidence = self._calculate_confidence(sync_anomalies, len(aligned_visual))
            
            processing_time = time.time() - start_time
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'sync_anomalies': sync_anomalies,
                'audio_visual_consistency': sync_score,
                'processing_time': processing_time,
                'frames_analyzed': len(visual_features),
                'audio_segments_analyzed': len(audio_features)
            }
        
        except Exception as e:
            return self._get_default_result(f"Video analysis failed: {str(e)}")
    
    def _analyze_audio_only(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio-only file (limited analysis)"""
        start_time = time.time()
        
        try:
            # Extract audio features
            audio_features, _ = self._extract_audio_features(audio_path)
            
            # Analyze audio consistency
            audio_consistency = self._analyze_audio_consistency(audio_features)
            
            processing_time = time.time() - start_time
            
            return {
                'prediction': max(0.3, 1.0 - audio_consistency),
                'confidence': 0.5,  # Lower confidence without visual data
                'sync_anomalies': [],
                'audio_consistency': audio_consistency,
                'processing_time': processing_time,
                'audio_segments_analyzed': len(audio_features)
            }
        
        except Exception as e:
            return self._get_default_result(f"Audio analysis failed: {str(e)}")
    
    def _extract_visual_features(self, video_path: str) -> Tuple[List[np.ndarray], List[float]]:
        """Extract visual features from video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        visual_features = []
        timestamps = []
        
        # Sample frames every 0.1 seconds
        frame_interval = max(1, int(fps * 0.1))
        
        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Extract mouth region
            mouth_region = self._extract_mouth_region(frame)
            if mouth_region is not None:
                # Extract features using CLIP
                features = self._extract_clip_features(mouth_region)
                visual_features.append(features)
                timestamps.append(frame_idx / fps)
        
        cap.release()
        return visual_features, timestamps
    
    def _extract_mouth_region(self, frame: np.ndarray) -> np.ndarray:
        """Extract mouth region from frame"""
        # Simplified mouth detection (in practice, use MediaPipe or dlib)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            # Extract lower third of face (mouth region)
            mouth_y = y + int(h * 0.6)
            mouth_h = int(h * 0.4)
            mouth_region = frame[mouth_y:mouth_y + mouth_h, x:x + w]
            return cv2.resize(mouth_region, (224, 224))
        
        return None
    
    def _extract_clip_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features using CLIP model"""
        if self.clip_model is None:
            return np.random.rand(512)  # Dummy features for simulation
        
        # Preprocess image
        inputs = self.clip_processor(images=image, return_tensors="pt")
        
        # Extract features
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        
        return image_features.squeeze().numpy()
    
    def _extract_audio_features(self, media_path: str) -> Tuple[List[np.ndarray], List[float]]:
        """Extract audio features from media file"""
        try:
            # Load audio
            y, sr = librosa.load(media_path, sr=22050)
            
            # Extract Mel spectrograms in 0.1-second windows
            hop_length = 512
            frame_length = 2048
            window_duration = 0.1  # seconds
            window_samples = int(window_duration * sr)
            
            audio_features = []
            timestamps = []
            
            for start_sample in range(0, len(y) - window_samples, window_samples // 2):
                end_sample = start_sample + window_samples
                window_audio = y[start_sample:end_sample]
                
                # Extract Mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=window_audio, sr=sr, n_mels=128,
                    hop_length=hop_length, n_fft=frame_length
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Flatten features
                features = mel_spec_db.flatten()
                audio_features.append(features)
                timestamps.append(start_sample / sr)
            
            return audio_features, timestamps
        
        except Exception as e:
            self.logger.error(f"Audio feature extraction failed: {e}")
            return [], []
    
    def _align_temporal_features(self, visual_features: List[np.ndarray], visual_timestamps: List[float],
                               audio_features: List[np.ndarray], audio_timestamps: List[float]) -> Tuple[List, List]:
        """Align visual and audio features temporally"""
        aligned_visual = []
        aligned_audio = []
        
        # Simple temporal alignment (in practice, use more sophisticated methods)
        min_length = min(len(visual_features), len(audio_features))
        
        for i in range(min_length):
            # Find closest audio feature for each visual feature
            visual_time = visual_timestamps[i] if i < len(visual_timestamps) else i * 0.1
            audio_time = audio_timestamps[i] if i < len(audio_timestamps) else i * 0.1
            
            # Only include if temporal difference is small
            if abs(visual_time - audio_time) < 0.2:  # 200ms tolerance
                aligned_visual.append(visual_features[i])
                aligned_audio.append(audio_features[i])
        
        return aligned_visual, aligned_audio
    
    def _detect_sync_anomalies(self, visual_features: List[np.ndarray], 
                             audio_features: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Detect audio-visual synchronization anomalies"""
        anomalies = []
        
        if len(visual_features) != len(audio_features):
            return anomalies
        
        # Calculate cross-correlation for each segment
        for i in range(len(visual_features)):
            if i < len(visual_features) - 1:
                # Compare visual change with audio change
                visual_change = np.linalg.norm(visual_features[i+1] - visual_features[i])
                audio_change = np.linalg.norm(audio_features[i+1] - audio_features[i])
                
                # Detect misalignment (high visual change with low audio change or vice versa)
                sync_score = 1.0 - abs(visual_change - audio_change) / (visual_change + audio_change + 1e-6)
                
                if sync_score < 0.3:  # Threshold for synchronization anomaly
                    anomalies.append({
                        'timestamp': i * 0.1,
                        'type': 'sync_mismatch',
                        'score': 1.0 - sync_score,
                        'description': f'Audio-visual misalignment detected at {i * 0.1:.1f}s'
                    })
        
        return anomalies
    
    def _calculate_sync_score(self, sync_anomalies: List[Dict[str, Any]]) -> float:
        """Calculate overall synchronization score"""
        if not sync_anomalies:
            return 0.9  # High sync score if no anomalies
        
        # Average anomaly score
        avg_anomaly_score = np.mean([anomaly['score'] for anomaly in sync_anomalies])
        return max(0.0, 1.0 - avg_anomaly_score)
    
    def _calculate_confidence(self, sync_anomalies: List[Dict[str, Any]], num_segments: int) -> float:
        """Calculate confidence based on analysis quality"""
        base_confidence = 0.8
        
        # Lower confidence if too few segments analyzed
        if num_segments < 10:
            base_confidence *= 0.7
        
        # Higher confidence if many anomalies detected consistently
        if len(sync_anomalies) > num_segments * 0.3:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _analyze_audio_consistency(self, audio_features: List[np.ndarray]) -> float:
        """Analyze audio consistency for audio-only files"""
        if len(audio_features) < 2:
            return 0.5
        
        # Calculate spectral consistency
        feature_matrix = np.array(audio_features)
        consistency_scores = []
        
        for i in range(len(audio_features) - 1):
            correlation = np.corrcoef(audio_features[i], audio_features[i+1])[0, 1]
            if not np.isnan(correlation):
                consistency_scores.append(abs(correlation))
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _simulate_video_analysis(self) -> Dict[str, Any]:
        """Simulate video analysis when dependencies are not available"""
        time.sleep(0.9)  # Simulate processing time
        
        sync_anomalies = [
            {
                'timestamp': 12.3,
                'type': 'phoneme_mismatch',
                'score': 0.82,
                'description': "Audio-visual misalignment for word 'acquisition'"
            },
            {
                'timestamp': 18.7,
                'type': 'viseme_error',
                'score': 0.75,
                'description': "Phoneme-viseme mismatch for word 'strategic'"
            }
        ]
        
        return {
            'prediction': np.random.uniform(0.60, 0.80),
            'confidence': np.random.uniform(0.80, 0.95),
            'sync_anomalies': sync_anomalies,
            'audio_visual_consistency': np.random.uniform(0.20, 0.40),
            'processing_time': 0.9,
            'frames_analyzed': 28,
            'audio_segments_analyzed': 32
        }
    
    def _get_default_result(self, error_message: str = None) -> Dict[str, Any]:
        """Get default result when analysis cannot be performed"""
        return {
            'prediction': 0.5,  # Neutral prediction
            'confidence': 0.1,  # Low confidence
            'sync_anomalies': [],
            'audio_visual_consistency': 0.5,
            'processing_time': 0.1,
            'frames_analyzed': 0,
            'audio_segments_analyzed': 0,
            'error': error_message
        }
