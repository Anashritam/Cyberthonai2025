"""
Media preprocessing utilities for the HEAR detection system
"""

import numpy as np
import cv2
import librosa
from PIL import Image
import io
from typing import Tuple, List, Optional, Union
import logging

class MediaPreprocessor:
    """
    Unified preprocessing pipeline for images, videos, and audio
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Standard preprocessing parameters
        self.image_size = (224, 224)
        self.audio_sample_rate = 22050
        self.mel_bands = 128
        self.video_max_frames = 30
        
    def preprocess_image(self, image_input: Union[str, np.ndarray, bytes]) -> Optional[np.ndarray]:
        """
        Preprocess image for deepfake detection
        
        Args:
            image_input: Image file path, numpy array, or bytes
            
        Returns:
            Preprocessed image array or None if failed
        """
        try:
            # Load image based on input type
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
            elif isinstance(image_input, bytes):
                image = cv2.imdecode(np.frombuffer(image_input, np.uint8), cv2.IMREAD_COLOR)
            else:
                image = image_input.copy()
            
            if image is None:
                self.logger.error("Failed to load image")
                return None
            
            # Resize to standard size
            image_resized = cv2.resize(image, self.image_size)
            
            # Normalize to [0, 1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2RGB)
            
            return image_rgb
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def preprocess_video(self, video_path: str) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
        """
        Preprocess video for deepfake detection
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (preprocessed_frames, audio_spectrogram) or (None, None) if failed
        """
        try:
            # Extract and preprocess video frames
            frames = self._extract_video_frames(video_path)
            
            # Extract and preprocess audio
            audio_spectrogram = self._extract_audio_spectrogram(video_path)
            
            return frames, audio_spectrogram
            
        except Exception as e:
            self.logger.error(f"Video preprocessing failed: {e}")
            return None, None
    
    def preprocess_audio(self, audio_input: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Preprocess audio for deepfake detection
        
        Args:
            audio_input: Audio file path or numpy array
            
        Returns:
            Mel spectrogram or None if failed
        """
        try:
            if isinstance(audio_input, str):
                return self._extract_audio_spectrogram(audio_input)
            else:
                # Process numpy array directly
                return self._audio_to_spectrogram(audio_input, self.audio_sample_rate)
                
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            return None
    
    def _extract_video_frames(self, video_path: str) -> Optional[List[np.ndarray]]:
        """Extract and preprocess frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Cannot open video: {video_path}")
                return None
            
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames evenly across the video
            if total_frames > self.video_max_frames:
                frame_indices = np.linspace(0, total_frames - 1, self.video_max_frames, dtype=int)
            else:
                frame_indices = list(range(total_frames))
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    preprocessed_frame = self.preprocess_image(frame)
                    if preprocessed_frame is not None:
                        frames.append(preprocessed_frame)
            
            cap.release()
            
            if len(frames) == 0:
                self.logger.error("No frames extracted from video")
                return None
            
            return frames
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            return None
    
    def _extract_audio_spectrogram(self, media_path: str) -> Optional[np.ndarray]:
        """Extract audio and convert to mel spectrogram"""
        try:
            # Load audio
            y, sr = librosa.load(media_path, sr=self.audio_sample_rate)
            
            if len(y) == 0:
                self.logger.error("No audio data extracted")
                return None
            
            return self._audio_to_spectrogram(y, sr)
            
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            return None
    
    def _audio_to_spectrogram(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Convert audio data to mel spectrogram"""
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_mels=self.mel_bands,
            fmax=8000,
            hop_length=512,
            n_fft=2048
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_normalized
    
    def detect_media_type(self, file_path: str) -> str:
        """
        Detect media type from file extension
        
        Args:
            file_path: Path to media file
            
        Returns:
            Media type: 'image', 'video', 'audio', or 'unknown'
        """
        extension = file_path.lower().split('.')[-1]
        
        image_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp'}
        video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v'}
        audio_extensions = {'mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg', 'wma'}
        
        if extension in image_extensions:
            return 'image'
        elif extension in video_extensions:
            return 'video'
        elif extension in audio_extensions:
            return 'audio'
        else:
            return 'unknown'
    
    def validate_media_file(self, file_path: str) -> bool:
        """
        Validate if media file can be processed
        
        Args:
            file_path: Path to media file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            media_type = self.detect_media_type(file_path)
            
            if media_type == 'image':
                image = cv2.imread(file_path)
                return image is not None
            
            elif media_type == 'video':
                cap = cv2.VideoCapture(file_path)
                is_valid = cap.isOpened()
                cap.release()
                return is_valid
            
            elif media_type == 'audio':
                try:
                    y, sr = librosa.load(file_path, duration=1)  # Test load 1 second
                    return len(y) > 0
                except:
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Media validation failed: {e}")
            return False
    
    def get_media_info(self, file_path: str) -> dict:
        """
        Get basic information about media file
        
        Args:
            file_path: Path to media file
            
        Returns:
            Dictionary with media information
        """
        try:
            import os
            
            info = {
                'file_size': os.path.getsize(file_path),
                'media_type': self.detect_media_type(file_path),
                'is_valid': self.validate_media_file(file_path)
            }
            
            media_type = info['media_type']
            
            if media_type == 'video':
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    info.update({
                        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                        'fps': cap.get(cv2.CAP_PROP_FPS),
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    })
                cap.release()
            
            elif media_type == 'image':
                image = cv2.imread(file_path)
                if image is not None:
                    info.update({
                        'width': image.shape[1],
                        'height': image.shape[0],
                        'channels': image.shape[2] if len(image.shape) > 2 else 1
                    })
            
            elif media_type == 'audio':
                try:
                    y, sr = librosa.load(file_path)
                    info.update({
                        'duration': len(y) / sr,
                        'sample_rate': sr,
                        'samples': len(y)
                    })
                except:
                    pass
            
            return info
            
        except Exception as e:
            self.logger.error(f"Getting media info failed: {e}")
            return {'error': str(e)}
