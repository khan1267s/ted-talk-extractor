#!/usr/bin/env python3
"""
OpenAI CLIP-based Person Detector
Uses OpenAI's CLIP model to determine if a frame contains a single speaker.
"""

import cv2
import numpy as np
import logging
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

class OpenAIPersonDetector:
    def __init__(self):
        """Initialize the OpenAI CLIP-based person detector."""
        self.initialized = False
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.initialized = True
            logger.info("OpenAI CLIP detector initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI CLIP detector: {e}")
            self.initialized = False
    
    def is_speaker_only_frame(self, frame: np.ndarray) -> bool:
        """
        Uses CLIP to classify if the frame contains a single speaker on a stage.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            True if the frame is classified as containing a single speaker.
        """
        if not self.initialized:
            return False
            
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Texts to classify against
            texts = [
                "a person speaking",
                "a person presenting", 
                "a single person on stage",
                "multiple people or crowd",
                "slides or text only"
            ]
            
            inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Check if any of the first 3 classes (speaker-related) have high probability
            speaker_prob = probs[0, :3].sum().item()  # Sum of first 3 probabilities
            
            # Return True if speaker probability is > 50%
            return speaker_prob > 0.5
            
        except Exception as e:
            logger.debug(f"CLIP detection failed: {e}")
            return False
