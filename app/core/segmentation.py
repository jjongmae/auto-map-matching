import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

class RoadSegmentor:
    def __init__(self, model_name="nvidia/segformer-b2-finetuned-cityscapes-1024-1024"):
        """
        Initialize SegFormer model.
        Args:
            model_name: HuggingFace model name. 
                        Default is b2 trained on Cityscapes (Class 0 = Road).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing RoadSegmentor on {self.device}...")
        
        # Local cache directory
        import os
        cache_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using model cache dir: {cache_dir}")
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name, cache_dir=cache_dir)
            self.model.to(self.device)
            self.model.eval()
            print("RoadSegmentor initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize RoadSegmentor: {e}")
            self.model = None

    def segment_road(self, image_bgr):
        """
        Segment road area from BGR image.
        Returns:
            mask: Binary mask (uint8), 255 for road, 0 for others.
        """
        if self.model is None:
            return None
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Prepare input
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process
        # Upsample logits to original image size
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image_bgr.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        
        # Get class with highest probability
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        pred_seg = pred_seg.cpu().numpy().astype(np.uint8)
        
        # Cityscapes classes: 0 = road, 1 = sidewalk, ...
        # We want road (0). Maybe sidewalk (1) too? 
        # For finding VPs, road is most important.
        mask = np.zeros_like(pred_seg)
        mask[pred_seg == 0] = 255
        
        return mask

_global_segmentor = None

def get_segmentor():
    global _global_segmentor
    if _global_segmentor is None:
        _global_segmentor = RoadSegmentor()
    return _global_segmentor
