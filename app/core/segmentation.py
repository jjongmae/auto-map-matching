import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

class RoadSegmentor:
    def __init__(self, model_name="nvidia/segformer-b2-finetuned-cityscapes-1024-1024"):
        """
        SegFormer 모델 초기화.
        인자:
            model_name: HuggingFace 모델 이름. 
                        기본값은 Cityscapes에서 훈련된 b2 (클래스 0 = 도로)입니다.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{self.device}에서 RoadSegmentor 초기화 중...")
        
        # 로컬 캐시 디렉토리
        import os
        cache_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"모델 캐시 디렉토리 사용: {cache_dir}")
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name, cache_dir=cache_dir)
            self.model.to(self.device)
            self.model.eval()
            print("RoadSegmentor가 성공적으로 초기화되었습니다.")
        except Exception as e:
            print(f"RoadSegmentor 초기화 실패: {e}")
            self.model = None

    def segment_road(self, image_bgr):
        """
        BGR 이미지에서 도로 영역을 분할합니다.
        반환:
            mask: 이진 마스크 (uint8), 도로는 255, 그 외에는 0.
        """
        if self.model is None:
            return None
            
        # BGR을 RGB로 변환
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 입력 준비
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 후처리
        # 로짓을 원본 이미지 크기로 업샘플링
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image_bgr.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        
        # 가장 높은 확률을 가진 클래스 가져오기
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        pred_seg = pred_seg.cpu().numpy().astype(np.uint8)
        
        # Cityscapes 클래스: 0 = 도로, 1 = 보도, ...
        # 우리는 도로(0)를 원합니다. 보도(1)도 포함할까요? 
        # 소실점(VP)을 찾기 위해 도로가 가장 중요합니다.
        mask = np.zeros_like(pred_seg)
        mask[pred_seg == 0] = 255
        
        return mask

_global_segmentor = None

def get_segmentor():
    global _global_segmentor
    if _global_segmentor is None:
        _global_segmentor = RoadSegmentor()
    return _global_segmentor
