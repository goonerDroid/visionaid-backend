from pydantic import BaseModel
from typing import Optional, List, Dict


class Tag(BaseModel):
    name: str
    confidence: Optional[float] = None


class DetectedObject(BaseModel):
    name: str
    confidence: Optional[float] = None
    bounding_box: Optional[dict] = None


class ImageAnalysisResponse(BaseModel):
    caption: Optional[str] = None
    text_content: Optional[str] = None
    tags: List[Tag] = []
    objects: List[DetectedObject] = []
    error: Optional[str] = None
