from pydantic import BaseModel
from typing import Optional, List, Dict


class DenseCaption(BaseModel):
    text: str
    confidence: Optional[float] = None


class ImageAnalysisResponse(BaseModel):
    caption: Optional[str] = None
    dense_captions: List[DenseCaption] = []
    text_content: Optional[str] = None
    error: Optional[str] = None
