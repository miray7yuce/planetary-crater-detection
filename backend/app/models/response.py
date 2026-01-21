from pydantic import BaseModel

class DetectionResponse(BaseModel):
    success: bool
    coverage: float | None = None
    image_base64: str | None = None
    message: str | None = None
