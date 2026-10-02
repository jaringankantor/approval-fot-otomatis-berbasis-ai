from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    service: str = Field(examples=["photo-check-backend"])


class Base64PhotoRequest(BaseModel):
    image_base64: str = Field(
        min_length=1,
        description="Isi file foto dalam format base64. Data URL juga diterima."
    )
    filename: str = Field(
        default="uploaded.jpg",
        description="Nama file untuk menentukan ekstensi gambar."
    )


class PredictResponse(BaseModel):
    success: bool
    data: dict


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
