import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.auth import require_api_key
from src.api.schemas import (
    Base64PhotoRequest,
    ErrorResponse,
    HealthResponse,
    PredictResponse
)
from src.api.services import decode_base64_photo, predict_image_bytes
from src.api.settings import ALLOWED_ORIGINS, APP_HOST, APP_PORT


app = FastAPI(
    title="Photo Approval Backend",
    description="Backend untuk menerima request foto, mengecek kelayakan foto, dan mengembalikan nilai hasil prediksi.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers=exc.headers
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": str(exc.detail)
        },
        headers=exc.headers
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "service": "photo-check-backend"
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse}
    }
)
async def predict_photo(
    authorized: bool = Depends(require_api_key),
    photo: UploadFile = File(...)
):
    try:
        image_bytes = await photo.read()
        result = predict_image_bytes(image_bytes, photo.filename)
        return {
            "success": True,
            "data": result
        }
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": str(exc)
            }
        )
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Gagal memproses foto.",
                "detail": str(exc)
            }
        )


@app.post(
    "/predict/base64",
    response_model=PredictResponse,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse}
    }
)
def predict_photo_base64(
    request: Base64PhotoRequest,
    authorized: bool = Depends(require_api_key)
):
    try:
        image_bytes = decode_base64_photo(request.image_base64)
        result = predict_image_bytes(image_bytes, request.filename)
        return {
            "success": True,
            "data": result
        }
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": str(exc)
            }
        )
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Gagal memproses foto.",
                "detail": str(exc)
            }
        )


if __name__ == "__main__":
    uvicorn.run(
        "07_backend_api:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False
    )
