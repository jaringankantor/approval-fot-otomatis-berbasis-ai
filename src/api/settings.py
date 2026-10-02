import os
from pathlib import Path

from dotenv import load_dotenv


# Memuat konfigurasi variabel lingkungan dari file .env di direktori root proyek
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

# Konfigurasi keamanan dan otentikasi API
API_KEY = os.getenv("API_KEY", "")
API_KEY_HEADER_NAME = os.getenv("API_KEY_HEADER_NAME", "X-API-Key")

# Konfigurasi Cross-Origin Resource Sharing (CORS)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

# Konfigurasi host dan port server FastAPI
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
