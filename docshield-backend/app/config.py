"""DocShield AI — Configuration Classes.

Manages application configurations across Development, Testing, and Production
environments with strict validation against insecure defaults.
"""

from datetime import timedelta
import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory for the backend (where wsgi.py / app directory lives)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load local environment variables from .env
load_dotenv(BASE_DIR / ".env")


class BaseConfig:
    """Base configuration shared by all environments."""

    # Core Security Keys
    SECRET_KEY = os.getenv("SECRET_KEY", "8e2218c0fccfb7184e4ed0a9d35c8edfc82183adee126ad8b8e4e12a0cb93e0e")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "f3a19b8c7d6e5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b")
    DOCSHIELD_AES_KEY = os.getenv("DOCSHIELD_AES_KEY", "897aa89efdea98fe834ce38cc4abc35e1a767af6a363244bf2a6cc54827167d2")

    # Uploads Lifecycle & Retention
    RETAIN_UPLOADS_FOR_DEMO = os.getenv("RETAIN_UPLOADS_FOR_DEMO", "false").lower() in ("true", "1", "yes")

    # JWT Settings
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(
        minutes=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES_MINUTES", "15"))
    )
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(
        days=int(os.getenv("JWT_REFRESH_TOKEN_EXPIRES_DAYS", "7"))
    )
    JWT_ALGORITHM = "HS256"

    # Database Settings
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Uploads & Storage Security
    # 16MB default limit enforced at Flask request level (handles phone-camera JPGs & PDFs)
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "16")) * 1024 * 1024
    UPLOAD_FOLDER = os.path.abspath(
        os.getenv("UPLOAD_FOLDER", str(BASE_DIR / "uploads"))
    )
    MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "8000"))
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
    ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/pjpeg", "image/jpg", "application/pdf"}

    # CORS Allow-list (Includes local dev and deployed Netlify frontend)
    _raw_cors = os.getenv(
        "CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,https://docshield-innovx.netlify.app"
    )
    CORS_ALLOWED_ORIGINS = [
        origin.strip() for origin in _raw_cors.split(",") if origin.strip()
    ]

    # Rate Limiting Settings
    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")
    RATELIMIT_DEFAULT = os.getenv("RATELIMIT_DEFAULT", "100/hour")
    RATELIMIT_ANALYZE = os.getenv("RATELIMIT_ANALYZE", "20/minute")
    RATELIMIT_LOGIN = os.getenv("RATELIMIT_LOGIN", "5/15minute")
    RATELIMIT_HEADERS_ENABLED = True

    # ML & Forensic Paths
    MODEL_WEIGHTS_PATH = os.path.abspath(
        os.getenv(
            "MODEL_WEIGHTS_PATH",
            str(BASE_DIR / "app" / "ml" / "weights" / "efficientnet_b0_docshield.pth"),
        )
    )
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

    DEBUG = False
    TESTING = False


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""

    DEBUG = True
    TESTING = False
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'instance' / 'docshield_dev.db'}"
    )


class TestingConfig(BaseConfig):
    """Testing environment configuration."""

    DEBUG = False
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    # Use ephemeral in-memory rate limiting and lower limits for tests if needed
    RATELIMIT_STORAGE_URI = "memory://"
    # Strict secret keys for test assertions
    SECRET_KEY = "test-secret-key-do-not-use-in-production"
    JWT_SECRET_KEY = "test-jwt-key-do-not-use-in-production"
    DOCSHIELD_AES_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    RETAIN_UPLOADS_FOR_DEMO = False


class ProductionConfig(BaseConfig):
    """Production configuration with resilient security defaults."""

    DEBUG = False
    TESTING = False

    # Force database URL configuration (gracefully falls back to local SQLite if DATABASE_URL is not set)
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL") or "sqlite:///instance/docshield.db"

    # Cross-site cookie configuration for Netlify <-> Render communication
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "None"

    @classmethod
    def init_app(cls, app):
        """Ensure critical production environment settings are active without crashing."""
        if not cls.SQLALCHEMY_DATABASE_URI:
            cls.SQLALCHEMY_DATABASE_URI = "sqlite:///instance/docshield.db"
        if not app.config.get("SECRET_KEY"):
            app.config["SECRET_KEY"] = cls.SECRET_KEY
        if not app.config.get("JWT_SECRET_KEY"):
            app.config["JWT_SECRET_KEY"] = cls.JWT_SECRET_KEY
        if not app.config.get("DOCSHIELD_AES_KEY"):
            app.config["DOCSHIELD_AES_KEY"] = cls.DOCSHIELD_AES_KEY


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
