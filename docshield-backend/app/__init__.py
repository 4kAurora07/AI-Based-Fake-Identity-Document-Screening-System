"""DocShield AI — Flask Application Factory.

Initializes extensions, configures security middlewares, registers blueprint routes,
and attaches unified, leak-free error handlers.
"""

import os
from flask import Flask, jsonify, g
from werkzeug.exceptions import HTTPException

from app.config import config_by_name
from app.extensions import db, limiter, cors
from app.utils.logging_config import setup_logging
from app.security.headers import init_security_headers


def create_app(config_name: str = None) -> Flask:
    """Application factory creating a configured Flask instance."""
    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "development").lower()

    app = Flask(__name__)
    config_cls = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(config_cls)

    # Allow configuration classes to run custom environment validation (e.g. Production checks)
    if hasattr(config_cls, "init_app"):
        config_cls.init_app(app)

    # Ensure upload directory exists securely
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(os.path.join(app.root_path, "..", "instance"), exist_ok=True)

    # Initialize Logging
    setup_logging(app)

    # Initialize Extensions
    db.init_app(app)
    limiter.init_app(app)
    cors.init_app(
        app,
        origins=app.config.get("CORS_ALLOWED_ORIGINS", []),
        supports_credentials=True,
    )

    # Initialize Security Headers Middleware
    init_security_headers(app)

    # Register Blueprints
    from app.api.health import health_bp
    from app.api.analyze import analyze_bp
    from app.auth.routes import auth_bp
    from app.api.admin import admin_bp
    from app.api.history import history_bp

    # Both /api/v1/ and /api/ prefixes supported
    app.register_blueprint(health_bp, url_prefix="/api/v1")
    app.register_blueprint(health_bp, url_prefix="/api", name="health_root")
    app.register_blueprint(analyze_bp, url_prefix="/api/v1")
    app.register_blueprint(analyze_bp, url_prefix="/api", name="analyze_root")
    app.register_blueprint(history_bp, url_prefix="/api")
    app.register_blueprint(history_bp, url_prefix="/api/v1", name="history_v1")
    app.register_blueprint(auth_bp, url_prefix="/api/v1/auth")
    app.register_blueprint(admin_bp, url_prefix="/api/v1/admin")


    # Register Global Error Handlers (Zero-Leakage Policy)
    register_error_handlers(app)

    # Ensure database schema is created
    with app.app_context():
        # Import models so SQLAlchemy metadata is aware of them
        from app.auth.models import AdminUser
        from app.models.scan import ScanResult
        db.create_all()

    # Pre-warm ML models at startup to eliminate cold-start penalty on first request.
    # Without this, the first /api/analyze call pays ~11s loading PyTorch + EasyOCR + model weights.
    if not app.config.get("TESTING"):
        app.logger.info("Pre-warming ML models at startup...")
        try:
            from app.ml.model_loader import get_ai_detector
            get_ai_detector()
            app.logger.info("EfficientNet-B0 detector loaded successfully.")
        except Exception as e:
            app.logger.warning("EfficientNet-B0 pre-warm skipped: %s", str(e))

    return app


def register_error_handlers(app: Flask):
    """Registers standard, sanitized JSON error responses preventing stack trace leakage."""

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "error": "Bad Request",
            "message": getattr(error, "description", "The request body or parameters are invalid."),
            "request_id": getattr(g, "request_id", None),
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Not Found",
            "message": "The requested API resource does not exist.",
            "request_id": getattr(g, "request_id", None),
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "error": "Method Not Allowed",
            "message": "The requested HTTP method is not permitted on this endpoint.",
            "request_id": getattr(g, "request_id", None),
        }), 405

    @app.errorhandler(413)
    def payload_too_large(error):
        return jsonify({
            "error": "Payload Too Large",
            "message": f"Uploaded file exceeds the maximum allowed limit of {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB.",
            "request_id": getattr(g, "request_id", None),
        }), 413

    @app.errorhandler(415)
    def unsupported_media_type(error):
        return jsonify({
            "error": "Unsupported Media Type",
            "message": "Only 'multipart/form-data' or 'application/json' are supported.",
            "request_id": getattr(g, "request_id", None),
        }), 415

    @app.errorhandler(429)
    def ratelimit_exceeded(error):
        return jsonify({
            "error": "Rate Limit Exceeded",
            "message": "Too many requests. Please throttle your traffic and try again later.",
            "request_id": getattr(g, "request_id", None),
        }), 429

    @app.errorhandler(Exception)
    def handle_unexpected_exception(error):
        # Pass HTTPExceptions directly if they have standard HTTP codes
        if isinstance(error, HTTPException):
            return jsonify({
                "error": error.name,
                "message": error.description,
                "request_id": getattr(g, "request_id", None),
            }), error.code

        # Log complete stack trace internally with request context
        app.logger.exception("Unhandled server exception: %s", str(error))

        # Return generic, non-revealing error to the external client
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing your request. Please contact support.",
            "request_id": getattr(g, "request_id", None),
        }), 500
