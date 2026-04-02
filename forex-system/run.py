"""
Forex Probability Intelligence System - Main Entry Point
Run both FastAPI backend and Flask frontend
"""
import asyncio
import threading
import uvicorn
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")


def run_fastapi():
    """Run FastAPI backend"""
    logger.info("Starting FastAPI backend on port 8000...")
    uvicorn.run(
        "app.api.routes:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


def run_flask():
    """Run Flask frontend"""
    logger.info("Starting Flask frontend on port 5000...")
    from app.frontend import app as flask_app
    flask_app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Forex Probability Intelligence System")
    parser.add_argument("--backend", action="store_true", help="Run only FastAPI backend")
    parser.add_argument("--frontend", action="store_true", help="Run only Flask frontend")
    parser.add_argument("--port", type=int, default=None, help="Override port")
    
    args = parser.parse_args()
    
    if args.backend:
        # Run only backend
        if args.port:
            import app.api.routes as api_routes
            uvicorn.run(api_routes.app, host="0.0.0.0", port=args.port)
        else:
            run_fastapi()
    
    elif args.frontend:
        # Run only frontend
        if args.port:
            from app.frontend import app as flask_app
            flask_app.run(host="0.0.0.0", port=args.port)
        else:
            run_flask()
    
    else:
        # Run both servers
        logger.info("=" * 60)
        logger.info("Forex Probability Intelligence System")
        logger.info("=" * 60)
        
        # Start FastAPI in a separate thread
        backend_thread = threading.Thread(target=run_fastapi, daemon=True)
        backend_thread.start()
        
        # Run Flask in the main thread
        run_flask()
