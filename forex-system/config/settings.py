"""
Forex Probability Intelligence System - Configuration Settings
MT5 Data Source Only
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Dict, Any
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    app_name: str = "Forex Probability Intelligence System"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    flask_host: str = "0.0.0.0"
    flask_port: int = 5000
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    
    # Trading Pairs
    trading_pairs: List[str] = ["EURUSD", "GBPUSD", "USDJPY"]
    default_pair: str = "EURUSD"
    
    # Model Settings
    xgboost_params: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    
    # Feature Engine Settings
    feature_window_sizes: List[int] = [5, 10, 20, 50]
    volatility_window: int = 20
    momentum_window: int = 14
    
    # Trading Sessions (UTC hours)
    trading_sessions: Dict[str, Dict[str, int]] = {
        "sydney": {"start": 22, "end": 7},
        "tokyo": {"start": 0, "end": 9},
        "london": {"start": 8, "end": 17},
        "new_york": {"start": 13, "end": 22}
    }
    
    # Risk Management
    max_risk_per_trade: float = 0.02  # 2%
    min_confidence_threshold: float = 0.65
    max_spread_pips: float = 3.0
    max_volatility_multiplier: float = 2.5
    
    # Signal Settings
    signal_expiry_minutes: int = 30
    min_probability_threshold: float = 0.55
    
    # Failure Detection Thresholds
    instant_failure_pips: float = 15.0
    slow_failure_minutes: float = 20.0
    fake_breakout_threshold: float = 0.3
    volatility_spike_threshold: float = 3.0
    
    # Feedback Loop
    retraining_day: str = "sunday"
    retraining_hour: int = 2
    min_samples_for_retraining: int = 100
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/forex_system.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Session definitions for trading hours
MARKET_SESSIONS = {
    "sydney": {"name": "Sydney", "start": 22, "end": 7, "color": "#4CAF50"},
    "tokyo": {"name": "Tokyo", "start": 0, "end": 9, "color": "#2196F3"},
    "london": {"name": "London", "start": 8, "end": 17, "color": "#FF9800"},
    "new_york": {"name": "New York", "start": 13, "end": 22, "color": "#F44336"}
}

# Feature configuration
FEATURE_CONFIG = {
    "price_action": {
        "break_of_structure": {"lookback": 20},
        "wick_rejection_strength": {"min_wick_ratio": 0.6},
        "range_expansion": {"atr_multiplier": 1.5}
    },
    "time_features": {
        "session_overlap_weight": True,
        "time_since_breakout": {"max_minutes": 60},
        "time_since_liquidity_event": {"max_minutes": 120}
    },
    "statistical": {
        "rolling_volatility": {"windows": [10, 20, 50]},
        "momentum_decay": {"periods": [5, 10, 20]},
        "mean_reversion_strength": {"zscore_window": 50}
    }
}

# Risk levels
RISK_LEVELS = {
    "low": {"color": "#4CAF50", "max_risk": 0.01},
    "medium": {"color": "#FF9800", "max_risk": 0.02},
    "high": {"color": "#F44336", "max_risk": 0.03}
}

# Signal status definitions
SIGNAL_STATUS = {
    "ok": {"color": "#4CAF50", "action": "trade", "message": "Signal is valid"},
    "low_confidence": {"color": "#FFC107", "action": "skip", "message": "Confidence below threshold"},
    "high_volatility": {"color": "#FF5722", "action": "pause", "message": "Market volatility too high"},
    "reversal_detected": {"color": "#9C27B0", "action": "exit_early", "message": "Market reversal detected"},
    "spread_too_high": {"color": "#795548", "action": "skip", "message": "Spread exceeds limit"}
}
