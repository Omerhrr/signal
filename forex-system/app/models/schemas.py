"""
Forex Probability Intelligence System - Data Models
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class SignalBias(str, Enum):
    """Trading signal bias direction"""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class SignalStatus(str, Enum):
    """Signal status types"""
    OK = "ok"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_VOLATILITY = "high_volatility"
    REVERSAL_DETECTED = "reversal_detected"
    SPREAD_TOO_HIGH = "spread_too_high"


class SignalAction(str, Enum):
    """Action to take on signal"""
    TRADE = "trade"
    SKIP = "skip"
    PAUSE = "pause"
    EXIT_EARLY = "exit_early"


class RiskLevel(str, Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MarketSession(str, Enum):
    """Trading session types"""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"


# ============== Market Data Models ==============

class TickData(BaseModel):
    """Real-time tick data from MT5"""
    symbol: str
    bid: float
    ask: float
    spread: float
    timestamp: datetime
    volume: Optional[int] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OHLCV(BaseModel):
    """OHLCV candlestick data"""
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MarketState(BaseModel):
    """Current market state snapshot"""
    symbol: str
    current_price: float
    spread: float
    volatility: float
    session: MarketSession
    trend: SignalBias
    momentum: float
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============== Feature Models ==============

class PriceActionFeatures(BaseModel):
    """Price action feature set"""
    break_of_structure: bool = False
    bos_direction: Optional[SignalBias] = None
    bos_strength: float = 0.0
    wick_rejection_strength: float = 0.0
    wick_rejection_type: Optional[str] = None  # 'bullish' or 'bearish'
    range_expansion: float = 0.0
    range_expansion_pct: float = 0.0
    higher_highs: int = 0
    lower_lows: int = 0
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None


class TimeFeatures(BaseModel):
    """Time-based feature set"""
    current_session: MarketSession
    session_overlap: List[MarketSession] = []
    time_since_breakout_minutes: Optional[float] = None
    time_since_liquidity_event_minutes: Optional[float] = None
    hour_of_day: int = 0
    day_of_week: int = 0
    is_weekend: bool = False
    session_progress: float = 0.0  # 0-1 progress within session


class StatisticalFeatures(BaseModel):
    """Statistical feature set"""
    rolling_volatility_10: float = 0.0
    rolling_volatility_20: float = 0.0
    rolling_volatility_50: float = 0.0
    momentum_decay_5: float = 0.0
    momentum_decay_10: float = 0.0
    momentum_decay_20: float = 0.0
    mean_reversion_strength: float = 0.0
    zscore: float = 0.0
    rsi: float = 50.0
    atr: float = 0.0
    atr_pct: float = 0.0


class FeatureSet(BaseModel):
    """Complete feature set for prediction"""
    symbol: str
    timestamp: datetime
    price_action: PriceActionFeatures
    time_features: TimeFeatures
    statistical: StatisticalFeatures
    raw_features: Dict[str, float] = {}
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============== Prediction Models ==============

class DirectionPrediction(BaseModel):
    """Direction prediction output"""
    prob_up: float = Field(ge=0, le=1)
    prob_down: float = Field(ge=0, le=1)
    predicted_direction: SignalBias
    confidence: float = Field(ge=0, le=1)
    model_version: str = "1.0.0"


class DurationPrediction(BaseModel):
    """Duration prediction from survival analysis"""
    expected_time_above_minutes: float
    expected_time_below_minutes: float
    hazard_rate: float
    survival_probability_5min: float
    survival_probability_10min: float
    survival_probability_15min: float
    model_version: str = "1.0.0"


class PredictionOutput(BaseModel):
    """Complete prediction output"""
    symbol: str
    timestamp: datetime
    direction: DirectionPrediction
    duration: DurationPrediction
    confidence_score: float = Field(ge=0, le=1)
    model_agreement: float = Field(ge=0, le=1)
    market_regime: str = "neutral"
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============== Signal Models ==============

class TradingSignal(BaseModel):
    """Complete trading signal"""
    signal_id: str
    symbol: str
    bias: SignalBias
    entry_zone_start: float
    entry_zone_end: float
    confidence: float = Field(ge=0, le=1)
    probability_hold_above: float = Field(ge=0, le=1)
    probability_hold_below: float = Field(ge=0, le=1)
    expected_duration_minutes: float
    stop_loss: float
    take_profit: float
    stop_loss_type: str = "dynamic"  # 'dynamic', 'fixed', 'volatility_based'
    take_profit_type: str = "adaptive"  # 'fixed', 'adaptive', 'confidence_weighted'
    risk_level: RiskLevel
    risk_reward_ratio: float
    status: SignalStatus
    action: SignalAction
    status_message: str = ""
    created_at: datetime
    expires_at: datetime
    features: Optional[FeatureSet] = None
    prediction: Optional[PredictionOutput] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SignalCard(BaseModel):
    """Lightweight signal card for dashboard display"""
    signal_id: str
    symbol: str
    bias: SignalBias
    entry_zone: str
    confidence: float
    probability: float
    expected_duration: str
    stop_loss: float
    take_profit: float
    risk_level: RiskLevel
    status: SignalStatus
    action: SignalAction
    status_color: str
    created_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============== Risk & Failure Models ==============

class FailureDetection(BaseModel):
    """Failure detection result"""
    failure_type: Optional[str] = None  # 'instant', 'slow', 'fake_breakout', 'high_volatility', 'low_confidence'
    detected: bool = False
    severity: str = "none"  # 'none', 'low', 'medium', 'high', 'critical'
    action_required: SignalAction = SignalAction.TRADE
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskAssessment(BaseModel):
    """Risk assessment output"""
    max_risk_per_trade: float
    position_size_multiplier: float
    stop_loss_pips: float
    take_profit_pips: float
    risk_reward_ratio: float
    volatility_adjusted: bool
    confidence_weighted_tp: bool


# ============== Feedback Models ==============

class TradeOutcome(BaseModel):
    """Trade outcome for feedback loop"""
    signal_id: str
    symbol: str
    bias: SignalBias
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime
    exit_time: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    outcome: str = "pending"  # 'win', 'loss', 'breakeven', 'pending', 'skipped'
    pips_gained: Optional[float] = None
    max_drawdown_pips: Optional[float] = None
    max_favorable_pips: Optional[float] = None
    confidence_at_entry: float
    market_condition: str = "normal"
    failure_type: Optional[str] = None


class ModelPerformance(BaseModel):
    """Model performance metrics"""
    model_type: str
    version: str
    total_signals: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float
    avg_duration_minutes: float
    avg_confidence: float
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# ============== API Response Models ==============

class APIResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SignalListResponse(BaseModel):
    """Signal list API response"""
    signals: List[TradingSignal]
    count: int
    page: int
    total_pages: int


class MarketRadarResponse(BaseModel):
    """Market radar data for dashboard"""
    pairs: Dict[str, MarketState]
    top_signals: List[SignalCard]
    market_sentiment: str
    active_session: MarketSession
    timestamp: datetime
