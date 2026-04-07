"""
Forex Probability Intelligence System - FastAPI Backend
API Endpoints for the trading system - MT5 Data Source Only
"""
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from app.models.schemas import (
    TradingSignal, SignalCard, MarketState, TickData, OHLCV,
    SignalBias, SignalStatus, SignalAction, RiskLevel, MarketSession,
    APIResponse, SignalListResponse, MarketRadarResponse
)
from app.services.data_ingestion import data_pipeline
from app.engines.decision_engine import decision_engine, signal_aggregator
from app.engines.risk_engine import risk_engine
from app.engines.direction_model import direction_model
from app.engines.duration_model import duration_model
from app.engines.feature_engine import feature_engine
from config.settings import get_settings

settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API for Forex Probability Intelligence System - MT5 Data Source"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models for API ==============

class SignalGenerateRequest(BaseModel):
    symbol: str
    timeframe: str = "M15"


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: str
    data_source: str
    mt5_connected: bool
    components: Dict[str, bool]


class MT5TickRequest(BaseModel):
    """MT5 tick data from Windows bridge"""
    symbol: str
    bid: float
    ask: float
    spread: float = 0.0
    volume: int = 0
    timestamp: Optional[str] = None


class MT5OHLCVRequest(BaseModel):
    """MT5 OHLCV data from Windows bridge"""
    symbol: str
    timeframe: str = "M15"
    candles: List[Dict[str, Any]]


# ============== Startup & Shutdown ==============

start_time = datetime.now(timezone.utc)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Forex Probability Intelligence System API")
    
    # Start data pipeline
    success = await data_pipeline.start()
    
    if success:
        logger.info("API started - Waiting for MT5 data from Windows bridge")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API")
    await data_pipeline.stop()


# ============== MT5 Data Ingestion Endpoints ==============

@app.post("/api/mt5/tick", tags=["MT5 Bridge"])
async def receive_mt5_tick(tick: MT5TickRequest):
    """Receive tick data from MT5 Windows bridge"""
    tick_data = {
        "bid": tick.bid,
        "ask": tick.ask,
        "spread": tick.spread,
        "volume": tick.volume,
        "timestamp": tick.timestamp or datetime.now(timezone.utc).isoformat()
    }
    
    received_tick = data_pipeline.receive_tick(tick.symbol, tick_data)
    
    return {"success": True, "symbol": tick.symbol, "bid": tick.bid, "ask": tick.ask}


@app.post("/api/mt5/ohlcv", tags=["MT5 Bridge"])
async def receive_mt5_ohlcv(data: MT5OHLCVRequest):
    """Receive OHLCV data from MT5 Windows bridge"""
    ohlcv = data_pipeline.receive_ohlcv(data.symbol, data.timeframe, data.candles)
    
    return {"success": True, "symbol": data.symbol, "timeframe": data.timeframe, "count": len(ohlcv)}


@app.get("/api/mt5/status", tags=["MT5 Bridge"])
async def get_mt5_status():
    """Get MT5 connection status"""
    symbols_status = {}
    for symbol in settings.trading_pairs:
        tick = data_pipeline.mt5_store.get_tick(symbol)
        symbols_status[symbol] = {
            "has_data": tick is not None,
            "last_update": data_pipeline.mt5_store.last_update.get(symbol, None),
            "is_fresh": data_pipeline.mt5_store.is_data_fresh(symbol)
        }
    
    return {
        "mt5_connected": data_pipeline.mt5_store.connected,
        "symbols": symbols_status
    }


# ============== Health & Status ==============

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health status"""
    uptime = datetime.now(timezone.utc) - start_time
    
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        uptime=str(uptime).split('.')[0],
        data_source="mt5",
        mt5_connected=data_pipeline.mt5_store.connected,
        components={
            "data_pipeline": data_pipeline.connected,
            "redis": data_pipeline.redis_manager.connected
        }
    )


@app.get("/api/status", tags=["System"])
async def get_system_status():
    """Get detailed system status"""
    failure_stats = risk_engine.get_failure_stats()
    should_pause, pause_reason = risk_engine.should_pause_trading()
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_source": "mt5",
        "mt5_connected": data_pipeline.mt5_store.connected,
        "active_signals": len(decision_engine.active_signals),
        "trading_pairs": settings.trading_pairs,
        "failure_stats": failure_stats,
        "trading_paused": should_pause,
        "pause_reason": pause_reason
    }


# ============== Market Data ==============

@app.get("/api/tick/{symbol}", response_model=APIResponse, tags=["Market Data"])
async def get_tick(symbol: str):
    """Get current tick for a symbol"""
    tick = await data_pipeline.get_tick(symbol.upper())
    
    if not tick:
        raise HTTPException(status_code=404, detail=f"No tick data for {symbol}. Is MT5 bridge running?")
    
    return APIResponse(
        success=True,
        message="Tick data retrieved",
        data=tick.dict()
    )


@app.get("/api/ticks", response_model=APIResponse, tags=["Market Data"])
async def get_all_ticks():
    """Get current ticks for all trading pairs"""
    ticks = {}
    
    for symbol in settings.trading_pairs:
        tick = await data_pipeline.get_tick(symbol)
        if tick:
            ticks[symbol] = tick.dict()
    
    return APIResponse(
        success=True,
        message="All ticks retrieved",
        data=ticks
    )


@app.get("/api/ohlcv/{symbol}", response_model=APIResponse, tags=["Market Data"])
async def get_ohlcv(
    symbol: str,
    timeframe: str = Query("M15", description="Timeframe: M1, M5, M15, M30, H1, H4, D1"),
    count: int = Query(100, ge=10, le=1000)
):
    """Get OHLCV data for a symbol"""
    ohlcv = await data_pipeline.get_ohlcv(symbol.upper(), timeframe, count)
    
    if not ohlcv:
        raise HTTPException(status_code=404, detail=f"No OHLCV data for {symbol}. Is MT5 bridge running?")
    
    return APIResponse(
        success=True,
        message=f"OHLCV data retrieved ({len(ohlcv)} candles)",
        data=[c.dict() for c in ohlcv]
    )


@app.get("/api/market-state/{symbol}", response_model=APIResponse, tags=["Market Data"])
async def get_market_state(symbol: str):
    """Get current market state for a symbol"""
    state = await data_pipeline.get_market_state(symbol.upper())
    
    if not state:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    
    return APIResponse(
        success=True,
        message="Market state retrieved",
        data=state
    )


@app.get("/api/market-radar", response_model=MarketRadarResponse, tags=["Market Data"])
async def get_market_radar():
    """Get market radar data for all pairs"""
    pairs = {}
    top_signals = []
    
    # Get states for all pairs
    for symbol in settings.trading_pairs:
        state = await data_pipeline.get_market_state(symbol)
        if state and 'current_price' in state and state.get('current_price', 0) > 0:
            # Convert trend string to SignalBias enum
            trend_str = state.get('trend', 'NEUTRAL')
            if trend_str == 'BUY':
                trend = SignalBias.BUY
            elif trend_str == 'SELL':
                trend = SignalBias.SELL
            else:
                trend = SignalBias.NEUTRAL
            
            # Convert session string to MarketSession enum
            session_str = state.get('session', 'london').lower()
            session_map = {
                'london': MarketSession.LONDON,
                'new_york': MarketSession.NEW_YORK,
                'tokyo': MarketSession.TOKYO,
                'sydney': MarketSession.SYDNEY,
                'new_york': MarketSession.NEW_YORK
            }
            session = session_map.get(session_str, MarketSession.LONDON)
            
            # Create MarketState with proper types
            try:
                pairs[symbol] = MarketState(
                    symbol=symbol,
                    current_price=state.get('current_price', 0),
                    spread=state.get('spread', 0),
                    volatility=state.get('volatility', 0),
                    session=session,
                    trend=trend,
                    momentum=state.get('momentum', 0),
                    timestamp=datetime.now(timezone.utc)
                )
            except Exception as e:
                logger.warning(f"Error creating MarketState for {symbol}: {e}")
                continue
    
    # Get top signals
    all_signals = decision_engine.get_signal_cards()
    top_signals = sorted(all_signals, key=lambda s: s.confidence, reverse=True)[:5]
    
    # Determine overall sentiment
    signal_agg = signal_aggregator.aggregate_signals(
        decision_engine.get_active_signals()
    )
    
    # Get current session
    hour = datetime.now(timezone.utc).hour
    if 8 <= hour < 17:
        session = "london"
    elif 13 <= hour < 22:
        session = "new_york"
    elif 0 <= hour < 9:
        session = "tokyo"
    else:
        session = "sydney"
    
    return MarketRadarResponse(
        pairs={k: v for k, v in pairs.items()},
        top_signals=top_signals,
        market_sentiment=signal_agg.get('bias', 'NEUTRAL'),
        active_session=session,
        timestamp=datetime.now(timezone.utc)
    )


# ============== Signals ==============

@app.post("/api/signals/generate", response_model=APIResponse, tags=["Signals"])
async def generate_signal(request: SignalGenerateRequest):
    """Generate a new trading signal"""
    symbol = request.symbol.upper()
    
    # Get market data
    tick = await data_pipeline.get_tick(symbol)
    ohlcv = await data_pipeline.get_ohlcv(symbol, request.timeframe, 100)
    market_state = await data_pipeline.get_market_state(symbol)
    
    if not tick or not ohlcv or not market_state:
        raise HTTPException(status_code=400, detail="Insufficient market data. Is MT5 bridge running?")
    
    # Generate signal
    signal = decision_engine.generate_signal(symbol, ohlcv, tick, market_state)
    
    if not signal:
        return APIResponse(
            success=False,
            message="No valid signal generated (criteria not met)",
            data=None
        )
    
    return APIResponse(
        success=True,
        message="Signal generated successfully",
        data=signal.dict()
    )


@app.get("/api/signals", response_model=SignalListResponse, tags=["Signals"])
async def get_signals(
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100)
):
    """Get trading signals with optional filters"""
    signals = decision_engine.get_active_signals(symbol)
    
    # Filter by status if provided
    if status:
        signals = [s for s in signals if s.status.value == status]
    
    # Sort by creation time (newest first)
    signals = sorted(signals, key=lambda s: s.created_at, reverse=True)
    
    # Pagination
    total = len(signals)
    total_pages = (total + limit - 1) // limit
    start = (page - 1) * limit
    end = start + limit
    
    return SignalListResponse(
        signals=signals[start:end],
        count=total,
        page=page,
        total_pages=total_pages
    )


@app.get("/api/signals/cards", response_model=APIResponse, tags=["Signals"])
async def get_signal_cards(symbol: Optional[str] = None):
    """Get lightweight signal cards for dashboard"""
    cards = decision_engine.get_signal_cards(symbol)
    
    return APIResponse(
        success=True,
        message=f"Retrieved {len(cards)} signal cards",
        data=[c.dict() for c in cards]
    )


@app.get("/api/signals/{signal_id}", response_model=APIResponse, tags=["Signals"])
async def get_signal(signal_id: str):
    """Get a specific signal by ID"""
    if signal_id not in decision_engine.active_signals:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    signal = decision_engine.active_signals[signal_id]
    
    return APIResponse(
        success=True,
        message="Signal retrieved",
        data=signal.dict()
    )


@app.delete("/api/signals/{signal_id}", response_model=APIResponse, tags=["Signals"])
async def delete_signal(signal_id: str):
    """Delete a signal"""
    if signal_id not in decision_engine.active_signals:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    del decision_engine.active_signals[signal_id]
    
    return APIResponse(
        success=True,
        message="Signal deleted",
        data=None
    )


# ============== Risk Management ==============

@app.get("/api/risk/failure-stats", response_model=APIResponse, tags=["Risk"])
async def get_failure_stats(hours: int = Query(24, ge=1, le=168)):
    """Get failure statistics"""
    stats = risk_engine.get_failure_stats(hours)
    
    return APIResponse(
        success=True,
        message="Failure statistics retrieved",
        data=stats
    )


@app.get("/api/risk/should-pause", response_model=APIResponse, tags=["Risk"])
async def check_trading_pause():
    """Check if trading should be paused"""
    should_pause, reason = risk_engine.should_pause_trading()
    
    return APIResponse(
        success=True,
        message="Pause status checked",
        data={
            "should_pause": should_pause,
            "reason": reason
        }
    )


# ============== WebSocket for Real-time Updates ==============

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    
    try:
        # Send initial data
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Forex Intelligence System (MT5)"
        })
        
        while True:
            # Wait for any message (ping/pong or commands)
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif data == "get_signals":
                signals = decision_engine.get_signal_cards()
                await websocket.send_json({
                    "type": "signals",
                    "data": [s.dict() for s in signals]
                })
            
            elif data == "get_ticks":
                ticks = {}
                for symbol in settings.trading_pairs:
                    tick = await data_pipeline.get_tick(symbol)
                    if tick:
                        ticks[symbol] = tick.dict()
                await websocket.send_json({
                    "type": "ticks",
                    "data": ticks
                })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_ticks():
    """Broadcast ticks to all WebSocket clients"""
    while True:
        try:
            ticks = {}
            for symbol in settings.trading_pairs:
                tick = await data_pipeline.get_tick(symbol)
                if tick:
                    ticks[symbol] = tick.dict()
            
            await manager.broadcast({
                "type": "ticks",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": ticks
            })
            
            await asyncio.sleep(1)  # Broadcast every second
            
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            await asyncio.sleep(5)


# ============== Performance Tracking ==============

from app.services.signal_tracker import signal_tracker

@app.get("/api/performance/stats", tags=["Performance"])
async def get_performance_stats(
    days: int = Query(30, ge=1, le=365),
    symbol: Optional[str] = None
):
    """Get performance statistics"""
    stats = signal_tracker.get_performance_stats(symbol, days)
    return stats


@app.get("/api/performance/signals", tags=["Performance"])
async def get_performance_signals(limit: int = Query(50, ge=1, le=200)):
    """Get signal history"""
    signals = signal_tracker.get_recent_signals(limit)
    return [s.to_dict() for s in signals]


@app.get("/api/performance/by-symbol", tags=["Performance"])
async def get_performance_by_symbol():
    """Get performance by symbol"""
    return signal_tracker.get_performance_by_symbol()


@app.get("/api/performance/confidence-accuracy", tags=["Performance"])
async def get_confidence_accuracy():
    """Get accuracy by confidence level"""
    return signal_tracker.calculate_confidence_accuracy()


class SignalOutcomeRequest(BaseModel):
    """Request model for updating signal outcome"""
    exit_price: float
    outcome: str  # win, loss, breakeven
    max_drawdown_pips: Optional[float] = None
    max_favorable_pips: Optional[float] = None


@app.post("/api/signals/{signal_id}/outcome", tags=["Signals"])
async def update_signal_outcome(signal_id: str, request: SignalOutcomeRequest):
    """Update signal outcome"""
    record = signal_tracker.update_outcome(
        signal_id,
        request.exit_price,
        request.outcome,
        request.max_drawdown_pips,
        request.max_favorable_pips
    )
    
    if not record:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    return APIResponse(
        success=True,
        message="Signal outcome updated",
        data=record.to_dict()
    )


# ============== HMM Regime Detection ==============

from app.engines.hmm_model import regime_analyzer

@app.get("/api/hmm/regime", tags=["HMM"])
async def get_hmm_regime(symbol: str = Query("EURUSD")):
    """Get HMM regime analysis for a symbol"""
    try:
        ohlcv = await data_pipeline.get_ohlcv(symbol, "M15", 100)
        
        if not ohlcv or len(ohlcv) < 20:
            return {
                "current_state": {
                    "regime": "unknown",
                    "probability": 0,
                    "transition_probabilities": {},
                    "confidence": 0,
                    "duration_expected": 0
                },
                "forecasts": [],
                "stability": {"score": 0},
                "trading_implications": {}
            }
        
        analysis = regime_analyzer.analyze(ohlcv)
        return analysis
        
    except Exception as e:
        logger.error(f"HMM regime error: {e}")
        return {
            "current_state": {
                "regime": "unknown",
                "probability": 0,
                "transition_probabilities": {},
                "confidence": 0,
                "duration_expected": 0
            },
            "forecasts": [],
            "stability": {"score": 0},
            "trading_implications": {}
        }


# ============== MCMC Probability Estimation ==============

from app.engines.mcmc_engine import mcmc_engine, uncertainty_quantifier
from app.engines.feature_engine import feature_engine

@app.get("/api/mcmc/estimate", tags=["MCMC"])
async def get_mcmc_estimate(symbol: str = Query("EURUSD")):
    """Get MCMC probability estimates for a symbol"""
    try:
        ohlcv = await data_pipeline.get_ohlcv(symbol, "M15", 100)
        
        if not ohlcv or len(ohlcv) < 50:
            return {
                "direction_probability": {"mean": 0.5, "std": 0.2, "ci_95": [0.3, 0.7], "uncertainty": 0.4},
                "duration_probability": {"mean": 10, "std": 5, "ci_95": [3, 25]},
                "volatility_forecast": {"mean": 0.01, "std": 0.005, "ci_95": [0.005, 0.02]},
                "regime_probability": {},
                "confidence_score": 0.3,
                "effective_sample_size": 100,
                "convergence_metric": 0.5
            }
        
        # Calculate features
        features = feature_engine.calculate_features(symbol, ohlcv)
        
        # Get direction prediction
        direction_pred = direction_model.predict(features)
        duration_pred = duration_model.predict(features, direction_pred.predicted_direction)
        
        # Get regime probabilities
        regime_analysis = regime_analyzer.analyze(ohlcv)
        regime_probs = regime_analysis.get('current_state', {}).get('transition_probabilities', {})
        
        # Get MCMC estimates
        mcmc_result = mcmc_engine.estimate_probabilities(
            features,
            direction_pred.confidence,
            duration_pred.expected_time_above_minutes,
            regime_probs
        )
        
        return mcmc_result.to_dict()
        
    except Exception as e:
        logger.error(f"MCMC estimate error: {e}")
        return {
            "direction_probability": {"mean": 0.5, "std": 0.2, "ci_95": [0.3, 0.7], "uncertainty": 0.4},
            "duration_probability": {"mean": 10, "std": 5, "ci_95": [3, 25]},
            "volatility_forecast": {"mean": 0.01, "std": 0.005, "ci_95": [0.005, 0.02]},
            "regime_probability": {},
            "confidence_score": 0.3,
            "effective_sample_size": 100,
            "convergence_metric": 0.5
        }


@app.get("/api/uncertainty/quantify", tags=["MCMC"])
async def quantify_uncertainty(symbol: str = Query("EURUSD")):
    """Quantify prediction uncertainty"""
    try:
        ohlcv = await data_pipeline.get_ohlcv(symbol, "M15", 100)
        
        if not ohlcv or len(ohlcv) < 50:
            return {
                "total_uncertainty": 0.5,
                "epistemic_uncertainty": 0.3,
                "aleatoric_uncertainty": 0.2,
                "confidence_bounds": {"lower_90": 0.3, "upper_90": 0.7, "width": 0.4},
                "prediction_quality": "low",
                "recommendation": "Insufficient data"
            }
        
        features = feature_engine.calculate_features(symbol, ohlcv)
        direction_pred = direction_model.predict(features)
        duration_pred = duration_model.predict(features, direction_pred.predicted_direction)
        
        uncertainty = uncertainty_quantifier.quantify_uncertainty(
            direction_pred.confidence,
            duration_pred.expected_time_above_minutes,
            features
        )
        
        return uncertainty
        
    except Exception as e:
        logger.error(f"Uncertainty quantification error: {e}")
        return {
            "total_uncertainty": 0.5,
            "epistemic_uncertainty": 0.3,
            "aleatoric_uncertainty": 0.2,
            "confidence_bounds": {"lower_90": 0.3, "upper_90": 0.7, "width": 0.4},
            "prediction_quality": "low",
            "recommendation": "Error in calculation"
        }


# ============== Volume Analysis ==============

from app.engines.volume_engine import volume_engine

@app.get("/api/volume/analyze", tags=["Volume"])
async def analyze_volume(
    symbol: str = Query("EURUSD"),
    timeframe: str = Query("M15")
):
    """Analyze volume for a symbol"""
    try:
        ohlcv = await data_pipeline.get_ohlcv(symbol, timeframe, 100)
        
        if not ohlcv or len(ohlcv) < 10:
            return {
                "current_volume": 0,
                "avg_volume": 0,
                "volume_ratio": 1.0,
                "volume_trend": "neutral",
                "signal_strength": 0,
                "message": "Insufficient data"
            }
        
        analysis = volume_engine.analyze(symbol, ohlcv)
        return analysis.to_dict()
        
    except Exception as e:
        logger.error(f"Volume analysis error: {e}")
        return {
            "current_volume": 0,
            "avg_volume": 0,
            "volume_ratio": 1.0,
            "volume_trend": "neutral",
            "signal_strength": 0,
            "message": str(e)
        }


@app.get("/api/volume/predict", tags=["Volume"])
async def predict_volume(
    symbol: str = Query("EURUSD"),
    timeframe: str = Query("M15")
):
    """Get volume-based prediction"""
    try:
        ohlcv = await data_pipeline.get_ohlcv(symbol, timeframe, 100)
        
        if not ohlcv or len(ohlcv) < 20:
            return {
                "direction_bias": "NEUTRAL",
                "confidence": 0.5,
                "breakout_probability": 0.3,
                "reversal_probability": 0.3,
                "message": "Insufficient data"
            }
        
        prediction = volume_engine.predict(ohlcv)
        return prediction.to_dict()
        
    except Exception as e:
        logger.error(f"Volume prediction error: {e}")
        return {
            "direction_bias": "NEUTRAL",
            "confidence": 0.5,
            "breakout_probability": 0.3,
            "reversal_probability": 0.3,
            "message": str(e)
        }


@app.get("/api/volume/profile", tags=["Volume"])
async def get_volume_profile(
    symbol: str = Query("EURUSD"),
    timeframe: str = Query("M15")
):
    """Get volume profile for a symbol"""
    try:
        ohlcv = await data_pipeline.get_ohlcv(symbol, timeframe, 100)
        
        if not ohlcv or len(ohlcv) < 10:
            return {
                "high_volume_nodes": [],
                "low_volume_nodes": [],
                "poc_price": 0,
                "message": "Insufficient data"
            }
        
        analysis = volume_engine.analyze(symbol, ohlcv)
        return {
            "high_volume_nodes": analysis.high_volume_nodes,
            "low_volume_nodes": analysis.low_volume_nodes,
            "poc_price": analysis.poc_price,
            "current_price": ohlcv[-1].close if ohlcv else 0
        }
        
    except Exception as e:
        logger.error(f"Volume profile error: {e}")
        return {
            "high_volume_nodes": [],
            "low_volume_nodes": [],
            "poc_price": 0,
            "message": str(e)
        }


# ============== Auto Scanner ==============

from app.services.auto_scanner import auto_scanner

# Set data pipeline reference
auto_scanner.set_data_pipeline(data_pipeline)

# Set duration outcome tracker reference (for automatic signal tracking)
from app.services.duration_outcome_tracker import duration_outcome_tracker
auto_scanner.set_outcome_tracker(duration_outcome_tracker)


@app.get("/api/scanner/status", tags=["Scanner"])
async def get_scanner_status():
    """Get auto scanner status"""
    return auto_scanner.get_scan_results()


@app.post("/api/scanner/start", tags=["Scanner"])
async def start_scanner():
    """Start the auto scanner"""
    if auto_scanner.state.is_running:
        return APIResponse(
            success=True,
            message="Scanner already running",
            data=auto_scanner.get_scan_results()["state"]
        )
    
    # Set symbols from settings or MT5 data
    auto_scanner.set_symbols(settings.trading_pairs)
    await auto_scanner.start()
    
    return APIResponse(
        success=True,
        message="Scanner started",
        data=auto_scanner.get_scan_results()["state"]
    )


@app.post("/api/scanner/stop", tags=["Scanner"])
async def stop_scanner():
    """Stop the auto scanner"""
    await auto_scanner.stop()
    
    return APIResponse(
        success=True,
        message="Scanner stopped",
        data=None
    )


@app.get("/api/scanner/signals", tags=["Scanner"])
async def get_scanner_signals(limit: int = Query(10, ge=1, le=50)):
    """Get top signals from scanner"""
    signals = auto_scanner.get_top_signals(limit)
    return APIResponse(
        success=True,
        message=f"Found {len(signals)} signals",
        data=signals
    )


@app.get("/api/scanner/confluence", tags=["Scanner"])
async def get_scanner_confluence(min_confluence: float = Query(0.6, ge=0.0, le=1.0)):
    """Get signals with high multi-timeframe confluence"""
    signals = auto_scanner.get_confluence_signals(min_confluence)
    return APIResponse(
        success=True,
        message=f"Found {len(signals)} high-confluence signals",
        data=signals
    )


@app.post("/api/scanner/symbols", tags=["Scanner"])
async def set_scanner_symbols(symbols: List[str]):
    """Set symbols for the scanner"""
    auto_scanner.set_symbols(symbols)
    
    return APIResponse(
        success=True,
        message=f"Scanner symbols updated: {symbols}",
        data=list(auto_scanner.symbols_to_scan)
    )


@app.post("/api/scanner/config", tags=["Scanner"])
async def configure_scanner(
    interval_seconds: Optional[int] = Query(None, ge=5, le=300),
    min_confidence: Optional[float] = Query(None, ge=0.5, le=1.0)
):
    """Configure scanner settings"""
    if interval_seconds:
        auto_scanner.scan_interval_seconds = interval_seconds
    if min_confidence:
        auto_scanner.min_signal_confidence = min_confidence
    
    return APIResponse(
        success=True,
        message="Scanner configuration updated",
        data={
            "interval_seconds": auto_scanner.scan_interval_seconds,
            "min_confidence": auto_scanner.min_signal_confidence,
            "timeframes": auto_scanner.timeframes_to_scan,
            "symbols": list(auto_scanner.symbols_to_scan)
        }
    )


@app.post("/api/scanner/scan-now", tags=["Scanner"])
async def trigger_scan_now():
    """Trigger an immediate scan"""
    if not auto_scanner.state.is_running:
        # One-time scan
        auto_scanner.set_symbols(settings.trading_pairs)
        await auto_scanner._run_scan_cycle()
    else:
        # Already running, just return current results
        pass
    
    return APIResponse(
        success=True,
        message="Scan completed",
        data=auto_scanner.get_scan_results()
    )


# ============== Duration-Based Predictions ==============

from app.engines.duration_predictor import duration_predictor
from app.services.duration_outcome_tracker import duration_outcome_tracker

# Set data pipeline reference for outcome tracker
duration_outcome_tracker.set_data_pipeline(data_pipeline)


@app.get("/api/duration/predict", tags=["Duration Predictions"])
async def get_duration_predictions(
    symbol: str = Query("EURUSD", description="Trading symbol"),
    timeframe: str = Query("M1", description="Source timeframe for analysis")
):
    """Get duration-based predictions for all time horizons (1min, 2min, 3min, 5min, etc.)"""
    try:
        ohlcv = await data_pipeline.get_ohlcv(symbol, timeframe, 100)
        tick = await data_pipeline.get_tick(symbol)
        
        if not ohlcv or len(ohlcv) < 20:
            return APIResponse(
                success=False,
                message="Insufficient data for predictions",
                data=[]
            )
        
        current_price = tick.bid if tick else ohlcv[-1].close
        
        # Get market state for volatility
        market_state = await data_pipeline.get_market_state(symbol)
        volatility = market_state.get('volatility', 0) if market_state else 0
        
        predictions = duration_predictor.predict_durations(symbol, ohlcv, current_price, volatility)
        
        return APIResponse(
            success=True,
            message=f"Generated {len(predictions)} duration predictions",
            data=[p.to_dict() for p in predictions]
        )
        
    except Exception as e:
        logger.error(f"Duration prediction error: {e}")
        return APIResponse(
            success=False,
            message=str(e),
            data=[]
        )


@app.post("/api/duration/generate-signal", tags=["Duration Predictions"])
async def generate_duration_signal(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query("M1", description="Source timeframe for analysis"),
    min_confidence: float = Query(0.55, ge=0.5, le=0.95, description="Minimum confidence threshold"),
    max_noise: float = Query(0.4, ge=0.1, le=0.8, description="Maximum noise score")
):
    """Generate a duration-based signal with entry, target, stop loss, and duration"""
    try:
        ohlcv = await data_pipeline.get_ohlcv(symbol.upper(), timeframe, 100)
        tick = await data_pipeline.get_tick(symbol.upper())
        
        if not ohlcv or len(ohlcv) < 20:
            return APIResponse(
                success=False,
                message="Insufficient market data",
                data=None
            )
        
        current_price = tick.bid if tick else ohlcv[-1].close
        
        # Get market state
        market_state = await data_pipeline.get_market_state(symbol.upper())
        volatility = market_state.get('volatility', 0) if market_state else 0
        session = market_state.get('session', 'unknown') if market_state else 'unknown'
        
        # Generate signal
        signal = duration_predictor.generate_signal(
            symbol.upper(),
            ohlcv,
            current_price,
            volatility,
            session,
            min_confidence,
            max_noise
        )
        
        if not signal:
            return APIResponse(
                success=False,
                message="No valid signal generated - criteria not met (confidence too low or noise too high)",
                data=None
            )
        
        # Add to outcome tracker
        duration_outcome_tracker.add_signal(signal)
        
        return APIResponse(
            success=True,
            message=f"Signal generated: {signal.direction.value} for {signal.duration_minutes}min",
            data=signal.to_dict()
        )
        
    except Exception as e:
        logger.error(f"Duration signal generation error: {e}")
        return APIResponse(
            success=False,
            message=str(e),
            data=None
        )


@app.get("/api/duration/active-signals", tags=["Duration Predictions"])
async def get_active_duration_signals():
    """Get all active duration signals being tracked"""
    signals = duration_outcome_tracker.get_active_signals()
    return APIResponse(
        success=True,
        message=f"Found {len(signals)} active signals",
        data=[s.to_dict() for s in signals]
    )


@app.get("/api/duration/history", tags=["Duration Predictions"])
async def get_duration_history(limit: int = Query(50, ge=1, le=200)):
    """Get historical duration signals with outcomes"""
    signals = duration_outcome_tracker.get_recent_signals(limit)
    return APIResponse(
        success=True,
        message=f"Retrieved {len(signals)} historical signals",
        data=signals
    )


@app.get("/api/duration/stats", tags=["Duration Predictions"])
async def get_duration_stats():
    """Get duration prediction performance statistics"""
    stats = duration_outcome_tracker.get_stats()
    return APIResponse(
        success=True,
        message="Statistics retrieved",
        data=stats
    )


@app.get("/api/duration/best-duration", tags=["Duration Predictions"])
async def get_best_duration():
    """Get the duration with the best historical win rate"""
    duration, win_rate = duration_outcome_tracker.get_best_duration()
    return APIResponse(
        success=True,
        message="Best duration found",
        data={
            "best_duration_minutes": duration,
            "win_rate": win_rate
        }
    )


@app.on_event("startup")
async def start_duration_tracker():
    """Start the duration outcome tracker on startup"""
    duration_outcome_tracker.load_signals()
    await duration_outcome_tracker.start_tracking()
    logger.info("Duration outcome tracker started")


@app.on_event("shutdown")
async def stop_duration_tracker():
    """Stop the duration outcome tracker on shutdown"""
    await duration_outcome_tracker.stop_tracking()
    logger.info("Duration outcome tracker stopped")


# Run with: uvicorn app.api.routes:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
