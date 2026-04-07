"""
Forex Probability Intelligence System - Auto Scanner Service
Continuously scans all symbols across multiple timeframes for signals
"""
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from loguru import logger
from collections import deque
import json

from app.models.schemas import TradingSignal, SignalBias, SignalStatus
from app.engines.decision_engine import decision_engine
from app.engines.feature_engine import feature_engine
from app.engines.direction_model import direction_model
from app.engines.duration_model import duration_model
from app.services.signal_tracker import signal_tracker
from config.settings import get_settings

settings = get_settings()

# All MT5 supported timeframes
MT5_TIMEFRAMES = [
    "M1", "M2", "M3", "M4", "M5", "M6", "M10", "M12", "M15", "M20", "M30",
    "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D1", "W1", "MN1"
]

# Priority timeframes for active scanning
PRIORITY_TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4", "D1"]


@dataclass
class ScanResult:
    """Result of a single symbol-timeframe scan"""
    symbol: str
    timeframe: str
    timestamp: datetime
    signal_generated: bool
    confidence: float = 0.0
    bias: Optional[SignalBias] = None
    prob_up: float = 0.0
    prob_down: float = 0.0
    volatility: float = 0.0
    trend: str = "NEUTRAL"
    features_score: float = 0.0
    message: str = ""
    signal_id: Optional[str] = None
    # Entry/Exit details
    entry_price: float = 0.0
    entry_zone_start: float = 0.0
    entry_zone_end: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    expected_duration_minutes: float = 0.0
    risk_reward_ratio: float = 0.0


@dataclass
class MultiTimeframeAnalysis:
    """Multi-timeframe analysis for a symbol"""
    symbol: str
    timestamp: datetime
    timeframes: Dict[str, ScanResult] = field(default_factory=dict)
    
    # Aggregated signals
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    
    # Consensus
    consensus_bias: SignalBias = SignalBias.NEUTRAL
    consensus_strength: float = 0.0
    
    # Best timeframe
    best_timeframe: str = ""
    best_confidence: float = 0.0
    
    # Confluence score
    confluence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "consensus_bias": self.consensus_bias.value if self.consensus_bias else "NEUTRAL",
            "consensus_strength": self.consensus_strength,
            "best_timeframe": self.best_timeframe,
            "best_confidence": self.best_confidence,
            "confluence_score": self.confluence_score,
            "timeframes": {tf: {
                "signal_generated": r.signal_generated,
                "confidence": r.confidence,
                "bias": r.bias.value if r.bias else "NEUTRAL",
                "prob_up": r.prob_up,
                "prob_down": r.prob_down,
                "trend": r.trend
            } for tf, r in self.timeframes.items()}
        }


@dataclass
class ScannerState:
    """Current state of the scanner"""
    is_running: bool = False
    last_scan_time: Optional[datetime] = None
    symbols_scanned: int = 0
    signals_found: int = 0
    scan_cycles: int = 0
    errors: int = 0
    current_status: str = "idle"


class AutoScanner:
    """Automatic signal scanner for all symbols and timeframes"""
    
    def __init__(self):
        self.state = ScannerState()
        self.scan_results: Dict[str, MultiTimeframeAnalysis] = {}
        self.recent_signals: deque = deque(maxlen=100)
        self.scan_history: deque = deque(maxlen=1000)
        
        # Store actual signal objects for detailed info
        self._generated_signals: Dict[str, TradingSignal] = {}
        
        # Configuration
        self.scan_interval_seconds = 30  # Scan every 30 seconds
        self.min_signal_confidence = 0.65
        self.timeframes_to_scan = PRIORITY_TIMEFRAMES
        self.symbols_to_scan: Set[str] = set()
        
        # Callbacks
        self.on_signal_callbacks: List[callable] = []
        self.on_scan_callbacks: List[callable] = []
        
        # Background task
        self._scan_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
        # Data pipeline reference
        self._data_pipeline = None
        
    def set_data_pipeline(self, pipeline):
        """Set the data pipeline reference"""
        self._data_pipeline = pipeline
        
    def add_symbol(self, symbol: str):
        """Add a symbol to scan"""
        self.symbols_to_scan.add(symbol.upper())
        
    def remove_symbol(self, symbol: str):
        """Remove a symbol from scanning"""
        self.symbols_to_scan.discard(symbol.upper())
        
    def set_symbols(self, symbols: List[str]):
        """Set symbols to scan"""
        self.symbols_to_scan = set(s.upper() for s in symbols)
        
    def add_signal_callback(self, callback: callable):
        """Add callback for when a signal is found"""
        self.on_signal_callbacks.append(callback)
        
    def add_scan_callback(self, callback: callable):
        """Add callback for when a scan completes"""
        self.on_scan_callbacks.append(callback)
        
    async def start(self):
        """Start the auto scanner"""
        if self.state.is_running:
            logger.warning("Auto scanner already running")
            return
            
        self.state.is_running = True
        self.state.current_status = "starting"
        self._stop_event.clear()
        
        logger.info("=" * 60)
        logger.info("🚀 Auto Scanner Starting")
        logger.info(f"   Symbols: {self.symbols_to_scan}")
        logger.info(f"   Timeframes: {self.timeframes_to_scan}")
        logger.info(f"   Scan interval: {self.scan_interval_seconds}s")
        logger.info("=" * 60)
        
        self._scan_task = asyncio.create_task(self._scan_loop())
        self.state.current_status = "running"
        
    async def stop(self):
        """Stop the auto scanner"""
        logger.info("Stopping auto scanner...")
        self._stop_event.set()
        self.state.is_running = False
        self.state.current_status = "stopped"
        
        if self._scan_task:
            try:
                await asyncio.wait_for(self._scan_task, timeout=5)
            except asyncio.TimeoutError:
                self._scan_task.cancel()
                
    async def _scan_loop(self):
        """Main scanning loop"""
        while not self._stop_event.is_set():
            try:
                await self._run_scan_cycle()
                self.state.scan_cycles += 1
            except Exception as e:
                logger.error(f"Scan cycle error: {e}")
                self.state.errors += 1
                
            # Wait for next scan
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.scan_interval_seconds
                )
                break  # Stop event was set
            except asyncio.TimeoutError:
                pass  # Continue scanning
                
    async def _run_scan_cycle(self):
        """Run a complete scan cycle for all symbols and timeframes"""
        self.state.current_status = "scanning"
        start_time = datetime.now(timezone.utc)
        
        logger.debug(f"Starting scan cycle {self.state.scan_cycles + 1}")
        
        # Get symbols from data pipeline if not set
        if not self.symbols_to_scan and self._data_pipeline:
            # Use symbols that have data
            available_symbols = set(self._data_pipeline.mt5_store.ticks.keys())
            if available_symbols:
                self.symbols_to_scan = available_symbols
                logger.info(f"Auto-detected symbols: {available_symbols}")
        
        # Scan all symbols in parallel
        scan_tasks = []
        for symbol in self.symbols_to_scan:
            scan_tasks.append(self._scan_symbol(symbol))
            
        results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        # Process results
        signals_found = 0
        for result in results:
            if isinstance(result, MultiTimeframeAnalysis):
                self.scan_results[result.symbol] = result
                if result.best_confidence >= self.min_signal_confidence:
                    signals_found += 1
                    
        # Update state
        self.state.last_scan_time = start_time
        self.state.symbols_scanned = len(self.symbols_to_scan)
        self.state.signals_found = signals_found
        self.state.current_status = "running"
        
        # Record scan history
        self.scan_history.append({
            "timestamp": start_time.isoformat(),
            "symbols": len(self.symbols_to_scan),
            "signals": signals_found,
            "cycle": self.state.scan_cycles
        })
        
        # Notify callbacks
        for callback in self.on_scan_callbacks:
            try:
                await callback(self.scan_results)
            except Exception as e:
                logger.error(f"Scan callback error: {e}")
                
        logger.info(f"Scan cycle complete: {signals_found} signals from {len(self.symbols_to_scan)} symbols")
        
    async def _scan_symbol(self, symbol: str) -> MultiTimeframeAnalysis:
        """Scan a symbol across all timeframes"""
        analysis = MultiTimeframeAnalysis(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Scan each timeframe
        tf_tasks = []
        for tf in self.timeframes_to_scan:
            tf_tasks.append(self._scan_timeframe(symbol, tf))
            
        tf_results = await asyncio.gather(*tf_tasks, return_exceptions=True)
        
        # Process timeframe results
        bullish = 0
        bearish = 0
        neutral = 0
        best_conf = 0
        best_tf = ""
        
        for i, result in enumerate(tf_results):
            if isinstance(result, ScanResult):
                tf = self.timeframes_to_scan[i]
                analysis.timeframes[tf] = result
                
                # Count bias
                if result.bias == SignalBias.BUY:
                    bullish += 1
                elif result.bias == SignalBias.SELL:
                    bearish += 1
                else:
                    neutral += 1
                    
                # Track best
                if result.confidence > best_conf:
                    best_conf = result.confidence
                    best_tf = tf
                    
                # If signal was generated, add to recent signals
                if result.signal_generated and result.signal_id:
                    self.recent_signals.append(result)
                    
                    # Notify callbacks
                    for callback in self.on_signal_callbacks:
                        try:
                            await callback(result)
                        except Exception as e:
                            logger.error(f"Signal callback error: {e}")
                            
        analysis.bullish_count = bullish
        analysis.bearish_count = bearish
        analysis.neutral_count = neutral
        analysis.best_timeframe = best_tf
        analysis.best_confidence = best_conf
        
        # Calculate consensus
        total = bullish + bearish + neutral
        if total > 0:
            if bullish > bearish and bullish > neutral:
                analysis.consensus_bias = SignalBias.BUY
                analysis.consensus_strength = bullish / total
            elif bearish > bullish and bearish > neutral:
                analysis.consensus_bias = SignalBias.SELL
                analysis.consensus_strength = bearish / total
            else:
                analysis.consensus_bias = SignalBias.NEUTRAL
                analysis.consensus_strength = 0.5
                
        # Calculate confluence score
        # Higher when multiple timeframes agree
        if bullish > bearish:
            analysis.confluence_score = bullish / total if total > 0 else 0
        elif bearish > bullish:
            analysis.confluence_score = bearish / total if total > 0 else 0
        else:
            analysis.confluence_score = 0.5
            
        return analysis
        
    async def _scan_timeframe(self, symbol: str, timeframe: str) -> ScanResult:
        """Scan a specific symbol-timeframe combination"""
        result = ScanResult(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            signal_generated=False
        )
        
        try:
            if not self._data_pipeline:
                result.message = "No data pipeline"
                return result
                
            # Get data
            tick = await self._data_pipeline.get_tick(symbol)
            ohlcv = await self._data_pipeline.get_ohlcv(symbol, timeframe, 100)
            market_state = await self._data_pipeline.get_market_state(symbol)
            
            if not tick or not ohlcv or len(ohlcv) < 20:
                result.message = "Insufficient data"
                return result
                
            # Calculate features
            features = feature_engine.calculate_features(symbol, ohlcv)
            
            # Get direction prediction
            direction_pred = direction_model.predict(features)
            
            # Get duration prediction
            duration_pred = duration_model.predict(features, direction_pred.predicted_direction)
            
            # Fill result
            result.confidence = direction_pred.confidence
            result.bias = direction_pred.predicted_direction
            result.prob_up = direction_pred.prob_up
            result.prob_down = direction_pred.prob_down
            result.trend = market_state.get('trend', 'NEUTRAL')
            result.volatility = market_state.get('volatility', 0)
            result.features_score = features.price_action.bos_strength if features.price_action.break_of_structure else 0
            
            # Check if signal should be generated
            if direction_pred.confidence >= self.min_signal_confidence:
                # Generate signal through decision engine
                signal = decision_engine.generate_signal(symbol, ohlcv, tick, market_state)
                
                if signal:
                    result.signal_generated = True
                    result.signal_id = signal.signal_id
                    result.message = f"Signal generated: {signal.bias.value}"
                    
                    # Fill in entry/exit details
                    result.entry_price = tick.bid
                    result.entry_zone_start = signal.entry_zone_start
                    result.entry_zone_end = signal.entry_zone_end
                    result.stop_loss = signal.stop_loss
                    result.take_profit = signal.take_profit
                    result.expected_duration_minutes = signal.expected_duration_minutes
                    result.risk_reward_ratio = signal.risk_reward_ratio
                    
                    # Store the full signal for later retrieval
                    self._generated_signals[signal.signal_id] = signal
                    
                    # Track signal with market state
                    signal_tracker.record_signal(signal, market_state)
                else:
                    result.message = "Criteria not met"
            else:
                result.message = f"Confidence {direction_pred.confidence:.2f} below threshold"
                
        except Exception as e:
            logger.error(f"Error scanning {symbol} {timeframe}: {e}")
            result.message = f"Error: {str(e)[:50]}"
            
        return result
        
    def get_scan_results(self) -> Dict[str, Any]:
        """Get current scan results"""
        return {
            "state": {
                "is_running": self.state.is_running,
                "last_scan_time": self.state.last_scan_time.isoformat() if self.state.last_scan_time else None,
                "symbols_scanned": self.state.symbols_scanned,
                "signals_found": self.state.signals_found,
                "scan_cycles": self.state.scan_cycles,
                "errors": self.state.errors,
                "current_status": self.state.current_status
            },
            "symbols": {symbol: analysis.to_dict() for symbol, analysis in self.scan_results.items()},
            "recent_signals": [
                {
                    "symbol": r.symbol,
                    "timeframe": r.timeframe,
                    "bias": r.bias.value if r.bias else "NEUTRAL",
                    "confidence": r.confidence,
                    "timestamp": r.timestamp.isoformat()
                } for r in self.recent_signals
            ]
        }
        
    def get_top_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top signals by confidence"""
        signals = []
        
        for symbol, analysis in self.scan_results.items():
            if analysis.best_confidence >= self.min_signal_confidence:
                # Get entry/exit details from best timeframe's signal
                best_tf_result = analysis.timeframes.get(analysis.best_timeframe)
                
                signal_data = {
                    "symbol": symbol,
                    "timeframe": analysis.best_timeframe,
                    "bias": analysis.consensus_bias.value,
                    "confidence": analysis.best_confidence,
                    "confluence": analysis.confluence_score,
                    "bullish_tf": analysis.bullish_count,
                    "bearish_tf": analysis.bearish_count,
                    "timestamp": analysis.timestamp.isoformat()
                }
                
                # Add entry/exit details if available
                if best_tf_result and best_tf_result.signal_generated:
                    signal_data["entry_price"] = best_tf_result.entry_price
                    signal_data["entry_zone_start"] = best_tf_result.entry_zone_start
                    signal_data["entry_zone_end"] = best_tf_result.entry_zone_end
                    signal_data["stop_loss"] = best_tf_result.stop_loss
                    signal_data["take_profit"] = best_tf_result.take_profit
                    signal_data["expected_duration_minutes"] = best_tf_result.expected_duration_minutes
                    signal_data["risk_reward_ratio"] = best_tf_result.risk_reward_ratio
                    signal_data["signal_id"] = best_tf_result.signal_id
                
                signals.append(signal_data)
                
        # Sort by confidence
        signals.sort(key=lambda x: x["confidence"], reverse=True)
        return signals[:limit]
        
    def get_confluence_signals(self, min_confluence: float = 0.6) -> List[Dict[str, Any]]:
        """Get signals with high multi-timeframe confluence"""
        signals = []
        
        for symbol, analysis in self.scan_results.items():
            if analysis.confluence_score >= min_confluence and analysis.best_confidence >= self.min_signal_confidence:
                # Get entry/exit details from best timeframe's signal
                best_tf_result = analysis.timeframes.get(analysis.best_timeframe)
                
                signal_data = {
                    "symbol": symbol,
                    "consensus_bias": analysis.consensus_bias.value,
                    "consensus_strength": analysis.consensus_strength,
                    "confluence_score": analysis.confluence_score,
                    "best_timeframe": analysis.best_timeframe,
                    "best_confidence": analysis.best_confidence,
                    "timeframe_alignment": {
                        "bullish": analysis.bullish_count,
                        "bearish": analysis.bearish_count,
                        "neutral": analysis.neutral_count
                    }
                }
                
                # Add entry/exit details if available
                if best_tf_result and best_tf_result.signal_generated:
                    signal_data["entry_price"] = best_tf_result.entry_price
                    signal_data["entry_zone_start"] = best_tf_result.entry_zone_start
                    signal_data["entry_zone_end"] = best_tf_result.entry_zone_end
                    signal_data["stop_loss"] = best_tf_result.stop_loss
                    signal_data["take_profit"] = best_tf_result.take_profit
                    signal_data["expected_duration_minutes"] = best_tf_result.expected_duration_minutes
                    signal_data["risk_reward_ratio"] = best_tf_result.risk_reward_ratio
                    signal_data["signal_id"] = best_tf_result.signal_id
                
                signals.append(signal_data)
                
        signals.sort(key=lambda x: x["confluence_score"], reverse=True)
        return signals


# Global instance
auto_scanner = AutoScanner()
