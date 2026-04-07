"""
Forex Probability Intelligence System - Duration Signal Outcome Tracker
Automatically tracks signal outcomes after duration expires
"""
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger
from collections import deque
import json
import os

from app.engines.duration_predictor import DurationSignal, duration_predictor


@dataclass
class PerformanceStats:
    """Performance statistics for duration signals"""
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    pending: int = 0
    
    win_rate: float = 0.0
    avg_pips_won: float = 0.0
    avg_pips_lost: float = 0.0
    total_pips: float = 0.0
    
    # By duration
    by_duration: Dict[int, Dict] = field(default_factory=dict)
    
    # By symbol
    by_symbol: Dict[str, Dict] = field(default_factory=dict)
    
    # By session
    by_session: Dict[str, Dict] = field(default_factory=dict)
    
    # Confidence accuracy
    high_confidence_win_rate: float = 0.0  # signals with > 70% confidence
    medium_confidence_win_rate: float = 0.0  # signals with 55-70% confidence


class DurationOutcomeTracker:
    """Tracks and evaluates duration-based signal outcomes"""
    
    def __init__(self):
        self.active_signals: Dict[str, DurationSignal] = {}
        self.completed_signals: deque = deque(maxlen=1000)
        self.stats = PerformanceStats()
        
        # Callbacks
        self.on_signal_complete_callbacks: List[callable] = []
        
        # Background task
        self._tracking_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
        # Data pipeline reference
        self._data_pipeline = None
        
        # Persistence
        self._storage_file = "/home/z/my-project/forex-system/data/duration_signals.json"
        self._ensure_storage_dir()
        
    def _ensure_storage_dir(self):
        """Ensure storage directory exists"""
        os.makedirs(os.path.dirname(self._storage_file), exist_ok=True)
    
    def set_data_pipeline(self, pipeline):
        """Set the data pipeline reference for price updates"""
        self._data_pipeline = pipeline
    
    def add_signal(self, signal: DurationSignal):
        """Add a new signal to track"""
        self.active_signals[signal.signal_id] = signal
        logger.info(f"Tracking signal: {signal.signal_id} - {signal.symbol} {signal.direction.value} for {signal.duration_minutes}min")
        
        # Persist
        self._persist_signals()
    
    async def start_tracking(self):
        """Start the outcome tracking background task"""
        if self._tracking_task:
            return
        
        self._stop_event.clear()
        self._tracking_task = asyncio.create_task(self._tracking_loop())
        logger.info("Duration outcome tracker started")
    
    async def stop_tracking(self):
        """Stop the tracking task"""
        self._stop_event.set()
        if self._tracking_task:
            try:
                await asyncio.wait_for(self._tracking_task, timeout=5)
            except asyncio.TimeoutError:
                self._tracking_task.cancel()
        logger.info("Duration outcome tracker stopped")
    
    async def _tracking_loop(self):
        """Main tracking loop"""
        while not self._stop_event.is_set():
            try:
                await self._check_signal_outcomes()
            except Exception as e:
                logger.error(f"Error checking outcomes: {e}")
            
            # Check every 10 seconds
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=10)
                break
            except asyncio.TimeoutError:
                pass
    
    async def _check_signal_outcomes(self):
        """Check all active signals for outcome"""
        now = datetime.now(timezone.utc)
        signals_to_complete = []
        
        for signal_id, signal in self.active_signals.items():
            # Check if duration has expired
            expiry_time = signal.timestamp + timedelta(minutes=signal.duration_minutes)
            
            if now >= expiry_time:
                # Time to evaluate
                await self._evaluate_signal(signal)
                signals_to_complete.append(signal_id)
            else:
                # Check if target or stop was hit early
                current_price = await self._get_current_price(signal.symbol)
                if current_price:
                    if self._check_target_hit(signal, current_price):
                        signal.status = "hit_target"
                        signal.outcome = "win"
                        signal.exit_price = current_price
                        signal.exit_time = now
                        signal.actual_duration_minutes = (now - signal.timestamp).total_seconds() / 60
                        signals_to_complete.append(signal_id)
                    elif self._check_stop_hit(signal, current_price):
                        signal.status = "hit_stop"
                        signal.outcome = "loss"
                        signal.exit_price = current_price
                        signal.exit_time = now
                        signal.actual_duration_minutes = (now - signal.timestamp).total_seconds() / 60
                        signals_to_complete.append(signal_id)
        
        # Complete signals
        for signal_id in signals_to_complete:
            signal = self.active_signals.pop(signal_id, None)
            if signal:
                self._record_completion(signal)
    
    async def _evaluate_signal(self, signal: DurationSignal):
        """Evaluate signal outcome at expiry"""
        current_price = await self._get_current_price(signal.symbol)
        
        if not current_price:
            signal.status = "expired"
            signal.outcome = "pending"
            signal.exit_price = signal.entry_price
            return
        
        signal.exit_price = current_price
        signal.exit_time = datetime.now(timezone.utc)
        signal.actual_duration_minutes = signal.duration_minutes
        
        # Calculate actual pips
        pip_value = 0.0001 if "JPY" not in signal.symbol else 0.01
        
        if signal.direction.value == "BUY":
            signal.actual_pips = (current_price - signal.entry_price) / pip_value
        else:
            signal.actual_pips = (signal.entry_price - current_price) / pip_value
        
        # Determine outcome
        # Win if price moved in predicted direction by at least 1 pip
        if signal.actual_pips >= 1:
            signal.status = "hit_target"
            signal.outcome = "win"
        elif signal.actual_pips <= -1:
            signal.status = "hit_stop"
            signal.outcome = "loss"
        else:
            signal.status = "expired"
            signal.outcome = "breakeven"
        
        logger.info(
            f"Signal {signal.signal_id} completed: {signal.outcome.upper()} "
            f"({signal.actual_pips:.1f} pips)"
        )
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        if not self._data_pipeline:
            return None
        
        try:
            tick = await self._data_pipeline.get_tick(symbol)
            if tick:
                return tick.bid
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
        
        return None
    
    def _check_target_hit(self, signal: DurationSignal, current_price: float) -> bool:
        """Check if target price was hit"""
        if signal.direction.value == "BUY":
            return current_price >= signal.target_price
        else:
            return current_price <= signal.target_price
    
    def _check_stop_hit(self, signal: DurationSignal, current_price: float) -> bool:
        """Check if stop loss was hit"""
        if signal.direction.value == "BUY":
            return current_price <= signal.stop_loss_price
        else:
            return current_price >= signal.stop_loss_price
    
    def _record_completion(self, signal: DurationSignal):
        """Record completed signal and update stats"""
        self.completed_signals.append(signal)
        
        # Update stats
        self.stats.total_signals += 1
        
        if signal.outcome == "win":
            self.stats.wins += 1
            self.stats.avg_pips_won = (
                (self.stats.avg_pips_won * (self.stats.wins - 1) + abs(signal.actual_pips or 0))
                / self.stats.wins
            )
            self.stats.total_pips += abs(signal.actual_pips or 0)
        elif signal.outcome == "loss":
            self.stats.losses += 1
            self.stats.avg_pips_lost = (
                (self.stats.avg_pips_lost * (self.stats.losses - 1) + abs(signal.actual_pips or 0))
                / self.stats.losses
            )
            self.stats.total_pips -= abs(signal.actual_pips or 0)
        else:
            self.stats.breakeven += 1
        
        # Calculate win rate
        decided = self.stats.wins + self.stats.losses
        self.stats.win_rate = self.stats.wins / decided if decided > 0 else 0
        
        # Update by duration
        if signal.duration_minutes not in self.stats.by_duration:
            self.stats.by_duration[signal.duration_minutes] = {
                "total": 0, "wins": 0, "losses": 0, "win_rate": 0, "avg_pips": 0
            }
        
        dur_stats = self.stats.by_duration[signal.duration_minutes]
        dur_stats["total"] += 1
        if signal.outcome == "win":
            dur_stats["wins"] += 1
        elif signal.outcome == "loss":
            dur_stats["losses"] += 1
        dur_stats["win_rate"] = dur_stats["wins"] / dur_stats["total"] if dur_stats["total"] > 0 else 0
        
        # Update predictor accuracy
        was_correct = signal.outcome == "win"
        duration_predictor.update_accuracy(signal.duration_minutes, was_correct, signal.actual_pips)
        
        # Update by symbol
        if signal.symbol not in self.stats.by_symbol:
            self.stats.by_symbol[signal.symbol] = {"total": 0, "wins": 0, "win_rate": 0}
        self.stats.by_symbol[signal.symbol]["total"] += 1
        if signal.outcome == "win":
            self.stats.by_symbol[signal.symbol]["wins"] += 1
        self.stats.by_symbol[signal.symbol]["win_rate"] = (
            self.stats.by_symbol[signal.symbol]["wins"] / 
            self.stats.by_symbol[signal.symbol]["total"]
        )
        
        # Update by session
        session = signal.session or "unknown"
        if session not in self.stats.by_session:
            self.stats.by_session[session] = {"total": 0, "wins": 0, "win_rate": 0}
        self.stats.by_session[session]["total"] += 1
        if signal.outcome == "win":
            self.stats.by_session[session]["wins"] += 1
        self.stats.by_session[session]["win_rate"] = (
            self.stats.by_session[session]["wins"] / 
            self.stats.by_session[session]["total"]
        )
        
        # Persist
        self._persist_signals()
        
        # Notify callbacks
        for callback in self.on_signal_complete_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _persist_signals(self):
        """Persist signals to file"""
        try:
            data = {
                "active": {sid: s.to_dict() for sid, s in self.active_signals.items()},
                "completed": [s.to_dict() for s in self.completed_signals],
                "stats": {
                    "total_signals": self.stats.total_signals,
                    "wins": self.stats.wins,
                    "losses": self.stats.losses,
                    "win_rate": self.stats.win_rate,
                    "total_pips": self.stats.total_pips,
                    "by_duration": self.stats.by_duration,
                    "by_symbol": self.stats.by_symbol
                }
            }
            
            with open(self._storage_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error persisting signals: {e}")
    
    def load_signals(self):
        """Load signals from file"""
        try:
            if os.path.exists(self._storage_file):
                with open(self._storage_file, 'r') as f:
                    data = json.load(f)
                
                # Load stats
                if "stats" in data:
                    self.stats.total_signals = data["stats"].get("total_signals", 0)
                    self.stats.wins = data["stats"].get("wins", 0)
                    self.stats.losses = data["stats"].get("losses", 0)
                    self.stats.win_rate = data["stats"].get("win_rate", 0)
                    self.stats.total_pips = data["stats"].get("total_pips", 0)
                    self.stats.by_duration = data["stats"].get("by_duration", {})
                    self.stats.by_symbol = data["stats"].get("by_symbol", {})
                
                logger.info(f"Loaded {self.stats.total_signals} historical signals")
        except Exception as e:
            logger.error(f"Error loading signals: {e}")
    
    def get_active_signals(self) -> List[DurationSignal]:
        """Get all active signals"""
        return list(self.active_signals.values())
    
    def get_recent_signals(self, limit: int = 50) -> List[Dict]:
        """Get recent completed signals"""
        signals = list(self.completed_signals)[-limit:]
        return [s.to_dict() for s in reversed(signals)]
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            "total_signals": self.stats.total_signals,
            "wins": self.stats.wins,
            "losses": self.stats.losses,
            "breakeven": self.stats.breakeven,
            "pending": len(self.active_signals),
            "win_rate": round(self.stats.win_rate * 100, 1),
            "avg_pips_won": round(self.stats.avg_pips_won, 1),
            "avg_pips_lost": round(self.stats.avg_pips_lost, 1),
            "total_pips": round(self.stats.total_pips, 1),
            "by_duration": self.stats.by_duration,
            "by_symbol": self.stats.by_symbol,
            "by_session": self.stats.by_session
        }
    
    def get_best_duration(self) -> Tuple[int, float]:
        """Get the duration with best win rate"""
        if not self.stats.by_duration:
            return (15, 0.5)  # Default
        
        best_dur = max(
            self.stats.by_duration.items(),
            key=lambda x: x[1].get("win_rate", 0) if x[1].get("total", 0) >= 5 else 0
        )
        return (best_dur[0], best_dur[1].get("win_rate", 0))


# Global instance
duration_outcome_tracker = DurationOutcomeTracker()
