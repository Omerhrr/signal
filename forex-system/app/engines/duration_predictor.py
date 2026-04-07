"""
Forex Probability Intelligence System - Duration-Based Prediction Engine
Predicts market direction for specific time durations (1min, 2min, 3min, 5min, etc.)
Reduces noise by focusing on predicted move duration rather than timeframe signals
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from loguru import logger
from collections import deque
import scipy.stats as stats

from app.models.schemas import OHLCV, SignalBias, TickData


# Prediction durations in minutes
PREDICTION_DURATIONS = [1, 2, 3, 5, 10, 15, 30, 60]


@dataclass
class DurationPrediction:
    """Prediction for a specific duration"""
    duration_minutes: int
    direction: SignalBias = SignalBias.NEUTRAL
    confidence: float = 0.0
    probability_up: float = 0.5
    probability_down: float = 0.5
    
    # Price predictions
    expected_move_pips: float = 0.0
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_loss_price: float = 0.0
    
    # Risk metrics
    risk_reward_ratio: float = 0.0
    probability_of_hit: float = 0.0
    
    # Quality metrics
    noise_score: float = 0.0  # Lower is better
    reliability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_minutes": self.duration_minutes,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 3),
            "probability_up": round(self.probability_up, 3),
            "probability_down": round(self.probability_down, 3),
            "expected_move_pips": round(self.expected_move_pips, 2),
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss_price": self.stop_loss_price,
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "probability_of_hit": round(self.probability_of_hit, 3),
            "noise_score": round(self.noise_score, 3),
            "reliability": round(self.reliability, 3)
        }


@dataclass
class DurationSignal:
    """Complete signal for a specific duration"""
    signal_id: str
    symbol: str
    timestamp: datetime
    
    # Prediction
    duration_minutes: int
    direction: SignalBias
    confidence: float
    
    # Prices
    entry_price: float
    target_price: float
    stop_loss_price: float
    expected_move_pips: float
    
    # Risk
    risk_reward_ratio: float
    
    # Status tracking
    status: str = "active"  # active, hit_target, hit_stop, expired, cancelled
    outcome: str = "pending"  # pending, win, loss, breakeven
    
    # Result tracking
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    actual_duration_minutes: Optional[float] = None
    actual_pips: Optional[float] = None
    
    # Market context
    market_regime: str = "unknown"
    volatility_at_entry: float = 0.0
    session: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "duration_minutes": self.duration_minutes,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 3),
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss_price": self.stop_loss_price,
            "expected_move_pips": round(self.expected_move_pips, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "status": self.status,
            "outcome": self.outcome,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "actual_duration_minutes": self.actual_duration_minutes,
            "actual_pips": round(self.actual_pips, 2) if self.actual_pips else None,
            "market_regime": self.market_regime,
            "volatility_at_entry": round(self.volatility_at_entry, 5),
            "session": self.session
        }


class DurationPredictor:
    """Predicts market direction for specific time durations"""
    
    def __init__(self):
        self.price_history: Dict[str, deque] = {}
        self.prediction_history: Dict[str, deque] = {}
        self.accuracy_by_duration: Dict[int, Dict[str, float]] = {}
        self.max_history = 1000
        
        # Initialize accuracy tracking for each duration
        for d in PREDICTION_DURATIONS:
            self.accuracy_by_duration[d] = {
                "total": 0,
                "correct": 0,
                "accuracy": 0.5,
                "avg_pips": 0.0
            }
    
    def predict_durations(
        self,
        symbol: str,
        ohlcv_data: List[OHLCV],
        current_price: float,
        volatility: float = 0.0
    ) -> List[DurationPrediction]:
        """Generate predictions for all durations"""
        
        if not ohlcv_data or len(ohlcv_data) < 20:
            return []
        
        predictions = []
        
        # Calculate base features
        features = self._calculate_features(ohlcv_data)
        
        # Generate prediction for each duration
        for duration in PREDICTION_DURATIONS:
            pred = self._predict_single_duration(
                symbol, duration, features, current_price, volatility
            )
            predictions.append(pred)
        
        return predictions
    
    def _calculate_features(self, ohlcv_data: List[OHLCV]) -> Dict[str, float]:
        """Calculate features for prediction"""
        
        closes = np.array([c.close for c in ohlcv_data])
        highs = np.array([c.high for c in ohlcv_data])
        lows = np.array([c.low for c in ohlcv_data])
        volumes = np.array([c.volume for c in ohlcv_data])
        
        features = {}
        
        # Price features
        features['returns'] = np.diff(closes[-20:]) / closes[-21:-1]
        features['log_returns'] = np.log(closes[-20:] / closes[-21:-1])
        
        # Momentum features
        features['momentum_5'] = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        features['momentum_10'] = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0
        features['momentum_20'] = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0
        
        # Volatility features
        features['volatility_5'] = np.std(features['returns'][-5:]) if len(features['returns']) >= 5 else 0
        features['volatility_10'] = np.std(features['returns'][-10:]) if len(features['returns']) >= 10 else 0
        features['volatility_20'] = np.std(features['returns']) if len(features['returns']) > 0 else 0
        
        # Range features
        ranges = highs[-20:] - lows[-20:]
        features['avg_range'] = np.mean(ranges)
        features['range_ratio'] = (highs[-1] - lows[-1]) / features['avg_range'] if features['avg_range'] > 0 else 1
        
        # Volume features
        features['volume_ratio'] = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1
        features['volume_trend'] = (np.mean(volumes[-5:]) - np.mean(volumes[-15:-5])) / np.mean(volumes[-15:-5]) if np.mean(volumes[-15:-5]) > 0 else 0
        
        # Price position
        features['price_position'] = (closes[-1] - lows[-20:]) / (highs[-20:] - lows[-20:])
        features['price_position'] = np.mean(features['price_position'])
        
        # Trend features
        features['higher_highs'] = np.sum(highs[-10:] > highs[-11:-1]) / 9 if len(highs) >= 11 else 0.5
        features['lower_lows'] = np.sum(lows[-10:] < lows[-11:-1]) / 9 if len(lows) >= 11 else 0.5
        
        # Z-score
        features['zscore'] = (closes[-1] - np.mean(closes[-20:])) / np.std(closes[-20:]) if np.std(closes[-20:]) > 0 else 0
        
        # RSI approximation
        gains = np.maximum(features['returns'], 0)
        losses = np.abs(np.minimum(features['returns'], 0))
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        features['rsi'] = 100 - (100 / (1 + rs))
        
        return features
    
    def _predict_single_duration(
        self,
        symbol: str,
        duration_minutes: int,
        features: Dict[str, Any],
        current_price: float,
        volatility: float
    ) -> DurationPrediction:
        """Generate prediction for a single duration"""
        
        pred = DurationPrediction(duration_minutes=duration_minutes)
        pred.entry_price = current_price
        
        # Calculate direction probabilities based on duration
        prob_up, prob_down = self._calculate_direction_probability(
            features, duration_minutes, volatility
        )
        
        pred.probability_up = prob_up
        pred.probability_down = prob_down
        
        # Determine direction
        if prob_up > prob_down:
            pred.direction = SignalBias.BUY
            pred.confidence = prob_up
        elif prob_down > prob_up:
            pred.direction = SignalBias.SELL
            pred.confidence = prob_down
        else:
            pred.direction = SignalBias.NEUTRAL
            pred.confidence = 0.5
        
        # Calculate expected move in pips
        expected_move = self._calculate_expected_move(
            features, duration_minutes, volatility
        )
        pred.expected_move_pips = expected_move
        
        # Calculate target and stop loss
        pip_value = 0.0001 if "JPY" not in symbol else 0.01
        
        if pred.direction == SignalBias.BUY:
            pred.target_price = current_price + (expected_move * pip_value)
            pred.stop_loss_price = current_price - (expected_move * 0.5 * pip_value)  # 0.5:1 R:R minimum
        elif pred.direction == SignalBias.SELL:
            pred.target_price = current_price - (expected_move * pip_value)
            pred.stop_loss_price = current_price + (expected_move * 0.5 * pip_value)
        else:
            pred.target_price = current_price
            pred.stop_loss_price = current_price
        
        # Calculate risk-reward ratio
        if pred.expected_move_pips > 0:
            pred.risk_reward_ratio = pred.expected_move_pips / (pred.expected_move_pips * 0.5)
        else:
            pred.risk_reward_ratio = 2.0  # Default 2:1
        
        # Calculate probability of hitting target
        pred.probability_of_hit = self._calculate_hit_probability(
            pred.confidence, duration_minutes, features
        )
        
        # Calculate noise score (lower is better)
        pred.noise_score = self._calculate_noise_score(features, duration_minutes)
        
        # Calculate reliability based on historical accuracy
        pred.reliability = self.accuracy_by_duration[duration_minutes]["accuracy"]
        
        return pred
    
    def _calculate_direction_probability(
        self,
        features: Dict[str, Any],
        duration_minutes: int,
        volatility: float
    ) -> Tuple[float, float]:
        """Calculate probability of price going up vs down for duration"""
        
        scores = []
        
        # Duration-adjusted momentum weight
        momentum_weight = min(1.0, duration_minutes / 10.0)  # Longer durations = more momentum impact
        
        # 1. Momentum score
        momentum = features.get('momentum_10', 0)
        if momentum > 0.001:
            prob_momentum = 0.55 + min(0.25, abs(momentum) * 20)
        elif momentum < -0.001:
            prob_momentum = 0.45 - min(0.25, abs(momentum) * 20)
        else:
            prob_momentum = 0.5
        scores.append(('momentum', prob_momentum, 0.25 * momentum_weight))
        
        # 2. Trend score
        hh = features.get('higher_highs', 0.5)
        ll = features.get('lower_lows', 0.5)
        if hh > 0.6:
            prob_trend = 0.55 + (hh - 0.5) * 0.3
        elif ll > 0.6:
            prob_trend = 0.45 - (ll - 0.5) * 0.3
        else:
            prob_trend = 0.5
        scores.append(('trend', prob_trend, 0.2))
        
        # 3. RSI score
        rsi = features.get('rsi', 50)
        if rsi < 35:
            prob_rsi = 0.6 + (35 - rsi) * 0.01  # Oversold = bullish
        elif rsi > 65:
            prob_rsi = 0.4 - (rsi - 65) * 0.01  # Overbought = bearish
        elif rsi < 45:
            prob_rsi = 0.53
        elif rsi > 55:
            prob_rsi = 0.47
        else:
            prob_rsi = 0.5
        scores.append(('rsi', prob_rsi, 0.15))
        
        # 4. Z-score score (mean reversion)
        zscore = features.get('zscore', 0)
        if zscore < -1.5:
            prob_z = 0.6 + abs(zscore) * 0.05
        elif zscore > 1.5:
            prob_z = 0.4 - abs(zscore) * 0.05
        else:
            prob_z = 0.5
        scores.append(('zscore', prob_z, 0.15))
        
        # 5. Volume score
        volume_ratio = features.get('volume_ratio', 1)
        volume_trend = features.get('volume_trend', 0)
        if volume_ratio > 1.3 and volume_trend > 0:
            # High increasing volume confirms momentum
            prob_vol = 0.55 if features.get('momentum_10', 0) > 0 else 0.45
        else:
            prob_vol = 0.5
        scores.append(('volume', prob_vol, 0.1))
        
        # 6. Price position score
        price_pos = features.get('price_position', 0.5)
        if price_pos < 0.3:
            prob_pos = 0.55  # Near low = bullish
        elif price_pos > 0.7:
            prob_pos = 0.45  # Near high = bearish
        else:
            prob_pos = 0.5
        scores.append(('position', prob_pos, 0.15))
        
        # Calculate weighted probability
        total_weight = sum(s[2] for s in scores)
        prob_up = sum(s[1] * s[2] for s in scores) / total_weight if total_weight > 0 else 0.5
        
        # Adjust for duration
        # Shorter durations are more random, longer durations follow trends more
        if duration_minutes <= 2:
            # Very short term - more noise, pull towards 0.5
            prob_up = 0.5 + (prob_up - 0.5) * 0.5
        elif duration_minutes >= 30:
            # Longer term - follow the calculated probability
            pass
        else:
            # Medium term - moderate adjustment
            prob_up = 0.5 + (prob_up - 0.5) * (0.5 + duration_minutes / 60)
        
        # Clamp to reasonable range
        prob_up = max(0.25, min(0.85, prob_up))
        prob_down = 1.0 - prob_up
        
        return prob_up, prob_down
    
    def _calculate_expected_move(
        self,
        features: Dict[str, Any],
        duration_minutes: int,
        volatility: float
    ) -> float:
        """Calculate expected move in pips for duration"""
        
        # Base move from average range
        avg_range = features.get('avg_range', 0.0005)  # Default 5 pips
        
        # Convert to pips (assuming non-JPY pair)
        avg_range_pips = avg_range * 10000
        
        # Scale by duration (square root rule - volatility scales with sqrt(time))
        # Normalize to per-minute, then scale to duration
        base_move = avg_range_pips * np.sqrt(duration_minutes) / np.sqrt(15)  # 15min as base
        
        # Adjust for volatility
        vol_adjustment = 1 + (volatility - 0.01) * 5 if volatility > 0 else 1
        base_move *= vol_adjustment
        
        # Adjust for volume
        volume_ratio = features.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            base_move *= 1.2  # Higher volume = bigger moves
        elif volume_ratio < 0.7:
            base_move *= 0.8  # Lower volume = smaller moves
        
        # Round to reasonable pip values
        base_move = max(3, min(50, base_move))  # Between 3 and 50 pips
        
        return base_move
    
    def _calculate_hit_probability(
        self,
        confidence: float,
        duration_minutes: int,
        features: Dict[str, Any]
    ) -> float:
        """Calculate probability of hitting target"""
        
        # Base probability from confidence
        prob = confidence
        
        # Adjust for volatility
        vol = features.get('volatility_20', 0)
        if vol > 0.02:
            prob *= 0.9  # High volatility reduces hit rate
        elif vol < 0.005:
            prob *= 1.05  # Low volatility increases hit rate
        
        # Adjust for duration
        # Shorter durations are harder to predict
        if duration_minutes <= 2:
            prob *= 0.85
        elif duration_minutes <= 5:
            prob *= 0.9
        elif duration_minutes >= 30:
            prob *= 0.95
        
        return min(0.9, max(0.3, prob))
    
    def _calculate_noise_score(
        self,
        features: Dict[str, Any],
        duration_minutes: int
    ) -> float:
        """Calculate noise score (lower is better, means cleaner signal)"""
        
        noise = 0.0
        
        # Volatility noise
        vol = features.get('volatility_20', 0)
        noise += min(0.3, vol * 10)
        
        # Range expansion noise
        range_ratio = features.get('range_ratio', 1)
        if range_ratio > 1.5:
            noise += 0.1
        
        # Conflicting signals noise
        momentum = features.get('momentum_10', 0)
        rsi = features.get('rsi', 50)
        if momentum > 0 and rsi > 70:
            noise += 0.15  # Bullish momentum but overbought
        elif momentum < 0 and rsi < 30:
            noise += 0.15  # Bearish momentum but oversold
        
        # Duration noise - shorter = more noise
        if duration_minutes <= 2:
            noise += 0.2
        elif duration_minutes <= 5:
            noise += 0.1
        
        # Volume noise
        volume_ratio = features.get('volume_ratio', 1)
        if volume_ratio < 0.5:
            noise += 0.1  # Low volume = unreliable
        
        return min(1.0, noise)
    
    def generate_signal(
        self,
        symbol: str,
        ohlcv_data: List[OHLCV],
        current_price: float,
        volatility: float = 0.0,
        session: str = "unknown",
        min_confidence: float = 0.55,
        max_noise: float = 0.4
    ) -> Optional[DurationSignal]:
        """Generate a duration-based signal for the best prediction"""
        
        predictions = self.predict_durations(symbol, ohlcv_data, current_price, volatility)
        
        if not predictions:
            return None
        
        # Filter predictions by confidence and noise
        valid_predictions = [
            p for p in predictions
            if p.confidence >= min_confidence 
            and p.noise_score <= max_noise
            and p.direction != SignalBias.NEUTRAL
        ]
        
        if not valid_predictions:
            return None
        
        # Select best prediction (highest confidence with acceptable noise)
        best_pred = max(valid_predictions, key=lambda p: p.confidence * (1 - p.noise_score))
        
        # Generate signal
        import uuid
        signal = DurationSignal(
            signal_id=f"SIG-{symbol}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}",
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            duration_minutes=best_pred.duration_minutes,
            direction=best_pred.direction,
            confidence=best_pred.confidence,
            entry_price=best_pred.entry_price,
            target_price=best_pred.target_price,
            stop_loss_price=best_pred.stop_loss_price,
            expected_move_pips=best_pred.expected_move_pips,
            risk_reward_ratio=best_pred.risk_reward_ratio,
            market_regime=self._determine_regime(ohlcv_data),
            volatility_at_entry=volatility,
            session=session
        )
        
        return signal
    
    def _determine_regime(self, ohlcv_data: List[OHLCV]) -> str:
        """Determine current market regime"""
        if len(ohlcv_data) < 20:
            return "unknown"
        
        closes = np.array([c.close for c in ohlcv_data])
        returns = np.diff(closes[-20:]) / closes[-21:-1]
        
        volatility = np.std(returns)
        trend = (closes[-1] - closes[-20]) / closes[-20]
        
        if volatility > 0.015:
            return "high_volatility"
        elif volatility < 0.005:
            return "low_volatility"
        elif trend > 0.005:
            return "uptrend"
        elif trend < -0.005:
            return "downtrend"
        else:
            return "ranging"
    
    def update_accuracy(
        self,
        duration_minutes: int,
        was_correct: bool,
        actual_pips: float
    ):
        """Update accuracy tracking for a duration"""
        if duration_minutes not in self.accuracy_by_duration:
            return
        
        stats = self.accuracy_by_duration[duration_minutes]
        stats["total"] += 1
        if was_correct:
            stats["correct"] += 1
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.5
        
        # Update average pips
        if actual_pips is not None:
            stats["avg_pips"] = (
                (stats["avg_pips"] * (stats["total"] - 1) + abs(actual_pips)) 
                / stats["total"]
            )


# Global instance
duration_predictor = DurationPredictor()
