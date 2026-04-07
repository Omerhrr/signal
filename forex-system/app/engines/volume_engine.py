"""
Forex Probability Intelligence System - Volume Analysis Engine
Analyze MT5 tick volume data for trading signals and predictions
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from loguru import logger
from collections import deque
import scipy.stats as stats

from app.models.schemas import OHLCV, SignalBias


@dataclass
class VolumeProfile:
    """Volume profile for a price level"""
    price_level: float
    volume: float
    percentage: float = 0.0


@dataclass
class VolumeAnalysisResult:
    """Result of volume analysis"""
    # Current volume metrics
    current_volume: float = 0
    avg_volume: float = 0
    volume_ratio: float = 1.0  # current / avg
    
    # Volume trend
    volume_trend: str = "neutral"  # increasing, decreasing, neutral
    volume_momentum: float = 0.0
    
    # Volume zones
    high_volume_nodes: List[Dict] = field(default_factory=list)
    low_volume_nodes: List[Dict] = field(default_factory=list)
    poc_price: float = 0.0  # Point of Control
    
    # Volume signals
    volume_spread_analysis: str = "neutral"
    accumulation_detected: bool = False
    distribution_detected: bool = False
    volume_climax: bool = False
    volume_divergence: bool = False
    
    # Predictions
    volume_forecast: float = 0.0
    volume_confidence: float = 0.5
    
    # Trading implications
    bullish_volume: bool = False
    bearish_volume: bool = False
    signal_strength: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_volume": self.current_volume,
            "avg_volume": self.avg_volume,
            "volume_ratio": self.volume_ratio,
            "volume_trend": self.volume_trend,
            "volume_momentum": self.volume_momentum,
            "high_volume_nodes": self.high_volume_nodes[:5],
            "low_volume_nodes": self.low_volume_nodes[:5],
            "poc_price": self.poc_price,
            "volume_spread_analysis": self.volume_spread_analysis,
            "accumulation_detected": self.accumulation_detected,
            "distribution_detected": self.distribution_detected,
            "volume_climax": self.volume_climax,
            "volume_divergence": self.volume_divergence,
            "volume_forecast": self.volume_forecast,
            "volume_confidence": self.volume_confidence,
            "bullish_volume": self.bullish_volume,
            "bearish_volume": self.bearish_volume,
            "signal_strength": self.signal_strength
        }


@dataclass
class VolumePrediction:
    """Volume-based prediction"""
    direction_bias: SignalBias = SignalBias.NEUTRAL
    confidence: float = 0.5
    expected_volume_change: float = 0.0
    breakout_probability: float = 0.0
    reversal_probability: float = 0.0
    volume_support: float = 0.0
    volume_resistance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction_bias": self.direction_bias.value,
            "confidence": self.confidence,
            "expected_volume_change": self.expected_volume_change,
            "breakout_probability": self.breakout_probability,
            "reversal_probability": self.reversal_probability,
            "volume_support": self.volume_support,
            "volume_resistance": self.volume_resistance
        }


class VolumeEngine:
    """Engine for analyzing MT5 tick volume data"""
    
    def __init__(self):
        self.volume_history: Dict[str, deque] = {}
        self.volume_profiles: Dict[str, Dict[float, float]] = {}
        self.max_history = 500
        
    def analyze(self, symbol: str, ohlcv_data: List[OHLCV]) -> VolumeAnalysisResult:
        """Perform comprehensive volume analysis"""
        result = VolumeAnalysisResult()
        
        if not ohlcv_data or len(ohlcv_data) < 10:
            return result
            
        # Extract volume data
        volumes = np.array([c.volume for c in ohlcv_data])
        closes = np.array([c.close for c in ohlcv_data])
        highs = np.array([c.high for c in ohlcv_data])
        lows = np.array([c.low for c in ohlcv_data])
        opens = np.array([c.open for c in ohlcv_data])
        
        # Basic volume metrics
        result.current_volume = volumes[-1]
        result.avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        result.volume_ratio = result.current_volume / result.avg_volume if result.avg_volume > 0 else 1.0
        
        # Volume trend
        result.volume_trend = self._analyze_volume_trend(volumes)
        result.volume_momentum = self._calculate_volume_momentum(volumes)
        
        # Volume Profile Analysis
        profile = self._build_volume_profile(closes, volumes, highs, lows)
        result.high_volume_nodes = profile['hvn']
        result.low_volume_nodes = profile['lvn']
        result.poc_price = profile['poc']
        
        # Volume Spread Analysis (VSA)
        result.volume_spread_analysis = self._analyze_volume_spread(
            opens, highs, lows, closes, volumes
        )
        
        # Accumulation/Distribution Detection
        result.accumulation_detected = self._detect_accumulation(closes, volumes)
        result.distribution_detected = self._detect_distribution(closes, volumes)
        
        # Volume Climax Detection
        result.volume_climax = self._detect_volume_climax(volumes, closes)
        
        # Volume Divergence
        result.volume_divergence = self._detect_volume_divergence(closes, volumes)
        
        # Volume Forecast
        result.volume_forecast = self._forecast_volume(volumes)
        result.volume_confidence = self._calculate_volume_confidence(volumes)
        
        # Trading signals
        result.bullish_volume = self._is_bullish_volume(closes, volumes)
        result.bearish_volume = self._is_bearish_volume(closes, volumes)
        result.signal_strength = self._calculate_signal_strength(result)
        
        # Store history for tracking
        self._update_history(symbol, volumes[-1], closes[-1])
        
        return result
    
    def _analyze_volume_trend(self, volumes: np.ndarray) -> str:
        """Analyze if volume is increasing or decreasing"""
        if len(volumes) < 10:
            return "neutral"
        
        recent_avg = np.mean(volumes[-5:])
        older_avg = np.mean(volumes[-10:-5])
        
        change_pct = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        
        if change_pct > 0.15:
            return "increasing"
        elif change_pct < -0.15:
            return "decreasing"
        return "neutral"
    
    def _calculate_volume_momentum(self, volumes: np.ndarray) -> float:
        """Calculate volume momentum indicator"""
        if len(volumes) < 10:
            return 0.0
        
        # Rate of change of volume
        vol_roc = np.diff(volumes[-10:]) / volumes[-11:-1]
        return float(np.mean(vol_roc) * 100)
    
    def _build_volume_profile(
        self, 
        closes: np.ndarray, 
        volumes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> Dict[str, Any]:
        """Build volume profile for price levels"""
        # Create price buckets
        price_range = highs.max() - lows.min()
        if price_range == 0:
            return {'hvn': [], 'lvn': [], 'poc': closes[-1]}
        
        num_buckets = 20
        bucket_size = price_range / num_buckets
        
        volume_at_price = {}
        
        for i in range(len(closes)):
            # Distribute volume across the candle range
            candle_low = lows[i]
            candle_high = highs[i]
            candle_volume = volumes[i]
            
            # Simple approximation: assign volume to close price
            price_bucket = round(closes[i] / bucket_size) * bucket_size
            volume_at_price[price_bucket] = volume_at_price.get(price_bucket, 0) + candle_volume
        
        if not volume_at_price:
            return {'hvn': [], 'lvn': [], 'poc': closes[-1]}
        
        # Find POC (Point of Control)
        poc_price = max(volume_at_price, key=volume_at_price.get)
        max_volume = max(volume_at_price.values())
        avg_volume = np.mean(list(volume_at_price.values()))
        
        # Identify HVN and LVN
        hvn = []
        lvn = []
        
        for price, vol in volume_at_price.items():
            node = {
                "price": round(price, 5),
                "volume": int(vol),
                "percentage": round(vol / max_volume * 100, 1)
            }
            
            if vol > avg_volume * 1.5:
                hvn.append(node)
            elif vol < avg_volume * 0.5:
                lvn.append(node)
        
        # Sort by volume
        hvn.sort(key=lambda x: x['volume'], reverse=True)
        lvn.sort(key=lambda x: x['volume'])
        
        return {
            'hvn': hvn[:5],
            'lvn': lvn[:5],
            'poc': round(poc_price, 5)
        }
    
    def _analyze_volume_spread(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> str:
        """Volume Spread Analysis (VSA)"""
        if len(volumes) < 3:
            return "neutral"
        
        spreads = highs - lows
        avg_spread = np.mean(spreads[-10:])
        avg_volume = np.mean(volumes[-10:])
        
        current_spread = spreads[-1]
        current_volume = volumes[-1]
        
        # Candle type
        is_bullish = closes[-1] > opens[-1]
        
        # VSA Patterns
        if current_volume > avg_volume * 1.5:
            if current_spread > avg_spread * 1.3:
                if is_bullish:
                    return "buying_climax" if closes[-1] > highs[-2] else "strength"
                else:
                    return "selling_climax" if closes[-1] < lows[-2] else "weakness"
            else:
                return "stopping_volume" if not is_bullish else "buying_pressure"
        elif current_volume < avg_volume * 0.5:
            if current_spread < avg_spread * 0.5:
                return "no_demand" if not is_bullish else "no_supply"
        
        return "neutral"
    
    def _detect_accumulation(self, closes: np.ndarray, volumes: np.ndarray) -> bool:
        """Detect accumulation pattern"""
        if len(volumes) < 10:
            return False
        
        # Look for: decreasing volume on down moves, increasing volume on up moves
        price_changes = np.diff(closes[-10:])
        volume_changes = volumes[-10:]
        
        down_moves = price_changes < 0
        up_moves = price_changes > 0
        
        if np.sum(down_moves) < 2 or np.sum(up_moves) < 2:
            return False
        
        avg_down_vol = np.mean(volume_changes[1:][down_moves])
        avg_up_vol = np.mean(volume_changes[1:][up_moves])
        
        # Accumulation: higher volume on up moves
        return avg_up_vol > avg_down_vol * 1.3
    
    def _detect_distribution(self, closes: np.ndarray, volumes: np.ndarray) -> bool:
        """Detect distribution pattern"""
        if len(volumes) < 10:
            return False
        
        price_changes = np.diff(closes[-10:])
        volume_changes = volumes[-10:]
        
        down_moves = price_changes < 0
        up_moves = price_changes > 0
        
        if np.sum(down_moves) < 2 or np.sum(up_moves) < 2:
            return False
        
        avg_down_vol = np.mean(volume_changes[1:][down_moves])
        avg_up_vol = np.mean(volume_changes[1:][up_moves])
        
        # Distribution: higher volume on down moves
        return avg_down_vol > avg_up_vol * 1.3
    
    def _detect_volume_climax(self, volumes: np.ndarray, closes: np.ndarray) -> bool:
        """Detect volume climax (exhaustion)"""
        if len(volumes) < 20:
            return False
        
        # Volume climax: extreme volume after a trend
        max_vol = np.max(volumes[-20:])
        current_vol = volumes[-1]
        
        is_extreme = current_vol >= max_vol * 0.95
        
        # Check for trend
        price_change = (closes[-1] - closes[-10]) / closes[-10]
        has_trend = abs(price_change) > 0.002
        
        return is_extreme and has_trend
    
    def _detect_volume_divergence(self, closes: np.ndarray, volumes: np.ndarray) -> bool:
        """Detect volume divergence"""
        if len(volumes) < 10:
            return False
        
        # Price making new highs/lows but volume decreasing
        price_trend = closes[-1] - closes[-5]
        volume_trend = np.mean(volumes[-5:]) - np.mean(volumes[-10:-5])
        
        # Bullish divergence: price down, volume increasing
        # Bearish divergence: price up, volume decreasing
        return (price_trend > 0 and volume_trend < 0) or (price_trend < 0 and volume_trend > 0)
    
    def _forecast_volume(self, volumes: np.ndarray) -> float:
        """Forecast next period volume using simple methods"""
        if len(volumes) < 10:
            return volumes[-1] if len(volumes) > 0 else 0
        
        # Use exponential weighted average
        weights = np.exp(np.linspace(-1, 0, len(volumes[-10:])))
        weights = weights / weights.sum()
        forecast = np.sum(volumes[-10:] * weights)
        
        return float(forecast)
    
    def _calculate_volume_confidence(self, volumes: np.ndarray) -> float:
        """Calculate confidence based on volume consistency"""
        if len(volumes) < 5:
            return 0.5
        
        # Lower volatility in volume = higher confidence
        vol_cv = np.std(volumes[-10:]) / np.mean(volumes[-10:]) if np.mean(volumes[-10:]) > 0 else 1
        
        # Convert to confidence (lower CV = higher confidence)
        confidence = max(0.2, min(0.9, 1 - vol_cv))
        
        return float(confidence)
    
    def _is_bullish_volume(self, closes: np.ndarray, volumes: np.ndarray) -> bool:
        """Determine if volume pattern is bullish"""
        if len(volumes) < 5:
            return False
        
        # Bullish: higher volume on up candles
        up_candles = closes[1:] > closes[:-1]
        if np.sum(up_candles[-5:]) == 0:
            return False
        
        up_vol = np.mean(volumes[1:][-5:][up_candles[-5:]])
        total_vol = np.mean(volumes[-5:])
        
        return up_vol > total_vol * 1.2
    
    def _is_bearish_volume(self, closes: np.ndarray, volumes: np.ndarray) -> bool:
        """Determine if volume pattern is bearish"""
        if len(volumes) < 5:
            return False
        
        # Bearish: higher volume on down candles
        down_candles = closes[1:] < closes[:-1]
        if np.sum(down_candles[-5:]) == 0:
            return False
        
        down_vol = np.mean(volumes[1:][-5:][down_candles[-5:]])
        total_vol = np.mean(volumes[-5:])
        
        return down_vol > total_vol * 1.2
    
    def _calculate_signal_strength(self, result: VolumeAnalysisResult) -> float:
        """Calculate overall volume signal strength"""
        strength = 0.0
        
        # Volume ratio contribution
        if result.volume_ratio > 1.5:
            strength += 0.2
        elif result.volume_ratio > 1.2:
            strength += 0.1
        
        # Volume trend contribution
        if result.volume_trend == "increasing":
            strength += 0.15
        
        # VSA pattern contribution
        if result.volume_spread_analysis in ["strength", "buying_pressure"]:
            strength += 0.2
        elif result.volume_spread_analysis in ["weakness", "selling_climax"]:
            strength += 0.15
        
        # Accumulation/Distribution
        if result.accumulation_detected:
            strength += 0.15
        if result.distribution_detected:
            strength += 0.1
        
        # Climax
        if result.volume_climax:
            strength += 0.1
        
        return min(1.0, strength)
    
    def _update_history(self, symbol: str, volume: float, price: float):
        """Update volume history for tracking"""
        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque(maxlen=self.max_history)
        
        self.volume_history[symbol].append({
            "volume": volume,
            "price": price,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def predict(self, ohlcv_data: List[OHLCV]) -> VolumePrediction:
        """Generate volume-based prediction"""
        prediction = VolumePrediction()
        
        if not ohlcv_data or len(ohlcv_data) < 20:
            return prediction
        
        analysis = self.analyze("symbol", ohlcv_data)
        
        # Determine direction bias
        if analysis.bullish_volume and not analysis.bearish_volume:
            prediction.direction_bias = SignalBias.BUY
            prediction.confidence = min(0.85, 0.5 + analysis.signal_strength * 0.5)
        elif analysis.bearish_volume and not analysis.bullish_volume:
            prediction.direction_bias = SignalBias.SELL
            prediction.confidence = min(0.85, 0.5 + analysis.signal_strength * 0.5)
        else:
            prediction.direction_bias = SignalBias.NEUTRAL
            prediction.confidence = 0.5
        
        # Volume forecast
        volumes = np.array([c.volume for c in ohlcv_data])
        prediction.expected_volume_change = (analysis.volume_forecast - volumes[-1]) / volumes[-1] * 100 if volumes[-1] > 0 else 0
        
        # Breakout probability
        if analysis.volume_ratio > 1.3 and analysis.volume_trend == "increasing":
            prediction.breakout_probability = min(0.8, 0.4 + analysis.signal_strength * 0.4)
        else:
            prediction.breakout_probability = 0.3
        
        # Reversal probability
        if analysis.volume_climax or analysis.volume_divergence:
            prediction.reversal_probability = 0.6
        else:
            prediction.reversal_probability = 0.3
        
        # Volume support/resistance
        if analysis.high_volume_nodes:
            prediction.volume_support = analysis.high_volume_nodes[0]['price']
        if analysis.low_volume_nodes:
            prediction.volume_resistance = analysis.low_volume_nodes[0]['price']
        
        return prediction


# Global instance
volume_engine = VolumeEngine()
