"""
Forex Probability Intelligence System - Feature Engine
Price Action + Time + Statistical Features
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from app.models.schemas import (
    OHLCV, TickData, FeatureSet, PriceActionFeatures, 
    TimeFeatures, StatisticalFeatures, MarketSession
)
from config.settings import get_settings, MARKET_SESSIONS

settings = get_settings()


class FeatureEngine:
    """Calculate trading features from market data"""
    
    def __init__(self):
        self.lookback_periods = settings.feature_window_sizes
        
    def calculate_features(
        self, 
        symbol: str,
        ohlcv: List[OHLCV],
        recent_ticks: Optional[List[TickData]] = None
    ) -> FeatureSet:
        """Calculate complete feature set from OHLCV data"""
        
        if not ohlcv:
            return self._empty_features(symbol)
        
        # Convert to DataFrame for easier calculations
        df = pd.DataFrame([{
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume,
            'timestamp': c.timestamp
        } for c in ohlcv])
        
        # Calculate feature groups
        price_action = self._calculate_price_action_features(df, symbol)
        time_features = self._calculate_time_features(symbol)
        statistical = self._calculate_statistical_features(df)
        raw_features = self._calculate_raw_features(df)
        
        return FeatureSet(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            price_action=price_action,
            time_features=time_features,
            statistical=statistical,
            raw_features=raw_features
        )
    
    def _empty_features(self, symbol: str) -> FeatureSet:
        """Return empty features when no data available"""
        return FeatureSet(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            price_action=PriceActionFeatures(),
            time_features=TimeFeatures(
                current_session=MarketSession.LONDON,
                hour_of_day=datetime.utcnow().hour,
                day_of_week=datetime.utcnow().weekday()
            ),
            statistical=StatisticalFeatures(),
            raw_features={}
        )
    
    def _calculate_price_action_features(self, df: pd.DataFrame, symbol: str) -> PriceActionFeatures:
        """Calculate price action features"""
        features = PriceActionFeatures()
        
        if len(df) < 20:
            return features
        
        # Break of Structure (BOS)
        bos = self._detect_break_of_structure(df)
        features.break_of_structure = bos['detected']
        features.bos_direction = bos['direction']
        features.bos_strength = bos['strength']
        
        # Wick Rejection
        wick = self._detect_wick_rejection(df)
        features.wick_rejection_strength = wick['strength']
        features.wick_rejection_type = wick['type']
        
        # Range Expansion
        expansion = self._calculate_range_expansion(df)
        features.range_expansion = expansion['value']
        features.range_expansion_pct = expansion['pct']
        
        # Swing points
        swing = self._find_swing_points(df)
        features.higher_highs = swing['higher_highs']
        features.lower_lows = swing['lower_lows']
        features.swing_high = swing['swing_high']
        features.swing_low = swing['swing_low']
        
        return features
    
    def _detect_break_of_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect break of structure patterns"""
        if len(df) < 20:
            return {'detected': False, 'direction': None, 'strength': 0.0}
        
        # Get recent highs and lows
        lookback = 20
        highs = df['high'].tail(lookback).values
        lows = df['low'].tail(lookback).values
        closes = df['close'].tail(lookback).values
        
        # Find highest high and lowest low in lookback
        highest_high = max(highs[:-1])
        lowest_low = min(lows[:-1])
        current_close = closes[-1]
        
        # Bullish BOS: Close above previous highest high
        bullish_bos = current_close > highest_high
        # Bearish BOS: Close below previous lowest low
        bearish_bos = current_close < lowest_low
        
        if bullish_bos:
            strength = (current_close - highest_high) / highest_high * 100
            return {
                'detected': True,
                'direction': 'BUY',
                'strength': min(strength, 10.0)  # Cap at 10%
            }
        elif bearish_bos:
            strength = (lowest_low - current_close) / lowest_low * 100
            return {
                'detected': True,
                'direction': 'SELL',
                'strength': min(strength, 10.0)
            }
        
        return {'detected': False, 'direction': None, 'strength': 0.0}
    
    def _detect_wick_rejection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect wick rejection patterns (pin bars)"""
        if len(df) < 5:
            return {'strength': 0.0, 'type': None}
        
        # Analyze last candle
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        total_range = last['high'] - last['low']
        
        if total_range == 0:
            return {'strength': 0.0, 'type': None}
        
        # Upper wick and lower wick
        upper_wick = last['high'] - max(last['open'], last['close'])
        lower_wick = min(last['open'], last['close']) - last['low']
        
        # Wick ratio (how much of the candle is wick)
        wick_ratio = (upper_wick + lower_wick) / total_range
        
        # Bullish rejection: Long lower wick, small upper wick
        bullish_rejection = lower_wick > upper_wick * 2 and lower_wick > body * 2
        # Bearish rejection: Long upper wick, small lower wick
        bearish_rejection = upper_wick > lower_wick * 2 and upper_wick > body * 2
        
        if bullish_rejection:
            return {
                'strength': min(wick_ratio, 1.0),
                'type': 'bullish'
            }
        elif bearish_rejection:
            return {
                'strength': min(wick_ratio, 1.0),
                'type': 'bearish'
            }
        
        return {'strength': 0.0, 'type': None}
    
    def _calculate_range_expansion(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate range expansion"""
        if len(df) < 20:
            return {'value': 0.0, 'pct': 0.0}
        
        # Current range vs average range
        current_range = df['high'].iloc[-1] - df['low'].iloc[-1]
        avg_range = (df['high'].tail(20).values - df['low'].tail(20).values).mean()
        
        if avg_range == 0:
            return {'value': 0.0, 'pct': 0.0}
        
        expansion = (current_range - avg_range) / avg_range
        
        return {
            'value': current_range - avg_range,
            'pct': expansion * 100
        }
    
    def _find_swing_points(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find swing highs and lows"""
        if len(df) < 10:
            return {'higher_highs': 0, 'lower_lows': 0, 'swing_high': None, 'swing_low': None}
        
        highs = df['high'].tail(20).values
        lows = df['low'].tail(20).values
        
        # Count higher highs and lower lows
        higher_highs = 0
        lower_lows = 0
        
        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                higher_highs += 1
            if lows[i] < lows[i-1]:
                lower_lows += 1
        
        # Find swing high/low (local extrema)
        swing_high = max(highs[-5:])
        swing_low = min(lows[-5:])
        
        return {
            'higher_highs': higher_highs,
            'lower_lows': lower_lows,
            'swing_high': float(swing_high),
            'swing_low': float(swing_low)
        }
    
    def _calculate_time_features(self, symbol: str) -> TimeFeatures:
        """Calculate time-based features"""
        now = datetime.utcnow()
        hour = now.hour
        
        # Determine current session
        current_session = self._get_session(hour)
        
        # Find overlapping sessions
        overlaps = self._get_session_overlaps(hour)
        
        # Time since market events (simulated)
        time_since_breakout = np.random.uniform(5, 60) if np.random.random() > 0.3 else None
        time_since_liquidity = np.random.uniform(10, 120) if np.random.random() > 0.5 else None
        
        # Session progress
        session_info = MARKET_SESSIONS.get(current_session.value, {})
        session_start = session_info.get('start', 0)
        session_end = session_info.get('end', 24)
        
        if session_end < session_start:  # Session crosses midnight
            if hour >= session_start:
                session_progress = (hour - session_start) / (24 - session_start + session_end)
            else:
                session_progress = (24 - session_start + hour) / (24 - session_start + session_end)
        else:
            if session_start <= hour < session_end:
                session_progress = (hour - session_start) / (session_end - session_start)
            else:
                session_progress = 0.0
        
        return TimeFeatures(
            current_session=current_session,
            session_overlap=overlaps,
            time_since_breakout_minutes=time_since_breakout,
            time_since_liquidity_event_minutes=time_since_liquidity,
            hour_of_day=hour,
            day_of_week=now.weekday(),
            is_weekend=now.weekday() >= 5,
            session_progress=min(session_progress, 1.0)
        )
    
    def _get_session(self, hour: int) -> MarketSession:
        """Determine primary trading session"""
        if 8 <= hour < 17:
            return MarketSession.LONDON
        elif 13 <= hour < 22:
            return MarketSession.NEW_YORK
        elif 0 <= hour < 9:
            return MarketSession.TOKYO
        elif 22 <= hour or hour < 7:
            return MarketSession.SYDNEY
        return MarketSession.LONDON
    
    def _get_session_overlaps(self, hour: int) -> List[MarketSession]:
        """Get overlapping trading sessions"""
        overlaps = []
        
        if 8 <= hour < 9:
            overlaps = [MarketSession.TOKYO, MarketSession.LONDON]
        elif 13 <= hour < 17:
            overlaps = [MarketSession.LONDON, MarketSession.NEW_YORK]
        
        return overlaps
    
    def _calculate_statistical_features(self, df: pd.DataFrame) -> StatisticalFeatures:
        """Calculate statistical features"""
        features = StatisticalFeatures()
        
        if len(df) < 50:
            return features
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # Rolling volatility
        returns = np.diff(np.log(closes))
        
        features.rolling_volatility_10 = float(np.std(returns[-10:]) * np.sqrt(252) if len(returns) >= 10 else 0)
        features.rolling_volatility_20 = float(np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0)
        features.rolling_volatility_50 = float(np.std(returns[-50:]) * np.sqrt(252) if len(returns) >= 50 else 0)
        
        # Momentum decay
        for period in [5, 10, 20]:
            if len(closes) >= period + 1:
                momentum = (closes[-1] - closes[-period-1]) / closes[-period-1]
                decay = momentum - (closes[-period] - closes[-period*2]) / closes[-period*2] if len(closes) >= period * 2 else 0
                
                if period == 5:
                    features.momentum_decay_5 = float(decay)
                elif period == 10:
                    features.momentum_decay_10 = float(decay)
                elif period == 20:
                    features.momentum_decay_20 = float(decay)
        
        # Mean reversion strength (z-score)
        window = 50
        if len(closes) >= window:
            sma = np.mean(closes[-window:])
            std = np.std(closes[-window:])
            features.zscore = float((closes[-1] - sma) / std) if std > 0 else 0
            features.mean_reversion_strength = float(abs(features.zscore))
        
        # RSI
        features.rsi = float(self._calculate_rsi(closes, 14))
        
        # ATR
        atr = self._calculate_atr(highs, lows, closes, 14)
        features.atr = float(atr)
        features.atr_pct = float(atr / closes[-1] * 100) if closes[-1] > 0 else 0
        
        return features
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return 0.0
        
        tr_values = []
        for i in range(1, min(period + 1, len(highs))):
            idx = -(i)
            tr = max(
                highs[idx] - lows[idx],
                abs(highs[idx] - closes[idx-1]),
                abs(lows[idx] - closes[idx-1])
            )
            tr_values.append(tr)
        
        return np.mean(tr_values) if tr_values else 0.0
    
    def _calculate_raw_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate raw numerical features for ML model"""
        raw = {}
        
        if len(df) < 20:
            return raw
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values if 'volume' in df.columns else None
        
        # Price features
        raw['close'] = float(closes[-1])
        raw['close_pct_change_1'] = float((closes[-1] - closes[-2]) / closes[-2]) if len(closes) >= 2 else 0
        raw['close_pct_change_5'] = float((closes[-1] - closes[-5]) / closes[-5]) if len(closes) >= 5 else 0
        raw['close_pct_change_10'] = float((closes[-1] - closes[-10]) / closes[-10]) if len(closes) >= 10 else 0
        raw['close_pct_change_20'] = float((closes[-1] - closes[-20]) / closes[-20]) if len(closes) >= 20 else 0
        
        # Moving averages
        raw['sma_5'] = float(np.mean(closes[-5:]))
        raw['sma_10'] = float(np.mean(closes[-10:]))
        raw['sma_20'] = float(np.mean(closes[-20:]))
        raw['ema_10'] = float(self._calculate_ema(closes, 10))
        
        # Price relative to MA
        raw['price_to_sma20'] = float(closes[-1] / raw['sma_20'] - 1) if raw['sma_20'] > 0 else 0
        
        # Range features
        raw['high_low_range'] = float(highs[-1] - lows[-1])
        raw['avg_range_10'] = float(np.mean([highs[-i] - lows[-i] for i in range(1, 11)]))
        raw['range_ratio'] = float(raw['high_low_range'] / raw['avg_range_10']) if raw['avg_range_10'] > 0 else 1
        
        # Candle features
        body = abs(closes[-1] - df['open'].iloc[-1])
        raw['candle_body'] = float(body)
        raw['body_to_range'] = float(body / raw['high_low_range']) if raw['high_low_range'] > 0 else 0
        
        # Volume features
        if volumes is not None and len(volumes) >= 10:
            raw['volume'] = float(volumes[-1])
            raw['volume_ma_10'] = float(np.mean(volumes[-10:]))
            raw['volume_ratio'] = float(volumes[-1] / raw['volume_ma_10']) if raw['volume_ma_10'] > 0 else 1
        
        return raw
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return float(np.mean(prices))
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return float(ema)
    
    def get_feature_vector(self, features: FeatureSet) -> np.ndarray:
        """Convert features to numpy array for ML model"""
        vector = []
        
        # Price action features
        vector.extend([
            float(features.price_action.break_of_structure),
            features.price_action.bos_strength,
            features.price_action.wick_rejection_strength,
            features.price_action.range_expansion_pct,
            features.price_action.higher_highs / 20.0,  # Normalize
            features.price_action.lower_lows / 20.0,
        ])
        
        # Time features
        vector.extend([
            features.time_features.hour_of_day / 24.0,  # Normalize
            features.time_features.day_of_week / 6.0,
            float(features.time_features.is_weekend),
            features.time_features.session_progress,
        ])
        
        # Statistical features
        vector.extend([
            features.statistical.rolling_volatility_20,
            features.statistical.momentum_decay_10,
            features.statistical.mean_reversion_strength,
            features.statistical.zscore,
            features.statistical.rsi / 100.0,  # Normalize
            features.statistical.atr_pct,
        ])
        
        # Raw features
        for key in ['close_pct_change_1', 'close_pct_change_5', 'price_to_sma20', 'range_ratio', 'body_to_range']:
            vector.append(features.raw_features.get(key, 0.0))
        
        return np.array(vector, dtype=np.float32)


# Global instance
feature_engine = FeatureEngine()
