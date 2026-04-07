"""
Forex Probability Intelligence System - Risk & Failure Engine
Detect and handle failures dynamically, manage risk controls
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger

from app.models.schemas import (
    TradingSignal, FeatureSet, PredictionOutput, FailureDetection, 
    RiskAssessment, RiskLevel, SignalStatus, SignalAction, SignalBias,
    TickData, MarketState
)
from config.settings import get_settings

settings = get_settings()


class RiskFailureEngine:
    """Engine for detecting failures and managing risk"""
    
    def __init__(self):
        self.failure_history: List[FailureDetection] = []
        self.risk_settings = {
            'max_risk_per_trade': settings.max_risk_per_trade,
            'min_confidence': settings.min_confidence_threshold,
            'max_spread_pips': settings.max_spread_pips,
            'instant_failure_pips': settings.instant_failure_pips,
            'slow_failure_minutes': settings.slow_failure_minutes,
            'fake_breakout_threshold': settings.fake_breakout_threshold,
            'volatility_spike_threshold': settings.volatility_spike_threshold
        }
    
    def detect_failure(
        self,
        signal: TradingSignal,
        current_tick: TickData,
        market_state: MarketState,
        time_elapsed_minutes: float,
        price_movement_pips: float
    ) -> FailureDetection:
        """Detect various failure conditions"""
        
        # 1. Instant Failure Detection
        instant_failure = self._detect_instant_failure(signal, price_movement_pips)
        if instant_failure.detected:
            self._log_failure(instant_failure)
            return instant_failure
        
        # 2. Slow Failure Detection (time exceeds expected)
        slow_failure = self._detect_slow_failure(signal, time_elapsed_minutes)
        if slow_failure.detected:
            self._log_failure(slow_failure)
            return slow_failure
        
        # 3. Fake Breakout Detection
        fake_breakout = self._detect_fake_breakout(signal, price_movement_pips, market_state)
        if fake_breakout.detected:
            self._log_failure(fake_breakout)
            return fake_breakout
        
        # 4. High Volatility Detection
        high_volatility = self._detect_high_volatility(signal, market_state, current_tick)
        if high_volatility.detected:
            self._log_failure(high_volatility)
            return high_volatility
        
        # 5. Low Confidence Detection
        low_confidence = self._detect_low_confidence(signal)
        if low_confidence.detected:
            self._log_failure(low_confidence)
            return low_confidence
        
        # No failure detected
        return FailureDetection(
            failure_type=None,
            detected=False,
            severity="none",
            action_required=SignalAction.TRADE,
            message="Signal is valid for trading",
            timestamp=datetime.utcnow()
        )
    
    def _detect_instant_failure(self, signal: TradingSignal, price_movement_pips: float) -> FailureDetection:
        """Detect instant failure - price moves against trade quickly"""
        
        # Determine adverse movement based on signal direction
        if signal.bias == SignalBias.BUY:
            adverse_movement = -price_movement_pips  # Negative = price went down
        else:
            adverse_movement = price_movement_pips  # Positive = price went up
        
        # Check if adverse movement exceeds threshold
        if adverse_movement < -self.risk_settings['instant_failure_pips']:
            return FailureDetection(
                failure_type="instant",
                detected=True,
                severity="critical",
                action_required=SignalAction.EXIT_EARLY,
                message=f"Instant failure: Price moved {abs(adverse_movement):.1f} pips against trade",
                timestamp=datetime.utcnow()
            )
        
        return FailureDetection(detected=False)
    
    def _detect_slow_failure(self, signal: TradingSignal, time_elapsed_minutes: float) -> FailureDetection:
        """Detect slow failure - time exceeds expected duration"""
        
        expected_duration = signal.expected_duration_minutes
        max_allowed = expected_duration * 1.5  # Allow 50% more time
        
        # Also check against absolute maximum
        absolute_max = self.risk_settings['slow_failure_minutes']
        
        if time_elapsed_minutes > max_allowed or time_elapsed_minutes > absolute_max:
            severity = "high" if time_elapsed_minutes > absolute_max else "medium"
            return FailureDetection(
                failure_type="slow",
                detected=True,
                severity=severity,
                action_required=SignalAction.EXIT_EARLY,
                message=f"Slow failure: {time_elapsed_minutes:.1f} min elapsed (expected {expected_duration:.1f} min)",
                timestamp=datetime.utcnow()
            )
        
        return FailureDetection(detected=False)
    
    def _detect_fake_breakout(
        self, 
        signal: TradingSignal, 
        price_movement_pips: float,
        market_state: MarketState
    ) -> FailureDetection:
        """Detect fake breakout - break without follow-through"""
        
        # Check if price moved in favorable direction initially but is reversing
        threshold = self.risk_settings['fake_breakout_threshold']
        
        # Get market momentum
        momentum = market_state.get('momentum', 0)
        
        # If signal direction conflicts with current momentum
        if signal.bias == SignalBias.BUY and momentum < -threshold:
            return FailureDetection(
                failure_type="fake_breakout",
                detected=True,
                severity="high",
                action_required=SignalAction.EXIT_EARLY,
                message="Fake breakout detected: Bullish signal with bearish momentum",
                timestamp=datetime.utcnow()
            )
        elif signal.bias == SignalBias.SELL and momentum > threshold:
            return FailureDetection(
                failure_type="fake_breakout",
                detected=True,
                severity="high",
                action_required=SignalAction.EXIT_EARLY,
                message="Fake breakout detected: Bearish signal with bullish momentum",
                timestamp=datetime.utcnow()
            )
        
        return FailureDetection(detected=False)
    
    def _detect_high_volatility(
        self, 
        signal: TradingSignal, 
        market_state: MarketState,
        current_tick: TickData
    ) -> FailureDetection:
        """Detect high volatility conditions"""
        
        # Check spread
        if current_tick.spread > self.risk_settings['max_spread_pips']:
            return FailureDetection(
                failure_type="high_volatility",
                detected=True,
                severity="medium",
                action_required=SignalAction.SKIP,
                message=f"Spread too high: {current_tick.spread:.1f} pips",
                timestamp=datetime.utcnow()
            )
        
        # Check volatility
        volatility = market_state.get('volatility_pct', 0)
        if volatility > self.risk_settings['volatility_spike_threshold']:
            return FailureDetection(
                failure_type="high_volatility",
                detected=True,
                severity="high",
                action_required=SignalAction.PAUSE,
                message=f"Volatility spike detected: {volatility:.2f}%",
                timestamp=datetime.utcnow()
            )
        
        return FailureDetection(detected=False)
    
    def _detect_low_confidence(self, signal: TradingSignal) -> FailureDetection:
        """Detect low confidence signals"""
        
        min_confidence = self.risk_settings['min_confidence']
        
        if signal.confidence < min_confidence:
            return FailureDetection(
                failure_type="low_confidence",
                detected=True,
                severity="low",
                action_required=SignalAction.SKIP,
                message=f"Confidence too low: {signal.confidence:.2f} < {min_confidence:.2f}",
                timestamp=datetime.utcnow()
            )
        
        # Check if probabilities are too close (uncertain market)
        prob_diff = abs(signal.probability_hold_above - signal.probability_hold_below)
        if prob_diff < 0.1:  # Less than 10% difference
            return FailureDetection(
                failure_type="low_confidence",
                detected=True,
                severity="medium",
                action_required=SignalAction.SKIP,
                message=f"Uncertain market: Probabilities too close ({prob_diff:.2%} diff)",
                timestamp=datetime.utcnow()
            )
        
        return FailureDetection(detected=False)
    
    def _log_failure(self, failure: FailureDetection):
        """Log failure detection"""
        self.failure_history.append(failure)
        
        # Keep only last 100 failures
        if len(self.failure_history) > 100:
            self.failure_history.pop(0)
        
        logger.warning(f"Failure detected: {failure.failure_type} - {failure.message}")
    
    def assess_risk(
        self,
        signal: TradingSignal,
        features: FeatureSet,
        account_balance: float = 10000.0
    ) -> RiskAssessment:
        """Assess risk for a trading signal"""
        
        # Determine risk level based on confidence and volatility
        risk_level = self._determine_risk_level(signal, features)
        
        # Calculate position size
        max_risk = account_balance * self.risk_settings['max_risk_per_trade']
        
        # Volatility-adjusted stop loss
        atr = features.statistical.atr
        atr_pct = features.statistical.atr_pct
        
        # Base stop loss in pips
        if atr_pct > 0:
            # ATR-based stop loss (1.5x ATR)
            sl_pips = atr_pct * 1.5 * 10000  # Convert to pips for non-JPY pairs
        else:
            sl_pips = 20  # Default 20 pips
        
        # Adjust for volatility
        volatility = features.statistical.rolling_volatility_20
        if volatility > 0.02:  # High volatility
            sl_pips *= 1.3
        elif volatility < 0.005:  # Low volatility
            sl_pips *= 0.7
        
        sl_pips = max(10, min(50, sl_pips))  # Clamp between 10-50 pips
        
        # Take profit based on confidence
        confidence = signal.confidence
        rr_ratio = 1.5 + (confidence - 0.5) * 2  # 1.5 to 2.5 based on confidence
        tp_pips = sl_pips * rr_ratio
        
        # Position size multiplier based on risk level
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.75,
            RiskLevel.HIGH: 0.5
        }
        position_multiplier = risk_multipliers.get(risk_level, 0.75)
        
        return RiskAssessment(
            max_risk_per_trade=max_risk * position_multiplier,
            position_size_multiplier=position_multiplier,
            stop_loss_pips=sl_pips,
            take_profit_pips=tp_pips,
            risk_reward_ratio=rr_ratio,
            volatility_adjusted=True,
            confidence_weighted_tp=True
        )
    
    def _determine_risk_level(self, signal: TradingSignal, features: FeatureSet) -> RiskLevel:
        """Determine overall risk level"""
        
        score = 0
        
        # Confidence contribution
        if signal.confidence >= 0.8:
            score -= 1
        elif signal.confidence < 0.65:
            score += 1
        
        # Volatility contribution
        if features.statistical.rolling_volatility_20 > 0.02:
            score += 2
        elif features.statistical.rolling_volatility_20 > 0.01:
            score += 1
        elif features.statistical.rolling_volatility_20 < 0.005:
            score -= 1
        
        # RSI contribution
        if features.statistical.rsi > 70 or features.statistical.rsi < 30:
            score += 1
        
        # Session contribution (high volatility sessions)
        hour = features.time_features.hour_of_day
        if 13 <= hour < 17:  # London-NY overlap
            score += 1
        
        # Mean reversion contribution
        if abs(features.statistical.zscore) > 2:
            score += 1
        
        # Determine level
        if score <= -1:
            return RiskLevel.LOW
        elif score <= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    def get_failure_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get failure statistics for the last N hours"""
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_failures = [f for f in self.failure_history if f.timestamp >= cutoff]
        
        if not recent_failures:
            return {
                'total_failures': 0,
                'by_type': {},
                'by_severity': {},
                'hours_analyzed': hours
            }
        
        # Group by type
        by_type = {}
        for f in recent_failures:
            if f.failure_type:
                by_type[f.failure_type] = by_type.get(f.failure_type, 0) + 1
        
        # Group by severity
        by_severity = {}
        for f in recent_failures:
            by_severity[f.severity] = by_severity.get(f.severity, 0) + 1
        
        return {
            'total_failures': len(recent_failures),
            'by_type': by_type,
            'by_severity': by_severity,
            'hours_analyzed': hours,
            'most_common': max(by_type.items(), key=lambda x: x[1])[0] if by_type else None
        }
    
    def should_pause_trading(self) -> tuple[bool, str]:
        """Determine if trading should be paused based on recent failures"""
        
        stats = self.get_failure_stats(hours=1)
        
        # Too many failures in last hour
        if stats['total_failures'] > 5:
            return True, f"Too many failures ({stats['total_failures']}) in last hour"
        
        # Critical failures
        critical_count = stats['by_severity'].get('critical', 0)
        if critical_count > 2:
            return True, f"Multiple critical failures ({critical_count}) in last hour"
        
        # High volatility dominated
        high_vol_count = stats['by_type'].get('high_volatility', 0)
        if high_vol_count > 3:
            return True, "Persistent high volatility conditions"
        
        return False, ""


class PositionSizer:
    """Calculate position sizes based on risk parameters"""
    
    def __init__(self):
        self.pip_values = {
            'EURUSD': 10.0,
            'GBPUSD': 10.0,
            'USDJPY': 9.07,
            'XAUUSD': 100.0,
            'AUDUSD': 10.0,
            'USDCAD': 7.45,
            'USDCHF': 11.15
        }
    
    def calculate_position_size(
        self,
        symbol: str,
        account_balance: float,
        risk_percent: float,
        stop_loss_pips: float,
        max_lot_size: float = 1.0
    ) -> float:
        """Calculate position size in lots"""
        
        # Get pip value
        pip_value = self.pip_values.get(symbol[:6], 10.0)
        
        # Calculate risk amount
        risk_amount = account_balance * (risk_percent / 100)
        
        # Calculate position size
        # Position = Risk / (Stop Loss Pips × Pip Value)
        if stop_loss_pips > 0 and pip_value > 0:
            position_size = risk_amount / (stop_loss_pips * pip_value)
        else:
            position_size = 0.01  # Minimum lot
        
        # Apply limits
        position_size = max(0.01, min(position_size, max_lot_size))
        
        # Round to 2 decimal places
        return round(position_size, 2)
    
    def calculate_stop_loss_price(
        self,
        entry_price: float,
        direction: SignalBias,
        stop_loss_pips: float,
        symbol: str
    ) -> float:
        """Calculate stop loss price"""
        
        # Determine pip size
        if 'JPY' in symbol:
            pip_size = 0.01
        else:
            pip_size = 0.0001
        
        sl_distance = stop_loss_pips * pip_size
        
        if direction == SignalBias.BUY:
            return entry_price - sl_distance
        else:
            return entry_price + sl_distance
    
    def calculate_take_profit_price(
        self,
        entry_price: float,
        direction: SignalBias,
        take_profit_pips: float,
        symbol: str
    ) -> float:
        """Calculate take profit price"""
        
        # Determine pip size
        if 'JPY' in symbol:
            pip_size = 0.01
        else:
            pip_size = 0.0001
        
        tp_distance = take_profit_pips * pip_size
        
        if direction == SignalBias.BUY:
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance


# Global instances
risk_engine = RiskFailureEngine()
position_sizer = PositionSizer()
