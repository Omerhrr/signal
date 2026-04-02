"""
Forex Probability Intelligence System - Decision Engine
Convert predictions into actionable trading signals
"""
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger

from app.models.schemas import (
    TradingSignal, SignalCard, FeatureSet, PredictionOutput,
    DirectionPrediction, DurationPrediction, SignalBias, SignalStatus,
    SignalAction, RiskLevel, TickData, MarketState
)
from app.engines.feature_engine import feature_engine, FeatureEngine
from app.engines.direction_model import direction_model
from app.engines.duration_model import duration_model
from app.engines.risk_engine import risk_engine, position_sizer
from config.settings import get_settings, SIGNAL_STATUS

settings = get_settings()


class DecisionEngine:
    """Engine for converting predictions into trading signals"""
    
    def __init__(self):
        self.min_confidence = settings.min_confidence_threshold
        self.min_probability = settings.min_probability_threshold
        self.signal_expiry = settings.signal_expiry_minutes
        self.active_signals: Dict[str, TradingSignal] = {}
    
    def generate_signal(
        self,
        symbol: str,
        ohlcv_data: List,
        current_tick: TickData,
        market_state: MarketState
    ) -> Optional[TradingSignal]:
        """Generate a trading signal for a symbol"""
        
        # 1. Calculate features
        features = feature_engine.calculate_features(symbol, ohlcv_data)
        
        # 2. Get direction prediction
        direction_pred = direction_model.predict(features)
        
        # 3. Get duration prediction
        duration_pred = duration_model.predict(features, direction_pred.predicted_direction)
        
        # 4. Create prediction output
        prediction = PredictionOutput(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            direction=direction_pred,
            duration=duration_pred,
            confidence_score=direction_pred.confidence,
            model_agreement=self._calculate_model_agreement(direction_pred),
            market_regime=self._determine_market_regime(features)
        )
        
        # 5. Check if signal meets minimum criteria
        if not self._meets_signal_criteria(prediction):
            return None
        
        # 6. Determine entry zone
        entry_zone = self._calculate_entry_zone(current_tick, direction_pred.predicted_direction)
        
        # 7. Calculate risk parameters
        risk_assessment = risk_engine.assess_risk(
            self._create_temp_signal(symbol, direction_pred, duration_pred, current_tick),
            features
        )
        
        # 8. Calculate SL and TP
        sl_price = position_sizer.calculate_stop_loss_price(
            current_tick.bid,
            direction_pred.predicted_direction,
            risk_assessment.stop_loss_pips,
            symbol
        )
        tp_price = position_sizer.calculate_take_profit_price(
            current_tick.bid,
            direction_pred.predicted_direction,
            risk_assessment.take_profit_pips,
            symbol
        )
        
        # 9. Determine signal status
        status, action, message = self._determine_signal_status(
            prediction, features, current_tick, market_state
        )
        
        # 10. Create trading signal
        signal = TradingSignal(
            signal_id=self._generate_signal_id(symbol),
            symbol=symbol,
            bias=direction_pred.predicted_direction,
            entry_zone_start=entry_zone['start'],
            entry_zone_end=entry_zone['end'],
            confidence=direction_pred.confidence,
            probability_hold_above=direction_pred.prob_up,
            probability_hold_below=direction_pred.prob_down,
            expected_duration_minutes=duration_pred.expected_time_above_minutes,
            stop_loss=sl_price,
            take_profit=tp_price,
            stop_loss_type="volatility_based",
            take_profit_type="confidence_weighted",
            risk_level=self._determine_risk_level(direction_pred.confidence, features),
            risk_reward_ratio=risk_assessment.risk_reward_ratio,
            status=status,
            action=action,
            status_message=message,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=self.signal_expiry),
            features=features,
            prediction=prediction
        )
        
        # Store signal
        self.active_signals[signal.signal_id] = signal
        
        return signal
    
    def _meets_signal_criteria(self, prediction: PredictionOutput) -> bool:
        """Check if prediction meets minimum signal criteria"""
        
        # Minimum confidence
        if prediction.confidence_score < self.min_confidence:
            logger.debug(f"Signal rejected: confidence {prediction.confidence_score:.2f} < {self.min_confidence}")
            return False
        
        # Minimum probability difference
        prob_diff = abs(
            prediction.direction.prob_up - prediction.direction.prob_down
        )
        if prob_diff < 0.1:  # Less than 10% difference
            logger.debug(f"Signal rejected: probabilities too close ({prob_diff:.2%})")
            return False
        
        # Direction must not be neutral
        if prediction.direction.predicted_direction == SignalBias.NEUTRAL:
            logger.debug("Signal rejected: neutral direction")
            return False
        
        return True
    
    def _calculate_model_agreement(self, direction_pred: DirectionPrediction) -> float:
        """Calculate model agreement score"""
        # For now, based on how decisive the prediction is
        # Higher agreement when probabilities are more extreme
        prob_diff = abs(direction_pred.prob_up - direction_pred.prob_down)
        return min(1.0, prob_diff * 2)  # Scale to 0-1
    
    def _determine_market_regime(self, features: FeatureSet) -> str:
        """Determine current market regime"""
        
        volatility = features.statistical.rolling_volatility_20
        momentum = features.statistical.momentum_decay_10
        rsi = features.statistical.rsi
        zscore = features.statistical.zscore
        
        # Trending market
        if abs(momentum) > 0.01:
            if momentum > 0:
                return "trending_bullish"
            else:
                return "trending_bearish"
        
        # Ranging market
        if abs(zscore) < 1.0 and volatility < 0.01:
            return "ranging"
        
        # Volatile market
        if volatility > 0.02:
            return "volatile"
        
        # Reversal potential
        if rsi > 70:
            return "overbought"
        elif rsi < 30:
            return "oversold"
        
        return "neutral"
    
    def _calculate_entry_zone(
        self, 
        current_tick: TickData, 
        direction: SignalBias
    ) -> Dict[str, float]:
        """Calculate optimal entry zone"""
        
        spread_pips = current_tick.spread
        mid_price = (current_tick.bid + current_tick.ask) / 2
        
        # For BUY: entry zone below current price
        # For SELL: entry zone above current price
        
        pip_size = 0.0001  # Standard for most pairs
        
        if direction == SignalBias.BUY:
            # Entry zone: current bid - 5 to current bid + 2 pips
            zone_start = current_tick.bid - (5 * pip_size)
            zone_end = current_tick.bid + (2 * pip_size)
        else:
            # Entry zone: current ask - 2 to current ask + 5 pips
            zone_start = current_tick.ask - (2 * pip_size)
            zone_end = current_tick.ask + (5 * pip_size)
        
        return {
            'start': round(zone_start, 5),
            'end': round(zone_end, 5)
        }
    
    def _create_temp_signal(
        self,
        symbol: str,
        direction_pred: DirectionPrediction,
        duration_pred: DurationPrediction,
        current_tick: TickData
    ) -> TradingSignal:
        """Create temporary signal for risk assessment"""
        return TradingSignal(
            signal_id="temp",
            symbol=symbol,
            bias=direction_pred.predicted_direction,
            entry_zone_start=current_tick.bid,
            entry_zone_end=current_tick.ask,
            confidence=direction_pred.confidence,
            probability_hold_above=direction_pred.prob_up,
            probability_hold_below=direction_pred.prob_down,
            expected_duration_minutes=duration_pred.expected_time_above_minutes,
            stop_loss=0,
            take_profit=0,
            risk_level=RiskLevel.MEDIUM,
            risk_reward_ratio=1.5,
            status=SignalStatus.OK,
            action=SignalAction.TRADE,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow()
        )
    
    def _determine_signal_status(
        self,
        prediction: PredictionOutput,
        features: FeatureSet,
        current_tick: TickData,
        market_state: MarketState
    ) -> tuple[SignalStatus, SignalAction, str]:
        """Determine signal status and recommended action"""
        
        # Check spread
        if current_tick.spread > settings.max_spread_pips:
            return (
                SignalStatus.SPREAD_TOO_HIGH,
                SignalAction.SKIP,
                f"Spread ({current_tick.spread:.1f} pips) exceeds maximum"
            )
        
        # Check confidence
        if prediction.confidence_score < self.min_confidence:
            return (
                SignalStatus.LOW_CONFIDENCE,
                SignalAction.SKIP,
                f"Confidence ({prediction.confidence_score:.2%}) below threshold"
            )
        
        # Check volatility
        volatility = features.statistical.rolling_volatility_20
        if volatility > settings.volatility_spike_threshold / 100:
            return (
                SignalStatus.HIGH_VOLATILITY,
                SignalAction.PAUSE,
                f"High volatility detected ({volatility:.2%})"
            )
        
        # Check for potential reversal
        if prediction.market_regime in ['overbought', 'oversold']:
            regime_direction = 'SELL' if prediction.market_regime == 'overbought' else 'BUY'
            if prediction.direction.predicted_direction.value != regime_direction:
                return (
                    SignalStatus.REVERSAL_DETECTED,
                    SignalAction.EXIT_EARLY,
                    f"Potential reversal: Market is {prediction.market_regime}"
                )
        
        # Signal is OK
        return (
            SignalStatus.OK,
            SignalAction.TRADE,
            "Signal is valid for trading"
        )
    
    def _determine_risk_level(self, confidence: float, features: FeatureSet) -> RiskLevel:
        """Determine risk level for the signal"""
        
        score = 0
        
        # Confidence contribution
        if confidence >= 0.8:
            score -= 1
        elif confidence < 0.65:
            score += 1
        
        # Volatility contribution
        if features.statistical.rolling_volatility_20 > 0.02:
            score += 2
        elif features.statistical.rolling_volatility_20 < 0.005:
            score -= 1
        
        # RSI contribution
        rsi = features.statistical.rsi
        if 40 <= rsi <= 60:
            score -= 1  # Neutral RSI is good
        elif rsi > 70 or rsi < 30:
            score += 1  # Extreme RSI adds risk
        
        # Determine level
        if score <= -1:
            return RiskLevel.LOW
        elif score <= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    def _generate_signal_id(self, symbol: str) -> str:
        """Generate unique signal ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        unique_part = uuid.uuid4().hex[:6]
        return f"{symbol}_{timestamp}_{unique_part}"
    
    def update_signal(
        self,
        signal_id: str,
        current_tick: TickData,
        market_state: MarketState,
        time_elapsed_minutes: float = 0
    ) -> Optional[TradingSignal]:
        """Update existing signal with new market data"""
        
        if signal_id not in self.active_signals:
            return None
        
        signal = self.active_signals[signal_id]
        
        # Check if signal expired
        if datetime.utcnow() > signal.expires_at:
            signal.status = SignalStatus.LOW_CONFIDENCE
            signal.action = SignalAction.SKIP
            signal.status_message = "Signal expired"
            return signal
        
        # Calculate price movement
        current_price = current_tick.bid if signal.bias == SignalBias.BUY else current_tick.ask
        entry_price = signal.entry_zone_start
        
        if signal.bias == SignalBias.BUY:
            price_movement_pips = (current_price - entry_price) / 0.0001
        else:
            price_movement_pips = (entry_price - current_price) / 0.0001
        
        # Check for failures
        if signal.features:
            failure = risk_engine.detect_failure(
                signal,
                current_tick,
                market_state,
                time_elapsed_minutes,
                price_movement_pips
            )
            
            if failure.detected:
                signal.status = SignalStatus.REVERSAL_DETECTED if failure.failure_type == "fake_breakout" else SignalStatus.HIGH_VOLATILITY
                signal.action = failure.action_required
                signal.status_message = failure.message
        
        return signal
    
    def get_active_signals(self, symbol: str = None) -> List[TradingSignal]:
        """Get all active signals, optionally filtered by symbol"""
        
        now = datetime.utcnow()
        active = []
        
        for signal in self.active_signals.values():
            # Remove expired signals
            if now > signal.expires_at:
                continue
            
            if symbol is None or signal.symbol == symbol:
                active.append(signal)
        
        return active
    
    def get_signal_cards(self, symbol: str = None) -> List[SignalCard]:
        """Get lightweight signal cards for dashboard"""
        
        signals = self.get_active_signals(symbol)
        cards = []
        
        for signal in signals:
            status_info = SIGNAL_STATUS.get(signal.status.value, {})
            
            cards.append(SignalCard(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                bias=signal.bias,
                entry_zone=f"{signal.entry_zone_start:.5f} - {signal.entry_zone_end:.5f}",
                confidence=signal.confidence,
                probability=signal.probability_hold_above if signal.bias == SignalBias.BUY else signal.probability_hold_below,
                expected_duration=f"{signal.expected_duration_minutes:.0f} min",
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                risk_level=signal.risk_level,
                status=signal.status,
                action=signal.action,
                status_color=status_info.get('color', '#9E9E9E'),
                created_at=signal.created_at
            ))
        
        return cards
    
    def clear_expired_signals(self):
        """Remove expired signals from memory"""
        now = datetime.utcnow()
        expired = [sid for sid, sig in self.active_signals.items() if now > sig.expires_at]
        
        for sid in expired:
            del self.active_signals[sid]
        
        if expired:
            logger.info(f"Cleared {len(expired)} expired signals")


class SignalAggregator:
    """Aggregate signals across multiple timeframes and symbols"""
    
    def __init__(self):
        self.decision_engine = DecisionEngine()
    
    def aggregate_signals(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """Aggregate multiple signals into a summary"""
        
        if not signals:
            return {
                'total': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'avg_confidence': 0,
                'best_signal': None
            }
        
        buy_count = sum(1 for s in signals if s.bias == SignalBias.BUY)
        sell_count = sum(1 for s in signals if s.bias == SignalBias.SELL)
        
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        
        best_signal = max(signals, key=lambda s: s.confidence)
        
        return {
            'total': len(signals),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'avg_confidence': avg_confidence,
            'best_signal': best_signal.signal_id,
            'bias': 'BULLISH' if buy_count > sell_count else 'BEARISH' if sell_count > buy_count else 'NEUTRAL'
        }


# Global instances
decision_engine = DecisionEngine()
signal_aggregator = SignalAggregator()
