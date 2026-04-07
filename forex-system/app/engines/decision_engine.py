"""
Forex Probability Intelligence System - Decision Engine
Convert predictions into actionable trading signals
Enhanced with advanced probability distribution, duration analysis,
signal confidence meter, and comprehensive signal generation.
"""
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger
from dataclasses import dataclass, field
from enum import Enum
import scipy.stats as stats
from collections import deque

from app.models.schemas import (
    TradingSignal, SignalCard, FeatureSet, PredictionOutput,
    DirectionPrediction, DurationPrediction, SignalBias, SignalStatus,
    SignalAction, RiskLevel, TickData, MarketState, TimeFeatures,
    MarketSession, OHLCV
)
from app.engines.feature_engine import feature_engine, FeatureEngine
from app.engines.direction_model import direction_model
from app.engines.duration_model import duration_model
from app.engines.risk_engine import risk_engine, position_sizer
from config.settings import get_settings, SIGNAL_STATUS

settings = get_settings()


# ============== Enhanced Data Classes ==============

class ProbabilityDistribution:
    """Advanced probability distribution for price movements"""
    
    def __init__(self, 
                 mean: float = 0.0, 
                 std: float = 1.0,
                 skewness: float = 0.0,
                 kurtosis: float = 3.0):
        self.mean = mean
        self.std = std
        self.skewness = skewness
        self.kurtosis = kurtosis
        self._distribution = None
    
    def fit_to_data(self, data: np.ndarray) -> None:
        """Fit distribution to historical data"""
        if len(data) < 10:
            return
        
        self.mean = np.mean(data)
        self.std = np.std(data)
        
        # Calculate higher moments
        if self.std > 0:
            standardized = (data - self.mean) / self.std
            self.skewness = np.mean(standardized ** 3)
            self.kurtosis = np.mean(standardized ** 4)
    
    def probability_above(self, threshold: float) -> float:
        """Calculate probability of value being above threshold"""
        if self.std <= 0:
            return 0.5
        
        z_score = (threshold - self.mean) / self.std
        
        # Adjust for skewness using Cornish-Fisher expansion
        if self.skewness != 0:
            z_adjusted = z_score + (self.skewness / 6) * (z_score ** 2 - 1)
        else:
            z_adjusted = z_score
        
        prob = 1 - stats.norm.cdf(z_adjusted)
        return float(np.clip(prob, 0.01, 0.99))
    
    def probability_below(self, threshold: float) -> float:
        """Calculate probability of value being below threshold"""
        return 1.0 - self.probability_above(threshold)
    
    def expected_value_in_range(self, lower: float, upper: float) -> float:
        """Calculate expected value within a range"""
        if self.std <= 0:
            return self.mean
        
        # Use truncated normal distribution
        a = (lower - self.mean) / self.std
        b = (upper - self.mean) / self.std
        
        # Expected value of truncated normal
        phi_a = stats.norm.pdf(a)
        phi_b = stats.norm.pdf(b)
        Phi_a = stats.norm.cdf(a)
        Phi_b = stats.norm.cdf(b)
        
        if Phi_b - Phi_a > 0:
            expected = self.mean + self.std * (phi_a - phi_b) / (Phi_b - Phi_a)
        else:
            expected = (lower + upper) / 2
        
        return float(expected)
    
    def percentile(self, p: float) -> float:
        """Get the p-th percentile of the distribution"""
        if self.std <= 0:
            return self.mean
        
        # Adjust for skewness
        z = stats.norm.ppf(p)
        if self.skewness != 0:
            z = z + (self.skewness / 6) * (z ** 2 - 1)
        
        return self.mean + z * self.std
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'mean': self.mean,
            'std': self.std,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis
        }


@dataclass
class ConfidenceMetrics:
    """Detailed confidence metrics for signal evaluation"""
    overall_confidence: float = 0.0
    directional_confidence: float = 0.0
    timing_confidence: float = 0.0
    volatility_confidence: float = 0.0
    session_confidence: float = 0.0
    feature_agreement: float = 0.0
    model_agreement: float = 0.0
    historical_accuracy: float = 0.0
    risk_adjusted_confidence: float = 0.0
    
    # Confidence breakdown
    bullish_signals: int = 0
    bearish_signals: int = 0
    neutral_signals: int = 0
    
    # Quality indicators
    signal_quality: str = "medium"  # low, medium, high, premium
    reliability_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_confidence': self.overall_confidence,
            'directional_confidence': self.directional_confidence,
            'timing_confidence': self.timing_confidence,
            'volatility_confidence': self.volatility_confidence,
            'session_confidence': self.session_confidence,
            'feature_agreement': self.feature_agreement,
            'model_agreement': self.model_agreement,
            'historical_accuracy': self.historical_accuracy,
            'risk_adjusted_confidence': self.risk_adjusted_confidence,
            'signal_quality': self.signal_quality,
            'reliability_score': self.reliability_score
        }


@dataclass
class DurationAnalysis:
    """Comprehensive duration analysis result"""
    expected_duration: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    median_duration: float = 0.0
    p25_duration: float = 0.0
    p75_duration: float = 0.0
    p90_duration: float = 0.0
    
    # Survival probabilities
    survival_1min: float = 0.0
    survival_5min: float = 0.0
    survival_10min: float = 0.0
    survival_15min: float = 0.0
    survival_30min: float = 0.0
    survival_60min: float = 0.0
    
    # Hazard analysis
    hazard_rate: float = 0.0
    cumulative_hazard: float = 0.0
    instantaneous_risk: float = 0.0
    
    # Duration scenarios
    optimistic_duration: float = 0.0
    base_duration: float = 0.0
    conservative_duration: float = 0.0
    pessimistic_duration: float = 0.0
    
    # Time to targets
    time_to_5pip: float = 0.0
    time_to_10pip: float = 0.0
    time_to_20pip: float = 0.0
    time_to_30pip: float = 0.0
    
    # Confidence
    duration_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'expected_duration': self.expected_duration,
            'min_duration': self.min_duration,
            'max_duration': self.max_duration,
            'median_duration': self.median_duration,
            'survival_probabilities': {
                '1min': self.survival_1min,
                '5min': self.survival_5min,
                '10min': self.survival_10min,
                '15min': self.survival_15min,
                '30min': self.survival_30min,
                '60min': self.survival_60min
            },
            'hazard_rate': self.hazard_rate,
            'scenarios': {
                'optimistic': self.optimistic_duration,
                'base': self.base_duration,
                'conservative': self.conservative_duration,
                'pessimistic': self.pessimistic_duration
            },
            'time_to_targets': {
                '5pip': self.time_to_5pip,
                '10pip': self.time_to_10pip,
                '20pip': self.time_to_20pip,
                '30pip': self.time_to_30pip
            },
            'duration_confidence': self.duration_confidence
        }


# ============== Enhanced Engines ==============

class ProbabilityDistributionEngine:
    """Advanced probability distribution engine for price movements"""
    
    def __init__(self):
        self.history_window = 100
        self.price_history: Dict[str, deque] = {}
        self.distributions: Dict[str, ProbabilityDistribution] = {}
    
    def update_distribution(self, symbol: str, price_change: float) -> None:
        """Update price change distribution for a symbol"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.history_window)
        
        self.price_history[symbol].append(price_change)
        
        # Update distribution
        if len(self.price_history[symbol]) >= 20:
            data = np.array(list(self.price_history[symbol]))
            if symbol not in self.distributions:
                self.distributions[symbol] = ProbabilityDistribution()
            self.distributions[symbol].fit_to_data(data)
    
    def calculate_probability_distribution(
        self, 
        features: FeatureSet,
        direction: SignalBias
    ) -> ProbabilityDistribution:
        """Calculate probability distribution for price movement"""
        
        symbol = features.symbol
        
        # Get base distribution from history
        if symbol in self.distributions:
            base_dist = self.distributions[symbol]
        else:
            base_dist = ProbabilityDistribution(mean=0, std=0.001)
        
        # Adjust based on features
        volatility = features.statistical.rolling_volatility_20
        momentum = features.statistical.momentum_decay_10
        rsi = features.statistical.rsi
        
        # Adjust mean based on momentum and RSI
        mean_adjustment = 0.0
        
        # Momentum contribution
        if abs(momentum) > 0.005:
            mean_adjustment += momentum * 0.5
        
        # RSI contribution (mean reversion)
        if rsi > 70:
            mean_adjustment -= 0.0005  # Expect pullback
        elif rsi < 30:
            mean_adjustment += 0.0005  # Expect bounce
        
        # Break of structure contribution
        if features.price_action.break_of_structure:
            bos_strength = features.price_action.bos_strength / 100
            if features.price_action.bos_direction == "BUY":
                mean_adjustment += bos_strength * 0.001
            elif features.price_action.bos_direction == "SELL":
                mean_adjustment -= bos_strength * 0.001
        
        # Wick rejection contribution
        if features.price_action.wick_rejection_strength > 0.5:
            if features.price_action.wick_rejection_type == "bullish":
                mean_adjustment += 0.0003
            elif features.price_action.wick_rejection_type == "bearish":
                mean_adjustment -= 0.0003
        
        # Adjust std based on volatility
        if volatility > 0:
            std_adjustment = volatility * 0.5
        else:
            std_adjustment = base_dist.std
        
        # Calculate skewness based on market conditions
        skewness = 0.0
        if direction == SignalBias.BUY:
            if momentum > 0:
                skewness = 0.2  # Positive skew for bullish momentum
            if rsi < 40:
                skewness += 0.1  # More upside potential when oversold
        else:
            if momentum < 0:
                skewness = -0.2  # Negative skew for bearish momentum
            if rsi > 60:
                skewness -= 0.1  # More downside potential when overbought
        
        return ProbabilityDistribution(
            mean=base_dist.mean + mean_adjustment,
            std=max(std_adjustment, 0.0001),
            skewness=skewness,
            kurtosis=base_dist.kurtosis
        )
    
    def calculate_direction_probability(
        self,
        features: FeatureSet,
        distribution: ProbabilityDistribution
    ) -> Tuple[float, float]:
        """Calculate probability of price going up vs down"""
        
        # Get multiple probability estimates
        scores = []
        
        # 1. Distribution-based probability
        prob_up_dist = distribution.probability_above(0)
        scores.append(('distribution', prob_up_dist, 0.3))
        
        # 2. Momentum-based probability
        momentum = features.statistical.momentum_decay_10
        if momentum > 0.005:
            prob_up_momentum = 0.6 + min(0.25, momentum * 10)
        elif momentum < -0.005:
            prob_up_momentum = 0.4 + max(-0.25, momentum * 10)
        else:
            prob_up_momentum = 0.5
        scores.append(('momentum', prob_up_momentum, 0.2))
        
        # 3. RSI-based probability
        rsi = features.statistical.rsi
        if rsi < 30:
            prob_up_rsi = 0.65  # Oversold, expect bounce
        elif rsi > 70:
            prob_up_rsi = 0.35  # Overbought, expect pullback
        elif rsi < 45:
            prob_up_rsi = 0.55
        elif rsi > 55:
            prob_up_rsi = 0.45
        else:
            prob_up_rsi = 0.5
        scores.append(('rsi', prob_up_rsi, 0.15))
        
        # 4. Mean reversion probability
        zscore = features.statistical.zscore
        if zscore < -2:
            prob_up_reversion = 0.7  # Price far below mean
        elif zscore > 2:
            prob_up_reversion = 0.3  # Price far above mean
        elif zscore < -1:
            prob_up_reversion = 0.6
        elif zscore > 1:
            prob_up_reversion = 0.4
        else:
            prob_up_reversion = 0.5
        scores.append(('reversion', prob_up_reversion, 0.15))
        
        # 5. Price action probability
        prob_up_pa = 0.5
        if features.price_action.break_of_structure:
            if features.price_action.bos_direction == "BUY":
                prob_up_pa += 0.15 * features.price_action.bos_strength
            else:
                prob_up_pa -= 0.15 * features.price_action.bos_strength
        
        if features.price_action.wick_rejection_strength > 0.5:
            if features.price_action.wick_rejection_type == "bullish":
                prob_up_pa += 0.1
            elif features.price_action.wick_rejection_type == "bearish":
                prob_up_pa -= 0.1
        scores.append(('price_action', prob_up_pa, 0.2))
        
        # Weighted average
        total_weight = sum(s[2] for s in scores)
        prob_up = sum(s[1] * s[2] for s in scores) / total_weight
        
        # Clamp to reasonable range
        prob_up = max(0.15, min(0.85, prob_up))
        prob_down = 1.0 - prob_up
        
        return prob_up, prob_down


class ConfidenceMeterEngine:
    """Engine for calculating comprehensive signal confidence"""
    
    def __init__(self):
        self.min_confidence = settings.min_confidence_threshold
        self.signal_history: deque = deque(maxlen=100)
        self.accuracy_history: Dict[str, deque] = {}
    
    def calculate_confidence(
        self,
        features: FeatureSet,
        direction_pred: DirectionPrediction,
        duration_pred: DurationPrediction,
        market_state: Optional[MarketState] = None
    ) -> ConfidenceMetrics:
        """Calculate comprehensive confidence metrics"""
        
        metrics = ConfidenceMetrics()
        
        # 1. Directional Confidence (from model prediction)
        metrics.directional_confidence = direction_pred.confidence
        
        # 2. Timing Confidence (based on session and time features)
        metrics.timing_confidence = self._calculate_timing_confidence(features)
        
        # 3. Volatility Confidence
        metrics.volatility_confidence = self._calculate_volatility_confidence(features)
        
        # 4. Session Confidence
        metrics.session_confidence = self._calculate_session_quality(features.time_features)
        
        # 5. Feature Agreement
        metrics.feature_agreement = self._calculate_feature_agreement(features, direction_pred)
        
        # 6. Model Agreement
        metrics.model_agreement = self._calculate_model_agreement(direction_pred)
        
        # 7. Historical Accuracy (if available)
        metrics.historical_accuracy = self._get_historical_accuracy(features.symbol)
        
        # Calculate overall confidence
        weights = {
            'directional': 0.25,
            'timing': 0.15,
            'volatility': 0.15,
            'session': 0.10,
            'feature': 0.15,
            'model': 0.10,
            'historical': 0.10
        }
        
        metrics.overall_confidence = (
            metrics.directional_confidence * weights['directional'] +
            metrics.timing_confidence * weights['timing'] +
            metrics.volatility_confidence * weights['volatility'] +
            metrics.session_confidence * weights['session'] +
            metrics.feature_agreement * weights['feature'] +
            metrics.model_agreement * weights['model'] +
            metrics.historical_accuracy * weights['historical']
        )
        
        # Risk-adjusted confidence
        risk_factor = 1.0 - self._calculate_risk_factor(features)
        metrics.risk_adjusted_confidence = metrics.overall_confidence * risk_factor
        
        # Determine signal quality
        metrics.signal_quality = self._determine_signal_quality(metrics)
        
        # Calculate reliability score
        metrics.reliability_score = self._calculate_reliability(metrics, features)
        
        # Count bullish/bearish signals
        self._count_signals(features, direction_pred, metrics)
        
        return metrics
    
    def _calculate_timing_confidence(self, features: FeatureSet) -> float:
        """Calculate timing confidence based on market timing"""
        score = 0.5
        
        hour = features.time_features.hour_of_day
        overlaps = features.time_features.session_overlap
        
        # Session overlap bonus
        if len(overlaps) >= 2:
            score += 0.2  # High activity during overlaps
        
        # Peak hours
        if 13 <= hour < 17:  # London-NY overlap
            score += 0.15
        elif 8 <= hour < 12:  # London session
            score += 0.1
        elif 0 <= hour < 9:  # Tokyo session
            score += 0.05
        
        # Session progress (avoid early/late session)
        progress = features.time_features.session_progress
        if 0.2 <= progress <= 0.8:
            score += 0.1
        
        # Weekend penalty
        if features.time_features.is_weekend:
            score -= 0.3
        
        return max(0.1, min(1.0, score))
    
    def _calculate_volatility_confidence(self, features: FeatureSet) -> float:
        """Calculate confidence based on volatility conditions"""
        score = 0.5
        
        vol = features.statistical.rolling_volatility_20
        atr_pct = features.statistical.atr_pct
        
        # Optimal volatility range
        if 0.005 <= vol <= 0.015:
            score += 0.3  # Good volatility
        elif vol < 0.005:
            score -= 0.1  # Too low
        elif vol > 0.02:
            score -= 0.2  # Too high
        elif vol > 0.03:
            score -= 0.4  # Very high
        
        # ATR contribution
        if atr_pct > 0:
            if 0.02 <= atr_pct <= 0.05:
                score += 0.2
            elif atr_pct > 0.08:
                score -= 0.2
        
        return max(0.1, min(1.0, score))
    
    def _calculate_session_quality(self, features: TimeFeatures) -> float:
        """Calculate session quality score"""
        score = 0.5
        
        session = features.current_session
        hour = features.hour_of_day
        
        # Session quality scores
        session_scores = {
            MarketSession.LONDON: 0.85,
            MarketSession.NEW_YORK: 0.80,
            MarketSession.TOKYO: 0.70,
            MarketSession.SYDNEY: 0.60
        }
        
        base_score = session_scores.get(session, 0.5)
        score = base_score
        
        # Adjust for session progress
        progress = features.session_progress
        if progress < 0.1 or progress > 0.9:
            score -= 0.1  # Early/late in session
        
        # Adjust for overlaps
        if len(features.session_overlap) >= 2:
            score = min(1.0, score + 0.15)
        
        return score
    
    def _calculate_feature_agreement(
        self, 
        features: FeatureSet, 
        direction_pred: DirectionPrediction
    ) -> float:
        """Calculate how many features agree with the prediction"""
        agreements = 0
        total = 0
        
        direction = direction_pred.predicted_direction
        
        # BOS agreement
        if features.price_action.break_of_structure:
            total += 1
            if direction == SignalBias.BUY and features.price_action.bos_direction == "BUY":
                agreements += 1
            elif direction == SignalBias.SELL and features.price_action.bos_direction == "SELL":
                agreements += 1
        
        # Wick rejection agreement
        if features.price_action.wick_rejection_strength > 0.5:
            total += 1
            if direction == SignalBias.BUY and features.price_action.wick_rejection_type == "bullish":
                agreements += 1
            elif direction == SignalBias.SELL and features.price_action.wick_rejection_type == "bearish":
                agreements += 1
        
        # RSI agreement
        total += 1
        rsi = features.statistical.rsi
        if direction == SignalBias.BUY and rsi < 55:
            agreements += 1
        elif direction == SignalBias.SELL and rsi > 45:
            agreements += 1
        
        # Momentum agreement
        total += 1
        momentum = features.statistical.momentum_decay_10
        if direction == SignalBias.BUY and momentum > -0.005:
            agreements += 1
        elif direction == SignalBias.SELL and momentum < 0.005:
            agreements += 1
        
        # Z-score agreement
        total += 1
        zscore = features.statistical.zscore
        if direction == SignalBias.BUY and zscore < 1:
            agreements += 1
        elif direction == SignalBias.SELL and zscore > -1:
            agreements += 1
        
        if total == 0:
            return 0.5
        
        return agreements / total
    
    def _calculate_model_agreement(self, direction_pred: DirectionPrediction) -> float:
        """Calculate model agreement score"""
        # Based on how decisive the prediction is
        prob_diff = abs(direction_pred.prob_up - direction_pred.prob_down)
        
        # Higher difference = higher agreement
        agreement = min(1.0, prob_diff * 2.5)
        
        return agreement
    
    def _get_historical_accuracy(self, symbol: str) -> float:
        """Get historical accuracy for a symbol"""
        if symbol not in self.accuracy_history:
            return 0.5  # Default for unknown symbols
        
        history = self.accuracy_history[symbol]
        if len(history) < 10:
            return 0.5
        
        return np.mean(list(history))
    
    def _calculate_risk_factor(self, features: FeatureSet) -> float:
        """Calculate risk factor (0-1, higher = more risk)"""
        risk = 0.0
        
        # Volatility risk
        vol = features.statistical.rolling_volatility_20
        if vol > 0.02:
            risk += 0.2
        elif vol > 0.015:
            risk += 0.1
        
        # RSI extreme risk
        rsi = features.statistical.rsi
        if rsi > 75 or rsi < 25:
            risk += 0.15
        elif rsi > 70 or rsi < 30:
            risk += 0.1
        
        # Z-score extreme risk
        zscore = abs(features.statistical.zscore)
        if zscore > 2.5:
            risk += 0.15
        elif zscore > 2:
            risk += 0.1
        
        # Session risk (outside main sessions)
        hour = features.time_features.hour_of_day
        if not (0 <= hour < 9 or 8 <= hour < 22):  # Outside major sessions
            risk += 0.1
        
        return min(0.5, risk)  # Cap at 0.5
    
    def _determine_signal_quality(self, metrics: ConfidenceMetrics) -> str:
        """Determine signal quality tier"""
        confidence = metrics.overall_confidence
        
        if confidence >= 0.8 and metrics.feature_agreement >= 0.7:
            return "premium"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.65:
            return "medium"
        else:
            return "low"
    
    def _calculate_reliability(self, metrics: ConfidenceMetrics, features: FeatureSet) -> float:
        """Calculate signal reliability score"""
        # Combine multiple reliability factors
        factors = []
        
        # Confidence consistency
        conf_values = [
            metrics.directional_confidence,
            metrics.timing_confidence,
            metrics.volatility_confidence,
            metrics.session_confidence
        ]
        consistency = 1.0 - np.std(conf_values) / (np.mean(conf_values) + 0.01)
        factors.append(max(0, consistency))
        
        # Feature agreement
        factors.append(metrics.feature_agreement)
        
        # Historical accuracy
        factors.append(metrics.historical_accuracy)
        
        # Volatility adjustment
        vol = features.statistical.rolling_volatility_20
        if vol < 0.01:
            factors.append(0.9)
        elif vol < 0.02:
            factors.append(0.7)
        else:
            factors.append(0.5)
        
        return np.mean(factors)
    
    def _count_signals(
        self, 
        features: FeatureSet, 
        direction_pred: DirectionPrediction,
        metrics: ConfidenceMetrics
    ) -> None:
        """Count bullish/bearish/neutral signals"""
        
        # Reset counts
        metrics.bullish_signals = 0
        metrics.bearish_signals = 0
        metrics.neutral_signals = 0
        
        # Check each feature for bullish/bearish signal
        
        # BOS
        if features.price_action.break_of_structure:
            if features.price_action.bos_direction == "BUY":
                metrics.bullish_signals += 1
            elif features.price_action.bos_direction == "SELL":
                metrics.bearish_signals += 1
        
        # Wick rejection
        if features.price_action.wick_rejection_strength > 0.5:
            if features.price_action.wick_rejection_type == "bullish":
                metrics.bullish_signals += 1
            elif features.price_action.wick_rejection_type == "bearish":
                metrics.bearish_signals += 1
        
        # RSI
        rsi = features.statistical.rsi
        if rsi < 35:
            metrics.bullish_signals += 1
        elif rsi > 65:
            metrics.bearish_signals += 1
        else:
            metrics.neutral_signals += 1
        
        # Momentum
        momentum = features.statistical.momentum_decay_10
        if momentum > 0.005:
            metrics.bullish_signals += 1
        elif momentum < -0.005:
            metrics.bearish_signals += 1
        else:
            metrics.neutral_signals += 1
        
        # Z-score
        zscore = features.statistical.zscore
        if zscore < -1:
            metrics.bullish_signals += 1
        elif zscore > 1:
            metrics.bearish_signals += 1
        else:
            metrics.neutral_signals += 1
    
    def record_outcome(self, symbol: str, was_correct: bool) -> None:
        """Record signal outcome for accuracy tracking"""
        if symbol not in self.accuracy_history:
            self.accuracy_history[symbol] = deque(maxlen=50)
        
        self.accuracy_history[symbol].append(1.0 if was_correct else 0.0)


class DurationAnalysisEngine:
    """Advanced duration analysis engine"""
    
    def __init__(self):
        self.duration_history: Dict[str, deque] = {}
        self.baseline_duration = 10.0
    
    def analyze_duration(
        self,
        features: FeatureSet,
        direction_pred: DirectionPrediction,
        duration_pred: DurationPrediction
    ) -> DurationAnalysis:
        """Perform comprehensive duration analysis"""
        
        analysis = DurationAnalysis()
        
        # Base duration from prediction
        base_duration = duration_pred.expected_time_above_minutes
        
        # Calculate duration adjustments
        adjustments = self._calculate_duration_adjustments(features)
        
        # Apply adjustments
        analysis.expected_duration = base_duration * adjustments['multiplier']
        analysis.expected_duration = max(3, min(60, analysis.expected_duration))
        
        # Calculate duration range
        std_factor = adjustments['uncertainty']
        analysis.min_duration = max(1, analysis.expected_duration * (1 - std_factor))
        analysis.max_duration = min(120, analysis.expected_duration * (1 + std_factor))
        
        # Percentiles
        analysis.median_duration = analysis.expected_duration * 0.9
        analysis.p25_duration = analysis.expected_duration * 0.7
        analysis.p75_duration = analysis.expected_duration * 1.3
        analysis.p90_duration = analysis.expected_duration * 1.6
        
        # Survival probabilities
        analysis.survival_1min = self._calculate_survival_prob(analysis.expected_duration, 1)
        analysis.survival_5min = self._calculate_survival_prob(analysis.expected_duration, 5)
        analysis.survival_10min = self._calculate_survival_prob(analysis.expected_duration, 10)
        analysis.survival_15min = self._calculate_survival_prob(analysis.expected_duration, 15)
        analysis.survival_30min = self._calculate_survival_prob(analysis.expected_duration, 30)
        analysis.survival_60min = self._calculate_survival_prob(analysis.expected_duration, 60)
        
        # Hazard analysis
        analysis.hazard_rate = duration_pred.hazard_rate
        analysis.cumulative_hazard = self._calculate_cumulative_hazard(
            analysis.hazard_rate, 
            analysis.expected_duration
        )
        analysis.instantaneous_risk = self._calculate_instantaneous_risk(features)
        
        # Duration scenarios
        analysis.optimistic_duration = analysis.expected_duration * 1.4
        analysis.base_duration = analysis.expected_duration
        analysis.conservative_duration = analysis.expected_duration * 0.7
        analysis.pessimistic_duration = analysis.expected_duration * 0.4
        
        # Time to targets (based on ATR)
        atr = features.statistical.atr
        if atr > 0:
            pip_equivalent = atr / 10  # Approximate pip value
            analysis.time_to_5pip = analysis.expected_duration * 0.25
            analysis.time_to_10pip = analysis.expected_duration * 0.45
            analysis.time_to_20pip = analysis.expected_duration * 0.75
            analysis.time_to_30pip = analysis.expected_duration * 1.0
        else:
            analysis.time_to_5pip = analysis.expected_duration * 0.25
            analysis.time_to_10pip = analysis.expected_duration * 0.45
            analysis.time_to_20pip = analysis.expected_duration * 0.75
            analysis.time_to_30pip = analysis.expected_duration * 1.0
        
        # Duration confidence
        analysis.duration_confidence = self._calculate_duration_confidence(features, duration_pred)
        
        return analysis
    
    def _calculate_duration_adjustments(self, features: FeatureSet) -> Dict[str, float]:
        """Calculate duration adjustment factors"""
        multiplier = 1.0
        uncertainty = 0.3  # Base uncertainty
        
        # Volatility adjustment
        vol = features.statistical.rolling_volatility_20
        if vol > 0.02:
            multiplier *= 0.7  # Shorter in high vol
            uncertainty += 0.1
        elif vol < 0.005:
            multiplier *= 1.3  # Longer in low vol
            uncertainty -= 0.05
        
        # ATR adjustment
        atr_pct = features.statistical.atr_pct
        if atr_pct > 0.05:
            multiplier *= 0.8
        elif atr_pct < 0.02:
            multiplier *= 1.2
        
        # Momentum adjustment
        momentum = abs(features.statistical.momentum_decay_10)
        if momentum > 0.01:
            multiplier *= 1.2  # Strong momentum = longer duration
        
        # Session adjustment
        hour = features.time_features.hour_of_day
        if 13 <= hour < 17:  # London-NY overlap
            multiplier *= 0.8  # Faster moves during overlap
        elif 0 <= hour < 9:  # Tokyo
            multiplier *= 1.2  # Slower moves
        
        # RSI adjustment
        rsi = features.statistical.rsi
        if rsi > 70 or rsi < 30:
            multiplier *= 0.7  # Potential reversal = shorter
            uncertainty += 0.15
        
        return {'multiplier': multiplier, 'uncertainty': min(0.5, uncertainty)}
    
    def _calculate_survival_prob(self, expected_duration: float, time: float) -> float:
        """Calculate survival probability at given time using Weibull-like distribution"""
        if expected_duration <= 0:
            return 0.5
        
        # Weibull survival function: S(t) = exp(-(t/lambda)^k)
        # Using k=1.5 for moderate hazard increase
        k = 1.5
        lambda_param = expected_duration / 0.9  # Scale parameter
        
        survival = np.exp(-((time / lambda_param) ** k))
        return float(max(0.05, min(0.95, survival)))
    
    def _calculate_cumulative_hazard(self, hazard_rate: float, duration: float) -> float:
        """Calculate cumulative hazard over duration"""
        if hazard_rate <= 0:
            return 0.0
        return hazard_rate * duration
    
    def _calculate_instantaneous_risk(self, features: FeatureSet) -> float:
        """Calculate instantaneous risk of reversal"""
        risk = 0.0
        
        # Volatility contribution
        vol = features.statistical.rolling_volatility_20
        if vol > 0.02:
            risk += 0.2
        elif vol > 0.015:
            risk += 0.1
        
        # RSI contribution
        rsi = features.statistical.rsi
        if rsi > 70:
            risk += 0.2  # High reversal risk
        elif rsi < 30:
            risk += 0.2
        elif rsi > 60 or rsi < 40:
            risk += 0.1
        
        # Z-score contribution
        zscore = abs(features.statistical.zscore)
        if zscore > 2:
            risk += 0.15
        
        # Range expansion contribution
        if features.price_action.range_expansion_pct > 50:
            risk += 0.1  # Extended range = higher reversal risk
        
        return min(1.0, risk)
    
    def _calculate_duration_confidence(
        self, 
        features: FeatureSet, 
        duration_pred: DurationPrediction
    ) -> float:
        """Calculate confidence in duration prediction"""
        confidence = 0.5
        
        # Higher confidence with lower volatility
        vol = features.statistical.rolling_volatility_20
        if vol < 0.01:
            confidence += 0.2
        elif vol < 0.015:
            confidence += 0.1
        elif vol > 0.02:
            confidence -= 0.15
        
        # Higher confidence during stable sessions
        hour = features.time_features.hour_of_day
        if 0 <= hour < 8 or 18 <= hour < 22:  # Off-peak hours
            confidence += 0.1  # More predictable
        
        # Lower confidence at session boundaries
        progress = features.time_features.session_progress
        if progress < 0.1 or progress > 0.9:
            confidence -= 0.1
        
        # Higher confidence with strong momentum direction
        momentum = abs(features.statistical.momentum_decay_10)
        if momentum > 0.01:
            confidence += 0.1
        
        return max(0.2, min(0.9, confidence))


class DecisionEngine:
    """Engine for converting predictions into trading signals"""
    
    def __init__(self):
        self.min_confidence = settings.min_confidence_threshold
        self.min_probability = settings.min_probability_threshold
        self.signal_expiry = settings.signal_expiry_minutes
        self.active_signals: Dict[str, TradingSignal] = {}
        
        # Initialize enhanced engines
        self.probability_engine = ProbabilityDistributionEngine()
        self.confidence_meter = ConfidenceMeterEngine()
        self.duration_analyzer = DurationAnalysisEngine()
    
    def generate_signal(
        self,
        symbol: str,
        ohlcv_data: List[OHLCV],
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
        
        # 4. Calculate probability distribution
        prob_distribution = self.probability_engine.calculate_probability_distribution(
            features, direction_pred.predicted_direction
        )
        
        # 5. Calculate enhanced direction probabilities
        prob_up, prob_down = self.probability_engine.calculate_direction_probability(
            features, prob_distribution
        )
        
        # Update direction prediction with enhanced probabilities
        direction_pred.prob_up = prob_up
        direction_pred.prob_down = prob_down
        direction_pred.confidence = max(prob_up, prob_down)
        direction_pred.predicted_direction = SignalBias.BUY if prob_up > prob_down else SignalBias.SELL
        
        # 6. Calculate confidence metrics
        confidence_metrics = self.confidence_meter.calculate_confidence(
            features, direction_pred, duration_pred, market_state
        )
        
        # 7. Perform duration analysis
        duration_analysis = self.duration_analyzer.analyze_duration(
            features, direction_pred, duration_pred
        )
        
        # 8. Create prediction output
        prediction = PredictionOutput(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            direction=direction_pred,
            duration=duration_pred,
            confidence_score=confidence_metrics.overall_confidence,
            model_agreement=confidence_metrics.model_agreement,
            market_regime=self._determine_market_regime(features)
        )
        
        # 9. Check if signal meets minimum criteria
        if not self._meets_signal_criteria(prediction, confidence_metrics):
            return None
        
        # 10. Determine entry zone
        entry_zone = self._calculate_entry_zone(current_tick, direction_pred.predicted_direction)
        
        # 11. Calculate risk parameters
        risk_assessment = risk_engine.assess_risk(
            self._create_temp_signal(symbol, direction_pred, duration_pred, current_tick),
            features
        )
        
        # 12. Calculate SL and TP
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
        
        # 13. Determine signal status
        status, action, message = self._determine_signal_status(
            prediction, features, current_tick, market_state, confidence_metrics
        )
        
        # 14. Create trading signal
        signal = TradingSignal(
            signal_id=self._generate_signal_id(symbol),
            symbol=symbol,
            bias=direction_pred.predicted_direction,
            entry_zone_start=entry_zone['start'],
            entry_zone_end=entry_zone['end'],
            confidence=confidence_metrics.overall_confidence,
            probability_hold_above=prob_up,
            probability_hold_below=prob_down,
            expected_duration_minutes=duration_analysis.expected_duration,
            stop_loss=sl_price,
            take_profit=tp_price,
            stop_loss_type="volatility_based",
            take_profit_type="confidence_weighted",
            risk_level=self._determine_risk_level(confidence_metrics, features),
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
    
    def _meets_signal_criteria(
        self, 
        prediction: PredictionOutput,
        confidence_metrics: ConfidenceMetrics
    ) -> bool:
        """Check if prediction meets minimum signal criteria"""
        
        # Minimum overall confidence
        if confidence_metrics.overall_confidence < self.min_confidence:
            logger.debug(f"Signal rejected: confidence {confidence_metrics.overall_confidence:.2f} < {self.min_confidence}")
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
        
        # Check signal quality
        if confidence_metrics.signal_quality == "low":
            logger.debug("Signal rejected: low signal quality")
            return False
        
        # Check feature agreement
        if confidence_metrics.feature_agreement < 0.3:
            logger.debug(f"Signal rejected: low feature agreement ({confidence_metrics.feature_agreement:.2f})")
            return False
        
        return True
    
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
        
        # Determine pip size based on symbol
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
        market_state: MarketState,
        confidence_metrics: ConfidenceMetrics
    ) -> tuple:
        """Determine signal status and recommended action"""
        
        # Check spread
        if current_tick.spread > settings.max_spread_pips:
            return (
                SignalStatus.SPREAD_TOO_HIGH,
                SignalAction.SKIP,
                f"Spread ({current_tick.spread:.1f} pips) exceeds maximum"
            )
        
        # Check confidence
        if confidence_metrics.overall_confidence < self.min_confidence:
            return (
                SignalStatus.LOW_CONFIDENCE,
                SignalAction.SKIP,
                f"Confidence ({confidence_metrics.overall_confidence:.2%}) below threshold"
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
        
        # Check risk-adjusted confidence
        if confidence_metrics.risk_adjusted_confidence < 0.5:
            return (
                SignalStatus.LOW_CONFIDENCE,
                SignalAction.SKIP,
                "Risk-adjusted confidence too low"
            )
        
        # Signal is OK
        return (
            SignalStatus.OK,
            SignalAction.TRADE,
            f"Signal valid - Quality: {confidence_metrics.signal_quality.upper()}"
        )
    
    def _determine_risk_level(
        self, 
        confidence_metrics: ConfidenceMetrics, 
        features: FeatureSet
    ) -> RiskLevel:
        """Determine risk level for the signal"""
        
        score = 0
        
        # Confidence contribution
        if confidence_metrics.overall_confidence >= 0.8:
            score -= 1
        elif confidence_metrics.overall_confidence < 0.65:
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
        
        # Signal quality contribution
        if confidence_metrics.signal_quality == "premium":
            score -= 1
        elif confidence_metrics.signal_quality == "low":
            score += 1
        
        # Feature agreement contribution
        if confidence_metrics.feature_agreement >= 0.7:
            score -= 1
        elif confidence_metrics.feature_agreement < 0.4:
            score += 1
        
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
