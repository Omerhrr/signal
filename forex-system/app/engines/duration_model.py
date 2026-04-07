"""
Forex Probability Intelligence System - Duration Prediction Model
Enhanced Survival Analysis for expected price movement duration
with multiple distribution models and confidence intervals.
"""
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from loguru import logger
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

try:
    from lifelines import WeibullAFTFitter, CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logger.warning("Lifelines not available, using enhanced heuristic duration prediction")

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from app.models.schemas import DurationPrediction, FeatureSet, SignalBias
from config.settings import get_settings

settings = get_settings()


class DurationModelType(str, Enum):
    """Types of duration models"""
    WEIBULL = "weibull"
    EXPONENTIAL = "exponential"
    LOG_NORMAL = "log_normal"
    HEURISTIC = "heuristic"
    ENSEMBLE = "ensemble"


@dataclass
class DurationMetrics:
    """Track duration prediction performance"""
    total_predictions: int = 0
    within_tolerance: int = 0
    avg_error: float = 0.0
    avg_absolute_error: float = 0.0
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, predicted: float, actual: float, tolerance: float = 0.3):
        """Update metrics with a new observation"""
        self.total_predictions += 1
        
        error = predicted - actual
        abs_error = abs(error)
        
        self.recent_errors.append(abs_error)
        self.avg_error = (self.avg_error * (self.total_predictions - 1) + error) / self.total_predictions
        self.avg_absolute_error = (self.avg_absolute_error * (self.total_predictions - 1) + abs_error) / self.total_predictions
        
        # Check if within tolerance
        if actual > 0:
            relative_error = abs_error / actual
            if relative_error <= tolerance:
                self.within_tolerance += 1
    
    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.within_tolerance / self.total_predictions
    
    @property
    def recent_mae(self) -> float:
        if len(self.recent_errors) == 0:
            return 0.0
        return float(np.mean(list(self.recent_errors)))


class WeibullDistribution:
    """Weibull distribution for survival analysis"""
    
    def __init__(self, shape: float = 1.5, scale: float = 10.0):
        self.shape = shape  # k parameter
        self.scale = scale  # lambda parameter
    
    def fit(self, durations: np.ndarray, events: np.ndarray):
        """Fit Weibull distribution to observed data"""
        if len(durations) < 10:
            return
        
        # Use method of moments for initial estimation
        positive_durations = durations[durations > 0]
        if len(positive_durations) < 5:
            return
        
        # Estimate scale (mean / gamma(1 + 1/k))
        mean_duration = np.mean(positive_durations)
        std_duration = np.std(positive_durations)
        
        if std_duration > 0 and mean_duration > 0:
            # Coefficient of variation
            cv = std_duration / mean_duration
            
            # Estimate shape from CV
            # For Weibull: CV^2 = gamma(1+2/k) / gamma(1+1/k)^2 - 1
            # Approximation
            self.shape = max(0.5, min(5.0, 1.0 / cv))
            
            # Estimate scale
            self.scale = mean_duration / np.exp(np.log(np.math.gamma(1 + 1/self.shape)))
    
    def survival_probability(self, t: float) -> float:
        """Calculate survival probability at time t"""
        if self.scale <= 0:
            return 0.5
        return float(np.exp(-(t / self.scale) ** self.shape))
    
    def hazard_rate(self, t: float) -> float:
        """Calculate hazard rate at time t"""
        if self.scale <= 0 or t <= 0:
            return 0.0
        return float((self.shape / self.scale) * (t / self.scale) ** (self.shape - 1))
    
    def expected_duration(self) -> float:
        """Calculate expected duration"""
        if self.scale <= 0:
            return 10.0
        return float(self.scale * np.exp(np.log(np.math.gamma(1 + 1/self.shape))))
    
    def percentile(self, p: float) -> float:
        """Calculate p-th percentile"""
        if self.scale <= 0:
            return 10.0
        return float(self.scale * (-np.log(1 - p)) ** (1 / self.shape))


class LogNormalDistribution:
    """Log-normal distribution for survival analysis"""
    
    def __init__(self, mu: float = 2.0, sigma: float = 0.5):
        self.mu = mu
        self.sigma = sigma
    
    def fit(self, durations: np.ndarray):
        """Fit log-normal distribution to observed data"""
        if len(durations) < 10:
            return
        
        positive_durations = durations[durations > 0]
        if len(positive_durations) < 5:
            return
        
        log_durations = np.log(positive_durations)
        self.mu = np.mean(log_durations)
        self.sigma = np.std(log_durations)
        self.sigma = max(0.1, min(2.0, self.sigma))
    
    def survival_probability(self, t: float) -> float:
        """Calculate survival probability at time t"""
        if t <= 0 or self.sigma <= 0:
            return 0.5
        z = (np.log(t) - self.mu) / self.sigma
        return float(1 - stats.norm.cdf(z)) if SCIPY_AVAILABLE else 0.5
    
    def expected_duration(self) -> float:
        """Calculate expected duration"""
        return float(np.exp(self.mu + 0.5 * self.sigma ** 2))
    
    def percentile(self, p: float) -> float:
        """Calculate p-th percentile"""
        return float(np.exp(self.mu + self.sigma * stats.norm.ppf(p))) if SCIPY_AVAILABLE else 10.0


class DurationHistoryTracker:
    """Track historical duration data for analysis"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.durations_by_symbol: Dict[str, deque] = {}
        self.durations_by_session: Dict[str, deque] = {}
        self.durations_by_volatility: Dict[str, deque] = {}
    
    def record(self, symbol: str, duration: float, session: str, volatility: float):
        """Record a duration observation"""
        # By symbol
        if symbol not in self.durations_by_symbol:
            self.durations_by_symbol[symbol] = deque(maxlen=self.max_history)
        self.durations_by_symbol[symbol].append(duration)
        
        # By session
        if session not in self.durations_by_session:
            self.durations_by_session[session] = deque(maxlen=self.max_history)
        self.durations_by_session[session].append(duration)
        
        # By volatility regime
        vol_regime = "high" if volatility > 0.02 else ("low" if volatility < 0.005 else "medium")
        if vol_regime not in self.durations_by_volatility:
            self.durations_by_volatility[vol_regime] = deque(maxlen=self.max_history)
        self.durations_by_volatility[vol_regime].append(duration)
    
    def get_symbol_duration_stats(self, symbol: str) -> Dict[str, float]:
        """Get duration statistics for a symbol"""
        if symbol not in self.durations_by_symbol:
            return {'mean': 10.0, 'std': 5.0, 'median': 8.0}
        
        data = np.array(list(self.durations_by_symbol[symbol]))
        if len(data) < 5:
            return {'mean': 10.0, 'std': 5.0, 'median': 8.0}
        
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'p25': float(np.percentile(data, 25)),
            'p75': float(np.percentile(data, 75))
        }
    
    def get_session_adjustment(self, session: str) -> float:
        """Get duration adjustment factor for session"""
        baseline = 10.0
        
        if session not in self.durations_by_session:
            # Default adjustments
            default_adjustments = {
                'london': 0.85,
                'new_york': 0.90,
                'tokyo': 1.15,
                'sydney': 1.20,
                'overlap': 0.70
            }
            return default_adjustments.get(session.lower(), 1.0)
        
        data = np.array(list(self.durations_by_session[session]))
        if len(data) < 5:
            return 1.0
        
        return float(np.mean(data) / baseline)


class DurationPredictionModel:
    """Enhanced survival analysis model for predicting price movement duration"""
    
    def __init__(self):
        self.weibull_model = None
        self.cox_model = None
        self.model_type = DurationModelType.HEURISTIC
        self.model_version = "2.0.0"
        self.is_trained = False
        self.baseline_duration = 10.0
        
        # Distribution models
        self.weibull_dist = WeibullDistribution()
        self.lognormal_dist = LogNormalDistribution()
        
        # Performance tracking
        self.metrics = DurationMetrics()
        self.history_tracker = DurationHistoryTracker()
        
        # Model path
        self.model_path = Path(__file__).parent.parent.parent / "data" / "models" / "duration_model.pkl"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing model
        self._load_model()
    
    def predict(self, features: FeatureSet, direction: SignalBias) -> DurationPrediction:
        """Predict duration of price movement with enhanced analysis"""
        
        if self.is_trained and LIFELINES_AVAILABLE:
            return self._predict_with_model(features, direction)
        else:
            return self._predict_heuristic(features, direction)
    
    def _predict_with_model(self, features: FeatureSet, direction: SignalBias) -> DurationPrediction:
        """Predict duration using trained survival model"""
        try:
            # Get base duration from distribution models
            weibull_duration = self.weibull_dist.expected_duration()
            lognormal_duration = self.lognormal_dist.expected_duration()
            
            # Ensemble the predictions
            base_duration = (weibull_duration + lognormal_duration) / 2
            
            # Apply feature-based adjustments
            adjustments = self._calculate_feature_adjustments(features)
            expected_duration = base_duration * adjustments['multiplier']
            
            # Calculate volatility factor
            volatility_factor = 1.0
            if features.statistical.atr_pct > 0:
                volatility_factor = max(0.5, min(1.5, 0.5 / features.statistical.atr_pct))
            
            expected_duration *= volatility_factor
            
            # Clamp to reasonable range
            expected_duration = max(3, min(60, expected_duration))
            
            # Calculate survival probabilities at multiple time points
            survival_probs = self._calculate_survival_probabilities(expected_duration)
            
            # Calculate hazard rates
            hazard_rate = self._calculate_dynamic_hazard_rate(features, expected_duration)
            
            return DurationPrediction(
                expected_time_above_minutes=expected_duration if direction == SignalBias.BUY else expected_duration * 0.7,
                expected_time_below_minutes=expected_duration if direction == SignalBias.SELL else expected_duration * 0.7,
                hazard_rate=hazard_rate,
                survival_probability_5min=survival_probs['5min'],
                survival_probability_10min=survival_probs['10min'],
                survival_probability_15min=survival_probs['15min'],
                model_version=self.model_version
            )
            
        except Exception as e:
            logger.error(f"Duration model prediction error: {e}")
            return self._predict_heuristic(features, direction)
    
    def _predict_heuristic(self, features: FeatureSet, direction: SignalBias) -> DurationPrediction:
        """Enhanced heuristic duration prediction based on market conditions"""
        
        # Start with baseline
        base_duration = self.baseline_duration
        
        # ===== FACTOR CALCULATIONS =====
        
        # 1. Volatility Adjustment
        volatility = features.statistical.rolling_volatility_20
        atr_pct = features.statistical.atr_pct
        
        vol_multiplier = 1.0
        if volatility > 0.02:
            vol_multiplier = 0.6  # High volatility = shorter duration
        elif volatility > 0.015:
            vol_multiplier = 0.75
        elif volatility < 0.005:
            vol_multiplier = 1.3  # Low volatility = longer duration
        elif volatility < 0.008:
            vol_multiplier = 1.15
        
        # ATR adjustment
        if atr_pct > 0.05:
            vol_multiplier *= 0.8
        elif atr_pct > 0.08:
            vol_multiplier *= 0.65
        
        # 2. Momentum Adjustment
        momentum = features.statistical.momentum_decay_10
        momentum_multiplier = 1.0
        
        if abs(momentum) > 0.01:
            # Strong momentum = longer duration
            momentum_multiplier = 1.25
        elif abs(momentum) > 0.005:
            momentum_multiplier = 1.1
        
        # 3. Session Adjustment
        session = features.time_features.current_session.value
        hour = features.time_features.hour_of_day
        
        session_multiplier = self.history_tracker.get_session_adjustment(session)
        
        # Fine-tune by hour
        if 13 <= hour < 17:  # London-NY overlap
            session_multiplier = 0.7  # High volatility overlap
        elif 8 <= hour < 12:  # London morning
            session_multiplier = 0.85
        elif 0 <= hour < 9:  # Tokyo
            session_multiplier = 1.2
        
        # 4. RSI Adjustment
        rsi = features.statistical.rsi
        rsi_multiplier = 1.0
        
        if rsi > 70 or rsi < 30:
            # Extreme RSI = potential reversal = shorter duration
            rsi_multiplier = 0.7
        elif rsi > 65 or rsi < 35:
            rsi_multiplier = 0.85
        
        # 5. Mean Reversion Adjustment
        zscore = features.statistical.zscore
        reversion_multiplier = 1.0
        
        if abs(zscore) > 2:
            # Price far from mean = stronger reversal pressure
            reversion_multiplier = 0.75
        elif abs(zscore) > 1.5:
            reversion_multiplier = 0.85
        
        # 6. Price Action Adjustment
        pa_multiplier = 1.0
        
        if features.price_action.break_of_structure:
            # BOS usually indicates stronger move
            bos_strength = features.price_action.bos_strength / 10
            pa_multiplier = 1.0 + bos_strength * 0.2
        
        if features.price_action.range_expansion_pct > 30:
            # Range expansion = strong momentum
            pa_multiplier *= 1.1
        
        # 7. Historical Adjustment
        symbol_stats = self.history_tracker.get_symbol_duration_stats(features.symbol)
        historical_adjustment = symbol_stats['mean'] / self.baseline_duration
        
        # ===== COMBINE ALL FACTORS =====
        
        combined_multiplier = (
            vol_multiplier * 
            momentum_multiplier * 
            session_multiplier * 
            rsi_multiplier * 
            reversion_multiplier * 
            pa_multiplier
        )
        
        # Blend with historical
        if historical_adjustment > 0.5 and historical_adjustment < 2.0:
            combined_multiplier = combined_multiplier * 0.7 + historical_adjustment * 0.3
        
        # Calculate expected duration
        expected_duration = base_duration * combined_multiplier
        
        # Add some controlled variance
        noise = np.random.normal(0, 0.1)
        expected_duration *= (1 + noise)
        
        # Clamp to reasonable range
        expected_duration = max(3, min(60, expected_duration))
        
        # ===== CALCULATE SURVIVAL PROBABILITIES =====
        
        survival_probs = self._calculate_survival_probabilities(expected_duration)
        
        # ===== CALCULATE HAZARD RATE =====
        
        hazard_rate = self._calculate_dynamic_hazard_rate(features, expected_duration)
        
        # ===== CREATE PREDICTION =====
        
        # Adjust for direction
        if direction == SignalBias.BUY:
            time_above = expected_duration
            time_below = expected_duration * 0.6
        else:
            time_above = expected_duration * 0.6
            time_below = expected_duration
        
        return DurationPrediction(
            expected_time_above_minutes=time_above,
            expected_time_below_minutes=time_below,
            hazard_rate=hazard_rate,
            survival_probability_5min=survival_probs['5min'],
            survival_probability_10min=survival_probs['10min'],
            survival_probability_15min=survival_probs['15min'],
            model_version="heuristic_v2"
        )
    
    def _calculate_feature_adjustments(self, features: FeatureSet) -> Dict[str, float]:
        """Calculate duration adjustment factors from features"""
        multiplier = 1.0
        uncertainty = 0.3
        
        # Volatility
        vol = features.statistical.rolling_volatility_20
        if vol > 0.02:
            multiplier *= 0.65
            uncertainty += 0.15
        elif vol < 0.005:
            multiplier *= 1.35
            uncertainty -= 0.05
        
        # ATR
        atr_pct = features.statistical.atr_pct
        if atr_pct > 0.05:
            multiplier *= 0.8
        
        # Momentum
        momentum = abs(features.statistical.momentum_decay_10)
        if momentum > 0.01:
            multiplier *= 1.2
        
        # Session
        hour = features.time_features.hour_of_day
        if 13 <= hour < 17:
            multiplier *= 0.7
        elif 0 <= hour < 9:
            multiplier *= 1.2
        
        # RSI
        rsi = features.statistical.rsi
        if rsi > 70 or rsi < 30:
            multiplier *= 0.7
            uncertainty += 0.15
        
        return {'multiplier': multiplier, 'uncertainty': min(0.5, uncertainty)}
    
    def _calculate_survival_probabilities(self, expected_duration: float) -> Dict[str, float]:
        """Calculate survival probabilities at multiple time points"""
        if expected_duration <= 0:
            return {'5min': 0.5, '10min': 0.5, '15min': 0.5}
        
        # Use Weibull-like survival function
        k = 1.5  # Shape parameter
        scale = expected_duration * 0.9
        
        probs = {}
        for t in [5, 10, 15]:
            survival = np.exp(-((t / scale) ** k))
            probs[f'{t}min'] = float(max(0.05, min(0.95, survival)))
        
        return probs
    
    def _calculate_dynamic_hazard_rate(self, features: FeatureSet, expected_duration: float) -> float:
        """Calculate dynamic hazard rate based on market conditions"""
        base_hazard = 1.0 / expected_duration if expected_duration > 0 else 0.1
        
        # Adjust based on volatility
        vol = features.statistical.rolling_volatility_20
        if vol > 0.02:
            base_hazard *= 1.5  # Higher hazard in high vol
        
        # Adjust based on RSI extremes
        rsi = features.statistical.rsi
        if rsi > 70 or rsi < 30:
            base_hazard *= 1.3  # Higher reversal risk
        
        # Adjust based on z-score
        zscore = abs(features.statistical.zscore)
        if zscore > 2:
            base_hazard *= 1.2
        
        return float(min(0.5, base_hazard))
    
    def train(self, duration_data: List[Dict]):
        """Train the survival model with historical duration data"""
        if not LIFELINES_AVAILABLE:
            logger.warning("Lifelines not available, skipping training")
            return False
        
        try:
            # Prepare training dataframe
            df_data = []
            durations = []
            events = []
            
            for entry in duration_data:
                duration = entry.get('duration_minutes', 0)
                if duration <= 0:
                    continue
                
                durations.append(duration)
                events.append(1 if entry.get('completed', True) else 0)
                
                df_data.append({
                    'duration': duration,
                    'event': entry.get('completed', 1),
                    'volatility': entry.get('volatility', 0),
                    'momentum': entry.get('momentum', 0),
                    'rsi': entry.get('rsi', 50) / 100.0,
                    'atr_pct': entry.get('atr_pct', 0),
                    'hour': entry.get('hour', 12),
                    'session_progress': entry.get('session_progress', 0.5)
                })
            
            if len(df_data) < 20:
                logger.warning("Not enough data for training")
                return False
            
            import pandas as pd
            df = pd.DataFrame(df_data)
            durations = np.array(durations)
            events = np.array(events)
            
            # Fit distribution models
            self.weibull_dist.fit(durations, events)
            self.lognormal_dist.fit(durations)
            
            # Train Weibull AFT model if lifelines available
            try:
                self.weibull_model = WeibullAFTFitter()
                self.weibull_model.fit(df, duration_col='duration', event_col='event')
            except Exception as e:
                logger.warning(f"Weibull AFT training failed: {e}")
            
            self.is_trained = True
            self.model_type = DurationModelType.ENSEMBLE
            self.model_version = f"trained_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            self._save_model()
            
            logger.info("Duration model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Duration model training error: {e}")
            return False
    
    def record_outcome(self, predicted_duration: float, actual_duration: float, 
                       symbol: str, session: str, volatility: float):
        """Record prediction outcome for performance tracking"""
        self.metrics.update(predicted_duration, actual_duration)
        self.history_tracker.record(symbol, actual_duration, session, volatility)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return {
            'total_predictions': self.metrics.total_predictions,
            'accuracy_within_30pct': self.metrics.accuracy,
            'avg_absolute_error': self.metrics.avg_absolute_error,
            'recent_mae': self.metrics.recent_mae,
            'model_version': self.model_version,
            'is_trained': self.is_trained
        }
    
    def _save_model(self):
        """Save model to disk"""
        if self.weibull_model is None and not self.is_trained:
            return
        
        try:
            model_data = {
                'weibull_model': self.weibull_model,
                'weibull_dist': self.weibull_dist,
                'lognormal_dist': self.lognormal_dist,
                'version': self.model_version,
                'trained_at': datetime.utcnow().isoformat(),
                'metrics': {
                    'total_predictions': self.metrics.total_predictions,
                    'accuracy': self.metrics.accuracy
                }
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Duration model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving duration model: {e}")
    
    def _load_model(self):
        """Load model from disk"""
        if not self.model_path.exists():
            logger.info("No existing duration model found")
            return
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.weibull_model = model_data.get('weibull_model')
            self.weibull_dist = model_data.get('weibull_dist', WeibullDistribution())
            self.lognormal_dist = model_data.get('lognormal_dist', LogNormalDistribution())
            self.model_version = model_data.get('version', 'unknown')
            self.is_trained = True
            
            logger.info(f"Loaded duration model version {self.model_version}")
            
        except Exception as e:
            logger.error(f"Error loading duration model: {e}")
            self.is_trained = False
    
    def generate_training_data(self, trade_history: List[Dict]) -> List[Dict]:
        """Generate training data from historical trade outcomes"""
        training_data = []
        
        for trade in trade_history:
            if trade.get('duration_minutes') and trade.get('outcome'):
                training_data.append({
                    'duration_minutes': trade['duration_minutes'],
                    'completed': 1 if trade['outcome'] in ['win', 'loss'] else 0,
                    'volatility': trade.get('volatility', 0),
                    'momentum': trade.get('momentum', 0),
                    'rsi': trade.get('rsi', 50),
                    'atr_pct': trade.get('atr_pct', 0),
                    'hour': trade.get('hour', 12),
                    'session_progress': trade.get('session_progress', 0.5)
                })
        
        return training_data


class DurationAnalyzer:
    """Analyze and predict multiple duration scenarios"""
    
    def __init__(self):
        self.duration_model = DurationPredictionModel()
    
    def analyze_duration_scenarios(
        self, 
        features: FeatureSet, 
        direction: SignalBias
    ) -> Dict[str, Any]:
        """Generate multiple duration scenarios with confidence intervals"""
        
        base_prediction = self.duration_model.predict(features, direction)
        
        # Generate scenarios with different confidence levels
        scenarios = {
            'optimistic': self._adjust_duration(base_prediction, 1.4),
            'base': base_prediction,
            'conservative': self._adjust_duration(base_prediction, 0.7),
            'pessimistic': self._adjust_duration(base_prediction, 0.4)
        }
        
        # Time-to-target estimates
        targets = {
            '5_pips': base_prediction.expected_time_above_minutes * 0.25,
            '10_pips': base_prediction.expected_time_above_minutes * 0.45,
            '20_pips': base_prediction.expected_time_above_minutes * 0.75,
            '30_pips': base_prediction.expected_time_above_minutes * 1.0
        }
        
        # Confidence intervals
        confidence_intervals = {
            '90_ci': (
                base_prediction.expected_time_above_minutes * 0.5,
                base_prediction.expected_time_above_minutes * 1.5
            ),
            '80_ci': (
                base_prediction.expected_time_above_minutes * 0.6,
                base_prediction.expected_time_above_minutes * 1.4
            ),
            '50_ci': (
                base_prediction.expected_time_above_minutes * 0.75,
                base_prediction.expected_time_above_minutes * 1.25
            )
        }
        
        # Risk assessment
        risk_level = self._assess_duration_risk(features, base_prediction)
        
        return {
            'scenarios': scenarios,
            'targets': targets,
            'confidence_intervals': confidence_intervals,
            'recommended_duration': base_prediction.expected_time_above_minutes,
            'confidence': base_prediction.survival_probability_10min,
            'risk_level': risk_level
        }
    
    def _adjust_duration(self, prediction: DurationPrediction, factor: float) -> DurationPrediction:
        """Adjust duration prediction by a factor"""
        return DurationPrediction(
            expected_time_above_minutes=prediction.expected_time_above_minutes * factor,
            expected_time_below_minutes=prediction.expected_time_below_minutes * factor,
            hazard_rate=prediction.hazard_rate / factor,
            survival_probability_5min=prediction.survival_probability_5min ** (1/factor),
            survival_probability_10min=prediction.survival_probability_10min ** (1/factor),
            survival_probability_15min=prediction.survival_probability_15min ** (1/factor),
            model_version=prediction.model_version
        )
    
    def _assess_duration_risk(self, features: FeatureSet, prediction: DurationPrediction) -> str:
        """Assess risk level for duration prediction"""
        risk_score = 0
        
        # Volatility risk
        vol = features.statistical.rolling_volatility_20
        if vol > 0.02:
            risk_score += 2
        elif vol > 0.015:
            risk_score += 1
        
        # RSI risk
        rsi = features.statistical.rsi
        if rsi > 70 or rsi < 30:
            risk_score += 1
        
        # Hazard rate risk
        if prediction.hazard_rate > 0.15:
            risk_score += 1
        
        # Duration risk (very short or very long)
        duration = prediction.expected_time_above_minutes
        if duration < 5 or duration > 45:
            risk_score += 1
        
        if risk_score >= 3:
            return 'high'
        elif risk_score >= 1:
            return 'medium'
        else:
            return 'low'


# Global instances
duration_model = DurationPredictionModel()
duration_analyzer = DurationAnalyzer()
