"""
Forex Probability Intelligence System - Duration Prediction Model
Survival Analysis for expected price movement duration
"""
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from loguru import logger

try:
    from lifelines import WeibullAFTFitter, CoxPHFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logger.warning("Lifelines not available, using simplified duration prediction")

from app.models.schemas import DurationPrediction, FeatureSet, SignalBias
from config.settings import get_settings

settings = get_settings()


class DurationPredictionModel:
    """Survival Analysis model for predicting price movement duration"""
    
    def __init__(self):
        self.weibull_model = None
        self.cox_model = None
        self.model_version = "1.0.0"
        self.is_trained = False
        self.baseline_duration = 10.0  # Default duration in minutes
        
        # Model path
        self.model_path = Path(__file__).parent.parent.parent / "data" / "models" / "duration_model.pkl"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing model
        self._load_model()
    
    def predict(self, features: FeatureSet, direction: SignalBias) -> DurationPrediction:
        """Predict duration of price movement"""
        
        if self.is_trained and LIFELINES_AVAILABLE:
            return self._predict_with_model(features, direction)
        else:
            return self._predict_heuristic(features, direction)
    
    def _predict_with_model(self, features: FeatureSet, direction: SignalBias) -> DurationPrediction:
        """Predict duration using trained survival model"""
        try:
            # Prepare feature dict for prediction
            feature_dict = self._prepare_features_for_survival(features)
            
            # Predict using Weibull model (if available)
            if self.weibull_model is not None:
                # Expected survival time
                expected_duration = self.weibull_model.predict_median(feature_dict)
                if np.isnan(expected_duration) or expected_duration <= 0:
                    expected_duration = self.baseline_duration
            else:
                expected_duration = self.baseline_duration
            
            # Adjust based on direction and volatility
            volatility_factor = 1.0
            if features.statistical.atr_pct > 0:
                # Higher volatility = shorter duration expected
                volatility_factor = max(0.5, min(1.5, 0.5 / features.statistical.atr_pct))
            
            expected_duration *= volatility_factor
            
            # Calculate survival probabilities at different time points
            survival_5min = self._calculate_survival_probability(expected_duration, 5)
            survival_10min = self._calculate_survival_probability(expected_duration, 10)
            survival_15min = self._calculate_survival_probability(expected_duration, 15)
            
            # Hazard rate (risk of price reversal)
            hazard_rate = 1.0 / expected_duration if expected_duration > 0 else 0.1
            
            return DurationPrediction(
                expected_time_above_minutes=expected_duration if direction == SignalBias.BUY else expected_duration * 0.7,
                expected_time_below_minutes=expected_duration if direction == SignalBias.SELL else expected_duration * 0.7,
                hazard_rate=hazard_rate,
                survival_probability_5min=survival_5min,
                survival_probability_10min=survival_10min,
                survival_probability_15min=survival_15min,
                model_version=self.model_version
            )
            
        except Exception as e:
            logger.error(f"Duration model prediction error: {e}")
            return self._predict_heuristic(features, direction)
    
    def _predict_heuristic(self, features: FeatureSet, direction: SignalBias) -> DurationPrediction:
        """Heuristic duration prediction based on market conditions"""
        
        base_duration = self.baseline_duration
        
        # Adjust based on volatility
        if features.statistical.rolling_volatility_20 > 0:
            # Higher volatility = shorter movements
            vol_adjustment = 1.0 / (1.0 + features.statistical.rolling_volatility_20 * 10)
            base_duration *= max(0.5, vol_adjustment)
        
        # Adjust based on ATR
        if features.statistical.atr_pct > 0:
            atr_adjustment = min(1.5, max(0.5, 0.3 / features.statistical.atr_pct))
            base_duration *= atr_adjustment
        
        # Adjust based on session
        hour = features.time_features.hour_of_day
        if 13 <= hour < 17:  # London-NY overlap - high volatility
            base_duration *= 0.7
        elif 8 <= hour < 12:  # London session
            base_duration *= 0.85
        elif 0 <= hour < 9:  # Tokyo session
            base_duration *= 1.2
        
        # Adjust based on momentum
        if features.statistical.momentum_decay_10 != 0:
            # Strong momentum = longer duration
            momentum_factor = 1.0 + abs(features.statistical.momentum_decay_10) * 5
            base_duration *= min(1.5, momentum_factor)
        
        # Adjust based on RSI
        rsi = features.statistical.rsi
        if 40 <= rsi <= 60:
            # Neutral RSI = normal duration
            pass
        elif rsi < 30 or rsi > 70:
            # Extreme RSI = potential reversal = shorter duration
            base_duration *= 0.7
        
        # Calculate expected duration with some variance
        expected_duration = base_duration * np.random.uniform(0.8, 1.2)
        expected_duration = max(3, min(60, expected_duration))  # Clamp between 3-60 minutes
        
        # Calculate survival probabilities (Weibull-like decay)
        survival_5min = self._calculate_survival_probability(expected_duration, 5)
        survival_10min = self._calculate_survival_probability(expected_duration, 10)
        survival_15min = self._calculate_survival_probability(expected_duration, 15)
        
        # Hazard rate
        hazard_rate = 1.0 / expected_duration
        
        return DurationPrediction(
            expected_time_above_minutes=expected_duration if direction == SignalBias.BUY else expected_duration * 0.6,
            expected_time_below_minutes=expected_duration if direction == SignalBias.SELL else expected_duration * 0.6,
            hazard_rate=hazard_rate,
            survival_probability_5min=survival_5min,
            survival_probability_10min=survival_10min,
            survival_probability_15min=survival_15min,
            model_version="heuristic_v1"
        )
    
    def _calculate_survival_probability(self, expected_duration: float, time_minutes: float) -> float:
        """Calculate survival probability using exponential decay"""
        if expected_duration <= 0:
            return 0.5
        
        # Simple exponential survival: S(t) = exp(-t/lambda)
        # where lambda is the expected duration
        survival = np.exp(-time_minutes / expected_duration)
        return float(max(0.05, min(0.95, survival)))
    
    def _prepare_features_for_survival(self, features: FeatureSet) -> Dict[str, float]:
        """Prepare features for survival model prediction"""
        return {
            'volatility': features.statistical.rolling_volatility_20,
            'momentum': features.statistical.momentum_decay_10,
            'rsi': features.statistical.rsi / 100.0,
            'atr_pct': features.statistical.atr_pct,
            'hour': features.time_features.hour_of_day,
            'session_progress': features.time_features.session_progress,
            'zscore': features.statistical.zscore
        }
    
    def train(self, duration_data: List[Dict]):
        """Train the survival model with historical duration data"""
        if not LIFELINES_AVAILABLE:
            logger.warning("Lifelines not available, skipping training")
            return False
        
        try:
            # Prepare training dataframe
            df_data = []
            for entry in duration_data:
                df_data.append({
                    'duration': entry['duration_minutes'],
                    'event': entry['completed'],  # 1 if movement completed, 0 if censored
                    'volatility': entry.get('volatility', 0),
                    'momentum': entry.get('momentum', 0),
                    'rsi': entry.get('rsi', 50) / 100.0,
                    'atr_pct': entry.get('atr_pct', 0),
                    'hour': entry.get('hour', 12),
                    'session_progress': entry.get('session_progress', 0.5)
                })
            
            import pandas as pd
            df = pd.DataFrame(df_data)
            
            # Train Weibull AFT model
            self.weibull_model = WeibullAFTFitter()
            self.weibull_model.fit(df, duration_col='duration', event_col='event')
            
            self.is_trained = True
            self.model_version = f"trained_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            self._save_model()
            
            logger.info("Duration model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Duration model training error: {e}")
            return False
    
    def _save_model(self):
        """Save model to disk"""
        if self.weibull_model is None:
            return
        
        try:
            model_data = {
                'weibull_model': self.weibull_model,
                'version': self.model_version,
                'trained_at': datetime.utcnow().isoformat()
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
        """Generate multiple duration scenarios"""
        
        base_prediction = self.duration_model.predict(features, direction)
        
        scenarios = {
            'optimistic': self._adjust_duration(base_prediction, 1.3),
            'base': base_prediction,
            'conservative': self._adjust_duration(base_prediction, 0.7),
            'pessimistic': self._adjust_duration(base_prediction, 0.4)
        }
        
        # Time-to-target estimates
        targets = {
            '5_pips': base_prediction.expected_time_above_minutes * 0.3,
            '10_pips': base_prediction.expected_time_above_minutes * 0.5,
            '20_pips': base_prediction.expected_time_above_minutes * 0.8,
            '30_pips': base_prediction.expected_time_above_minutes * 1.0
        }
        
        return {
            'scenarios': scenarios,
            'targets': targets,
            'recommended_duration': base_prediction.expected_time_above_minutes,
            'confidence': base_prediction.survival_probability_10min
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


# Global instances
duration_model = DurationPredictionModel()
duration_analyzer = DurationAnalyzer()
