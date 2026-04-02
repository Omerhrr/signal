"""
Forex Probability Intelligence System - Direction Prediction Model
Enhanced XGBoost-based prediction for price direction with ensemble methods,
advanced feature engineering, and confidence calibration.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import json
from loguru import logger
from dataclasses import dataclass, field
from collections import deque
import scipy.stats as stats

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, using advanced heuristic prediction")

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from app.models.schemas import DirectionPrediction, SignalBias, FeatureSet
from app.engines.feature_engine import feature_engine
from config.settings import get_settings

settings = get_settings()


@dataclass
class ModelMetrics:
    """Track model performance metrics"""
    total_predictions: int = 0
    correct_predictions: int = 0
    avg_confidence: float = 0.0
    calibration_error: float = 0.0
    recent_accuracy: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, was_correct: bool, confidence: float):
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1
        self.recent_accuracy.append(1.0 if was_correct else 0.0)
        self.avg_confidence = (self.avg_confidence * (self.total_predictions - 1) + confidence) / self.total_predictions
    
    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions
    
    @property
    def recent_performance(self) -> float:
        if len(self.recent_accuracy) == 0:
            return 0.5
        return np.mean(list(self.recent_accuracy))


class FeatureImportanceTracker:
    """Track and analyze feature importance over time"""
    
    def __init__(self):
        self.importance_history: Dict[str, deque] = {}
        self.current_importance: Dict[str, float] = {}
    
    def update(self, feature_names: List[str], importances: np.ndarray):
        """Update feature importance tracking"""
        for name, imp in zip(feature_names, importances):
            if name not in self.importance_history:
                self.importance_history[name] = deque(maxlen=50)
            self.importance_history[name].append(float(imp))
            self.current_importance[name] = float(imp)
    
    def get_stable_importance(self) -> Dict[str, float]:
        """Get averaged feature importance"""
        stable = {}
        for name, history in self.importance_history.items():
            if len(history) >= 5:
                stable[name] = float(np.mean(list(history)))
            else:
                stable[name] = self.current_importance.get(name, 0.0)
        return stable
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features"""
        stable = self.get_stable_importance()
        sorted_features = sorted(stable.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]


class ConfidenceCalibrator:
    """Calibrate model confidence to match actual accuracy"""
    
    def __init__(self):
        self.calibration_data: deque = deque(maxlen=1000)
        self.isotonic_regressor = None
        self.calibration_curve: Optional[np.ndarray] = None
    
    def add_observation(self, predicted_prob: float, actual_outcome: bool):
        """Add a calibration observation"""
        self.calibration_data.append((predicted_prob, 1.0 if actual_outcome else 0.0))
    
    def calibrate(self, confidence: float) -> float:
        """Calibrate a confidence value"""
        if len(self.calibration_data) < 50:
            return confidence
        
        if self.isotonic_regressor is None and SKLEARN_AVAILABLE:
            try:
                self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
                probs, outcomes = zip(*self.calibration_data)
                self.isotonic_regressor.fit(np.array(probs), np.array(outcomes))
            except Exception as e:
                logger.warning(f"Calibration fitting failed: {e}")
                return confidence
        
        if self.isotonic_regressor is not None:
            try:
                calibrated = self.isotonic_regressor.predict(np.array([confidence]))[0]
                return float(calibrated)
            except Exception:
                return confidence
        
        return confidence
    
    def get_calibration_metrics(self) -> Dict[str, float]:
        """Get calibration performance metrics"""
        if len(self.calibration_data) < 20:
            return {'calibration_error': 0.0, 'samples': 0}
        
        # Calculate expected calibration error
        probs, outcomes = zip(*self.calibration_data)
        probs = np.array(probs)
        outcomes = np.array(outcomes)
        
        # Bin predictions
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        calibration_error = 0.0
        
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_confidence = np.mean(probs[mask])
                bin_accuracy = np.mean(outcomes[mask])
                calibration_error += np.abs(bin_confidence - bin_accuracy) * np.sum(mask)
        
        calibration_error /= len(probs)
        
        return {
            'calibration_error': float(calibration_error),
            'samples': len(self.calibration_data)
        }


class DirectionPredictionModel:
    """Enhanced XGBoost model for predicting price direction"""
    
    def __init__(self):
        self.model = None
        self.model_version = "2.0.0"
        self.feature_names: List[str] = []
        self.is_trained = False
        self.training_stats: Dict[str, Any] = {}
        
        # Enhanced components
        self.metrics = ModelMetrics()
        self.feature_tracker = FeatureImportanceTracker()
        self.calibrator = ConfidenceCalibrator()
        
        # Initialize model path
        self.model_path = Path(__file__).parent.parent.parent / "data" / "models" / "direction_model.pkl"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing model
        self._load_model()
    
    def predict(self, features: FeatureSet) -> DirectionPrediction:
        """Predict price direction from features with enhanced analysis"""
        
        # Get feature vector
        feature_vector = feature_engine.get_feature_vector(features)
        
        if self.is_trained and self.model is not None and XGBOOST_AVAILABLE:
            prediction = self._predict_with_model(feature_vector)
        else:
            prediction = self._predict_heuristic(features)
        
        # Apply calibration
        calibrated_confidence = self.calibrator.calibrate(prediction.confidence)
        
        # Adjust confidence based on recent model performance
        performance_adjustment = self.metrics.recent_performance
        if performance_adjustment > 0.5:
            # Model is performing well, maintain confidence
            adjusted_confidence = calibrated_confidence
        else:
            # Model is underperforming, reduce confidence
            adjusted_confidence = calibrated_confidence * (0.7 + 0.3 * performance_adjustment * 2)
        
        # Update prediction with adjusted values
        prediction.confidence = float(max(0.2, min(0.95, adjusted_confidence)))
        
        # Recalculate probabilities if needed
        if prediction.predicted_direction == SignalBias.BUY:
            prediction.prob_up = prediction.confidence
            prediction.prob_down = 1.0 - prediction.confidence
        else:
            prediction.prob_down = prediction.confidence
            prediction.prob_up = 1.0 - prediction.confidence
        
        return prediction
    
    def _predict_with_model(self, feature_vector: np.ndarray) -> DirectionPrediction:
        """Make prediction using trained XGBoost model"""
        try:
            # Reshape for single prediction
            X = feature_vector.reshape(1, -1)
            
            # Get probability predictions
            proba = self.model.predict_proba(X)[0]
            
            # proba[0] = prob_down, proba[1] = prob_up
            prob_up = float(proba[1])
            prob_down = float(proba[0])
            
            # Determine direction
            if prob_up > prob_down:
                direction = SignalBias.BUY
                confidence = prob_up
            else:
                direction = SignalBias.SELL
                confidence = prob_down
            
            # Update feature importance tracking
            if hasattr(self.model, 'feature_importances_'):
                self.feature_tracker.update(
                    self.feature_names or [f"f{i}" for i in range(len(feature_vector))],
                    self.model.feature_importances_
                )
            
            return DirectionPrediction(
                prob_up=prob_up,
                prob_down=prob_down,
                predicted_direction=direction,
                confidence=confidence,
                model_version=self.model_version
            )
            
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return self._predict_heuristic(None)
    
    def _predict_heuristic(self, features: Optional[FeatureSet]) -> DirectionPrediction:
        """Enhanced heuristic prediction when model not available"""
        
        if features is None:
            # Random prediction with slight bullish bias (market tendency)
            prob_up = np.random.uniform(0.45, 0.65)
            prob_down = 1 - prob_up
        else:
            # Use comprehensive feature-based heuristics
            score = 0.5
            confidence_weight = 0.0
            
            # ===== PRICE ACTION SIGNALS =====
            
            # 1. Break of structure signal (strong)
            if features.price_action.break_of_structure:
                bos_strength = min(features.price_action.bos_strength / 10, 1.0)
                if features.price_action.bos_direction == "BUY":
                    score += 0.15 * bos_strength
                elif features.price_action.bos_direction == "SELL":
                    score -= 0.15 * bos_strength
                confidence_weight += 0.2
            
            # 2. Wick rejection signal
            if features.price_action.wick_rejection_strength > 0.5:
                wick_strength = features.price_action.wick_rejection_strength
                if features.price_action.wick_rejection_type == "bullish":
                    score += 0.1 * wick_strength
                elif features.price_action.wick_rejection_type == "bearish":
                    score -= 0.1 * wick_strength
                confidence_weight += 0.1
            
            # 3. Swing points analysis
            if features.price_action.higher_highs > features.price_action.lower_lows:
                score += 0.05
            elif features.price_action.lower_lows > features.price_action.higher_highs:
                score -= 0.05
            
            # 4. Range expansion
            if features.price_action.range_expansion_pct > 30:
                # Strong momentum in current direction
                confidence_weight += 0.1
            
            # ===== STATISTICAL SIGNALS =====
            
            # 5. RSI signal
            rsi = features.statistical.rsi
            if rsi < 25:
                score += 0.15  # Strong oversold
                confidence_weight += 0.15
            elif rsi < 30:
                score += 0.1   # Oversold
                confidence_weight += 0.1
            elif rsi > 75:
                score -= 0.15  # Strong overbought
                confidence_weight += 0.15
            elif rsi > 70:
                score -= 0.1   # Overbought
                confidence_weight += 0.1
            
            # 6. Z-score mean reversion
            zscore = features.statistical.zscore
            if zscore < -2:
                score += 0.12  # Price far below mean
                confidence_weight += 0.1
            elif zscore < -1.5:
                score += 0.08
            elif zscore > 2:
                score -= 0.12  # Price far above mean
                confidence_weight += 0.1
            elif zscore > 1.5:
                score -= 0.08
            
            # 7. Momentum analysis
            momentum = features.statistical.momentum_decay_10
            if abs(momentum) > 0.01:
                if momentum > 0:
                    score += 0.08
                else:
                    score -= 0.08
                confidence_weight += 0.15
            
            # 8. Volatility analysis
            volatility = features.statistical.rolling_volatility_20
            if volatility > 0.02:
                # High volatility - reduce confidence
                confidence_weight *= 0.8
            elif volatility < 0.005:
                # Low volatility - clearer direction
                confidence_weight *= 1.1
            
            # 9. ATR analysis
            atr_pct = features.statistical.atr_pct
            if atr_pct > 0.05:
                confidence_weight *= 0.85  # High volatility
            
            # ===== TIME-BASED SIGNALS =====
            
            # 10. Session adjustments
            hour = features.time_features.hour_of_day
            session = features.time_features.current_session
            
            # London-NY overlap - high volatility, less predictable
            if 13 <= hour < 17:
                confidence_weight *= 0.9
            
            # Asian session - more predictable
            elif 0 <= hour < 8:
                confidence_weight *= 1.1
            
            # Weekend penalty
            if features.time_features.is_weekend:
                confidence_weight *= 0.5
            
            # ===== RAW FEATURES =====
            
            # 11. Price relative to moving average
            price_to_sma20 = features.raw_features.get('price_to_sma20', 0)
            if abs(price_to_sma20) > 0.01:
                if price_to_sma20 > 0:
                    score += 0.03  # Above MA
                else:
                    score -= 0.03  # Below MA
            
            # 12. Volume confirmation
            volume_ratio = features.raw_features.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:
                confidence_weight += 0.1  # High volume confirms
            
            # ===== FINAL SCORE CALCULATION =====
            
            # Add controlled noise
            noise = np.random.uniform(-0.05, 0.05)
            score += noise
            
            # Calculate final probabilities
            prob_up = max(0.15, min(0.85, score))
            prob_down = 1.0 - prob_up
            
            # Calculate confidence based on signal strength
            prob_diff = abs(prob_up - prob_down)
            base_confidence = max(prob_up, prob_down)
            
            # Adjust confidence by weight
            final_confidence = base_confidence * (0.7 + 0.3 * min(confidence_weight, 1.0))
            final_confidence = max(0.4, min(0.85, final_confidence))
        
        direction = SignalBias.BUY if prob_up > prob_down else SignalBias.SELL
        confidence = final_confidence if 'final_confidence' in dir() else max(prob_up, prob_down)
        
        return DirectionPrediction(
            prob_up=prob_up,
            prob_down=prob_down,
            predicted_direction=direction,
            confidence=confidence,
            model_version="heuristic_v2"
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Train the XGBoost model with enhanced configuration"""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping training")
            return False
        
        try:
            logger.info(f"Training direction model with {len(X)} samples")
            
            # Store feature names
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            
            # Enhanced XGBoost parameters
            params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            
            # Update with settings if available
            if hasattr(settings, 'xgboost_params'):
                params.update(settings.xgboost_params)
            
            # Create and train model
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X, y, verbose=False)
            
            self.is_trained = True
            self.model_version = f"trained_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Store training stats
            self.training_stats = {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
                'trained_at': datetime.utcnow().isoformat()
            }
            
            # Update feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_tracker.update(self.feature_names, self.model.feature_importances_)
            
            # Save model
            self._save_model()
            
            logger.info("Direction model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def record_outcome(self, prediction: DirectionPrediction, actual_direction: SignalBias):
        """Record prediction outcome for performance tracking"""
        was_correct = prediction.predicted_direction == actual_direction
        self.metrics.update(was_correct, prediction.confidence)
        self.calibrator.add_observation(prediction.confidence, was_correct)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        calibration = self.calibrator.get_calibration_metrics()
        top_features = self.feature_tracker.get_top_features(10)
        
        return {
            'accuracy': self.metrics.accuracy,
            'recent_performance': self.metrics.recent_performance,
            'total_predictions': self.metrics.total_predictions,
            'avg_confidence': self.metrics.avg_confidence,
            'calibration_error': calibration['calibration_error'],
            'top_features': top_features,
            'model_version': self.model_version,
            'is_trained': self.is_trained
        }
    
    def _save_model(self):
        """Save model to disk with enhanced metadata"""
        if self.model is None:
            return
        
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'version': self.model_version,
                'training_stats': self.training_stats,
                'trained_at': datetime.utcnow().isoformat(),
                'performance_metrics': self.get_performance_metrics()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self):
        """Load model from disk"""
        if not self.model_path.exists():
            logger.info("No existing model found, will use heuristic predictions")
            return
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            self.feature_names = model_data.get('feature_names', [])
            self.model_version = model_data.get('version', 'unknown')
            self.training_stats = model_data.get('training_stats', {})
            self.is_trained = True
            
            logger.info(f"Loaded model version {self.model_version}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_trained = False
    
    def generate_training_data(self, ohlcv_data: List[Dict], lookahead: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data from historical OHLCV data with enhanced features"""
        X_list = []
        y_list = []
        
        for data in ohlcv_data:
            symbol = data['symbol']
            candles = data['candles']
            
            if len(candles) < lookahead + 50:
                continue
            
            for i in range(50, len(candles) - lookahead):
                # Get features up to current candle
                current_candles = candles[:i]
                features = feature_engine.calculate_features(symbol, current_candles)
                feature_vector = feature_engine.get_feature_vector(features)
                
                # Determine label: price went up or down in lookahead period
                current_close = candles[i-1]['close']
                future_close = candles[i + lookahead - 1]['close']
                label = 1 if future_close > current_close else 0
                
                X_list.append(feature_vector)
                y_list.append(label)
        
        if not X_list:
            return np.array([]), np.array([])
        
        return np.array(X_list), np.array(y_list)


class EnsembleDirectionModel:
    """Ensemble of multiple direction prediction models with adaptive weighting"""
    
    def __init__(self):
        self.models: List[DirectionPredictionModel] = []
        self.weights: List[float] = []
        self.model_performance: Dict[int, deque] = {}
    
    def add_model(self, model: DirectionPredictionModel, weight: float = 1.0):
        """Add a model to the ensemble"""
        idx = len(self.models)
        self.models.append(model)
        self.weights.append(weight)
        self.model_performance[idx] = deque(maxlen=50)
    
    def predict(self, features: FeatureSet) -> DirectionPrediction:
        """Get ensemble prediction with adaptive weighting"""
        if not self.models:
            # Return default heuristic prediction
            return DirectionPredictionModel().predict(features)
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(features)
            predictions.append(pred)
        
        # Adaptive weighting based on recent performance
        adaptive_weights = []
        for i, (pred, base_weight) in enumerate(zip(predictions, self.weights)):
            performance = self._get_model_performance(i)
            # Adjust weight by performance
            adjusted_weight = base_weight * (0.5 + performance)
            adaptive_weights.append(adjusted_weight)
        
        # Normalize weights
        total_weight = sum(adaptive_weights)
        adaptive_weights = [w / total_weight for w in adaptive_weights]
        
        # Weighted average of probabilities
        prob_up = sum(p.prob_up * w for p, w in zip(predictions, adaptive_weights))
        prob_down = sum(p.prob_down * w for p, w in zip(predictions, adaptive_weights))
        
        # Normalize
        total = prob_up + prob_down
        if total > 0:
            prob_up /= total
            prob_down /= total
        
        direction = SignalBias.BUY if prob_up > prob_down else SignalBias.SELL
        confidence = max(prob_up, prob_down)
        
        # Calculate model agreement
        agreement = self._calculate_agreement(predictions)
        
        return DirectionPrediction(
            prob_up=prob_up,
            prob_down=prob_down,
            predicted_direction=direction,
            confidence=confidence,
            model_version=f"ensemble_v2_agreement_{agreement:.2f}"
        )
    
    def _get_model_performance(self, model_idx: int) -> float:
        """Get recent performance for a model"""
        if model_idx not in self.model_performance:
            return 0.5
        
        history = self.model_performance[model_idx]
        if len(history) < 5:
            return 0.5
        
        return np.mean(list(history))
    
    def _calculate_agreement(self, predictions: List[DirectionPrediction]) -> float:
        """Calculate agreement between models"""
        if not predictions:
            return 0.5
        
        directions = [p.predicted_direction for p in predictions]
        buy_count = sum(1 for d in directions if d == SignalBias.BUY)
        sell_count = sum(1 for d in directions if d == SignalBias.SELL)
        
        agreement = max(buy_count, sell_count) / len(predictions)
        return agreement
    
    def record_outcome(self, actual_direction: SignalBias):
        """Record outcomes for all models"""
        for i, model in enumerate(self.models):
            if hasattr(model, 'metrics'):
                was_correct = model.metrics.recent_performance > 0.5
                self.model_performance[i].append(1.0 if was_correct else 0.0)


# Global instance
direction_model = DirectionPredictionModel()
