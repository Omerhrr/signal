"""
Forex Probability Intelligence System - Direction Prediction Model
XGBoost-based prediction for price direction
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import json
from loguru import logger

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, using simple prediction")

from app.models.schemas import DirectionPrediction, SignalBias, FeatureSet
from app.engines.feature_engine import feature_engine
from config.settings import get_settings

settings = get_settings()


class DirectionPredictionModel:
    """XGBoost model for predicting price direction"""
    
    def __init__(self):
        self.model = None
        self.model_version = "1.0.0"
        self.feature_names: List[str] = []
        self.is_trained = False
        self.training_stats: Dict[str, Any] = {}
        
        # Initialize model path
        self.model_path = Path(__file__).parent.parent.parent / "data" / "models" / "direction_model.pkl"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing model
        self._load_model()
    
    def predict(self, features: FeatureSet) -> DirectionPrediction:
        """Predict price direction from features"""
        
        # Get feature vector
        feature_vector = feature_engine.get_feature_vector(features)
        
        if self.is_trained and self.model is not None and XGBOOST_AVAILABLE:
            return self._predict_with_model(feature_vector)
        else:
            return self._predict_heuristic(features)
    
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
        """Heuristic prediction when model not available"""
        
        if features is None:
            # Random prediction with slight bullish bias (market tendency)
            prob_up = np.random.uniform(0.45, 0.65)
            prob_down = 1 - prob_up
        else:
            # Use feature-based heuristics
            score = 0.5
            
            # Break of structure signal
            if features.price_action.break_of_structure:
                if features.price_action.bos_direction == "BUY":
                    score += 0.15 * features.price_action.bos_strength
                elif features.price_action.bos_direction == "SELL":
                    score -= 0.15 * features.price_action.bos_strength
            
            # Wick rejection signal
            if features.price_action.wick_rejection_strength > 0.5:
                if features.price_action.wick_rejection_type == "bullish":
                    score += 0.1
                elif features.price_action.wick_rejection_type == "bearish":
                    score -= 0.1
            
            # RSI signal
            rsi = features.statistical.rsi
            if rsi < 30:
                score += 0.1  # Oversold - bullish
            elif rsi > 70:
                score -= 0.1  # Overbought - bearish
            
            # Mean reversion
            if features.statistical.zscore < -2:
                score += 0.1  # Price below mean - potential bounce
            elif features.statistical.zscore > 2:
                score -= 0.1  # Price above mean - potential drop
            
            # Session adjustment
            hour = features.time_features.hour_of_day
            if 8 <= hour < 12:  # London morning
                score += 0.05  # Slightly more volatile/bullish
            
            # Add some noise
            score += np.random.uniform(-0.1, 0.1)
            
            prob_up = max(0.1, min(0.9, score))
            prob_down = 1 - prob_up
        
        direction = SignalBias.BUY if prob_up > prob_down else SignalBias.SELL
        confidence = max(prob_up, prob_down)
        
        return DirectionPrediction(
            prob_up=prob_up,
            prob_down=prob_down,
            predicted_direction=direction,
            confidence=confidence,
            model_version="heuristic_v1"
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Train the XGBoost model"""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping training")
            return False
        
        try:
            logger.info(f"Training direction model with {len(X)} samples")
            
            # Store feature names
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            
            # Create and train model
            self.model = xgb.XGBClassifier(**settings.xgboost_params)
            self.model.fit(X, y)
            
            self.is_trained = True
            self.model_version = f"trained_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Save model
            self._save_model()
            
            logger.info("Direction model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def _save_model(self):
        """Save model to disk"""
        if self.model is None:
            return
        
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'version': self.model_version,
                'training_stats': self.training_stats,
                'trained_at': datetime.utcnow().isoformat()
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
        """Generate training data from historical OHLCV data"""
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
    """Ensemble of multiple direction prediction models"""
    
    def __init__(self):
        self.models: List[DirectionPredictionModel] = []
        self.weights: List[float] = []
        
    def add_model(self, model: DirectionPredictionModel, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.weights.append(weight)
    
    def predict(self, features: FeatureSet) -> DirectionPrediction:
        """Get ensemble prediction"""
        if not self.models:
            # Return default heuristic prediction
            return DirectionPredictionModel().predict(features)
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(features)
            predictions.append(pred)
        
        # Weighted average of probabilities
        total_weight = sum(self.weights)
        prob_up = sum(p.prob_up * w for p, w in zip(predictions, self.weights)) / total_weight
        prob_down = sum(p.prob_down * w for p, w in zip(predictions, self.weights)) / total_weight
        
        # Normalize
        total = prob_up + prob_down
        if total > 0:
            prob_up /= total
            prob_down /= total
        
        direction = SignalBias.BUY if prob_up > prob_down else SignalBias.SELL
        confidence = max(prob_up, prob_down)
        
        return DirectionPrediction(
            prob_up=prob_up,
            prob_down=prob_down,
            predicted_direction=direction,
            confidence=confidence,
            model_version="ensemble_v1"
        )


# Global instance
direction_model = DirectionPredictionModel()
