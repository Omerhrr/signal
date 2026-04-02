"""
Forex Probability Intelligence System - Hidden Markov Model Engine
Market regime detection using HMM for trend/range identification
"""
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from loguru import logger
from dataclasses import dataclass
from collections import deque
from enum import Enum

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    logger.warning("hmmlearn not available, using simplified regime detection")

from app.models.schemas import FeatureSet, OHLCV
from config.settings import get_settings

settings = get_settings()


class MarketRegime(str, Enum):
    """Market regime states"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


@dataclass
class RegimeState:
    """Current regime state information"""
    regime: MarketRegime
    probability: float
    duration_expected: float  # Expected duration in this state
    transition_probabilities: Dict[str, float]
    confidence: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'regime': self.regime.value,
            'probability': self.probability,
            'duration_expected': self.duration_expected,
            'transition_probabilities': self.transition_probabilities,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class HMMFeatures:
    """Features for HMM model"""
    returns: float
    volatility: float
    momentum: float
    volume_change: float
    range_ratio: float
    rsi_normalized: float
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.returns,
            self.volatility,
            self.momentum,
            self.volume_change,
            self.range_ratio,
            self.rsi_normalized
        ])


class HMMRegimeDetector:
    """Hidden Markov Model for market regime detection"""
    
    def __init__(self, n_states: int = 6):
        self.n_states = n_states
        self.model = None
        self.is_trained = False
        self.model_version = "1.0.0"
        
        # State mapping
        self.state_mapping = {
            0: MarketRegime.TRENDING_UP,
            1: MarketRegime.TRENDING_DOWN,
            2: MarketRegime.RANGING,
            3: MarketRegime.VOLATILE,
            4: MarketRegime.BREAKOUT,
            5: MarketRegime.REVERSAL
        }
        
        # History for training
        self.feature_history: deque = deque(maxlen=1000)
        
        # Model path
        self.model_path = Path(__file__).parent.parent.parent / "data" / "models" / "hmm_model.pkl"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing model
        self._load_model()
    
    def extract_features(self, ohlcv: List[OHLCV]) -> HMMFeatures:
        """Extract features for HMM from OHLCV data"""
        if len(ohlcv) < 20:
            return HMMFeatures(0, 0, 0, 0, 0, 0.5)
        
        closes = np.array([c.close for c in ohlcv])
        highs = np.array([c.high for c in ohlcv])
        lows = np.array([c.low for c in ohlcv])
        volumes = np.array([c.volume for c in ohlcv])
        
        # Returns (log returns)
        returns = np.log(closes[-1] / closes[-2]) if closes[-2] > 0 else 0
        
        # Volatility (rolling std of returns)
        log_returns = np.diff(np.log(closes[-20:]))
        volatility = np.std(log_returns) if len(log_returns) > 1 else 0
        
        # Momentum (price change rate)
        momentum = (closes[-1] - closes[-10]) / closes[-10] if closes[-10] > 0 else 0
        
        # Volume change
        vol_mean = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
        volume_change = (volumes[-1] - vol_mean) / vol_mean if vol_mean > 0 else 0
        
        # Range ratio (current range vs average range)
        current_range = highs[-1] - lows[-1]
        avg_range = np.mean(highs[-20:] - lows[-20:])
        range_ratio = current_range / avg_range if avg_range > 0 else 1
        
        # RSI normalized (0-1)
        rsi = self._calculate_rsi(closes[-15:])
        rsi_normalized = rsi / 100.0
        
        return HMMFeatures(
            returns=float(returns),
            volatility=float(volatility),
            momentum=float(momentum),
            volume_change=float(volume_change),
            range_ratio=float(range_ratio),
            rsi_normalized=float(rsi_normalized)
        )
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def detect_regime(self, ohlcv: List[OHLCV]) -> RegimeState:
        """Detect current market regime using HMM"""
        features = self.extract_features(ohlcv)
        feature_array = features.to_array().reshape(1, -1)
        
        # Store in history
        self.feature_history.append(features.to_array())
        
        if self.is_trained and self.model is not None and HMMLEARN_AVAILABLE:
            return self._detect_with_model(feature_array)
        else:
            return self._detect_heuristic(features, ohlcv)
    
    def _detect_with_model(self, feature_array: np.ndarray) -> RegimeState:
        """Detect regime using trained HMM model"""
        try:
            # Get state probabilities
            state_probs = self.model.predict_proba(feature_array)[0]
            predicted_state = np.argmax(state_probs)
            
            regime = self.state_mapping.get(predicted_state, MarketRegime.RANGING)
            probability = float(state_probs[predicted_state])
            
            # Calculate transition probabilities
            transition_probs = {}
            for i, prob in enumerate(state_probs):
                state_regime = self.state_mapping.get(i, MarketRegime.RANGING)
                transition_probs[state_regime.value] = float(prob)
            
            # Estimate expected duration in this state
            # From transition matrix diagonal
            duration = 1.0 / (1.0 - self.model.transmat_[predicted_state, predicted_state])
            
            return RegimeState(
                regime=regime,
                probability=probability,
                duration_expected=float(duration),
                transition_probabilities=transition_probs,
                confidence=min(1.0, probability * 1.5),
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"HMM detection error: {e}")
            return self._detect_heuristic(None, None)
    
    def _detect_heuristic(self, features: HMMFeatures, ohlcv: List[OHLCV]) -> RegimeState:
        """Heuristic regime detection when model not available"""
        
        if features is None:
            return RegimeState(
                regime=MarketRegime.RANGING,
                probability=0.5,
                duration_expected=10.0,
                transition_probabilities={r.value: 1/6 for r in MarketRegime},
                confidence=0.3,
                timestamp=datetime.now(timezone.utc)
            )
        
        # Score each regime
        scores = {r: 0.0 for r in MarketRegime}
        
        # Trending Up
        if features.momentum > 0.005:
            scores[MarketRegime.TRENDING_UP] += 0.3
        if features.returns > 0:
            scores[MarketRegime.TRENDING_UP] += 0.2
        if features.rsi_normalized > 0.55:
            scores[MarketRegime.TRENDING_UP] += 0.1
        
        # Trending Down
        if features.momentum < -0.005:
            scores[MarketRegime.TRENDING_DOWN] += 0.3
        if features.returns < 0:
            scores[MarketRegime.TRENDING_DOWN] += 0.2
        if features.rsi_normalized < 0.45:
            scores[MarketRegime.TRENDING_DOWN] += 0.1
        
        # Ranging
        if abs(features.momentum) < 0.003:
            scores[MarketRegime.RANGING] += 0.3
        if features.volatility < 0.005:
            scores[MarketRegime.RANGING] += 0.2
        if 0.4 <= features.rsi_normalized <= 0.6:
            scores[MarketRegime.RANGING] += 0.2
        
        # Volatile
        if features.volatility > 0.01:
            scores[MarketRegime.VOLATILE] += 0.4
        if features.range_ratio > 1.5:
            scores[MarketRegime.VOLATILE] += 0.2
        
        # Breakout
        if features.range_ratio > 1.3:
            scores[MarketRegime.BREAKOUT] += 0.3
        if abs(features.volume_change) > 0.3:
            scores[MarketRegime.BREAKOUT] += 0.2
        if abs(features.momentum) > 0.008:
            scores[MarketRegime.BREAKOUT] += 0.2
        
        # Reversal
        if features.rsi_normalized > 0.75 or features.rsi_normalized < 0.25:
            scores[MarketRegime.REVERSAL] += 0.3
        if features.volatility > 0.008:
            scores[MarketRegime.REVERSAL] += 0.1
        
        # Find best regime
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]
        
        # Normalize probabilities
        total_score = sum(scores.values())
        transition_probs = {r.value: s/total_score for r, s in scores.items()} if total_score > 0 else {r.value: 1/6 for r in MarketRegime}
        
        # Estimate duration based on regime
        duration_map = {
            MarketRegime.TRENDING_UP: 30.0,
            MarketRegime.TRENDING_DOWN: 30.0,
            MarketRegime.RANGING: 60.0,
            MarketRegime.VOLATILE: 10.0,
            MarketRegime.BREAKOUT: 5.0,
            MarketRegime.REVERSAL: 15.0
        }
        
        return RegimeState(
            regime=best_regime,
            probability=min(0.9, best_score + 0.3),
            duration_expected=duration_map.get(best_regime, 20.0),
            transition_probabilities=transition_probs,
            confidence=min(0.8, best_score + 0.2),
            timestamp=datetime.now(timezone.utc)
        )
    
    def train(self, ohlcv_data: List[List[OHLCV]]) -> bool:
        """Train HMM model on historical data"""
        if not HMMLEARN_AVAILABLE:
            logger.warning("hmmlearn not available, skipping training")
            return False
        
        try:
            # Extract features from all data
            all_features = []
            for ohlcv_list in ohlcv_data:
                for i in range(20, len(ohlcv_list)):
                    features = self.extract_features(ohlcv_list[:i+1])
                    all_features.append(features.to_array())
            
            if len(all_features) < 100:
                logger.warning("Not enough data for HMM training")
                return False
            
            X = np.array(all_features)
            
            # Normalize features
            self.feature_mean = np.mean(X, axis=0)
            self.feature_std = np.std(X, axis=0) + 1e-8
            X_normalized = (X - self.feature_mean) / self.feature_std
            
            # Create and train Gaussian HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type='full',
                n_iter=100,
                random_state=42
            )
            
            self.model.fit(X_normalized)
            self.is_trained = True
            self.model_version = f"trained_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            # Save model
            self._save_model()
            
            logger.info(f"HMM model trained with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"HMM training error: {e}")
            return False
    
    def _save_model(self):
        """Save HMM model to disk"""
        if self.model is None:
            return
        
        try:
            model_data = {
                'model': self.model,
                'n_states': self.n_states,
                'state_mapping': self.state_mapping,
                'feature_mean': getattr(self, 'feature_mean', None),
                'feature_std': getattr(self, 'feature_std', None),
                'version': self.model_version,
                'trained_at': datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"HMM model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving HMM model: {e}")
    
    def _load_model(self):
        """Load HMM model from disk"""
        if not self.model_path.exists():
            logger.info("No existing HMM model found")
            return
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            self.n_states = model_data.get('n_states', 6)
            self.state_mapping = model_data.get('state_mapping', self.state_mapping)
            self.feature_mean = model_data.get('feature_mean')
            self.feature_std = model_data.get('feature_std')
            self.model_version = model_data.get('version', 'unknown')
            self.is_trained = True
            
            logger.info(f"Loaded HMM model version {self.model_version}")
            
        except Exception as e:
            logger.error(f"Error loading HMM model: {e}")
            self.is_trained = False
    
    def get_regime_transition_forecast(self, current_regime: MarketRegime, steps: int = 3) -> List[Dict]:
        """Forecast regime transitions"""
        if not self.is_trained or self.model is None:
            return []
        
        try:
            # Get current state index
            current_idx = None
            for idx, regime in self.state_mapping.items():
                if regime == current_regime:
                    current_idx = idx
                    break
            
            if current_idx is None:
                return []
            
            forecasts = []
            state_probs = np.zeros(self.n_states)
            state_probs[current_idx] = 1.0
            
            for step in range(1, steps + 1):
                # Multiply by transition matrix
                state_probs = state_probs @ self.model.transmat_
                
                # Get most likely state
                most_likely = np.argmax(state_probs)
                regime = self.state_mapping.get(most_likely, MarketRegime.RANGING)
                
                forecasts.append({
                    'step': step,
                    'regime': regime.value,
                    'probability': float(state_probs[most_likely]),
                    'all_probabilities': {
                        self.state_mapping[i].value: float(p) 
                        for i, p in enumerate(state_probs)
                    }
                })
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Forecast error: {e}")
            return []


class RegimeAnalyzer:
    """Analyze regime patterns and provide insights"""
    
    def __init__(self):
        self.hmm_detector = HMMRegimeDetector()
        self.regime_history: deque = deque(maxlen=100)
    
    def analyze(self, ohlcv: List[OHLCV]) -> Dict[str, Any]:
        """Comprehensive regime analysis"""
        current_state = self.hmm_detector.detect_regime(ohlcv)
        
        # Store in history
        self.regime_history.append({
            'regime': current_state.regime.value,
            'probability': current_state.probability,
            'timestamp': current_state.timestamp.isoformat()
        })
        
        # Get forecasts
        forecasts = self.hmm_detector.get_regime_transition_forecast(current_state.regime)
        
        # Analyze regime stability
        stability = self._analyze_stability()
        
        return {
            'current_state': current_state.to_dict(),
            'forecasts': forecasts,
            'stability': stability,
            'trading_implications': self._get_trading_implications(current_state)
        }
    
    def _analyze_stability(self) -> Dict[str, Any]:
        """Analyze regime stability from history"""
        if len(self.regime_history) < 5:
            return {'score': 0.5, 'regime_changes': 0}
        
        regimes = [h['regime'] for h in self.regime_history]
        
        # Count regime changes
        changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
        
        # Stability score (fewer changes = more stable)
        stability_score = 1.0 - (changes / (len(regimes) - 1))
        
        return {
            'score': stability_score,
            'regime_changes': changes,
            'current_regime_duration': self._get_current_duration()
        }
    
    def _get_current_duration(self) -> int:
        """Get duration of current regime in observations"""
        if not self.regime_history:
            return 0
        
        current = self.regime_history[-1]['regime']
        duration = 1
        
        for i in range(len(self.regime_history) - 2, -1, -1):
            if self.regime_history[i]['regime'] == current:
                duration += 1
            else:
                break
        
        return duration
    
    def _get_trading_implications(self, state: RegimeState) -> Dict[str, Any]:
        """Get trading implications for current regime"""
        implications = {
            'trend_follow': False,
            'counter_trend': False,
            'range_trade': False,
            'increase_stop': False,
            'tighten_stop': False,
            'avoid_trading': False,
            'recommended_action': 'neutral',
            'risk_level': 'medium'
        }
        
        if state.regime == MarketRegime.TRENDING_UP:
            implications['trend_follow'] = True
            implications['recommended_action'] = 'buy_dips'
            implications['risk_level'] = 'low'
        
        elif state.regime == MarketRegime.TRENDING_DOWN:
            implications['trend_follow'] = True
            implications['recommended_action'] = 'sell_rallies'
            implications['risk_level'] = 'low'
        
        elif state.regime == MarketRegime.RANGING:
            implications['range_trade'] = True
            implications['counter_trend'] = True
            implications['recommended_action'] = 'buy_support_sell_resistance'
            implications['risk_level'] = 'medium'
        
        elif state.regime == MarketRegime.VOLATILE:
            implications['tighten_stop'] = True
            implications['recommended_action'] = 'reduce_size'
            implications['risk_level'] = 'high'
        
        elif state.regime == MarketRegime.BREAKOUT:
            implications['trend_follow'] = True
            implications['recommended_action'] = 'follow_breakout'
            implications['risk_level'] = 'medium'
        
        elif state.regime == MarketRegime.REVERSAL:
            implications['counter_trend'] = True
            implications['recommended_action'] = 'potential_reversal_trade'
            implications['risk_level'] = 'high'
        
        return implications


# Global instances
hmm_detector = HMMRegimeDetector()
regime_analyzer = RegimeAnalyzer()
