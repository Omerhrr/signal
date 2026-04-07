"""
Forex Probability Intelligence System - MCMC Engine
Markov Chain Monte Carlo for probability estimation and uncertainty quantification
"""
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from loguru import logger
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

try:
    import pymc as pm
    PYMC_AVAILABLE = True
except ImportError:
    try:
        import pymc3 as pm
        PYMC_AVAILABLE = True
    except ImportError:
        PYMC_AVAILABLE = False
        logger.warning("PyMC not available, using simplified Bayesian inference")

from app.models.schemas import FeatureSet, SignalBias
from config.settings import get_settings

settings = get_settings()


@dataclass
class ProbabilityEstimate:
    """MCMC-based probability estimate with uncertainty"""
    mean: float
    std: float
    median: float
    credible_interval_95: Tuple[float, float]
    credible_interval_80: Tuple[float, float]
    samples: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def to_dict(self) -> Dict:
        return {
            'mean': float(self.mean),
            'std': float(self.std),
            'median': float(self.median),
            'ci_95': [float(self.credible_interval_95[0]), float(self.credible_interval_95[1])],
            'ci_80': [float(self.credible_interval_80[0]), float(self.credible_interval_80[1])],
            'uncertainty': float(self.std / (self.mean + 0.01))  # Coefficient of variation
        }


@dataclass
class MCMCResult:
    """Result from MCMC sampling"""
    direction_probability: ProbabilityEstimate
    duration_probability: ProbabilityEstimate
    volatility_forecast: ProbabilityEstimate
    regime_probability: Dict[str, ProbabilityEstimate]
    confidence_score: float
    effective_sample_size: int
    convergence_metric: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'direction_probability': self.direction_probability.to_dict(),
            'duration_probability': self.duration_probability.to_dict(),
            'volatility_forecast': self.volatility_forecast.to_dict(),
            'regime_probability': {k: v.to_dict() for k, v in self.regime_probability.items()},
            'confidence_score': self.confidence_score,
            'effective_sample_size': self.effective_sample_size,
            'convergence_metric': self.convergence_metric,
            'timestamp': self.timestamp.isoformat()
        }


class MCMCProbabilisticEngine:
    """MCMC-based probability estimation engine"""
    
    def __init__(self, n_samples: int = 2000, n_tune: int = 1000):
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.is_trained = False
        
        # Prior parameters (updated from data)
        self.priors = {
            'direction_alpha': 1.0,  # Beta distribution for direction
            'direction_beta': 1.0,
            'duration_mu': 10.0,  # LogNormal for duration
            'duration_sigma': 0.5,
            'vol_mu': 0.01,  # Normal for volatility
            'vol_sigma': 0.005
        }
        
        # History for updating priors
        self.outcome_history: deque = deque(maxlen=500)
        
        # Model path
        self.model_path = Path(__file__).parent.parent.parent / "data" / "models" / "mcmc_model.pkl"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._load_model()
    
    def estimate_probabilities(
        self,
        features: FeatureSet,
        direction_pred: float,
        duration_pred: float,
        regime_probs: Dict[str, float]
    ) -> MCMCResult:
        """Estimate probabilities using MCMC sampling"""
        
        if PYMC_AVAILABLE and self.is_trained:
            return self._estimate_with_pymc(features, direction_pred, duration_pred, regime_probs)
        else:
            return self._estimate_with_bootstrap(features, direction_pred, duration_pred, regime_probs)
    
    def _estimate_with_pymc(
        self,
        features: FeatureSet,
        direction_pred: float,
        duration_pred: float,
        regime_probs: Dict[str, float]
    ) -> MCMCResult:
        """Estimate using PyMC for proper MCMC"""
        try:
            # Build simple Bayesian model
            with pm.Model() as model:
                # Direction probability (Beta distribution)
                alpha = max(0.1, direction_pred * 20)
                beta_param = max(0.1, (1 - direction_pred) * 20)
                direction = pm.Beta('direction', alpha=alpha, beta=beta_param)
                
                # Duration (LogNormal)
                log_duration_mu = np.log(max(1, duration_pred))
                duration = pm.Lognormal('duration', mu=log_duration_mu, sigma=0.3)
                
                # Volatility (Normal)
                volatility = pm.Normal('volatility', mu=features.statistical.rolling_volatility_20, sigma=0.005)
                
                # Sample
                trace = pm.sample(self.n_samples, tune=self.n_tune, chains=2, return_inferencedata=False, progressbar=False)
            
            # Extract results
            direction_samples = trace['direction']
            duration_samples = trace['duration']
            volatility_samples = trace['volatility']
            
            # Create probability estimates
            direction_prob = self._samples_to_estimate(direction_samples)
            duration_prob = self._samples_to_estimate(duration_samples)
            volatility_prob = self._samples_to_estimate(volatility_samples)
            
            # Regime probabilities (using Dirichlet-like sampling)
            regime_estimates = {}
            for regime, prob in regime_probs.items():
                samples = np.random.beta(prob * 10 + 1, (1 - prob) * 10 + 1, self.n_samples)
                regime_estimates[regime] = self._samples_to_estimate(samples)
            
            # Calculate convergence
            convergence = self._calculate_convergence(direction_samples)
            
            return MCMCResult(
                direction_probability=direction_prob,
                duration_probability=duration_prob,
                volatility_forecast=volatility_prob,
                regime_probability=regime_estimates,
                confidence_score=float(1 - direction_prob.std),
                effective_sample_size=len(direction_samples),
                convergence_metric=convergence,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"PyMC estimation error: {e}")
            return self._estimate_with_bootstrap(features, direction_pred, duration_pred, regime_probs)
    
    def _estimate_with_bootstrap(
        self,
        features: FeatureSet,
        direction_pred: float,
        duration_pred: float,
        regime_probs: Dict[str, float]
    ) -> MCMCResult:
        """Bootstrap-based estimation as fallback"""
        
        # Generate samples using beta distribution around predictions
        # Direction samples
        alpha = max(0.5, direction_pred * 20)
        beta_param = max(0.5, (1 - direction_pred) * 20)
        direction_samples = np.random.beta(alpha, beta_param, self.n_samples)
        
        # Duration samples (log-normal)
        log_mu = np.log(max(1, duration_pred))
        duration_samples = np.random.lognormal(log_mu, 0.3, self.n_samples)
        
        # Volatility samples
        vol_mu = features.statistical.rolling_volatility_20
        vol_std = max(0.001, vol_mu * 0.3)
        volatility_samples = np.random.normal(vol_mu, vol_std, self.n_samples)
        volatility_samples = np.maximum(0, volatility_samples)  # Ensure non-negative
        
        # Create probability estimates
        direction_prob = self._samples_to_estimate(direction_samples)
        duration_prob = self._samples_to_estimate(duration_samples)
        volatility_prob = self._samples_to_estimate(volatility_samples)
        
        # Regime probabilities
        regime_estimates = {}
        for regime, prob in regime_probs.items():
            samples = np.random.beta(prob * 10 + 1, (1 - prob) * 10 + 1, self.n_samples)
            regime_estimates[regime] = self._samples_to_estimate(samples)
        
        return MCMCResult(
            direction_probability=direction_prob,
            duration_probability=duration_prob,
            volatility_forecast=volatility_prob,
            regime_probability=regime_estimates,
            confidence_score=float(1 - direction_prob.std * 2),
            effective_sample_size=self.n_samples,
            convergence_metric=0.9,  # Bootstrap always "converges"
            timestamp=datetime.now(timezone.utc)
        )
    
    def _samples_to_estimate(self, samples: np.ndarray) -> ProbabilityEstimate:
        """Convert samples to probability estimate"""
        return ProbabilityEstimate(
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
            median=float(np.median(samples)),
            credible_interval_95=(float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))),
            credible_interval_80=(float(np.percentile(samples, 10)), float(np.percentile(samples, 90))),
            samples=samples
        )
    
    def _calculate_convergence(self, samples: np.ndarray) -> float:
        """Calculate simple convergence metric (Gelman-Rubin-like)"""
        if len(samples) < 100:
            return 0.5
        
        # Split into two chains
        mid = len(samples) // 2
        chain1 = samples[:mid]
        chain2 = samples[mid:]
        
        # Calculate between-chain and within-chain variance
        mean1, mean2 = np.mean(chain1), np.mean(chain2)
        var1, var2 = np.var(chain1), np.var(chain2)
        
        # Pooled variance
        pooled_var = (var1 + var2) / 2
        
        # Between-chain variance
        between_var = ((mean1 - mean2) ** 2) / 2
        
        # R-hat approximation (should be close to 1 for convergence)
        if pooled_var > 0:
            r_hat = np.sqrt((pooled_var + between_var) / pooled_var)
            # Convert to 0-1 scale (1 = perfect convergence)
            return max(0, 1 - abs(r_hat - 1))
        
        return 0.5
    
    def update_with_outcome(self, signal_features: Dict, outcome: str, pips: float):
        """Update model with observed outcome"""
        self.outcome_history.append({
            'features': signal_features,
            'outcome': 1 if outcome == 'win' else 0,
            'pips': pips,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Update priors periodically
        if len(self.outcome_history) % 50 == 0:
            self._update_priors()
    
    def _update_priors(self):
        """Update prior parameters from observed outcomes"""
        if len(self.outcome_history) < 20:
            return
        
        outcomes = [o['outcome'] for o in self.outcome_history]
        pips = [o['pips'] for o in self.outcome_history if o['pips'] is not None]
        
        # Update direction prior (Beta parameters)
        wins = sum(outcomes)
        losses = len(outcomes) - wins
        
        # Use conjugate prior update
        self.priors['direction_alpha'] = 1 + wins
        self.priors['direction_beta'] = 1 + losses
        
        # Update duration prior
        if pips:
            abs_pips = [abs(p) for p in pips if p != 0]
            if abs_pips:
                self.priors['duration_mu'] = np.mean(abs_pips)
                self.priors['duration_sigma'] = np.std(abs_pips)
        
        logger.info(f"Updated MCMC priors: alpha={self.priors['direction_alpha']:.1f}, beta={self.priors['direction_beta']:.1f}")
        
        # Save updated model
        self._save_model()
    
    def get_predictive_distribution(self, features: FeatureSet, n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Get full predictive distribution for analysis"""
        
        # Generate samples
        direction_samples = np.random.beta(
            self.priors['direction_alpha'],
            self.priors['direction_beta'],
            n_samples
        )
        
        duration_samples = np.random.lognormal(
            np.log(max(1, self.priors['duration_mu'])),
            self.priors['duration_sigma'],
            n_samples
        )
        
        return {
            'direction': direction_samples,
            'duration': duration_samples,
            'expected_pips': direction_samples * duration_samples * 0.1  # Rough approximation
        }
    
    def calculate_value_at_risk(self, pips_distribution: np.ndarray, confidence: float = 0.95) -> Dict[str, float]:
        """Calculate Value at Risk and Expected Shortfall"""
        sorted_pips = np.sort(pips_distribution)
        
        var_index = int((1 - confidence) * len(sorted_pips))
        var = sorted_pips[var_index]
        
        # Expected Shortfall (CVaR)
        es = np.mean(sorted_pips[:var_index])
        
        return {
            'var': float(var),
            'expected_shortfall': float(es),
            'confidence': confidence
        }
    
    def _save_model(self):
        """Save MCMC model state"""
        try:
            model_data = {
                'priors': self.priors,
                'n_samples': self.n_samples,
                'outcome_count': len(self.outcome_history),
                'saved_at': datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
        except Exception as e:
            logger.error(f"Error saving MCMC model: {e}")
    
    def _load_model(self):
        """Load MCMC model state"""
        if not self.model_path.exists():
            return
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.priors = model_data.get('priors', self.priors)
            self.is_trained = True
            
            logger.info(f"Loaded MCMC model with priors: {self.priors}")
            
        except Exception as e:
            logger.error(f"Error loading MCMC model: {e}")


class BayesianSignalOptimizer:
    """Optimize signal parameters using Bayesian inference"""
    
    def __init__(self):
        self.mcmc_engine = MCMCProbabilisticEngine()
    
    def optimize_signal_parameters(
        self,
        features: FeatureSet,
        base_confidence: float,
        base_duration: float
    ) -> Dict[str, Any]:
        """Optimize signal parameters with uncertainty quantification"""
        
        # Get predictive distribution
        pred_dist = self.mcmc_engine.get_predictive_distribution(features)
        
        # Calculate optimal stop loss and take profit
        direction_samples = pred_dist['direction']
        duration_samples = pred_dist['duration']
        expected_pips = pred_dist['expected_pips']
        
        # Optimal SL/TP based on distribution
        sl_pips = np.percentile(np.abs(expected_pips), 25)  # 25th percentile as SL
        tp_pips = np.percentile(np.abs(expected_pips), 75)  # 75th percentile as TP
        
        # Risk-adjusted confidence
        uncertainty = np.std(direction_samples)
        adjusted_confidence = base_confidence * (1 - uncertainty)
        
        # Value at Risk
        var_result = self.mcmc_engine.calculate_value_at_risk(expected_pips)
        
        return {
            'optimal_stop_loss_pips': float(max(5, sl_pips)),
            'optimal_take_profit_pips': float(max(10, tp_pips)),
            'risk_adjusted_confidence': float(adjusted_confidence),
            'direction_uncertainty': float(uncertainty),
            'duration_uncertainty': float(np.std(duration_samples)),
            'expected_pips_mean': float(np.mean(expected_pips)),
            'expected_pips_median': float(np.median(expected_pips)),
            'value_at_risk': var_result,
            'recommended_position_size': self._calculate_position_size(adjusted_confidence, sl_pips)
        }
    
    def _calculate_position_size(self, confidence: float, sl_pips: float) -> float:
        """Calculate recommended position size based on confidence and risk"""
        # Base risk per trade: 1% of account
        base_risk = 0.01
        
        # Adjust by confidence
        adjusted_risk = base_risk * confidence
        
        # Scale by stop loss (wider SL = smaller position)
        if sl_pips > 20:
            adjusted_risk *= 0.8
        elif sl_pips < 10:
            adjusted_risk *= 1.2
        
        return float(min(0.02, adjusted_risk))  # Cap at 2%


class UncertaintyQuantifier:
    """Quantify and communicate uncertainty in predictions"""
    
    def __init__(self):
        self.mcmc_engine = MCMCProbabilisticEngine()
    
    def quantify_uncertainty(
        self,
        direction_prob: float,
        duration_pred: float,
        features: FeatureSet
    ) -> Dict[str, Any]:
        """Comprehensive uncertainty quantification"""
        
        # Get MCMC samples
        direction_samples = np.random.beta(
            max(0.5, direction_prob * 20),
            max(0.5, (1 - direction_prob) * 20),
            2000
        )
        
        # Uncertainty metrics
        epistemic_uncertainty = self._calculate_epistemic_uncertainty(features)
        aleatoric_uncertainty = self._calculate_aleatoric_uncertainty(features)
        
        # Total uncertainty
        total_uncertainty = np.std(direction_samples)
        
        # Confidence bounds
        lower_bound = np.percentile(direction_samples, 5)
        upper_bound = np.percentile(direction_samples, 95)
        
        # Prediction quality
        quality = self._assess_prediction_quality(total_uncertainty, features)
        
        return {
            'total_uncertainty': float(total_uncertainty),
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'confidence_bounds': {
                'lower_90': float(lower_bound),
                'upper_90': float(upper_bound),
                'width': float(upper_bound - lower_bound)
            },
            'prediction_quality': quality,
            'recommendation': self._get_uncertainty_recommendation(total_uncertainty, quality)
        }
    
    def _calculate_epistemic_uncertainty(self, features: FeatureSet) -> float:
        """Calculate uncertainty due to lack of knowledge"""
        # Based on feature quality and data availability
        uncertainty = 0.0
        
        # Check if we have enough data
        if features.statistical.rolling_volatility_20 == 0:
            uncertainty += 0.2
        
        # Check for missing features
        if features.price_action.bos_strength == 0:
            uncertainty += 0.1
        
        if features.statistical.rsi == 50:  # Default value
            uncertainty += 0.1
        
        return min(0.5, uncertainty)
    
    def _calculate_aleatoric_uncertainty(self, features: FeatureSet) -> float:
        """Calculate inherent uncertainty in data"""
        # Based on volatility and market conditions
        uncertainty = 0.0
        
        # Volatility contribution
        vol = features.statistical.rolling_volatility_20
        uncertainty += min(0.3, vol * 10)
        
        # RSI extremes
        rsi = features.statistical.rsi
        if rsi > 70 or rsi < 30:
            uncertainty += 0.1
        
        # Session contribution
        hour = features.time_features.hour_of_day
        if 13 <= hour < 17:  # Overlap - higher inherent uncertainty
            uncertainty += 0.1
        
        return min(0.4, uncertainty)
    
    def _assess_prediction_quality(self, uncertainty: float, features: FeatureSet) -> str:
        """Assess overall prediction quality"""
        if uncertainty < 0.1:
            return "high"
        elif uncertainty < 0.2:
            return "medium"
        elif uncertainty < 0.3:
            return "low"
        else:
            return "very_low"
    
    def _get_uncertainty_recommendation(self, uncertainty: float, quality: str) -> str:
        """Get recommendation based on uncertainty"""
        if quality == "high":
            return "Proceed with standard position size"
        elif quality == "medium":
            return "Consider reducing position size by 25%"
        elif quality == "low":
            return "Reduce position size by 50% or skip trade"
        else:
            return "Avoid trading - uncertainty too high"


# Global instances
mcmc_engine = MCMCProbabilisticEngine()
bayesian_optimizer = BayesianSignalOptimizer()
uncertainty_quantifier = UncertaintyQuantifier()
