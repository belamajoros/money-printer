#!/usr/bin/env python3
"""
Trading Statistics Manager

Tracks trading performance, win rates, P&L by model, and provides
comprehensive analytics for the trading bot ecosystem.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

# Model file path mapping
MODEL_PATHS = {
    "random_forest_v1": "data/models/random_forest/trained_model.pkl",
    "xgboost_v1": "data/models/xgboost/trained_model.pkl",
    # Add more models here as needed
}

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Performance metrics for a trading model"""
    model_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_profit_per_trade: float = 0.0
    max_drawdown: float = 0.0
    consecutive_losses: int = 0
    last_trade_time: Optional[str] = None
    is_flagged: bool = False
    flag_reason: str = ""
    sessions_underperforming: int = 0

@dataclass
class TrainingMetrics:
    """Training diagnostics and metrics"""
    model_name: str
    train_loss: float
    val_loss: float
    overfit_risk: str  # Low, Medium, High
    best_features: List[str]
    avg_confidence: float
    winrate_predicted: float
    winrate_actual: float
    training_time: float
    timestamp: str
    feature_importance: Dict[str, float]
    confidence_distribution: Dict[str, int]

@dataclass
class ModelPerformanceMetrics:
    """Detailed performance metrics for a trading model"""
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    win_rate: float
    avg_profit: float
    total_trades: int
    confidence_correlation: float
    model_version: str

class TradingStatsManager:
    """Comprehensive trading statistics and performance tracking"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.diagnostics_dir = os.path.join(data_dir, "diagnostics")
        self.stats_file = os.path.join(data_dir, "trading_stats.json")
        self.models_performance = {}
        self.training_history = {}
        
        # Ensure directories exist
        os.makedirs(self.diagnostics_dir, exist_ok=True)
        
        # Load existing stats
        self._load_stats()
        
    def _load_stats(self):
        """Load existing trading statistics"""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                    
                # Load model performances
                for model_name, perf_data in data.get('models', {}).items():
                    self.models_performance[model_name] = ModelPerformance(**perf_data)
                    
                # Load training history
                self.training_history = data.get('training_history', {})
                
                logger.info(f"✅ Loaded stats for {len(self.models_performance)} models")
        except Exception as e:
            logger.warning(f"Could not load existing stats: {e}")
            
    def _save_stats(self):
        """Save trading statistics to file"""
        try:
            data = {
                'models': {name: asdict(perf) for name, perf in self.models_performance.items()},
                'training_history': self.training_history,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            with open(self.stats_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
            
    def record_trade(self, model_name: str, was_successful: bool, pnl: float, 
                    trade_data: Optional[Dict] = None):
        """Record a completed trade"""
        if model_name not in self.models_performance:
            self.models_performance[model_name] = ModelPerformance(model_name=model_name)
            
        perf = self.models_performance[model_name]
        perf.total_trades += 1
        perf.total_pnl += pnl
        perf.last_trade_time = datetime.utcnow().isoformat()
        
        if was_successful:
            perf.winning_trades += 1
            perf.consecutive_losses = 0
        else:
            perf.losing_trades += 1
            perf.consecutive_losses += 1
            
        # Update calculated metrics
        perf.win_rate = (perf.winning_trades / perf.total_trades) if perf.total_trades > 0 else 0
        perf.avg_profit_per_trade = perf.total_pnl / perf.total_trades if perf.total_trades > 0 else 0
        
        # Check for flagging conditions
        self._check_model_flags(model_name)
        
        # Save updated stats
        self._save_stats()
        
        logger.info(f"📊 Trade recorded for {model_name}: {'WIN' if was_successful else 'LOSS'} | P&L: ${pnl:+.2f}")
        
    def _check_model_flags(self, model_name: str):
        """Check if model should be flagged for underperformance"""
        perf = self.models_performance[model_name]
        
        # Reset flag status
        perf.is_flagged = False
        perf.flag_reason = ""
        
        # Minimum trades threshold before flagging
        if perf.total_trades < 10:
            return
            
        # Flag conditions
        if perf.win_rate < 0.50:
            perf.is_flagged = True
            perf.flag_reason = f"Low win rate: {perf.win_rate:.1%}"
            
        elif perf.total_pnl < -50:  # More than $50 loss
            perf.is_flagged = True
            perf.flag_reason = f"Negative P&L: ${perf.total_pnl:.2f}"
            
        elif perf.consecutive_losses >= 5:
            perf.is_flagged = True
            perf.flag_reason = f"Consecutive losses: {perf.consecutive_losses}"
            
        if perf.is_flagged:
            logger.warning(f"🚩 {model_name} flagged: {perf.flag_reason}")
            
    def record_training_metrics(self, metrics: TrainingMetrics):
        """Record training completion metrics"""
        model_name = metrics.model_name
        
        # Save detailed metrics to diagnostics folder
        metrics_file = os.path.join(self.diagnostics_dir, f"{model_name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(asdict(metrics), f, indent=2)
                
            # Store latest metrics in memory
            if model_name not in self.training_history:
                self.training_history[model_name] = []
                
            self.training_history[model_name].append(asdict(metrics))
            
            # Keep only last 10 training sessions
            if len(self.training_history[model_name]) > 10:
                self.training_history[model_name] = self.training_history[model_name][-10:]
                
            self._save_stats()
            
            logger.info(f"📈 Training metrics saved for {model_name}: {metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to save training metrics for {model_name}: {e}")
            
    def get_dashboard_stats(self, wallet_balance: float) -> Dict[str, Any]:
        """Get comprehensive dashboard statistics"""
        total_trades = sum(perf.total_trades for perf in self.models_performance.values())
        total_pnl = sum(perf.total_pnl for perf in self.models_performance.values())
        
        # Calculate overall win rate
        total_wins = sum(perf.winning_trades for perf in self.models_performance.values())
        overall_win_rate = (total_wins / total_trades) if total_trades > 0 else 0
        
        # Rank models by P&L
        ranked_models = sorted(
            self.models_performance.values(),
            key=lambda x: x.total_pnl,
            reverse=True
        )
        
        # Get flagged models
        flagged_models = [perf for perf in self.models_performance.values() if perf.is_flagged]
        
        return {
            'wallet_balance': wallet_balance,
            'total_trades': total_trades,
            'overall_win_rate': overall_win_rate,
            'total_pnl': total_pnl,
            'ranked_models': [asdict(model) for model in ranked_models],
            'flagged_models': [asdict(model) for model in flagged_models],
            'last_updated': datetime.utcnow().isoformat()
        }
        
    def get_model_leaderboard(self) -> List[Dict[str, Any]]:
        """Get ranked leaderboard of model performance"""
        ranked_models = sorted(
            self.models_performance.values(),
            key=lambda x: (x.total_pnl, x.win_rate),
            reverse=True
        )
        
        leaderboard = []
        for rank, model in enumerate(ranked_models, 1):
            leaderboard.append({
                'rank': rank,
                'model_name': model.model_name,
                'total_pnl': model.total_pnl,
                'win_rate': model.win_rate,
                'total_trades': model.total_trades,
                'is_flagged': model.is_flagged,
                'flag_reason': model.flag_reason
            })
            
        return leaderboard
        
    def get_underperforming_models(self) -> List[str]:
        """Get list of underperforming models that need retraining"""
        return [model.model_name for model in self.models_performance.values() if model.is_flagged]
        
    def get_model_diagnostics(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed diagnostics for a specific model"""
        if model_name not in self.models_performance:
            return None
            
        perf = self.models_performance[model_name]
        training_history = self.training_history.get(model_name, [])
        
        return {
            'performance': asdict(perf),
            'training_history': training_history,
            'latest_training': training_history[-1] if training_history else None
        }
        
    def reset_model_flags(self, model_name: str):
        """Reset flags for a model (after retraining)"""
        if model_name in self.models_performance:
            perf = self.models_performance[model_name]
            perf.is_flagged = False
            perf.flag_reason = ""
            perf.sessions_underperforming = 0
            perf.consecutive_losses = 0
            self._save_stats()
            logger.info(f"🔄 Flags reset for {model_name}")
            
    def format_dashboard_display(self, wallet_balance: float) -> str:
        """Format dashboard stats for display"""
        stats = self.get_dashboard_stats(wallet_balance)
        
        output = []
        output.append("💰 TRADING DASHBOARD")
        output.append("=" * 40)
        output.append(f"Wallet Balance: ${stats['wallet_balance']:.2f}")
        output.append(f"Total Trades: {stats['total_trades']}")
        output.append(f"Overall Win Rate: {stats['overall_win_rate']:.1%}")
        output.append(f"Total P&L: ${stats['total_pnl']:+.2f}")
        output.append("")
        
        if stats['ranked_models']:
            output.append("🏆 Bot Rankings (by Total P&L):")
            for i, model in enumerate(stats['ranked_models'], 1):
                flag_indicator = " ❌" if model['is_flagged'] else ""
                output.append(f"{i}. {model['model_name']}: ${model['total_pnl']:+.2f} "
                            f"(Win Rate: {model['win_rate']:.1%}){flag_indicator}")
        
        if stats['flagged_models']:
            output.append("")
            output.append("⚠️ FLAGGED MODELS:")
            for model in stats['flagged_models']:
                output.append(f"• {model['model_name']}: {model['flag_reason']}")
                
        return "\n".join(output)

class ModelDriftDetector:
    """Handles model loading, performance tracking, and drift detection"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.performance_history: List[ModelPerformanceMetrics] = []
        self.model_version: Optional[str] = None
        self.model_load_time: Optional[datetime] = None
        self.is_model_valid: bool = False
        self.drift_detection_window: int = 50
        self.validation_threshold: float = 50.0  # Min acceptable win rate %
        self.current_model = None

        self._load_model()

    def _load_model(self) -> bool:
        """Load the current model from the path defined in MODEL_PATHS"""
        model_path = MODEL_PATHS.get(self.model_name)
        if not model_path:
            logger.error(f"❌ No model path defined for {self.model_name} in MODEL_PATHS")
            self.is_model_valid = False
            return False

        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            self.is_model_valid = False
            return False

        try:
            with open(model_path, 'rb') as f:
                self.current_model = pickle.load(f)
            self.model_version = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y%m%d_%H%M%S")
            self.model_load_time = datetime.utcnow()
            self.is_model_valid = True
            logger.info(f"✅ Loaded model {self.model_name} from {model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model {self.model_name}: {e}")
            self.is_model_valid = False
            return False


    def calculate_current_performance(self) -> Optional[ModelPerformanceMetrics]:
        """Compute metrics from recent trades"""
        trades_file = f"data/transactions/{self.model_name}_trades.csv"
        if not os.path.exists(trades_file):
            logger.warning(f"No trades found for model: {self.model_name}")
            return None

        df = pd.read_csv(trades_file)
        if len(df) < 10:
            logger.warning(f"Too few trades for analysis: {len(df)}")
            return None

        recent_df = df.tail(self.drift_detection_window)
        win_rate = recent_df["was_successful"].mean() * 100
        avg_profit = recent_df["pnl_percent"].mean()
        total_trades = len(recent_df)

        if "confidence" in recent_df.columns:
            y_true = recent_df["was_successful"].astype(int)
            y_pred_proba = recent_df["confidence"]
            y_pred = (y_pred_proba > 0.5).astype(int)

            if len(np.unique(y_true)) > 1:
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                cc = np.corrcoef(y_pred_proba, recent_df["pnl_percent"])[0, 1]
                if np.isnan(cc):
                    cc = 0.0
            else:
                accuracy = precision = recall = f1 = 0.0
                cc = 0.0
        else:
            accuracy = precision = recall = f1 = cc = 0.0

        return ModelPerformanceMetrics(
            timestamp=datetime.utcnow(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            win_rate=win_rate,
            avg_profit=avg_profit,
            total_trades=total_trades,
            confidence_correlation=cc,
            model_version=self.model_version or "unknown",
        )

    def detect_drift(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Determine whether drift has occurred"""
        metrics = self.calculate_current_performance()
        if not metrics:
            return True, "Insufficient trade data", {}

        self.performance_history.append(metrics)

        # 1. Win rate threshold
        if metrics.win_rate < self.validation_threshold:
            return True, f"Win rate too low: {metrics.win_rate:.1f}%", {
                "current_win_rate": metrics.win_rate
            }

        # 2. Decline trend
        if len(self.performance_history) >= 3:
            recent = self.performance_history[-3:]
            win_rates = [m.win_rate for m in recent]
            if win_rates[-1] < win_rates[-2] < win_rates[-3]:
                decline = win_rates[-3] - win_rates[-1]
                if decline > 10:
                    return True, f"Performance dropped {decline:.1f}% over 3 windows", {
                        "trend": win_rates,
                        "decline": decline
                    }

        # 3. Confidence correlation
        if metrics.confidence_correlation < -0.2:
            return True, f"Negative confidence correlation: {metrics.confidence_correlation:.2f}", {
                "confidence_correlation": metrics.confidence_correlation
            }

        # 4. Model age
        if self.model_load_time:
            age_days = (datetime.utcnow() - self.model_load_time).days
            if age_days > 7:
                return True, f"Model older than 7 days ({age_days} days)", {
                    "model_age_days": age_days
                }

        return False, "Performance acceptable", {
            "win_rate": metrics.win_rate,
            "confidence_correlation": metrics.confidence_correlation,
            "avg_profit": metrics.avg_profit
        }

    def validate_model_for_trading(self) -> Tuple[bool, str]:
        if not self.is_model_valid:
            return False, "Model not loaded or invalid"

        has_drift, reason, _ = self.detect_drift()
        if has_drift:
            return False, f"Drift detected: {reason}"

        return True, "Model is valid for trading"

    def get_performance_summary(self) -> Dict[str, Any]:
        current = self.calculate_current_performance()
        has_drift, reason, details = self.detect_drift()
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "is_valid": self.is_model_valid,
            "model_age_hours": (
                (datetime.utcnow() - self.model_load_time).total_seconds() / 3600
                if self.model_load_time else None
            ),
            "current_performance": asdict(current) if current else None,
            "drift_status": {
                "has_drift": has_drift,
                "reason": reason,
                "details": details
            }
        }


# Global instance
_stats_manager = None

def get_stats_manager() -> TradingStatsManager:
    """Get singleton stats manager instance"""
    global _stats_manager
    if _stats_manager is None:
        _stats_manager = TradingStatsManager()
    return _stats_manager

