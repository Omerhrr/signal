"""
Forex Probability Intelligence System - Signal Tracking & Performance
Track all signals, outcomes, and calculate performance metrics
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np

from app.models.schemas import (
    TradingSignal, SignalBias, SignalStatus, SignalAction,
    TradeOutcome, ModelPerformance
)
from config.settings import get_settings

settings = get_settings()


@dataclass
class SignalRecord:
    """Record of a signal for tracking"""
    signal_id: str
    symbol: str
    bias: str  # BUY, SELL
    entry_price: float
    entry_zone_start: float
    entry_zone_end: float
    stop_loss: float
    take_profit: float
    confidence: float
    probability_up: float
    probability_down: float
    expected_duration_minutes: float
    risk_level: str
    created_at: str
    expires_at: str
    
    # Outcome tracking
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    outcome: str = "pending"  # pending, win, loss, breakeven, expired
    pips_gained: Optional[float] = None
    pips_target: Optional[float] = None
    max_drawdown_pips: Optional[float] = None
    max_favorable_pips: Optional[float] = None
    duration_minutes: Optional[float] = None
    
    # Market conditions at entry
    volatility_at_entry: Optional[float] = None
    rsi_at_entry: Optional[float] = None
    session_at_entry: Optional[str] = None
    trend_at_entry: Optional[str] = None
    
    # Feature data for ML
    feature_data: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SignalTracker:
    """Track and analyze signal performance"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Path(__file__).parent.parent.parent / "data" / "signal_tracking.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        
        # In-memory cache for recent signals
        self.recent_signals: Dict[str, SignalRecord] = {}
        self.max_cache = 100
        
        # Performance metrics cache
        self.metrics_cache: Dict[str, deque] = {
            'wins': deque(maxlen=1000),
            'losses': deque(maxlen=1000),
            'confidences': deque(maxlen=1000),
            'durations': deque(maxlen=1000)
        }
    
    def _init_database(self):
        """Initialize SQLite database for signal tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                bias TEXT NOT NULL,
                entry_price REAL,
                entry_zone_start REAL,
                entry_zone_end REAL,
                stop_loss REAL,
                take_profit REAL,
                confidence REAL,
                probability_up REAL,
                probability_down REAL,
                expected_duration_minutes REAL,
                risk_level TEXT,
                created_at TEXT,
                expires_at TEXT,
                exit_price REAL,
                exit_time TEXT,
                outcome TEXT DEFAULT 'pending',
                pips_gained REAL,
                pips_target REAL,
                max_drawdown_pips REAL,
                max_favorable_pips REAL,
                duration_minutes REAL,
                volatility_at_entry REAL,
                rsi_at_entry REAL,
                session_at_entry TEXT,
                trend_at_entry TEXT,
                feature_data TEXT
            )
        ''')
        
        # Performance stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                total_signals INTEGER,
                wins INTEGER,
                losses INTEGER,
                breakeven INTEGER,
                win_rate REAL,
                avg_pips REAL,
                avg_confidence REAL,
                UNIQUE(symbol, date)
            )
        ''')
        
        # Feature importance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL,
                signal_id TEXT,
                feature_value REAL,
                outcome TEXT,
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_signal(self, signal: TradingSignal, market_state: Dict = None) -> SignalRecord:
        """Record a new signal for tracking"""
        record = SignalRecord(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            bias=signal.bias.value,
            entry_price=(signal.entry_zone_start + signal.entry_zone_end) / 2,
            entry_zone_start=signal.entry_zone_start,
            entry_zone_end=signal.entry_zone_end,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
            probability_up=signal.probability_hold_above,
            probability_down=signal.probability_hold_below,
            expected_duration_minutes=signal.expected_duration_minutes,
            risk_level=signal.risk_level.value,
            created_at=signal.created_at.isoformat(),
            expires_at=signal.expires_at.isoformat(),
            outcome="pending",
            volatility_at_entry=market_state.get('volatility') if market_state else None,
            rsi_at_entry=market_state.get('rsi') if market_state else None,
            session_at_entry=market_state.get('session') if market_state else None,
            trend_at_entry=market_state.get('trend') if market_state else None,
            feature_data=signal.features.dict() if signal.features else None
        )
        
        # Save to database
        self._save_signal(record)
        
        # Cache in memory
        self.recent_signals[signal.signal_id] = record
        if len(self.recent_signals) > self.max_cache:
            # Remove oldest
            oldest = min(self.recent_signals.items(), key=lambda x: x[1].created_at)
            del self.recent_signals[oldest[0]]
        
        logger.info(f"Recorded signal {signal.signal_id} for tracking")
        return record
    
    def _save_signal(self, record: SignalRecord):
        """Save signal to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO signals (
                signal_id, symbol, bias, entry_price, entry_zone_start, entry_zone_end,
                stop_loss, take_profit, confidence, probability_up, probability_down,
                expected_duration_minutes, risk_level, created_at, expires_at,
                outcome, volatility_at_entry, rsi_at_entry, session_at_entry, 
                trend_at_entry, feature_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.signal_id, record.symbol, record.bias, record.entry_price,
            record.entry_zone_start, record.entry_zone_end, record.stop_loss,
            record.take_profit, record.confidence, record.probability_up,
            record.probability_down, record.expected_duration_minutes,
            record.risk_level, record.created_at, record.expires_at,
            record.outcome, record.volatility_at_entry, record.rsi_at_entry,
            record.session_at_entry, record.trend_at_entry,
            json.dumps(record.feature_data) if record.feature_data else None
        ))
        
        conn.commit()
        conn.close()
    
    def update_outcome(
        self,
        signal_id: str,
        exit_price: float,
        outcome: str,
        max_drawdown_pips: float = None,
        max_favorable_pips: float = None
    ) -> Optional[SignalRecord]:
        """Update signal outcome"""
        # Get existing record
        record = self._get_signal(signal_id)
        if not record:
            logger.warning(f"Signal {signal_id} not found for update")
            return None
        
        # Calculate pips
        pip_size = 0.01 if 'JPY' in record.symbol else 0.0001
        
        if record.bias == "BUY":
            pips_gained = (exit_price - record.entry_price) / pip_size
        else:
            pips_gained = (record.entry_price - exit_price) / pip_size
        
        # Calculate duration
        entry_time = datetime.fromisoformat(record.created_at.replace('Z', '+00:00'))
        exit_time = datetime.now(timezone.utc)
        duration = (exit_time - entry_time).total_seconds() / 60
        
        # Update record
        record.exit_price = exit_price
        record.exit_time = exit_time.isoformat()
        record.outcome = outcome
        record.pips_gained = round(pips_gained, 1)
        record.pips_target = round(abs(record.take_profit - record.entry_price) / pip_size, 1)
        record.max_drawdown_pips = max_drawdown_pips
        record.max_favorable_pips = max_favorable_pips
        record.duration_minutes = round(duration, 1)
        
        # Save to database
        self._update_signal_outcome(record)
        
        # Update cache
        self.recent_signals[signal_id] = record
        
        # Update metrics
        self._update_metrics(record)
        
        logger.info(f"Updated signal {signal_id}: {outcome} ({pips_gained:.1f} pips)")
        return record
    
    def _get_signal(self, signal_id: str) -> Optional[SignalRecord]:
        """Get signal from database"""
        if signal_id in self.recent_signals:
            return self.recent_signals[signal_id]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM signals WHERE signal_id = ?', (signal_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_record(row)
        return None
    
    def _row_to_record(self, row: tuple) -> SignalRecord:
        """Convert database row to SignalRecord"""
        return SignalRecord(
            signal_id=row[0],
            symbol=row[1],
            bias=row[2],
            entry_price=row[3],
            entry_zone_start=row[4],
            entry_zone_end=row[5],
            stop_loss=row[6],
            take_profit=row[7],
            confidence=row[8],
            probability_up=row[9],
            probability_down=row[10],
            expected_duration_minutes=row[11],
            risk_level=row[12],
            created_at=row[13],
            expires_at=row[14],
            exit_price=row[15],
            exit_time=row[16],
            outcome=row[17],
            pips_gained=row[18],
            pips_target=row[19],
            max_drawdown_pips=row[20],
            max_favorable_pips=row[21],
            duration_minutes=row[22],
            volatility_at_entry=row[23],
            rsi_at_entry=row[24],
            session_at_entry=row[25],
            trend_at_entry=row[26],
            feature_data=json.loads(row[27]) if row[27] else None
        )
    
    def _update_signal_outcome(self, record: SignalRecord):
        """Update signal outcome in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE signals SET
                exit_price = ?, exit_time = ?, outcome = ?,
                pips_gained = ?, pips_target = ?, max_drawdown_pips = ?,
                max_favorable_pips = ?, duration_minutes = ?
            WHERE signal_id = ?
        ''', (
            record.exit_price, record.exit_time, record.outcome,
            record.pips_gained, record.pips_target, record.max_drawdown_pips,
            record.max_favorable_pips, record.duration_minutes,
            record.signal_id
        ))
        
        conn.commit()
        conn.close()
    
    def _update_metrics(self, record: SignalRecord):
        """Update performance metrics cache"""
        if record.outcome == 'win':
            self.metrics_cache['wins'].append(record.signal_id)
        elif record.outcome == 'loss':
            self.metrics_cache['losses'].append(record.signal_id)
        
        self.metrics_cache['confidences'].append(record.confidence)
        if record.duration_minutes:
            self.metrics_cache['durations'].append(record.duration_minutes)
    
    def get_performance_stats(self, symbol: str = None, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        where_clause = "WHERE created_at >= ?"
        params = [(datetime.now(timezone.utc) - timedelta(days=days)).isoformat()]
        
        if symbol:
            where_clause += " AND symbol = ?"
            params.append(symbol)
        
        # Get all signals in period
        cursor.execute(f'''
            SELECT outcome, confidence, pips_gained, duration_minutes, 
                   volatility_at_entry, rsi_at_entry, session_at_entry
            FROM signals {where_clause}
        ''', params)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                'total_signals': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'avg_pips': 0,
                'total_pips': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_confidence': 0,
                'avg_duration': 0,
                'by_session': {},
                'by_volatility': {}
            }
        
        # Calculate statistics
        outcomes = [r[0] for r in rows]
        confidences = [r[1] for r in rows if r[1]]
        pips = [r[2] for r in rows if r[2] is not None]
        durations = [r[3] for r in rows if r[3]]
        
        wins = outcomes.count('win')
        losses = outcomes.count('loss')
        breakeven = outcomes.count('breakeven')
        pending = outcomes.count('pending')
        
        # By session
        by_session = {}
        for r in rows:
            session = r[6] or 'unknown'
            if session not in by_session:
                by_session[session] = {'wins': 0, 'losses': 0, 'total': 0}
            by_session[session]['total'] += 1
            if r[0] == 'win':
                by_session[session]['wins'] += 1
            elif r[0] == 'loss':
                by_session[session]['losses'] += 1
        
        # Calculate win rates by session
        for session in by_session:
            total = by_session[session]['total']
            by_session[session]['win_rate'] = by_session[session]['wins'] / total if total > 0 else 0
        
        return {
            'total_signals': len(rows),
            'wins': wins,
            'losses': losses,
            'breakeven': breakeven,
            'pending': pending,
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
            'avg_pips': np.mean(pips) if pips else 0,
            'total_pips': sum(pips) if pips else 0,
            'best_trade': max(pips) if pips else 0,
            'worst_trade': min(pips) if pips else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'profit_factor': self._calculate_profit_factor(pips),
            'sharpe_ratio': self._calculate_sharpe_ratio(pips),
            'by_session': by_session
        }
    
    def _calculate_profit_factor(self, pips: List[float]) -> float:
        """Calculate profit factor"""
        if not pips:
            return 0
        
        profits = sum(p for p in pips if p > 0)
        losses = abs(sum(p for p in pips if p < 0))
        
        return profits / losses if losses > 0 else float('inf') if profits > 0 else 0
    
    def _calculate_sharpe_ratio(self, pips: List[float], risk_free: float = 0) -> float:
        """Calculate Sharpe ratio of returns"""
        if len(pips) < 2:
            return 0
        
        returns = np.array(pips)
        std = np.std(returns)
        
        if std == 0:
            return 0
        
        return (np.mean(returns) - risk_free) / std
    
    def get_recent_signals(self, limit: int = 50, symbol: str = None) -> List[SignalRecord]:
        """Get recent signals"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute('''
                SELECT * FROM signals 
                WHERE symbol = ?
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (symbol, limit))
        else:
            cursor.execute('''
                SELECT * FROM signals 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_record(row) for row in rows]
    
    def get_signal_history_dataframe(self, days: int = 30) -> List[Dict]:
        """Get signal history as list of dicts for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM signals 
            WHERE created_at >= ?
            ORDER BY created_at DESC
        ''', ((datetime.now(timezone.utc) - timedelta(days=days)).isoformat(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [asdict(self._row_to_record(row)) for row in rows]
    
    def calculate_confidence_accuracy(self) -> Dict[str, float]:
        """Calculate accuracy by confidence level"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN confidence >= 0.8 THEN 'high'
                    WHEN confidence >= 0.65 THEN 'medium'
                    ELSE 'low'
                END as conf_level,
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins
            FROM signals
            WHERE outcome IN ('win', 'loss')
            GROUP BY conf_level
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        result = {}
        for row in rows:
            level, total, wins = row
            result[level] = {
                'total': total,
                'wins': wins,
                'accuracy': wins / total if total > 0 else 0
            }
        
        return result
    
    def get_performance_by_symbol(self) -> Dict[str, Dict]:
        """Get performance breakdown by symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                symbol,
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                AVG(pips_gained) as avg_pips,
                SUM(pips_gained) as total_pips,
                AVG(confidence) as avg_confidence
            FROM signals
            WHERE outcome IN ('win', 'loss')
            GROUP BY symbol
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        result = {}
        for row in rows:
            symbol, total, wins, losses, avg_pips, total_pips, avg_conf = row
            result[symbol] = {
                'total': total,
                'wins': wins,
                'losses': losses,
                'win_rate': wins / total if total > 0 else 0,
                'avg_pips': avg_pips or 0,
                'total_pips': total_pips or 0,
                'avg_confidence': avg_conf or 0
            }
        
        return result
    
    def check_expired_signals(self) -> List[str]:
        """Check and mark expired signals"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        
        # Get expired pending signals
        cursor.execute('''
            SELECT signal_id FROM signals 
            WHERE outcome = 'pending' AND expires_at < ?
        ''', (now,))
        
        expired_ids = [row[0] for row in cursor.fetchall()]
        
        # Mark as expired
        for signal_id in expired_ids:
            cursor.execute('''
                UPDATE signals SET outcome = 'expired', exit_time = ?
                WHERE signal_id = ?
            ''', (now, signal_id))
        
        conn.commit()
        conn.close()
        
        return expired_ids


# Global instance
signal_tracker = SignalTracker()
