"""
Forex Probability Intelligence System - Data Ingestion Layer
MT5 Data Source Only - Receives data from Windows MT5 Bridge
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Callable
from loguru import logger
import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config.settings import get_settings
from app.models.schemas import TickData, OHLCV

settings = get_settings()


class MT5DataStore:
    """Store and manage MT5 data received from Windows bridge"""

    def __init__(self):
        self.ticks: Dict[str, TickData] = {}
        self.ohlcv_data: Dict[str, List[OHLCV]] = {}
        self.last_update: Dict[str, datetime] = {}
        self.connected = False
        self._max_ohlcv_cache = 500

    def update_tick(self, symbol: str, tick_data: dict) -> TickData:
        """Update tick data from MT5 bridge"""
        tick = TickData(
            symbol=symbol.upper(),
            bid=float(tick_data.get('bid', 0)),
            ask=float(tick_data.get('ask', 0)),
            spread=float(tick_data.get('spread', 0)),
            timestamp=datetime.fromisoformat(tick_data['timestamp']) if isinstance(tick_data.get('timestamp'), str) else datetime.now(timezone.utc),
            volume=int(tick_data.get('volume', 0))
        )
        self.ticks[symbol.upper()] = tick
        self.last_update[symbol.upper()] = datetime.now(timezone.utc)
        self.connected = True
        return tick

    def update_ohlcv(self, symbol: str, timeframe: str, candles: list) -> List[OHLCV]:
        """Update OHLCV data from MT5 bridge"""
        symbol = symbol.upper()
        key = f"{symbol}_{timeframe}"
        
        ohlcv_list = []
        for candle in candles:
            ohlcv = OHLCV(
                symbol=symbol,
                timeframe=timeframe,
                open=float(candle.get('open', 0)),
                high=float(candle.get('high', 0)),
                low=float(candle.get('low', 0)),
                close=float(candle.get('close', 0)),
                volume=int(candle.get('volume', 0)),
                timestamp=datetime.fromisoformat(candle['timestamp']) if isinstance(candle.get('timestamp'), str) else datetime.now(timezone.utc)
            )
            ohlcv_list.append(ohlcv)
        
        self.ohlcv_data[key] = ohlcv_list[-self._max_ohlcv_cache:]
        self.last_update[symbol] = datetime.now(timezone.utc)
        return ohlcv_list

    def get_tick(self, symbol: str) -> Optional[TickData]:
        """Get latest tick for symbol"""
        return self.ticks.get(symbol.upper())

    def get_ohlcv(self, symbol: str, timeframe: str = "M15") -> List[OHLCV]:
        """Get OHLCV data for symbol"""
        key = f"{symbol.upper()}_{timeframe}"
        return self.ohlcv_data.get(key, [])

    def is_data_fresh(self, symbol: str, max_age_seconds: int = 30) -> bool:
        """Check if data is fresh"""
        symbol = symbol.upper()
        if symbol not in self.last_update:
            return False
        age = (datetime.now(timezone.utc) - self.last_update[symbol]).total_seconds()
        return age < max_age_seconds


class RedisStreamManager:
    """Manage Redis Streams for real-time data"""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.connected = False

    async def connect(self) -> bool:
        if not REDIS_AVAILABLE:
            return False

        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password if settings.redis_password else None,
                decode_responses=True
            )
            self.redis_client.ping()
            self.connected = True
            logger.info("Redis connected successfully")
            return True
        except Exception as e:
            logger.warning(f"Redis connection error: {e}")
            return False

    async def disconnect(self):
        if self.redis_client:
            self.redis_client.close()
            self.connected = False

    async def publish_tick(self, tick: TickData):
        if not self.connected:
            return

        stream_key = f"ticks:{tick.symbol}"
        try:
            self.redis_client.xadd(stream_key, {
                "bid": str(tick.bid),
                "ask": str(tick.ask),
                "spread": str(tick.spread),
                "timestamp": tick.timestamp.isoformat(),
                "volume": str(tick.volume or 0)
            })
        except Exception as e:
            logger.error(f"Error publishing tick: {e}")


class DataIngestionPipeline:
    """Main data ingestion pipeline - MT5 only"""

    def __init__(self):
        self.mt5_store = MT5DataStore()
        self.redis_manager = RedisStreamManager()
        self.tick_callbacks: List[Callable[[TickData], None]] = []
        self.running = False
        self.connected = False
        self.data_source = "mt5"

    async def start(self):
        """Start the data ingestion pipeline"""
        self.running = True
        self.connected = True
        
        logger.info("=" * 50)
        logger.info("✓ MT5 Data Pipeline Started")
        logger.info("  Waiting for data from Windows MT5 Bridge...")
        logger.info("  Make sure mt5_windows_bridge.py is running on Windows")
        logger.info("=" * 50)

        # Try Redis (optional)
        await self.redis_manager.connect()
        
        return True

    async def stop(self):
        """Stop the data ingestion pipeline"""
        self.running = False
        await self.redis_manager.disconnect()
        self.connected = False

    def receive_tick(self, symbol: str, tick_data: dict) -> TickData:
        """Receive tick data from MT5 bridge"""
        tick = self.mt5_store.update_tick(symbol, tick_data)
        
        # Publish to Redis
        asyncio.create_task(self.redis_manager.publish_tick(tick))
        
        # Call registered callbacks
        for callback in self.tick_callbacks:
            try:
                callback(tick)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        return tick

    def receive_ohlcv(self, symbol: str, timeframe: str, candles: list) -> List[OHLCV]:
        """Receive OHLCV data from MT5 bridge"""
        return self.mt5_store.update_ohlcv(symbol, timeframe, candles)

    def register_tick_callback(self, callback: Callable[[TickData], None]):
        """Register callback for new ticks"""
        self.tick_callbacks.append(callback)

    async def get_tick(self, symbol: str) -> Optional[TickData]:
        """Get latest tick for a symbol"""
        return self.mt5_store.get_tick(symbol)

    async def get_ohlcv(self, symbol: str, timeframe: str = "M1", count: int = 100) -> List[OHLCV]:
        """Get OHLCV data for a symbol"""
        ohlcv = self.mt5_store.get_ohlcv(symbol, timeframe)
        return ohlcv[-count:] if ohlcv else []

    async def get_market_state(self, symbol: str) -> Dict[str, Any]:
        """Get current market state for a symbol"""
        tick = self.mt5_store.get_tick(symbol)
        ohlcv = self.mt5_store.get_ohlcv(symbol, "M15")
        
        if not tick:
            return {
                "symbol": symbol,
                "status": "no_data",
                "message": "Waiting for MT5 data...",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        base_state = {
            "symbol": symbol,
            "current_price": tick.bid,
            "spread": tick.spread,
            "session": self._get_current_session(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_source": "mt5",
            "data_fresh": self.mt5_store.is_data_fresh(symbol)
        }
        
        if not ohlcv or len(ohlcv) < 20:
            return base_state
        
        # Calculate indicators
        closes = [c.close for c in ohlcv[-50:]]
        highs = [c.high for c in ohlcv[-50:]]
        lows = [c.low for c in ohlcv[-50:]]
        
        # ATR
        ranges = [h - l for h, l in zip(highs, lows)]
        atr = sum(ranges[-14:]) / 14 if len(ranges) >= 14 else sum(ranges) / len(ranges)
        
        # Trend (SMA crossover)
        sma_short = sum(closes[-5:]) / 5
        sma_long = sum(closes[-20:]) / 20 if len(closes) >= 20 else sum(closes) / len(closes)
        trend = "BUY" if sma_short > sma_long else "SELL"
        
        # Momentum
        momentum = (closes[-1] - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0
        
        # RSI
        rsi = self._calculate_rsi(closes)
        
        return {
            **base_state,
            "volatility": atr,
            "volatility_pct": atr / tick.bid * 100 if tick.bid else 0,
            "trend": trend,
            "momentum": momentum,
            "rsi": rsi
        }

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    def _get_current_session(self) -> str:
        """Determine current trading session"""
        hour = datetime.now(timezone.utc).hour
        
        if 13 <= hour < 17:
            return "new_york"
        elif 8 <= hour < 13:
            return "london"
        elif 0 <= hour < 9:
            return "tokyo"
        elif 22 <= hour or hour < 7:
            return "sydney"
        else:
            return "london"


# Global instance
data_pipeline = DataIngestionPipeline()
