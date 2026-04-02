#!/usr/bin/env python3
"""
MT5 Windows Bridge for WSL Backend
This script runs on Windows (native) and sends MT5 data to the WSL backend.

Setup:
1. Install MetaTrader5 on Windows
2. Install Python on Windows: https://www.python.org/downloads/
3. Install MT5 Python package: pip install MetaTrader5 requests
4. Run this script on Windows: python mt5_windows_bridge.py

Usage:
    python mt5_windows_bridge.py --wsl-host <WSL_IP> --port 8000

To find your WSL IP address, run in WSL: hostname -I
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Check if MT5 is available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.error("MetaTrader5 package not installed!")
    logger.error("Install with: pip install MetaTrader5")
    sys.exit(1)

# HTTP client
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.error("requests package not installed!")
    logger.error("Install with: pip install requests")
    sys.exit(1)


# Symbol mapping
SYMBOL_MAP = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD",
    "EURGBP": "EURGBP",
    "EURJPY": "EURJPY",
    "GBPJPY": "GBPJPY",
    "USDCHF": "USDCHF",
    "NZDUSD": "NZDUSD",
    "USDCAD": "USDCAD",
    "XAUUSD": "XAUUSD",
}

# MT5 Timeframe constants
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class MT5Bridge:
    """Bridge between MT5 and WSL backend"""

    def __init__(self, wsl_host: str, port: int = 8000):
        self.wsl_host = wsl_host
        self.port = port
        self.base_url = f"http://{wsl_host}:{port}"
        self.running = False
        self.connected = False

    def connect_mt5(self) -> bool:
        """Connect to MT5 terminal"""
        if not MT5_AVAILABLE:
            return False

        # Initialize MT5
        if not mt5.initialize():
            logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False

        # Check version
        version = mt5.version()
        if version is None:
            logger.error("MT5 not found or not running")
            return False

        logger.info(f"MT5 connected: version {version[0]}.{version[1]}.{version[2]}")
        self.connected = True
        return True

    def disconnect_mt5(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        self.connected = False
        logger.info("MT5 disconnected")

    def get_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current tick from MT5"""
        if not self.connected:
            return None

        mt5_symbol = SYMBOL_MAP.get(symbol, symbol)

        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick is None:
            logger.warning(f"No tick data for {mt5_symbol}")
            return None

        return {
            "symbol": symbol,
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "spread": float((tick.ask - tick.bid) * (10000 if "JPY" not in symbol else 100)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "volume": int(tick.volume)
        }

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "M15",
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """Get OHLCV candles from MT5"""
        if not self.connected:
            return []

        mt5_symbol = SYMBOL_MAP.get(symbol, symbol)
        mt5_timeframe = TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_M15)

        rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_timeframe, 0, count)
        if rates is None:
            logger.warning(f"No OHLCV data for {mt5_symbol}")
            return []

        result = []
        for rate in rates:
            # Convert numpy types to Python native types for JSON serialization
            result.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "open": float(rate['open']),
                "high": float(rate['high']),
                "low": float(rate['low']),
                "close": float(rate['close']),
                "volume": int(rate['tick_volume']),
                "timestamp": datetime.fromtimestamp(int(rate['time']), tz=timezone.utc).isoformat()
            })

        return result

    def send_tick_to_wsl(self, tick: Dict[str, Any]) -> bool:
        """Send tick data to WSL backend"""
        if not REQUESTS_AVAILABLE:
            return False

        try:
            url = f"{self.base_url}/api/mt5/tick"
            response = requests.post(url, json=tick, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending tick to WSL: {e}")
            return False

    def send_ohlcv_to_wsl(self, symbol: str, timeframe: str, ohlcv: List[Dict]) -> bool:
        """Send OHLCV data to WSL backend"""
        if not REQUESTS_AVAILABLE:
            return False

        try:
            url = f"{self.base_url}/api/mt5/ohlcv"
            response = requests.post(url, json={
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": ohlcv
            }, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending OHLCV to WSL: {e}")
            return False

    def check_wsl_connection(self) -> bool:
        """Check if WSL backend is reachable"""
        try:
            url = f"{self.base_url}/health"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"WSL backend connected: {self.base_url}")
                logger.info(f"  Status: {data.get('status')}, Version: {data.get('version')}")
                return True
        except Exception as e:
            logger.error(f"Cannot connect to WSL backend: {e}")
        return False

    def run(self, symbols: List[str], interval: float = 1.0):
        """Main loop to stream data"""
        self.running = True

        logger.info(f"Starting MT5 -> WSL bridge for symbols: {symbols}")
        logger.info(f"WSL endpoint: {self.base_url}")

        # Check WSL connection
        if not self.check_wsl_connection():
            logger.warning("WSL backend not reachable. Data will be logged only.")

        tick_count = 0
        last_ohlcv_time = 0

        while self.running:
            try:
                for symbol in symbols:
                    # Get and send tick
                    tick = self.get_tick(symbol)
                    if tick:
                        success = self.send_tick_to_wsl(tick)
                        tick_count += 1

                        if tick_count % 10 == 0:
                            status = "✓" if success else "✗"
                            logger.info(
                                f"{status} {symbol}: {tick['bid']:.5f} | "
                                f"Spread: {tick['spread']:.1f} pips"
                            )

                    # Send OHLCV every 60 seconds
                    if time.time() - last_ohlcv_time > 60:
                        ohlcv = self.get_ohlcv(symbol, "M15", 100)
                        if ohlcv:
                            success = self.send_ohlcv_to_wsl(symbol, "M15", ohlcv)
                            if success:
                                logger.info(f"✓ Sent {len(ohlcv)} M15 candles for {symbol}")
                        last_ohlcv_time = time.time()

                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("Stopping bridge...")
                self.running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)

        self.disconnect_mt5()
        logger.info("Bridge stopped")


def main():
    parser = argparse.ArgumentParser(
        description="MT5 Windows Bridge for WSL Forex Backend"
    )
    parser.add_argument(
        "--wsl-host",
        required=True,
        help="WSL IP address (find with 'hostname -I' in WSL)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="WSL backend port (default: 8000)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["EURUSD", "GBPUSD", "USDJPY"],
        help="Trading symbols to stream"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Tick interval in seconds (default: 1.0)"
    )

    args = parser.parse_args()

    # Create and run bridge
    bridge = MT5Bridge(args.wsl_host, args.port)

    if not bridge.connect_mt5():
        logger.error("Failed to connect to MT5")
        sys.exit(1)

    bridge.run(args.symbols, args.interval)


if __name__ == "__main__":
    main()
