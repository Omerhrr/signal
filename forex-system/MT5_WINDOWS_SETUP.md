# MetaTrader 5 (MT5) Windows Setup Guide for WSL

This guide explains how to set up MT5 on Windows and connect it to your Forex Probability Intelligence System running in WSL.

---

## Quick Start

### Step 1: Get WSL IP Address

```bash
# In WSL terminal
hostname -I
```

Note the IP address (e.g., `21.0.7.148`).

### Step 2: Install MT5 on Windows

1. Download MT5 from your broker or [MetaQuotes](https://www.metatrader5.com/en/download)
2. Install and open MT5
3. Log in to your account (demo or live)

### Step 3: Install Python Dependencies on Windows

Open Command Prompt on Windows:

```cmd
pip install MetaTrader5 requests
```

### Step 4: Run the Bridge

On Windows, navigate to the project folder and run:

```cmd
cd C:\Users\USER\Desktop\signal\forex-system
python mt5_windows_bridge.py --wsl-host 21.0.7.148
```

Replace `21.0.7.148` with your actual WSL IP from Step 1.

### Step 5: Start the WSL Backend

In WSL:

```bash
cd /home/z/my-project/forex-system
python run.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Windows Host                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              MetaTrader 5 Terminal                       │    │
│  │   - Real-time forex data                                 │    │
│  │   - Trading execution                                    │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │ Python API                          │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            mt5_windows_bridge.py                         │    │
│  │   - Sends data via HTTP POST                             │    │
│  │   - http://WSL_IP:8000/api/mt5/tick                      │    │
│  └────────────────────────┬────────────────────────────────┘    │
└───────────────────────────┼─────────────────────────────────────┘
                            │ HTTP POST
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WSL (Ubuntu)                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │          Forex Probability Intelligence System           │    │
│  │   - FastAPI backend (port 8000)                          │    │
│  │   - Flask frontend (port 5000)                           │    │
│  │   - ML prediction engine                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Setup

### MT5 Installation on Windows

1. **Choose your broker** (recommended brokers for API access):
   - **MetaQuotes Demo**: Good for testing
   - **IC Markets**: Low spreads, supports API
   - **Pepperstone**: Good API support
   - **FXCM**: Institutional-grade API

2. **Download & Install MT5**:
   - Go to your broker's website
   - Download the MT5 terminal for Windows
   - Run the installer with default settings

3. **Create Account**:
   - Open MT5 terminal
   - Go to **File → Open an Account**
   - Search for your broker's demo server
   - Create a demo account (recommended for testing)
   - Log in with your credentials

### Broker Account Setup

**Demo Account (Recommended for Testing)**:
1. In MT5: **File → Open an Account**
2. Search for your broker
3. Select "Demo" server
4. Fill in registration form
5. Initial deposit: $10,000 (virtual)

**Live Account (For Real Trading)**:
- Contact your broker to open a live account
- Complete KYC verification
- ⚠️ Only use live accounts after thorough testing

---

## Command Reference

### MT5 Bridge Options

```cmd
python mt5_windows_bridge.py --wsl-host <IP> [options]

Required:
  --wsl-host <IP>       WSL IP address (find with 'hostname -I' in WSL)

Optional:
  --port <PORT>         Backend port (default: 8000)
  --symbols EURUSD GBPUSD USDJPY   Symbols to stream (default: EURUSD GBPUSD USDJPY)
  --interval <SECONDS>  Tick interval (default: 1.0)
```

### Example Commands

```cmd
# Basic usage
python mt5_windows_bridge.py --wsl-host 21.0.7.148

# With custom symbols
python mt5_windows_bridge.py --wsl-host 21.0.7.148 --symbols EURUSD GBPUSD USDJPY XAUUSD

# Faster tick rate
python mt5_windows_bridge.py --wsl-host 21.0.7.148 --interval 0.5
```

---

## API Endpoints

The MT5 bridge sends data to these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mt5/tick` | POST | Receive tick data |
| `/api/mt5/ohlcv` | POST | Receive OHLCV candles |
| `/api/mt5/status` | GET | Check MT5 connection status |
| `/health` | GET | System health check |

### Tick Data Format

```json
{
  "symbol": "EURUSD",
  "bid": 1.08500,
  "ask": 1.08502,
  "spread": 2.0,
  "volume": 150,
  "timestamp": "2024-01-15T10:30:00+00:00"
}
```

### OHLCV Data Format

```json
{
  "symbol": "EURUSD",
  "timeframe": "M15",
  "candles": [
    {
      "open": 1.08500,
      "high": 1.08550,
      "low": 1.08480,
      "close": 1.08530,
      "volume": 1250,
      "timestamp": "2024-01-15T10:15:00+00:00"
    }
  ]
}
```

---

## Troubleshooting

### Issue: "Cannot connect to WSL backend"

**Solution**:
1. Verify WSL IP is correct: `hostname -I`
2. Make sure the backend is running: `python run.py`
3. Check Windows Firewall allows Python

### Issue: "MT5 initialize() failed"

**Solution**:
1. Make sure MT5 terminal is running and logged in
2. Run the bridge script with administrator privileges
3. Check MT5 terminal is 64-bit version

### Issue: "No tick data for SYMBOL"

**Solution**:
1. Open MT5 Market Watch (View → Market Watch)
2. Right-click → Show All
3. Ensure the symbol is visible in Market Watch

### Issue: Windows Firewall Blocking

**Solution** - Run in PowerShell (Admin):
```powershell
New-NetFirewallRule -DisplayName "Python WSL" -Direction Outbound -Action Allow
```

### Issue: Wrong WSL IP

The correct WSL IP can be found with:
```bash
# In WSL
hostname -I
# OR
ip route show | grep -i default | awk '{print $3}'
```

---

## Testing the Connection

### Test 1: Check Backend Health

In WSL:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "data_source": "mt5",
  "mt5_connected": false
}
```

### Test 2: Start MT5 Bridge

On Windows:
```cmd
python mt5_windows_bridge.py --wsl-host <WSL_IP>
```

Expected output:
```
MT5 connected: version 500.5723.31 Mar 2026
WSL backend connected: http://21.0.7.148:8000
Starting MT5 -> WSL bridge for symbols: ['EURUSD', 'GBPUSD', 'USDJPY']
✓ EURUSD: 1.08500 | Spread: 1.5 pips
```

### Test 3: Verify Data in Backend

In WSL:
```bash
curl http://localhost:8000/api/mt5/status
```

Expected response:
```json
{
  "mt5_connected": true,
  "symbols": {
    "EURUSD": {"has_data": true, "is_fresh": true},
    "GBPUSD": {"has_data": true, "is_fresh": true},
    "USDJPY": {"has_data": true, "is_fresh": true}
  }
}
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Get WSL IP | `hostname -I` |
| Start backend | `python run.py` |
| Start MT5 bridge | `python mt5_windows_bridge.py --wsl-host <IP>` |
| Check health | `curl http://localhost:8000/health` |
| Check MT5 status | `curl http://localhost:8000/api/mt5/status` |
| View dashboard | Open `http://localhost:5000` in browser |

---

## Additional Resources

- [MetaTrader 5 Python API Documentation](https://www.mql5.com/en/docs/python_metatrader5)
- [MT5 Platform Download](https://www.metatrader5.com/en/download)
