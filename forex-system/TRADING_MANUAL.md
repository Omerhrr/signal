# Forex Probability Intelligence System
## Complete Trading Manual

**Version 1.0 | April 2026**

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Getting Started](#getting-started)
4. [Dashboard Navigation](#dashboard-navigation)
5. [Understanding Signals](#understanding-signals)
6. [Duration-Based Predictions](#duration-based-predictions)
7. [Market Analysis Page](#market-analysis-page)
8. [Risk Management](#risk-management)
9. [Performance Tracking](#performance-tracking)
10. [Trading Strategies](#trading-strategies)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)
13. [FAQ](#faq)

---

## Introduction

### What is the Forex Probability Intelligence System?

The Forex Probability Intelligence System is an advanced trading analysis platform that combines multiple analytical approaches to generate high-probability trading signals. Unlike traditional trading systems that rely on a single methodology, this system integrates:

- **Machine Learning Predictions**: XGBoost and LightGBM models trained on historical market patterns
- **Statistical Analysis**: Monte Carlo Markov Chain (MCMC) probability estimation
- **Hidden Markov Models**: Market regime detection (trending, ranging, volatile)
- **Volume Analysis**: Smart money flow and volume profile analysis
- **Multi-Timeframe Confluence**: Signal confirmation across multiple timeframes
- **Duration-Based Predictions**: Time-horizon specific forecasts (1, 2, 3, 5, 15 minutes)

### Key Philosophy

The system is built on the principle that **probability, not certainty**, drives successful trading. Every signal comes with:

- Confidence scores
- Uncertainty quantification
- Risk/reward ratios
- Duration estimates
- Win probability based on historical performance

---

## System Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WINDOWS HOST                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           MetaTrader 5 Terminal                      │    │
│  │        (Real-time Market Data Source)                │    │
│  └────────────────────────┬────────────────────────────┘    │
│                           │                                  │
│  ┌────────────────────────▼────────────────────────────┐    │
│  │         MT5 Windows Bridge                           │    │
│  │    (Sends data to WSL via HTTP)                      │    │
│  └────────────────────────┬────────────────────────────┘    │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            │ HTTP POST (ticks & OHLCV)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      WSL (Ubuntu)                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          FastAPI Backend (Port 8000)                 │    │
│  │    - Data Processing                                 │    │
│  │    - ML Predictions                                  │    │
│  │    - Signal Generation                               │    │
│  │    - Auto Scanner                                    │    │
│  └────────────────────────┬────────────────────────────┘    │
│                           │                                  │
│  ┌────────────────────────▼────────────────────────────┐    │
│  │          Flask Frontend (Port 5000)                  │    │
│  │    - Web Dashboard                                   │    │
│  │    - Real-time Updates                               │    │
│  │    - Performance Tracking                            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Purpose | Access |
|-----------|---------|--------|
| **Dashboard** | Overview of all trading pairs and signals | http://localhost:5000 |
| **Analysis Page** | Deep analysis of a single pair | http://localhost:5000/analysis |
| **Duration Page** | Time-based predictions | http://localhost:5000/duration |
| **Performance Page** | Historical performance tracking | http://localhost:5000/performance |
| **Signals Page** | Active and recent signals | http://localhost:5000/signals |
| **API** | Backend endpoints | http://localhost:8000 |

---

## Getting Started

### Prerequisites

1. **MetaTrader 5** installed and running on Windows
2. **Python 3.10+** installed on both Windows and WSL
3. **Active trading account** (demo recommended for testing)

### Step 1: Configure Trading Pairs

Edit the `.env` file to set your trading pairs:

```bash
# In /home/z/my-project/.env
TRADING_PAIRS=EURUSD,GBPUSD,USDJPY,AUDUSD,XAUUSD
DEFAULT_PAIR=EURUSD
```

**Recommended Pairs for Beginners:**
- EURUSD (most liquid, tightest spreads)
- GBPUSD (good volatility)
- USDJPY (clear trends)

**Advanced Pairs:**
- XAUUSD (Gold - higher volatility)
- GBPJPY (high volatility, larger moves)
- EURJPY, AUDJPY (carry trade pairs)

### Step 2: Start the Backend

In WSL terminal:

```bash
cd /home/z/my-project/forex-system
python run.py
```

You should see:
```
Starting Forex Probability Intelligence System...
API running on http://0.0.0.0:8000
Frontend running on http://0.0.0.0:5000
```

### Step 3: Start MT5 Bridge

On Windows:

```cmd
cd C:\Users\USER\Desktop\signal\forex-system
python mt5_windows_bridge.py --wsl-host <WSL_IP>
```

Replace `<WSL_IP>` with your WSL IP address (find with `hostname -I` in WSL).

### Step 4: Access the Dashboard

Open your browser and navigate to:
- **Dashboard**: http://localhost:5000
- **API Health**: http://localhost:8000/health

---

## Dashboard Navigation

### Main Dashboard (`/`)

The main dashboard provides a real-time overview of all configured trading pairs.

#### Market Radar Section

Shows all trading pairs with key metrics:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Symbol** | Trading pair name | - |
| **Price** | Current bid price | - |
| **Spread** | Bid-ask spread in pips | Lower is better (< 2 pips ideal) |
| **Trend** | Market direction | BUY, SELL, or NEUTRAL |
| **Volatility** | Current market volatility | High = more risk/opportunity |
| **Session** | Current market session | Sydney, Tokyo, London, New York |

**Session Overlaps (Best Trading Times):**
- **London + New York** (13:00-17:00 UTC): Highest liquidity
- **Tokyo + London** (08:00-09:00 UTC): Moderate activity
- **Sydney + Tokyo** (00:00-07:00 UTC): Asian session

#### Signal Cards Section

Displays active trading signals generated by the system:

```
┌─────────────────────────────────────────┐
│  EURUSD - BUY                           │
│  Confidence: 72%                        │
│  Entry Zone: 1.0850 - 1.0855           │
│  Target: 1.0890 (+40 pips)             │
│  Stop Loss: 1.0830 (-20 pips)          │
│  R:R Ratio: 2:1                         │
│  Duration: ~15 minutes                  │
└─────────────────────────────────────────┘
```

### Understanding Signal Quality

**Confidence Levels:**
- **70%+**: High confidence - consider trading
- **60-70%**: Moderate confidence - use additional confirmation
- **55-60%**: Low confidence - wait for better setup
- **<55%**: No clear direction - skip

---

## Understanding Signals

### Signal Components

Every signal includes:

#### 1. Direction (BUY/SELL)
The predicted market direction based on multiple factors:
- Technical analysis patterns
- ML model predictions
- Multi-timeframe confluence
- Volume flow analysis

#### 2. Entry Zone
A price range for optimal entry:
```
Entry Zone: 1.0850 - 1.0855
Current Price: 1.0852 (within zone - good to enter)
```

**Best Practice:** Enter when price is within or near the entry zone.

#### 3. Stop Loss (SL)
Price level to exit if the trade goes against you:
```
Stop Loss: 1.0830
Risk: 20 pips from entry
```

**Rule:** Never move stop loss further from entry. Only trail it to lock in profits.

#### 4. Take Profit (TP)
Target price level to exit with profit:
```
Take Profit: 1.0890
Reward: 40 pips from entry
```

#### 5. Risk/Reward Ratio
The relationship between potential loss and gain:
```
Risk: 20 pips
Reward: 40 pips
R:R = 1:2 (Good!)
```

**Minimum R:R:** Never take trades with R:R worse than 1:1.5

#### 6. Expected Duration
How long the trade is expected to take:
```
Expected Duration: 15 minutes
```

### Signal Status Indicators

| Status | Color | Meaning |
|--------|-------|---------|
| **OK** | Green | Signal is valid and active |
| **LOW_CONFIDENCE** | Yellow | Confidence below threshold |
| **HIGH_VOLATILITY** | Orange | Market too volatile |
| **REVERSAL_DETECTED** | Purple | Market direction changed |
| **SPREAD_TOO_HIGH** | Brown | Spread exceeds limit |

---

## Duration-Based Predictions

### What Are Duration Predictions?

Unlike traditional signals that predict direction alone, duration predictions tell you **how long** to hold a position. The system generates forecasts for specific time horizons:

| Duration | Use Case |
|----------|----------|
| **1 minute** | Scalping, quick momentum plays |
| **2 minutes** | Very short-term trades |
| **3 minutes** | Short-term momentum |
| **5 minutes** | Standard short-term trade |
| **15 minutes** | Extended short-term trade |

### Duration Page (`/duration`)

Access the duration predictions at: http://localhost:5000/duration

#### Reading Duration Signals

```
┌──────────────────────────────────────────────────────┐
│  EURUSD - SELL for 3 minutes                         │
│  Entry: 1.0850                                       │
│  Target: 1.0835 (15 pips in 3 min)                  │
│  Confidence: 68%                                     │
│  Win Rate (Historical): 64%                          │
│                                                      │
│  [START TIME: 14:30] [END TIME: 14:33]              │
│  Status: ACTIVE                                      │
│  Outcome: Pending...                                 │
└──────────────────────────────────────────────────────┘
```

#### How Duration Tracking Works

1. **Signal Generated**: System identifies opportunity
2. **Countdown Starts**: Timer begins for the specified duration
3. **Auto-Tracking**: System monitors price action
4. **Outcome Recorded**: Win/Loss automatically determined

**Win Criteria:**
- Price moves in predicted direction by target amount
- Duration completes without hitting stop loss

**Loss Criteria:**
- Price moves against prediction beyond threshold
- Stop loss hit before duration ends

### Best Durations to Trade

The system tracks win rates by duration. Check the **Best Duration** metric:

```bash
curl http://localhost:8000/api/duration/best-duration
```

Example response:
```json
{
  "best_duration_minutes": 5,
  "win_rate": 0.67
}
```

This indicates 5-minute trades have historically performed best.

---

## Market Analysis Page

### Accessing Analysis

Navigate to: http://localhost:5000/analysis

Or click **Analyze** on any pair in the dashboard.

### Analysis Components

#### 1. Probability Meter

Shows the current probability of price moving UP or DOWN:

```
┌─────────────────────────────────────────────┐
│           PROBABILITY METER                 │
│                                             │
│   BEARISH ◄────────────► BULLISH           │
│         [████████░░░░░░░░░░░] 35%          │
│                                             │
│   Probability UP: 65%                       │
│   Probability DOWN: 35%                     │
│   Confidence: 72%                           │
└─────────────────────────────────────────────┘
```

**Interpretation:**
- **65%+ UP**: Look for BUY opportunities
- **65%+ DOWN**: Look for SELL opportunities
- **50-65% either way**: Market uncertain, wait

#### 2. Duration Meter

Shows expected price movement duration:

```
┌─────────────────────────────────────────────┐
│           DURATION FORECAST                 │
│                                             │
│   Expected Move Time: 8 minutes            │
│   Confidence: 70%                           │
│                                             │
│   ┌─────┬─────┬─────┬─────┬─────┐          │
│   │ 1m  │ 3m  │ 5m  │10m  │15m  │          │
│   │ 45% │ 62% │ 75% │ 68% │ 55% │          │
│   └─────┴─────┴─────┴─────┴─────┘          │
│          Best: 5 minutes                    │
└─────────────────────────────────────────────┘
```

#### 3. Regime Analysis (HMM)

Market regime detection using Hidden Markov Models:

| Regime | Description | Trading Approach |
|--------|-------------|------------------|
| **Trending** | Clear directional movement | Follow the trend |
| **Ranging** | Price moving sideways | Trade boundaries |
| **Volatile** | Large price swings | Reduce position size |
| **Transition** | Regime changing | Wait for clarity |

#### 4. Volume Analysis

Volume profile and smart money indicators:

```
Volume Ratio: 1.5x average (Elevated)
Volume Trend: Increasing
Smart Money Flow: BUYING
Breakout Probability: 45%
```

**Interpretation:**
- **High Volume + Price Up**: Strong bullish move
- **High Volume + Price Down**: Strong bearish move
- **Low Volume + Price Move**: Weak move, may reverse

#### 5. MCMC Probability Estimation

Monte Carlo simulations for probability bounds:

```json
{
  "direction_probability": {
    "mean": 0.65,
    "std": 0.08,
    "ci_95": [0.52, 0.78]
  },
  "confidence_score": 0.72
}
```

**95% Confidence Interval:** The true probability is between 52% and 78% with 95% certainty.

---

## Risk Management

### Position Sizing

The system provides risk management recommendations. Never risk more than 2% of your account on a single trade.

**Position Size Formula:**
```
Position Size = (Account Balance × Risk %) / Stop Loss in pips
```

**Example:**
```
Account Balance: $10,000
Risk per Trade: 2% = $200
Stop Loss: 20 pips
Position Size: $200 / 20 pips = $10 per pip
```

### Risk Settings in .env

```bash
# Maximum risk per trade (2% recommended)
MAX_RISK_PER_TRADE=0.02

# Minimum confidence to generate signal
MIN_CONFIDENCE_THRESHOLD=0.65

# Maximum spread allowed (in pips)
MAX_SPREAD_PIPS=3.0

# Volatility spike threshold
VOLATILITY_SPIKE_THRESHOLD=3.0
```

### Failure Detection

The system automatically detects potential failures:

| Failure Type | Detection | Action |
|--------------|-----------|--------|
| **Instant Failure** | Price moves 15+ pips against entry immediately | Exit immediately |
| **Slow Failure** | Trade stuck for 20+ minutes without progress | Consider exit |
| **Fake Breakout** | Price reverses after initial move | Trail stop tightly |
| **Volatility Spike** | Volatility exceeds 3x normal | Pause trading |

### When to Pause Trading

The system will recommend pausing when:
1. Multiple consecutive losses detected
2. Market volatility is abnormally high
3. Spread widens significantly
4. Regime transition detected

Check pause status:
```bash
curl http://localhost:8000/api/risk/should-pause
```

---

## Performance Tracking

### Performance Page

Navigate to: http://localhost:5000/performance

#### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Win Rate** | Percentage of winning trades | >55% |
| **Profit Factor** | Gross profit / Gross loss | >1.5 |
| **Average Win** | Average profit per winning trade | - |
| **Average Loss** | Average loss per losing trade | Should be < Average Win |
| **Expectancy** | Expected profit per trade | >0 |
| **Max Drawdown** | Largest peak-to-trough decline | <10% |

#### Performance by Symbol

Shows win rate and profitability for each trading pair:

```
┌──────────┬──────────┬──────────┬───────────┐
│ Symbol   │ Win Rate │ Trades   │ Profit    │
├──────────┼──────────┼──────────┼───────────┤
│ EURUSD   │ 62%      │ 45       │ +$320     │
│ GBPUSD   │ 58%      │ 38       │ +$180     │
│ USDJPY   │ 55%      │ 22       │ +$45      │
│ XAUUSD   │ 48%      │ 15       │ -$120     │
└──────────┴──────────┴──────────┴───────────┘
```

**Action:** Consider removing pairs with win rate <50% from your config.

#### Confidence Accuracy

Shows actual win rate grouped by predicted confidence:

```
Confidence 60-65%: Actual Win Rate 58%
Confidence 65-70%: Actual Win Rate 64%
Confidence 70-75%: Actual Win Rate 71%
Confidence 75%+  : Actual Win Rate 76%
```

**Ideal:** Higher confidence should correlate with higher win rates.

### API Endpoints for Performance

```bash
# Overall statistics
curl http://localhost:8000/api/performance/stats

# Performance by symbol
curl http://localhost:8000/api/performance/by-symbol

# Confidence accuracy
curl http://localhost:8000/api/performance/confidence-accuracy

# Duration statistics
curl http://localhost:8000/api/duration/stats
```

---

## Trading Strategies

### Strategy 1: High Confidence Scalping (1-3 minutes)

**Best For:** Active traders who can monitor charts

**Setup:**
1. Wait for signal with 70%+ confidence
2. Check duration prediction shows 1-3 minutes
3. Verify volume is above average
4. Enter at entry zone price

**Rules:**
- Take profit at predicted target
- Exit if duration expires without reaching target
- Never hold beyond predicted duration

**Risk Management:**
- Use tight stop loss (10-15 pips)
- Risk max 1% per trade
- Max 5 trades per session

### Strategy 2: Standard Duration Trading (5-15 minutes)

**Best For:** Part-time traders

**Setup:**
1. Wait for signal with 65%+ confidence
2. Duration prediction 5-15 minutes
3. Check multi-timeframe confluence (multiple timeframes agree)
4. Enter at entry zone

**Rules:**
- Set take profit and stop loss
- Let trade run for predicted duration
- Trail stop loss after 50% profit

**Risk Management:**
- Risk max 2% per trade
- R:R minimum 1:1.5

### Strategy 3: Confluence Trading

**Best For:** Conservative traders

**Setup:**
1. Wait for signal on multiple timeframes (M5, M15, M30)
2. All timeframes must show same direction
3. Check HMM regime is "Trending"
4. Enter when all conditions align

**Rules:**
- Only trade when 3+ timeframes agree
- Skip if any timeframe shows opposite direction
- Higher position size allowed (2-3% risk)

**Risk Management:**
- Trail stop loss aggressively
- Exit if any timeframe reverses

### Strategy 4: Session Overlap Trading

**Best For:** Time-specific traders

**Setup:**
1. Trade only during session overlaps:
   - London + New York: 13:00-17:00 UTC
   - Tokyo + London: 08:00-09:00 UTC
2. Focus on pairs relevant to active sessions
3. Higher volume and volatility expected

**Rules:**
- EURUSD, GBPUSD for London/New York overlap
- USDJPY, EURJPY for Tokyo session
- XAUUSD during London/New York overlap

---

## Best Practices

### DO's

1. **Wait for Confirmation**
   - Don't chase signals that already moved
   - Enter only at entry zone prices

2. **Check Multiple Indicators**
   - Probability meter
   - Duration prediction
   - Volume analysis
   - Regime detection

3. **Respect Stop Losses**
   - Never widen stop loss
   - Trail stops to lock profits

4. **Track Your Performance**
   - Review weekly performance
   - Remove underperforming pairs
   - Adjust confidence thresholds

5. **Trade During Active Sessions**
   - London + New York overlap is best
   - Avoid trading during low liquidity

6. **Start with Demo**
   - Test system thoroughly before live trading
   - Understand signal quality and timing

### DON'Ts

1. **Don't Overtrade**
   - Quality over quantity
   - Max 10 trades per day

2. **Don't Ignore Risk Management**
   - Never risk more than 2% per trade
   - Don't add to losing positions

3. **Don't Trade During News**
   - High volatility during news releases
   - Wait 15 minutes after major news

4. **Don't Chase Losses**
   - After 3 consecutive losses, take a break
   - Review what went wrong

5. **Don't Blindly Follow Signals**
   - Use signals as guidance, not commands
   - Apply your own judgment

### Daily Routine

**Morning (Pre-Market):**
1. Check overnight signals and outcomes
2. Review performance stats
3. Check current regime and volatility
4. Identify pairs with active signals

**During Trading:**
1. Monitor active signals
2. Enter at entry zones
3. Set alerts for take profit and stop loss
4. Track duration countdown

**Evening (Post-Market):**
1. Review all trades taken
2. Update your trading journal
3. Note any lessons learned
4. Plan for next session

---

## Troubleshooting

### No Signals Appearing

**Possible Causes:**
1. MT5 bridge not running
2. Market closed (weekend)
3. Confidence threshold too high
4. No clear market conditions

**Solutions:**
```bash
# Check MT5 connection
curl http://localhost:8000/api/mt5/status

# Check if data is coming through
curl http://localhost:8000/api/ticks

# Lower confidence threshold in .env
MIN_CONFIDENCE_THRESHOLD=0.55
```

### Probability Meter Stuck at 50%

**Possible Causes:**
1. Insufficient data for prediction
2. Model not trained
3. Market in consolidation

**Solutions:**
```bash
# Check data availability
curl http://localhost:8000/api/ohlcv/EURUSD

# Check if model exists (first run may need training time)
# The system uses heuristic predictions initially
```

### Signals Not Being Tracked

**Possible Causes:**
1. Outcome tracker not started
2. Database issues
3. Auto scanner not running

**Solutions:**
```bash
# Check scanner status
curl http://localhost:8000/api/scanner/status

# Start scanner manually
curl -X POST http://localhost:8000/api/scanner/start

# Check duration signals
curl http://localhost:8000/api/duration/active-signals
```

### Backend Not Starting

**Possible Causes:**
1. Port already in use
2. Missing dependencies
3. Database locked

**Solutions:**
```bash
# Check if ports are in use
lsof -i :8000
lsof -i :5000

# Kill existing process
kill -9 <PID>

# Reinstall dependencies
pip install -r requirements.txt

# Clear database lock
rm db/custom.db-journal
```

### MT5 Bridge Connection Failed

**Possible Causes:**
1. Wrong WSL IP address
2. Firewall blocking
3. Backend not running

**Solutions:**
```bash
# Get correct WSL IP
hostname -I

# Check if backend is accessible
curl http://localhost:8000/health

# On Windows, check firewall
# Allow Python through Windows Firewall
```

---

## FAQ

### General Questions

**Q: How accurate are the signals?**
A: Signal accuracy varies by pair and market conditions. Check the Performance page for current win rates. Typically expect 55-65% win rate with proper risk management.

**Q: Can I use this for live trading?**
A: Yes, but thoroughly test on demo first. Understand the system, its strengths, and limitations before risking real money.

**Q: Which pairs work best?**
A: Major pairs (EURUSD, GBPUSD, USDJPY) typically perform best due to liquidity and predictable patterns. XAUUSD can be profitable but requires experience.

**Q: How often are signals generated?**
A: Depends on market conditions. During active sessions, expect 5-15 signals per hour across all pairs. Quiet markets may produce fewer signals.

### Technical Questions

**Q: What's the minimum confidence I should trade?**
A: Start with 65%+. As you gain experience, you may adjust based on performance data.

**Q: How do duration predictions work?**
A: The system analyzes how long similar market conditions took to reach targets historically. It's a statistical estimate, not a guarantee.

**Q: Can I add more trading pairs?**
A: Yes! Edit the `TRADING_PAIRS` in `.env` file. Make sure the pair is available in your MT5 Market Watch.

**Q: Why do I see "No existing model found" in logs?**
A: This is normal for new installations. The system uses heuristic predictions initially. Models are trained automatically as data accumulates.

### Risk Questions

**Q: What's the maximum drawdown I should expect?**
A: With proper risk management, max drawdown should stay under 10%. If it exceeds this, review your position sizing and pair selection.

**Q: Should I take every signal?**
A: No. Use signals as a starting point for analysis. Consider current market conditions, news events, and your own judgment.

**Q: How do I know when to stop trading?**
A: The system will recommend pausing when conditions are unfavorable. Also pause after 3+ consecutive losses or during major news events.

---

## Quick Reference Card

### Key URLs
| Page | URL |
|------|-----|
| Dashboard | http://localhost:5000 |
| Analysis | http://localhost:5000/analysis |
| Duration | http://localhost:5000/duration |
| Performance | http://localhost:5000/performance |
| Signals | http://localhost:5000/signals |
| API Docs | http://localhost:8000/docs |

### Key Commands
```bash
# Start backend
python run.py

# Start MT5 bridge
python mt5_windows_bridge.py --wsl-host <IP>

# Check health
curl http://localhost:8000/health

# Get current signals
curl http://localhost:8000/api/signals

# Start scanner
curl -X POST http://localhost:8000/api/scanner/start
```

### Risk Management Rules
| Rule | Value |
|------|-------|
| Max Risk per Trade | 2% |
| Min R:R Ratio | 1:1.5 |
| Min Confidence | 65% |
| Max Spread | 3 pips |
| Max Daily Trades | 10 |
| Stop after Losses | 3 consecutive |

### Trading Session Times (UTC)
| Session | Start | End | Best Pairs |
|---------|-------|-----|------------|
| Sydney | 22:00 | 07:00 | AUD pairs |
| Tokyo | 00:00 | 09:00 | JPY pairs |
| London | 08:00 | 17:00 | EUR, GBP pairs |
| New York | 13:00 | 22:00 | USD pairs |
| **L+N Overlap** | **13:00** | **17:00** | **All pairs** |

---

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review logs: `logs/forex_system.log`
3. Check API health: `http://localhost:8000/health`

---

*Happy Trading! Remember: Probability, not certainty, drives success.*
