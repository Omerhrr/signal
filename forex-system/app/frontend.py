"""
Forex Probability Intelligence System - Flask Frontend
Dashboard with HTMX + Alpine.js + TailwindCSS + ECharts
"""
import asyncio
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_cors import CORS
from loguru import logger
import httpx

from config.settings import get_settings, SIGNAL_STATUS, RISK_LEVELS, MARKET_SESSIONS

settings = get_settings()

# Get the project root directory (forex-system folder)
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "templates"

# Create Flask app with explicit template folder
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.secret_key = "forex-intelligence-secret-key"
CORS(app)

# API base URL
API_BASE = f"http://localhost:{settings.api_port}"


# ============== Template Helpers ==============

@app.template_filter('datetime_format')
def datetime_format(value):
    """Format datetime for display"""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except:
            return value
    if isinstance(value, datetime):
        return value.strftime('%H:%M:%S')
    return str(value)


@app.template_filter('price_format')
def price_format(value):
    """Format price for display"""
    try:
        return f"{float(value):.5f}"
    except:
        return str(value)


@app.context_processor
def inject_globals():
    """Inject global variables into templates"""
    return {
        'signal_status': SIGNAL_STATUS,
        'risk_levels': RISK_LEVELS,
        'market_sessions': MARKET_SESSIONS,
        'trading_pairs': settings.trading_pairs,
        'app_name': settings.app_name
    }


# ============== Async API Calls ==============

async def fetch_api(endpoint: str, method: str = "GET", data: dict = None) -> Dict:
    """Fetch data from FastAPI backend"""
    async with httpx.AsyncClient() as client:
        url = f"{API_BASE}{endpoint}"
        try:
            if method == "GET":
                response = await client.get(url, timeout=5.0)
            elif method == "POST":
                response = await client.post(url, json=data, timeout=10.0)
            else:
                return {"error": "Invalid method"}
            
            return response.json()
        except Exception as e:
            logger.error(f"API fetch error: {e}")
            return {"error": str(e)}


def run_async(coro):
    """Run async function in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


# ============== Routes ==============

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('pages/dashboard.html')


@app.route('/signals')
def signals():
    """Signals page"""
    return render_template('pages/signals.html')


@app.route('/analysis/')
@app.route('/analysis/<symbol>')
def analysis(symbol=None):
    """Analysis page for specific symbol"""
    # Symbol can come from URL path or query parameter
    # Query parameter takes precedence
    query_symbol = request.args.get('symbol')
    if query_symbol:
        symbol = query_symbol.upper()
    elif symbol:
        symbol = symbol.upper()
    else:
        symbol = settings.default_pair
        
    return render_template('pages/analysis.html', symbol=symbol)


@app.route('/settings')
def settings_page():
    """Settings page"""
    return render_template('pages/settings.html')


@app.route('/performance')
def performance():
    """Performance tracking page"""
    return render_template('pages/performance.html')


@app.route('/duration')
def duration():
    """Duration-based predictions page"""
    return render_template('pages/duration.html')


# ============== HTMX Partial Routes ==============

@app.route('/partials/signal-cards')
def partial_signal_cards():
    """Partial: Signal cards for HTMX"""
    result = run_async(fetch_api("/api/signals/cards"))
    signals = result.get('data', [])
    return render_template('components/signal_cards.html', signals=signals)


@app.route('/partials/market-radar')
def partial_market_radar():
    """Partial: Market radar for HTMX"""
    result = run_async(fetch_api("/api/market-radar"))
    return render_template('components/market_radar.html', 
                          radar=result if 'error' not in result else None)


@app.route('/partials/tick-table')
def partial_tick_table():
    """Partial: Tick data table for HTMX"""
    result = run_async(fetch_api("/api/ticks"))
    ticks = result.get('data', {})
    return render_template('components/tick_table.html', ticks=ticks)


@app.route('/partials/confidence-meter')
def partial_confidence_meter():
    """Partial: Confidence meter visualization"""
    result = run_async(fetch_api("/api/signals/cards"))
    signals = result.get('data', [])
    return render_template('components/confidence_meter.html', signals=signals)


@app.route('/partials/failure-stats')
def partial_failure_stats():
    """Partial: Failure statistics"""
    result = run_async(fetch_api("/api/risk/failure-stats"))
    stats = result.get('data', {})
    return render_template('components/failure_stats.html', stats=stats)


# ============== API Proxy Routes ==============

@app.route('/api/ticks')
def api_ticks():
    """Proxy: Get all ticks"""
    result = run_async(fetch_api("/api/ticks"))
    return jsonify(result)


@app.route('/api/tick/<symbol>')
def api_tick(symbol):
    """Proxy: Get tick for symbol"""
    result = run_async(fetch_api(f"/api/tick/{symbol}"))
    return jsonify(result)


@app.route('/api/ohlcv/<symbol>')
def api_ohlcv(symbol):
    """Proxy: Get OHLCV data"""
    timeframe = request.args.get('timeframe', 'M15')
    count = request.args.get('count', 100)
    result = run_async(fetch_api(f"/api/ohlcv/{symbol}?timeframe={timeframe}&count={count}"))
    return jsonify(result)


@app.route('/api/signals')
def api_signals():
    """Proxy: Get signals"""
    symbol = request.args.get('symbol')
    status = request.args.get('status')
    endpoint = "/api/signals"
    params = []
    if symbol:
        params.append(f"symbol={symbol}")
    if status:
        params.append(f"status={status}")
    if params:
        endpoint += "?" + "&".join(params)
    
    result = run_async(fetch_api(endpoint))
    return jsonify(result)


@app.route('/api/signals/cards')
def api_signals_cards():
    """Proxy: Get signal cards"""
    result = run_async(fetch_api("/api/signals/cards"))
    return jsonify(result)


@app.route('/api/signals/generate', methods=['POST'])
def api_generate_signal():
    """Proxy: Generate signal"""
    data = request.json
    result = run_async(fetch_api("/api/signals/generate", "POST", data))
    return jsonify(result)


@app.route('/api/market-state/<symbol>')
def api_market_state(symbol):
    """Proxy: Get market state"""
    result = run_async(fetch_api(f"/api/market-state/{symbol}"))
    return jsonify(result)


@app.route('/api/market-radar')
def api_market_radar():
    """Proxy: Get market radar"""
    result = run_async(fetch_api("/api/market-radar"))
    return jsonify(result)


@app.route('/api/health')
def api_health():
    """Proxy: Health check"""
    result = run_async(fetch_api("/health"))
    return jsonify(result)


# ============== Performance API Routes ==============

@app.route('/api/performance/stats')
def api_performance_stats():
    """Get performance statistics"""
    days = request.args.get('days', 30)
    symbol = request.args.get('symbol', '')
    endpoint = f"/api/performance/stats?days={days}"
    if symbol:
        endpoint += f"&symbol={symbol}"
    result = run_async(fetch_api(endpoint))
    return jsonify(result)


@app.route('/api/performance/signals')
def api_performance_signals():
    """Get signal history"""
    limit = request.args.get('limit', 50)
    result = run_async(fetch_api(f"/api/performance/signals?limit={limit}"))
    return jsonify(result)


@app.route('/api/performance/by-symbol')
def api_performance_by_symbol():
    """Get performance by symbol"""
    result = run_async(fetch_api("/api/performance/by-symbol"))
    return jsonify(result)


@app.route('/api/hmm/regime')
def api_hmm_regime():
    """Get HMM regime state"""
    symbol = request.args.get('symbol', 'EURUSD')
    result = run_async(fetch_api(f"/api/hmm/regime?symbol={symbol}"))
    return jsonify(result)


@app.route('/api/mcmc/estimate')
def api_mcmc_estimate():
    """Get MCMC probability estimate"""
    symbol = request.args.get('symbol', 'EURUSD')
    result = run_async(fetch_api(f"/api/mcmc/estimate?symbol={symbol}"))
    return jsonify(result)


@app.route('/api/signals/<signal_id>/outcome', methods=['POST'])
def api_signal_outcome(signal_id):
    """Update signal outcome"""
    data = request.json
    result = run_async(fetch_api(f"/api/signals/{signal_id}/outcome", "POST", data))
    return jsonify(result)


# ============== Scanner API Routes ==============

@app.route('/api/scanner/status')
def api_scanner_status():
    """Get scanner status"""
    result = run_async(fetch_api("/api/scanner/status"))
    return jsonify(result)


@app.route('/api/scanner/start', methods=['POST'])
def api_scanner_start():
    """Start the scanner"""
    result = run_async(fetch_api("/api/scanner/start", "POST", {}))
    return jsonify(result)


@app.route('/api/scanner/stop', methods=['POST'])
def api_scanner_stop():
    """Stop the scanner"""
    result = run_async(fetch_api("/api/scanner/stop", "POST", {}))
    return jsonify(result)


@app.route('/api/scanner/signals')
def api_scanner_signals():
    """Get top scanner signals"""
    limit = request.args.get('limit', 10)
    result = run_async(fetch_api(f"/api/scanner/signals?limit={limit}"))
    return jsonify(result)


@app.route('/api/scanner/confluence')
def api_scanner_confluence():
    """Get confluence signals"""
    min_confluence = request.args.get('min_confluence', 0.6)
    result = run_async(fetch_api(f"/api/scanner/confluence?min_confluence={min_confluence}"))
    return jsonify(result)


@app.route('/api/scanner/scan-now', methods=['POST'])
def api_scanner_scan_now():
    """Trigger immediate scan"""
    result = run_async(fetch_api("/api/scanner/scan-now", "POST", {}))
    return jsonify(result)


@app.route('/api/scanner/symbols', methods=['POST'])
def api_scanner_symbols():
    """Set scanner symbols"""
    data = request.json
    result = run_async(fetch_api("/api/scanner/symbols", "POST", data))
    return jsonify(result)


@app.route('/api/scanner/config', methods=['POST'])
def api_scanner_config():
    """Configure scanner"""
    interval = request.args.get('interval_seconds')
    confidence = request.args.get('min_confidence')
    endpoint = "/api/scanner/config"
    params = []
    if interval:
        params.append(f"interval_seconds={interval}")
    if confidence:
        params.append(f"min_confidence={confidence}")
    if params:
        endpoint += "?" + "&".join(params)
    result = run_async(fetch_api(endpoint, "POST", {}))
    return jsonify(result)


# ============== Volume API Routes ==============

@app.route('/api/volume/analyze')
def api_volume_analyze():
    """Get volume analysis"""
    symbol = request.args.get('symbol', 'EURUSD')
    timeframe = request.args.get('timeframe', 'M15')
    result = run_async(fetch_api(f"/api/volume/analyze?symbol={symbol}&timeframe={timeframe}"))
    return jsonify(result)


@app.route('/api/volume/predict')
def api_volume_predict():
    """Get volume prediction"""
    symbol = request.args.get('symbol', 'EURUSD')
    timeframe = request.args.get('timeframe', 'M15')
    result = run_async(fetch_api(f"/api/volume/predict?symbol={symbol}&timeframe={timeframe}"))
    return jsonify(result)


@app.route('/api/volume/profile')
def api_volume_profile():
    """Get volume profile"""
    symbol = request.args.get('symbol', 'EURUSD')
    timeframe = request.args.get('timeframe', 'M15')
    result = run_async(fetch_api(f"/api/volume/profile?symbol={symbol}&timeframe={timeframe}"))
    return jsonify(result)


# ============== Duration Prediction API Routes ==============

@app.route('/api/duration/predict')
def api_duration_predict():
    """Get duration predictions"""
    symbol = request.args.get('symbol', 'EURUSD')
    timeframe = request.args.get('timeframe', 'M1')
    result = run_async(fetch_api(f"/api/duration/predict?symbol={symbol}&timeframe={timeframe}"))
    return jsonify(result)


@app.route('/api/duration/generate-signal', methods=['POST'])
def api_duration_generate_signal():
    """Generate duration signal"""
    symbol = request.args.get('symbol', 'EURUSD')
    timeframe = request.args.get('timeframe', 'M1')
    min_confidence = request.args.get('min_confidence', 0.55)
    max_noise = request.args.get('max_noise', 0.4)
    endpoint = f"/api/duration/generate-signal?symbol={symbol}&timeframe={timeframe}&min_confidence={min_confidence}&max_noise={max_noise}"
    result = run_async(fetch_api(endpoint, "POST", {}))
    return jsonify(result)


@app.route('/api/duration/active-signals')
def api_duration_active_signals():
    """Get active duration signals"""
    result = run_async(fetch_api("/api/duration/active-signals"))
    return jsonify(result)


@app.route('/api/duration/history')
def api_duration_history():
    """Get duration signal history"""
    limit = request.args.get('limit', 50)
    result = run_async(fetch_api(f"/api/duration/history?limit={limit}"))
    return jsonify(result)


@app.route('/api/duration/stats')
def api_duration_stats():
    """Get duration prediction stats"""
    result = run_async(fetch_api("/api/duration/stats"))
    return jsonify(result)


@app.route('/api/duration/best-duration')
def api_duration_best_duration():
    """Get best performing duration"""
    result = run_async(fetch_api("/api/duration/best-duration"))
    return jsonify(result)


# ============== Error Handlers ==============

@app.errorhandler(404)
def not_found(e):
    return render_template('pages/error.html', error="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('pages/error.html', error="Server error"), 500


if __name__ == "__main__":
    app.run(host=settings.flask_host, port=settings.flask_port, debug=settings.debug)
