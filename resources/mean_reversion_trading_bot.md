Create a production-ready Python trading bot for MetaTrader5 (MT5) **exclusively optimized for Lirunex Demo Account trading XAUUSD.std (Gold)** with a built-in backtesting module (no test mode, pure production + backtest). The bot must follow strict Lirunex compliance rules, use a mean reversion strategy (14 RSI + 20 SMA), and include a fully error-proof backtest module with visualization. All code must be runnable with no syntax/logic errors (fix IndexError, division by zero, empty data).

## CORE PRODUCTION REQUIREMENTS (LIRUNEX COMPLIANT)
### 1. Broker Compliance (Fix Retcode 10015/10022/10004)
- **Price Rules**:
  - BUY LIMIT: 0.2 pips BELOW current market BID (tick.bid - 0.2) — validate price < bid
  - SELL LIMIT: 0.2 pips ABOVE current market BID (tick.bid + 0.2) — validate price > bid
  - Never use ask price for pending order calculations (fixes retcode 10015)
- **Order Rules**:
  - No `type_filling` field for pending orders (fixes retcode 10022)
  - Deviation = 0 (Lirunex requirement)
  - Order time = mt5.ORDER_TIME_GTC (no expiry)
  - Comments ≤15 chars (no special characters: "XAUUSD Bot Order")
- **Universal Rules**:
  - 0.01 lot size (Lirunex demo minimum) with step validation
  - Demo-only lock (block real account trading: trade_mode == 0)
  - Price normalization to 2 decimal places (XAUUSD.std requirement)

### 2. Trading Strategy (1-Minute Timeframe)
- **Indicators**: 14-period RSI (NumPy-only) + 20-period SMA + 10-period ATR
- **BUY SIGNAL**: RSI < 30 (oversold) + latest close < 20SMA → place BUY LIMIT
- **SELL SIGNAL**: RSI > 70 (overbought) + latest close > 20SMA → place SELL LIMIT
- **SL/TP**: 1xATR (SL), 2xATR (TP) → strict 2:1 risk-reward ratio

### 3. Risk Management (Mandatory)
- Pre-order cleanup: Close opposite positions + cancel pending orders (magic-number matched)
- Duplicate prevention: No new orders if existing positions/orders exist (same magic number)
- Unique magic number (123456) to isolate bot trades
- 60-second signal check interval (configurable)

### 4. Production Infrastructure
- Modular functions (MT5 init, indicator calc, order placement, risk management)
- Robust error handling (no crashes from None returns, retry on data errors)
- Clean MT5 shutdown (mt5.shutdown() in finally block)
- Human-readable logging (market updates, signal triggers, order status)
- All settings configurable (top-level variables: no hardcoding)

## BACKTEST MODULE (FULLY ERROR-PROOF)
### 1. Backtest Configuration (Top-Level Variables)
- BACKTEST_ENABLED (bool): Toggle backtest/production mode
- BACKTEST_START_DATE: 7 days of historical data (datetime.now() - timedelta(days=7))
- BACKTEST_LOT_SIZE: 0.01 (same as production)
- BACKTEST_PIP_VALUE: 1.0 (XAUUSD.std: 1 pip = $1 for 0.01 lot)

### 2. Historical Data Fetch
- Fetch 1-minute OHLC data via mt5.copy_rates_range()
- Calculate RSI/SMA/ATR for every row (rolling windows)
- Drop rows with missing indicators (initial period)

### 3. Backtest Execution (Error-Proof)
- **Critical Fixes**:
  - Skip last 5 candles to avoid empty next_candles
  - Check for len(next_candles) == 0 (return 0 pips if empty)
  - Add division by zero checks (win rate, avg profit)
  - Handle empty balance series (max drawdown)
- Simulate trades based on production strategy signals
- Track trade history (timestamp, type, entry/exit price, pips, profit, balance)
- Calculate key stats: total trades, win rate, total profit, max drawdown, avg profit per trade

### 4. Trade Outcome Simulation
- Check if SL/TP is hit in subsequent candles
- If no SL/TP hit: exit at last candle close
- If no next candles: exit at entry price (0 pips)

### 5. Visualization (Matplotlib)
- 4-panel plot:
  1. Account balance over time (line chart)
  2. Profit/loss per trade (bar chart, green/red)
  3. Win/loss distribution (pie chart)
  4. Pips per trade (bar chart, green/red)
- Save plot as 'backtest_results.png' (300 DPI)

## TECHNICAL REQUIREMENTS
- Python 3.8+ compatible (Lirunex MT5 package support)
- Use only official MT5/Pandas/NumPy/Matplotlib libraries
- No unused code/comments
- In-line comments for critical logic
- Production/backtest mode separation (no overlap)
- Fix all errors: IndexError (empty next_candles), ZeroDivisionError, empty data

## OUTPUT REQUIREMENTS
- Full runnable Python code (single file)
- Production mode: Pure live trading (no test orders)
- Backtest mode: Run backtest → plot results → exit (no production trading)
- Detailed backtest summary (print stats)
- Production troubleshooting notes (Lirunex retcodes)