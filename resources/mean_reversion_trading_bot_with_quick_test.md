Create a production-ready Python trading bot for MetaTrader5 (MT5) **exclusively optimized for Lirunex Demo Account trading XAUUSD.std (Gold)** with a dedicated TEST MODE (no syntax errors) + post-test production mode. The bot must follow strict variable scoping rules, Lirunex broker compliance, and a mean reversion strategy — all code must be runnable with no syntax/logic errors.

## CORE SCRIPT STRUCTURE (NON-NEGOTIABLE)
1. **Global Configuration Section**: Top-level editable variables for all strategy/broker settings (no hardcoded values).
2. **Modular Function Design**: Isolated functions for every core task (MT5 init, lot validation, indicator calculation, order placement, test mode) — no monolithic code.
3. **Variable Scoping Fix**: Declare `global` variables **only at the start of the function** where they’re modified (eliminates "used prior to global declaration" syntax errors).
4. **Isolated Test Mode**: Test mode logic in a separate `run_test_mode()` function; main loop calls this function first (if enabled) then switches to production mode.
5. **Clean Shutdown**: MT5 connection closed via `mt5.shutdown()` in a `finally` block (guaranteed execution on exit/Ctrl+C).
6. **Robust Error Handling**: No crashes from `None` returns, retries on data/indicator errors, human-readable error messages for all MT5 operations.

## TEST MODE REQUIREMENTS (CONFIGURABLE + AUTO-SWITCH)
1. **Top-Level Toggles**: Editable `TEST_MODE (bool)` and `TEST_TRADE_TYPE (mt5.ORDER_TYPE_BUY/SELL)` for one-click test order type selection.
2. **Test Mode First Execution**: If `TEST_MODE=True`, run test mode **immediately on startup** (before production mode signal monitoring).
3. **Auto-Disable Test Mode**: Set `TEST_MODE=False` after test order execution (no repeated test orders in the main loop).
4. **Minimal Test Data**: Fetch only the required bars (ATR_PERIOD +10) for test order SL/TP calculation (no unnecessary data fetching).
5. **Test Order Logic**: Reuse the production `place_pending_order()` function for test orders (ensures Lirunex compliance for test/prod orders).
6. **Test Mode Logging**: Detailed debug output for test order execution (current bid/ask, pending price validation, order status, ticket number).
7. **Post-Test Instructions**: Print clear steps to disable test mode/change test order type on script exit.

## LIRUNEX BROKER COMPLIANCE (FIX RETCODE 10015/10022/10004)
### Critical Price Fix (Retcode 10015: Invalid Price)
- **BUY LIMIT**: 0.2 pips **BELOW the current market BID** (`tick.bid - 0.2`) — validate price < bid before sending.
- **SELL LIMIT**: 0.2 pips **ABOVE the current market BID** (`tick.bid + 0.2`) — validate price > bid before sending.
- **NO ASK PRICE USAGE**: Never use `tick.ask` for pending order price calculation (Lirunex rejects ask-based SELL LIMIT prices).

### Order Request Rules (Retcode 10022: Invalid Filling Mode)
- **Pending Orders**: Omit the `type_filling` field entirely (use Lirunex’s default RETURN mode; explicit filling mode is rejected).
- **Market/Close Orders**: Use `mt5.ORDER_FILLING_FOK` (only valid for market/close operations on Lirunex).
- **Deviation**: Set `deviation=0` for all orders (Lirunex rejects any deviation > 0).
- **Order Time**: `mt5.ORDER_TIME_GTC` (no expiry — avoids timestamp validation errors).
- **Comments**: Max 15 characters, no special characters (e.g., "XAUUSD Bot Order" — Lirunex rejects long/special character comments).
- **Lot Size**: 0.01 (Lirunex demo minimum for Gold) with **automatic step validation** (round to the broker’s `volume_step`).

### Universal Lirunex Rules
- **Demo-Only Lock**: Block execution if connected to a real account (validate `account_info.trade_mode == 0`).
- **Symbol Validation**: Enable XAUUSD.std in Market Watch if hidden (Lirunex requires explicit symbol selection).
- **Price Normalization**: Round all prices (pending, SL, TP) to XAUUSD.std’s decimal places (2) — no unnormalized prices.

## TRADING STRATEGY (1-MINUTE TIMEFRAME)
### Mean Reversion with Trend Confirmation
- **Timeframe**: `mt5.TIMEFRAME_M1` (1-minute OHLC bars).
- **Indicators**: 14-period RSI (oversold/overbought) + 20-period SMA (trend filter) + 10-period ATR (SL/TP).
- **BUY SIGNAL**: RSI < 30 (oversold) **AND** latest close price < 20-period SMA (bearish trend) → place BUY LIMIT.
- **SELL SIGNAL**: RSI > 70 (overbought) **AND** latest close price > 20-period SMA (bullish trend) → place SELL LIMIT.
- **No Signal**: Print clear "no trading signal" message if RSI is in the 30-70 range or price/SMA trend filter fails.

### Indicator Calculation Rules
- **NumPy-Only Calculations**: RSI, SMA, ATR calculated with NumPy (no heavy libraries like TA-Lib — fast execution).
- **SMA**: Simple moving average of close prices (convolve method for efficiency).
- **RSI**: Classic wilder’s RSI (gain/loss smoothing, no division by zero).
- **ATR**: Average True Range (max of high-low, high-prev close, low-prev close) — 10-period smoothing.
- **Indicator Return**: A single dictionary with the **latest value** of SMA, RSI, ATR, and last close/high/low (no unused data).

## RISK MANAGEMENT (MANDATORY BEFORE ANY ORDER)
1. **Pre-Order Cleanup**: Close all **magic number-matched** opposite open positions **and** cancel all **magic number-matched** pending orders for XAUUSD.std — run this before test/production orders.
2. **Duplicate Prevention**: Check for existing magic number-matched open positions **OR** pending orders for the signal type — block new orders if duplicates exist.
3. **Fixed Risk-Reward**: 1x ATR for Stop Loss (SL), 2x ATR for Take Profit (TP) — strict 2:1 RR for all orders.
4. **Unique Magic Number**: Editable `MAGIC_NUMBER` to isolate bot trades from manual trades (no cross-contamination).
5. **Signal Check Interval**: Editable `CHECK_INTERVAL` (seconds) for production mode signal monitoring (default 60s).

## LOGGING REQUIREMENTS (HUMAN-READABLE + DETAILED)
1. **Startup Logs**: MT5 connection status, demo account login, symbol validation, lot size validation (green checkmarks for success).
2. **Market Updates**: Timestamp, current RSI, close price, SMA, market bid (core strategy values) — printed on every signal check.
3. **Order Debug**: Current market bid/ask, pending price, SL/TP, lot size, deviation — printed before every order send.
4. **Signal Triggers**: Highlighted BUY/SELL signal messages (emojis: 📈/📉) with RSI/price/SMA values that triggered the signal.
5. **Order Status**: Clear success/failure messages for all orders (test/prod/close/cancel) + MT5 ticket number (if successful).
6. **Error Logs**: MT5 retcode, error message, and actionable fix (e.g., "Enable automated trading in MT5") for all failed operations.
7. **Mode Switches**: Highlighted messages for test mode start/complete and production mode activation (emoji: 🔄).

## TECHNICAL REQUIREMENTS
- **MT5 Compatibility**: Use only official `MetaTrader5` Python package functions (no custom MT5 wrappers).
- **Pandas for Data Handling**: Minimal Pandas usage (only to convert MT5 rate data to a usable format — no complex operations).
- **Python Version**: Compatible with Python 3.8+ (Lirunex MT5 Python package support).
- **No Unused Code**: Remove all commented-out code, unused variables, and test/production mode placeholders.
- **In-Line Comments**: Concise, explanatory comments for all functions and critical code blocks (no over-commenting).
- **Configurable All**: Every strategy/broker/execution setting is an editable top-level variable (no hardcoding).

## OUTPUT REQUIREMENTS
1. **Full Runnable Code**: A single Python file with no external dependencies (excluding MT5/Pandas/NumPy — standard for MT5 trading).
2. **1:1 Function Matching**: Replicate the function names and structure from the core script (initialize_mt5, validate_lot_size, calculate_indicators, etc.).
3. **Test Mode Instructions**: A printed block on script exit with steps to disable test mode, change test order type, and locate test orders in MT5.
4. **No Syntax Errors**: Strict adherence to Python variable scoping rules (global declarations at function start — no "used prior to global declaration" errors).
5. **Lirunex-Specific Troubleshooting**: Implicit error handling for Lirunex’s most common retcodes (10015, 10022, 10004) with actionable fixes.