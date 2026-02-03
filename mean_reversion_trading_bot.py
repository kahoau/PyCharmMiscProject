import MetaTrader5 as mt5
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set matplotlib style for better visuals
plt.style.use('seaborn-v0_8-darkgrid')

# -------------------------- CONFIGURATION --------------------------
# Core Trading Config
SYMBOL = "XAUUSD.std"  # Lirunex gold symbol (do not change)
LOT_SIZE = 0.01  # Lirunex demo minimum lot size
TIMEFRAME = mt5.TIMEFRAME_M1  # 1-minute chart for signal generation
RSI_PERIOD = 14  # RSI period for mean reversion signals
RSI_OVERSOLD = 30  # RSI BUY threshold (oversold)
RSI_OVERBOUGHT = 70  # RSI SELL threshold (overbought)
MA_PERIOD = 20  # SMA period for trend confirmation
ATR_PERIOD = 10  # ATR period for risk management
ATR_MULTIPLIER_SL = 1.0  # SL = 1xATR (risk management)
ATR_MULTIPLIER_TP = 2.0  # TP = 2xATR (2:1 risk-reward ratio)
MAGIC_NUMBER = 123456  # Unique ID for bot trades
CHECK_INTERVAL = 60  # Signal check interval (seconds)
PENDING_ORDER_OFFSET = 0.2  # Pending order offset (pips) from current bid

# Backtest Config
BACKTEST_DAYS = 7  # Number of days of historical data to backtest
BACKTEST_STARTING_BALANCE = 10000.0  # Starting balance for backtest
BACKTEST_OUTPUT_PATH = "backtest_results.png"  # Path to save backtest graph

# Mode Switch (KEY FEATURE: Set to "BACKTEST" or "PRODUCTION")
RUN_MODE = "BACKTEST"  # Options: "BACKTEST" (only run backtest) or "PRODUCTION" (backtest + live trading)


# ---------------------------------------------------------------------------------------

def initialize_mt5():
    """
    Initialize MT5 connection and validate Lirunex demo account/symbol
    Critical Lirunex Fix: Ensures connection to demo (not real) account
    """
    # Initialize MT5 (required for all MT5 operations)
    if not mt5.initialize():
        error = mt5.last_error()
        print(f"❌ MT5 Initialization Failed | Code: {error[0]} | Message: {error[1]}")
        return False

    # Verify demo account connection (Lirunex demo = trade_mode 0)
    account_info = mt5.account_info()
    if account_info is None:
        error = mt5.last_error()
        print(f"❌ Failed to Fetch Account Info | Error: {error[1]}")
        mt5.shutdown()
        return False

    # Critical check: Prevent real account trading
    if account_info.trade_mode != 0:
        print(f"⚠️ WARNING: Connected to REAL account (trade_mode={account_info.trade_mode})!")
        print("   Switch to Lirunex demo account before proceeding!")
        mt5.shutdown()
        return False
    else:
        print(f"✅ Connected to Lirunex Demo Account | Login: {account_info.login}")

    # Validate XAUUSD.std symbol (fixes "symbol not found" errors)
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        error = mt5.last_error()
        print(f"❌ {SYMBOL} Not Found | Error: {error[1]}")
        mt5.shutdown()
        return False

    # Enable symbol if hidden in MT5 Market Watch
    if not symbol_info.visible:
        if not mt5.symbol_select(SYMBOL, True):
            error = mt5.last_error()
            print(f"❌ Failed to Enable {SYMBOL} | Error: {error[1]}")
            mt5.shutdown()
            return False

    print(f"✅ Validated Symbol: {SYMBOL} | Decimal Places: {symbol_info.digits}")
    return True


def validate_lot_size():
    """
    Validate lot size against Lirunex rules (0.01 min, 0.01 step for XAUUSD.std)
    Critical Fix: Prevents "invalid volume" errors from Lirunex
    """
    global LOT_SIZE
    symbol_info = mt5.symbol_info(SYMBOL)

    # Get Lirunex's lot size parameters
    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lot_step = symbol_info.volume_step

    # Validate lot size range
    if LOT_SIZE < min_lot or LOT_SIZE > max_lot:
        print(f"❌ Invalid Lot Size | Lirunex Allows: {min_lot} - {max_lot} lots")
        return False

    # Round to Lirunex's required step (critical for 0.01 lot size)
    LOT_SIZE = round(LOT_SIZE / lot_step) * lot_step
    print(f"✅ Valid Lot Size: {LOT_SIZE} (Broker Step: {lot_step})")
    return True


def normalize_price(symbol: str, price: float) -> float:
    """
    Normalize price to Lirunex's decimal places (2 for XAUUSD.std)
    Used for pending order price/SL/TP calculations (critical for Lirunex validation)
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"⚠️ Could not fetch symbol info | Using original price: {price}")
        return price

    decimal_places = symbol_info.digits
    normalized_price = round(price, decimal_places)
    return normalized_price


def fetch_m1_data(periods: int):
    """
    Fetch 1-minute OHLC data for XAUUSD.std (required for indicator calculations)
    Fix: Uses mt5.copy_rates_from to avoid dtype errors with pandas
    """
    utc_now = datetime.now()
    rates = mt5.copy_rates_from(SYMBOL, TIMEFRAME, utc_now, periods)

    if rates is None or len(rates) < periods:
        error = mt5.last_error()
        print(f"❌ Failed to Fetch 1-Min Data | Error: {error[1]}")
        return None

    # Convert to DataFrame (only keep necessary columns)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    return {
        'time': df['time'].values,
        'open': df['open'].values,
        'high': df['high'].values,
        'low': df['low'].values,
        'close': df['close'].values
    }


def calculate_indicators(data):
    """
    Calculate core indicators: 14-period RSI, 20-period SMA, 10-period ATR
    Core Strategy: Mean reversion (RSI) + trend confirmation (SMA)
    """
    closes = data['close']
    highs = data['high']
    lows = data['low']

    # 1. Calculate 20-period SMA (trend confirmation)
    sma = np.convolve(closes, np.ones(MA_PERIOD) / MA_PERIOD, mode='valid')
    if len(sma) == 0:
        print("❌ Insufficient Data for SMA Calculation")
        return None

    # 2. Calculate 14-period RSI (mean reversion signal)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Smoothed average gain/loss (RSI formula)
    avg_gain = np.convolve(gains, np.ones(RSI_PERIOD) / RSI_PERIOD, mode='valid')
    avg_loss = np.convolve(losses, np.ones(RSI_PERIOD) / RSI_PERIOD, mode='valid')

    # Avoid division by zero (critical edge case)
    avg_loss[avg_loss == 0] = 0.0001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    if len(rsi) == 0:
        print("❌ Insufficient Data for RSI Calculation")
        return None

    # 3. Calculate 10-period ATR (risk management)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1])
    )
    atr = np.convolve(tr, np.ones(ATR_PERIOD) / ATR_PERIOD, mode='valid')
    if len(atr) == 0:
        print("❌ Insufficient Data for ATR Calculation")
        return None

    # Return latest indicator values (most recent candle)
    return {
        'sma': sma[-1],
        'rsi': rsi[-1],
        'atr': atr[-1],
        'last_close': closes[-1],
        'last_high': highs[-1],
        'last_low': lows[-1]
    }


def has_open_position_or_pending_order(order_type: int):
    """
    Check for existing positions OR pending orders (prevent duplicates)
    Critical for pending order strategy (avoids multiple pending orders)
    """
    # Check open positions
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is not None:
        for pos in positions:
            if pos.magic == MAGIC_NUMBER and pos.type == order_type:
                print(
                    f"⚠️ Open {('BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL')} Position | Ticket: {pos.ticket}")
                return True

    # Check pending orders
    pending_orders = mt5.orders_get(symbol=SYMBOL)
    if pending_orders is not None:
        target_order_type = mt5.ORDER_TYPE_BUY_LIMIT if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_SELL_LIMIT
        for order in pending_orders:
            if order.magic == MAGIC_NUMBER and order.type == target_order_type:
                print(
                    f"⚠️ Open {('BUY LIMIT' if target_order_type == mt5.ORDER_TYPE_BUY_LIMIT else 'SELL LIMIT')} Pending Order | Ticket: {order.ticket}")
                return True

    return False


def close_opposite_position_and_cancel_pending():
    """
    Close opposite positions AND cancel pending orders (risk management)
    Critical for pending order strategy (no hedging + no conflicting pending orders)
    """
    # Step 1: Close opposite positions
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is not None:
        for pos in positions:
            if pos.magic == MAGIC_NUMBER:
                # Get real-time close price (critical for Lirunex)
                tick = mt5.symbol_info_tick(SYMBOL)
                if tick is None:
                    print(f"❌ Failed to Fetch Tick Data | Position {pos.ticket}")
                    return False

                close_price = tick.bid if pos.type == mt5.ORDER_TYPE_SELL else tick.ask
                close_price = normalize_price(SYMBOL, close_price)

                # Build close request (Lirunex-optimized)
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": SYMBOL,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_BUY if pos.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                    "position": pos.ticket,
                    "price": close_price,
                    "deviation": 0,  # Critical Fix: No slippage (Lirunex rejects deviation>0)
                    "magic": MAGIC_NUMBER,
                    "comment": "Close Bot Position",  # Critical Fix: Short, clean comment
                    "type_filling": mt5.ORDER_FILLING_FOK,  # FOK is allowed for market orders
                    "type_time": mt5.ORDER_TIME_GTC
                }

                # Execute close request
                result = mt5.order_send(close_request)

                # Critical Fix: Check for None result (prevents crash)
                if result is None:
                    error = mt5.last_error()
                    print(f"❌ Failed to Send Close Request | Error: {error[1]}")
                    return False

                # Check if close was successful
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"✅ Closed Opposite Position | Ticket: {pos.ticket}")
                else:
                    print(f"❌ Failed to Close Position | Retcode: {result.retcode} | Error: {mt5.last_error()[1]}")
                    return False

    # Step 2: Cancel all pending orders from the bot
    pending_orders = mt5.orders_get(symbol=SYMBOL)
    if pending_orders is not None:
        for order in pending_orders:
            if order.magic == MAGIC_NUMBER:
                # Build cancel request
                cancel_request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                    "symbol": SYMBOL,
                    "magic": MAGIC_NUMBER,
                    "comment": "Cancel Pending Order"
                }

                # Execute cancel request
                result = mt5.order_send(cancel_request)

                if result is None:
                    error = mt5.last_error()
                    print(f"❌ Failed to Cancel Order | Error: {error[1]}")
                    return False

                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"✅ Cancelled Pending Order | Ticket: {order.ticket}")
                else:
                    print(f"❌ Failed to Cancel Order | Retcode: {result.retcode} | Error: {mt5.last_error()[1]}")
                    return False

    return True


def place_pending_order(order_type: int, indicators):
    """
    Place BUY LIMIT/SELL LIMIT pending order (fixes retcode 10015 for SELL LIMIT)
    Critical Fix: Uses bid price for both BUY/SELL LIMIT (BUY = bid-0.2, SELL = bid+0.2)
    """
    # Get real-time tick data (critical for valid pending order price)
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        error = mt5.last_error()
        print(f"❌ Failed to Fetch Tick Data | Error: {error[1]}")
        return False

    # Log current bid/ask for debugging
    print(f"\n📝 Pending Order Debug (Lirunex Fix)")
    print(f"   Current Market: Bid = {tick.bid:.2f}, Ask = {tick.ask:.2f}")

    # Calculate CORRECT pending order prices (fixes retcode 10015)
    if order_type == mt5.ORDER_TYPE_BUY:
        # BUY LIMIT: 0.2 pips below current bid (valid for Lirunex)
        pending_order_type = mt5.ORDER_TYPE_BUY_LIMIT
        pending_price = normalize_price(SYMBOL, tick.bid - PENDING_ORDER_OFFSET)
        sl = normalize_price(SYMBOL, pending_price - (ATR_MULTIPLIER_SL * indicators['atr']))
        tp = normalize_price(SYMBOL, pending_price + (ATR_MULTIPLIER_TP * indicators['atr']))
        order_label = "BUY LIMIT"

    else:
        # SELL LIMIT: 0.2 pips above current bid (VALID - fixes retcode 10015)
        pending_order_type = mt5.ORDER_TYPE_SELL_LIMIT
        pending_price = normalize_price(SYMBOL, tick.bid + PENDING_ORDER_OFFSET)
        sl = normalize_price(SYMBOL, pending_price + (ATR_MULTIPLIER_SL * indicators['atr']))
        tp = normalize_price(SYMBOL, pending_price - (ATR_MULTIPLIER_TP * indicators['atr']))
        order_label = "SELL LIMIT"

    # Validate pending price (debugging)
    if pending_order_type == mt5.ORDER_TYPE_BUY_LIMIT and pending_price >= tick.bid:
        print(f"⚠️ Invalid {order_label} Price: {pending_price:.2f} ≥ Bid ({tick.bid:.2f})")
        return False
    if pending_order_type == mt5.ORDER_TYPE_SELL_LIMIT and pending_price <= tick.bid:
        print(f"⚠️ Invalid {order_label} Price: {pending_price:.2f} ≤ Bid ({tick.bid:.2f})")
        return False

    # Build pending order request (100% Lirunex compliant)
    pending_request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": pending_order_type,
        "price": pending_price,
        "sl": sl,
        "tp": tp,
        "deviation": 0,
        "magic": MAGIC_NUMBER,
        "comment": "XAUUSD Bot Order",  # Short compliant comment
        "type_time": mt5.ORDER_TIME_GTC  # No expiry (GTC)
    }

    # Print order details
    print(f"   {order_label} Price: {pending_price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} (2:1 RR)")
    print(f"   Lot Size: {LOT_SIZE} | Deviation: 0 | No Expiry (GTC)")

    # Send order to MT5
    result = mt5.order_send(pending_request)

    # Check for errors
    if result is None:
        error = mt5.last_error()
        print(f"❌ Order Request Failed | Error: {error[1]}")
        return False

    # Verify order success
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ {order_label} Failed | Retcode: {result.retcode} | Error: {mt5.last_error()[1]}")
        print("   ⚠️ Ensure 'Allow automated trading' is enabled in MT5!")
        return False

    # Log successful order
    order_ticket = result.order
    print(f"\n✅ {order_label} Order Created Successfully!")
    print(f"   Ticket Number: {order_ticket} | Check MT5's 'Orders' tab to confirm!")
    return True


# -------------------------- BACKTEST MODULE (WITH VISUALIZATION) --------------------------
def run_simple_backtest():
    """
    Run a simple backtest using historical MT5 data + generate interactive visualization
    Returns: Backtest results dict (for production mode if needed)
    """
    print("=" * 60)
    print(f"📊 Running Backtest (Last {BACKTEST_DAYS} Days of 1-Min Data)")
    print("=" * 60)

    # Fetch historical data for backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=BACKTEST_DAYS)
    rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_date, end_date)

    if rates is None or len(rates) == 0:
        print(f"❌ Failed to Fetch Historical Data | Error: {mt5.last_error()[1]}")
        return None

    # Convert to DataFrame for analysis
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Precompute indicators for all candles
    df['sma'] = df['close'].rolling(window=MA_PERIOD).mean()

    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=RSI_PERIOD).mean()
    avg_loss = avg_loss.replace(0, 0.0001)
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Calculate ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.abs(df['high'] - df['close'].shift(1)),
        np.abs(df['low'] - df['close'].shift(1))
    )
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()

    # Backtest logic
    backtest_results = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_profit': 0.0,
        'balance': BACKTEST_STARTING_BALANCE,
        'max_balance': BACKTEST_STARTING_BALANCE,
        'max_drawdown': 0.0,
        'balance_history': [BACKTEST_STARTING_BALANCE],
        'drawdown_history': [0.0],
        'trade_times': [],
        'trades': []
    }

    in_position = False
    position_type = None  # 'BUY' or 'SELL'
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0

    # Iterate through candles (skip first N candles with incomplete indicators)
    for idx, row in df.iloc[max(RSI_PERIOD, MA_PERIOD, ATR_PERIOD):].iterrows():
        if not in_position:
            # BUY SIGNAL
            if row['rsi'] < RSI_OVERSOLD and row['close'] < row['sma'] and not np.isnan(row['atr']):
                in_position = True
                position_type = 'BUY'
                entry_price = row['close'] - PENDING_ORDER_OFFSET  # Simulate BUY LIMIT entry
                sl_price = entry_price - (ATR_MULTIPLIER_SL * row['atr'])
                tp_price = entry_price + (ATR_MULTIPLIER_TP * row['atr'])
                backtest_results['total_trades'] += 1
                backtest_results['trade_times'].append(idx)

            # SELL SIGNAL
            elif row['rsi'] > RSI_OVERBOUGHT and row['close'] > row['sma'] and not np.isnan(row['atr']):
                in_position = True
                position_type = 'SELL'
                entry_price = row['close'] + PENDING_ORDER_OFFSET  # Simulate SELL LIMIT entry
                sl_price = entry_price + (ATR_MULTIPLIER_SL * row['atr'])
                tp_price = entry_price - (ATR_MULTIPLIER_TP * row['atr'])
                backtest_results['total_trades'] += 1
                backtest_results['trade_times'].append(idx)

        else:
            # Check if SL/TP hit for BUY position
            if position_type == 'BUY':
                if row['low'] <= sl_price:
                    # SL hit (loss)
                    profit = (sl_price - entry_price) * LOT_SIZE * 100  # XAUUSD pip value = $10 per lot
                    backtest_results['balance'] += profit
                    backtest_results['losing_trades'] += 1
                    in_position = False
                    backtest_results['trades'].append(('BUY', entry_price, sl_price, 'LOSS', profit, idx))
                    backtest_results['balance_history'].append(backtest_results['balance'])
                elif row['high'] >= tp_price:
                    # TP hit (win)
                    profit = (tp_price - entry_price) * LOT_SIZE * 100
                    backtest_results['balance'] += profit
                    backtest_results['winning_trades'] += 1
                    in_position = False
                    backtest_results['trades'].append(('BUY', entry_price, tp_price, 'WIN', profit, idx))
                    backtest_results['balance_history'].append(backtest_results['balance'])

            # Check if SL/TP hit for SELL position
            elif position_type == 'SELL':
                if row['high'] >= sl_price:
                    # SL hit (loss)
                    profit = (entry_price - sl_price) * LOT_SIZE * 100
                    backtest_results['balance'] += profit
                    backtest_results['losing_trades'] += 1
                    in_position = False
                    backtest_results['trades'].append(('SELL', entry_price, sl_price, 'LOSS', profit, idx))
                    backtest_results['balance_history'].append(backtest_results['balance'])
                elif row['low'] <= tp_price:
                    # TP hit (win)
                    profit = (entry_price - tp_price) * LOT_SIZE * 100
                    backtest_results['balance'] += profit
                    backtest_results['winning_trades'] += 1
                    in_position = False
                    backtest_results['trades'].append(('SELL', entry_price, tp_price, 'WIN', profit, idx))
                    backtest_results['balance_history'].append(backtest_results['balance'])

        # Update max balance and drawdown
        backtest_results['max_balance'] = max(backtest_results['max_balance'], backtest_results['balance'])
        drawdown = (backtest_results['max_balance'] - backtest_results['balance']) / backtest_results[
            'max_balance'] * 100
        backtest_results['max_drawdown'] = max(backtest_results['max_drawdown'], drawdown)
        backtest_results['drawdown_history'].append(drawdown)

    # Calculate final metrics
    backtest_results['total_profit'] = backtest_results['balance'] - BACKTEST_STARTING_BALANCE
    win_rate = (backtest_results['winning_trades'] / backtest_results['total_trades'] * 100) if backtest_results[
                                                                                                    'total_trades'] > 0 else 0

    # Print backtest results
    print("\n📈 Backtest Results Summary")
    print(f"Starting Balance: ${BACKTEST_STARTING_BALANCE:.2f}")
    print(f"Final Balance: ${backtest_results['balance']:.2f}")
    print(f"Net Profit: ${backtest_results['total_profit']:.2f}")
    print(f"Total Trades: {backtest_results['total_trades']}")
    print(f"Winning Trades: {backtest_results['winning_trades']}")
    print(f"Losing Trades: {backtest_results['losing_trades']}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Max Drawdown: {backtest_results['max_drawdown']:.1f}%")
    print("=" * 60)

    # Generate backtest visualization
    plot_backtest_results(backtest_results)

    return backtest_results


def plot_backtest_results(results):
    """
    Generate interactive backtest visualization with 3 subplots:
    1. Balance over time
    2. Drawdown percentage
    3. Trade distribution (win/loss)
    """
    if results['total_trades'] == 0:
        print("⚠️ No trades to visualize - skipping backtest graph")
        return

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'XAUUSD Mean Reversion Backtest Results ({BACKTEST_DAYS} Days)', fontsize=16, fontweight='bold')

    # Subplot 1: Balance History
    ax1.plot(results['balance_history'], color='#2ecc71', linewidth=2, label='Account Balance')
    ax1.axhline(y=BACKTEST_STARTING_BALANCE, color='#e74c3c', linestyle='--', label='Starting Balance')
    ax1.set_title('Account Balance Over Time', fontweight='bold')
    ax1.set_ylabel('Balance ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Drawdown History
    ax2.plot(results['drawdown_history'], color='#f39c12', linewidth=2)
    ax2.set_title('Drawdown Percentage', fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.axhline(y=results['max_drawdown'], color='#e74c3c', linestyle='--',
                label=f'Max Drawdown: {results["max_drawdown"]:.1f}%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Trade Distribution (Pie Chart)
    trade_labels = ['Winning Trades', 'Losing Trades']
    trade_values = [results['winning_trades'], results['losing_trades']]
    colors = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax3.pie(
        trade_values,
        labels=trade_labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05)
    )
    ax3.set_title('Trade Distribution (Win/Loss)', fontweight='bold')

    # Customize pie chart text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(BACKTEST_OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✅ Backtest graph saved to: {BACKTEST_OUTPUT_PATH}")
    plt.show()


# -------------------------- PRODUCTION TRADING LOOP --------------------------
def main_trading_loop():
    """
    Main production trading loop: Monitor for RSI/SMA signals and place pending orders
    Only runs if RUN_MODE = "PRODUCTION"
    """
    print("\n🚀 Starting Lirunex XAUUSD.std Trading Bot (Production Mode)")
    print(f"   Strategy: 14 RSI + 20 SMA Mean Reversion | ATR 10 (2:1 RR)")
    print(f"   Signal Check Interval: {CHECK_INTERVAL}s | Pending Offset: {PENDING_ORDER_OFFSET} pip")
    print("----------------------------------------------------------------------")

    while True:
        try:
            # Fetch 1-minute OHLC data
            data = fetch_m1_data(periods=max(RSI_PERIOD, MA_PERIOD, ATR_PERIOD) + 10)
            if data is None:
                time.sleep(CHECK_INTERVAL)
                continue

            # Calculate core indicators
            indicators = calculate_indicators(data)
            if indicators is None:
                time.sleep(CHECK_INTERVAL)
                continue

            # Extract indicator values
            rsi = indicators['rsi']
            sma = indicators['sma']
            last_close = indicators['last_close']

            # Log market conditions
            tick = mt5.symbol_info_tick(SYMBOL)
            market_bid = tick.bid if tick else "N/A"
            print(f"\n⏳ Market Update | Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"   RSI: {rsi:.1f} | Price: {last_close:.2f} | SMA: {sma:.2f} | Bid: {market_bid}")

            # BUY SIGNAL
            if (rsi < RSI_OVERSOLD and last_close < sma and not has_open_position_or_pending_order(mt5.ORDER_TYPE_BUY)):
                print(f"\n📈 BUY SIGNAL TRIGGERED! (RSI: {rsi:.1f} < {RSI_OVERSOLD}, Price < SMA)")
                if close_opposite_position_and_cancel_pending():
                    place_pending_order(mt5.ORDER_TYPE_BUY, indicators)

            # SELL SIGNAL
            elif (rsi > RSI_OVERBOUGHT and last_close > sma and not has_open_position_or_pending_order(
                    mt5.ORDER_TYPE_SELL)):
                print(f"\n📉 SELL SIGNAL TRIGGERED! (RSI: {rsi:.1f} > {RSI_OVERBOUGHT}, Price > SMA)")
                if close_opposite_position_and_cancel_pending():
                    place_pending_order(mt5.ORDER_TYPE_SELL, indicators)

            # NO SIGNAL
            else:
                print(f"   No trading signal (RSI range: {RSI_OVERSOLD}-{RSI_OVERBOUGHT})")

            # Wait for next check
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n⚠️ Bot Stopped by User")
            break
        except Exception as e:
            print(f"❌ Unexpected Error: {str(e)} | Retrying in {CHECK_INTERVAL} seconds")
            time.sleep(CHECK_INTERVAL)


# -------------------------- MAIN EXECUTION --------------------------
if __name__ == "__main__":
    # Validate MT5 connection and lot size
    if not initialize_mt5():
        exit(1)
    if not validate_lot_size():
        mt5.shutdown()
        exit(1)

    # Run backtest (always runs first)
    backtest_results = run_simple_backtest()

    # Run production mode only if flag is set
    if RUN_MODE == "PRODUCTION" and backtest_results is not None:
        print("\n🔄 Switching to Production Mode (after backtest completion)")
        try:
            main_trading_loop()
        finally:
            mt5.shutdown()
            print("\n🔌 MT5 Connection Closed | Bot Terminated")
    else:
        # Only backtest mode - shutdown MT5
        mt5.shutdown()
        print("\n✅ Backtest Complete | MT5 Connection Closed (Backtest Mode Only)")