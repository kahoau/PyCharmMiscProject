import MetaTrader5 as mt5
import time
import numpy as np
import pandas as pd  # Add pandas for data handling
from datetime import datetime

# -------------------------- CONFIGURATION (EDIT THESE) --------------------------
SYMBOL = "XAUUSD.std"  # Exact symbol name for your broker
LOT_SIZE = 0.01
TIMEFRAME = mt5.TIMEFRAME_M1  # 1-minute chart
RSI_PERIOD = 14
RSI_OVERSOLD = 30  # Buy signal when RSI < 30
RSI_OVERBOUGHT = 70  # Sell signal when RSI > 70
MA_PERIOD = 20  # 20-period simple moving average (SMA)
ATR_PERIOD = 10  # ATR for volatility filter/SL/TP
ATR_MULTIPLIER_SL = 1.0  # SL = 1x ATR
ATR_MULTIPLIER_TP = 2.0  # TP = 2x ATR (2:1 risk-reward)
MAGIC_NUMBER = 123456  # Unique ID for bot trades


# --------------------------------------------------------------------------------

def initialize_mt5():
    """Initialize MT5 connection and verify symbol/account status"""
    if not mt5.initialize():
        print(f"MT5 Initialization Failed! Error: {mt5.last_error()}")
        return False

    # Verify account connection
    account_info = mt5.account_info()
    if account_info is None:
        print(f"Failed to get account info! Error: {mt5.last_error()}")
        mt5.shutdown()
        return False

    # Verify XAUUSD.std exists
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"{SYMBOL} not found in MT5! Error: {mt5.last_error()}")
        mt5.shutdown()
        return False

    # Enable symbol if disabled
    if not symbol_info.visible:
        if not mt5.symbol_select(SYMBOL, True):
            print(f"Failed to enable {SYMBOL}! Error: {mt5.last_error()}")
            mt5.shutdown()
            return False

    print(f"✅ MT5 Connected Successfully")
    print(f"   Account: {account_info.login} | Symbol: {SYMBOL} | Type: {account_info.trade_mode} (0=demo, 1=real)")
    return True


def validate_lot_size():
    """Ensure LOT_SIZE is allowed by your broker for XAUUSD.std"""
    # Declare global FIRST (fixes syntax error)
    global LOT_SIZE

    symbol_info = mt5.symbol_info(SYMBOL)
    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lot_step = symbol_info.volume_step

    # Check lot size range
    if LOT_SIZE < min_lot or LOT_SIZE > max_lot:
        print(f"❌ Invalid Lot Size! Broker allows {min_lot} - {max_lot} lots for {SYMBOL}")
        return False

    # Round to broker's lot step (e.g., 0.01, 0.05)
    LOT_SIZE = round(LOT_SIZE / lot_step) * lot_step
    print(f"✅ Valid Lot Size: {LOT_SIZE} (Broker step: {lot_step})")
    return True


def get_historical_data(periods: int):
    """Fetch 1-min OHLC data for XAUUSD.std (fixed dtype issue)"""
    utc_from = datetime.now()
    rates = mt5.copy_rates_from(SYMBOL, TIMEFRAME, utc_from, periods)

    if rates is None or len(rates) < periods:
        print(f"❌ Failed to fetch {SYMBOL} data! Error: {mt5.last_error()}")
        return None

    # FIX: Use pandas to avoid dtype casting errors
    df = pd.DataFrame(rates)

    # Convert MT5's Unix timestamp (seconds) to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Extract only the fields we need (ignore extra fields like spread/real_volume)
    data = {
        'time': df['time'].values,
        'open': df['open'].values,
        'high': df['high'].values,
        'low': df['low'].values,
        'close': df['close'].values
    }
    return data


def calculate_indicators(data):
    """Calculate RSI, SMA, and ATR for strategy signals (fixed data format)"""
    closes = data['close']
    highs = data['high']
    lows = data['low']

    # 1. SMA (20-period)
    sma = np.convolve(closes, np.ones(MA_PERIOD) / MA_PERIOD, mode='valid')
    if len(sma) == 0:
        print("❌ Insufficient data for SMA!")
        return None

    # 2. RSI (14-period)
    deltas = np.diff(closes)
    gains = deltas.copy()
    losses = deltas.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = np.abs(losses)

    avg_gain = np.convolve(gains, np.ones(RSI_PERIOD) / RSI_PERIOD, mode='valid')
    avg_loss = np.convolve(losses, np.ones(RSI_PERIOD) / RSI_PERIOD, mode='valid')

    # Avoid division by zero
    avg_loss[avg_loss == 0] = 0.0001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    if len(rsi) == 0:
        print("❌ Insufficient data for RSI!")
        return None

    # 3. ATR (10-period)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1])
    )
    atr = np.convolve(tr, np.ones(ATR_PERIOD) / ATR_PERIOD, mode='valid')
    if len(atr) == 0:
        print("❌ Insufficient data for ATR!")
        return None

    # Return latest values
    return {
        'sma': sma[-1],
        'rsi': rsi[-1],
        'atr': atr[-1],
        'last_close': closes[-1],
        'last_high': highs[-1],
        'last_low': lows[-1]
    }


def has_open_position(symbol: str, order_type: int):
    """Check for open bot positions"""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False

    for pos in positions:
        if pos.magic == MAGIC_NUMBER and pos.type == order_type:
            return True
    return False


def execute_trade(order_type: int, sl: float, tp: float):
    """Execute buy/sell trade with SL/TP for XAUUSD.std"""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print(f"❌ Failed to get {SYMBOL} tick data! Error: {mt5.last_error()}")
        return False

    # Entry price (buy=ask, sell=bid)
    entry_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    # Normalize prices to broker's decimal places
    symbol_info = mt5.symbol_info(SYMBOL)
    entry_price = mt5.symbol_normalize_price(SYMBOL, entry_price)
    sl = mt5.symbol_normalize_price(SYMBOL, sl)
    tp = mt5.symbol_normalize_price(SYMBOL, tp)

    # Trade request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": order_type,
        "price": entry_price,
        "sl": sl,
        "tp": tp,
        "deviation": 3,  # Slippage tolerance for XAUUSD.std
        "magic": MAGIC_NUMBER,
        "comment": "XAUUSD Python Bot",
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    # Send trade
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Trade Failed! Retcode: {result.retcode} | Error: {mt5.last_error()}")
        return False

    # Success message
    trade_type = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
    print(f"✅ {trade_type} Trade Executed")
    print(f"   Ticket: {result.order} | Entry: {entry_price} | SL: {sl} | TP: {tp}")
    return True


def close_position(order_type: int):
    """Close open buy/sell position for XAUUSD.std"""
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return False

    for pos in positions:
        if pos.magic == MAGIC_NUMBER and pos.type == order_type:
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            close_price = mt5.symbol_info_tick(
                SYMBOL).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(SYMBOL).ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": close_price,
                "deviation": 3,
                "magic": MAGIC_NUMBER,
                "comment": "Close Bot Position",
                "type_filling": mt5.ORDER_FILLING_IOC
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Closed Position | Ticket: {pos.ticket}")
                return True
            else:
                print(f"❌ Failed to Close Position! Retcode: {result.retcode}")
                return False
    return False


def main_strategy_loop():
    """Main loop for XAUUSD.std trading strategy"""
    print("\n🚀 Starting XAUUSD.std Trading Bot (1-Min Chart) | Press Ctrl+C to stop")
    print("----------------------------------------------------------------------")

    while True:
        try:
            # Fetch enough data for indicators
            data = get_historical_data(periods=max(RSI_PERIOD, MA_PERIOD, ATR_PERIOD) + 10)
            if data is None:
                time.sleep(60)
                continue

            # Calculate indicators
            indicators = calculate_indicators(data)
            if indicators is None:
                time.sleep(60)
                continue

            # Extract values
            sma = indicators['sma']
            rsi = indicators['rsi']
            atr = indicators['atr']
            last_close = indicators['last_close']

            # -------------------------- BUY SIGNAL --------------------------
            if rsi < RSI_OVERSOLD and last_close < sma and not has_open_position(SYMBOL, mt5.ORDER_TYPE_BUY):
                # Calculate SL/TP (buy: SL = entry - ATR, TP = entry + 2*ATR)
                sl = last_close - (ATR_MULTIPLIER_SL * atr)
                tp = last_close + (ATR_MULTIPLIER_TP * atr)
                # Close opposite position
                if has_open_position(SYMBOL, mt5.ORDER_TYPE_SELL):
                    print("🔄 Closing open SELL position for BUY signal")
                    close_position(mt5.ORDER_TYPE_SELL)
                # Execute buy
                execute_trade(mt5.ORDER_TYPE_BUY, sl, tp)

            # -------------------------- SELL SIGNAL -------------------------
            elif rsi > RSI_OVERBOUGHT and last_close > sma and not has_open_position(SYMBOL, mt5.ORDER_TYPE_SELL):
                # Calculate SL/TP (sell: SL = entry + ATR, TP = entry - 2*ATR)
                sl = last_close + (ATR_MULTIPLIER_SL * atr)
                tp = last_close - (ATR_MULTIPLIER_TP * atr)
                # Close opposite position
                if has_open_position(SYMBOL, mt5.ORDER_TYPE_BUY):
                    print("🔄 Closing open BUY position for SELL signal")
                    close_position(mt5.ORDER_TYPE_BUY)
                # Execute sell
                execute_trade(mt5.ORDER_TYPE_SELL, sl, tp)

            # No signal
            else:
                print(
                    f"⏳ No Signal | Time: {datetime.now().strftime('%H:%M:%S')} | RSI: {rsi:.1f} | SMA: {sma:.2f} | Close: {last_close:.2f}")

            # Wait 1 minute (match 1-min chart)
            time.sleep(60)

        except KeyboardInterrupt:
            print("\n⚠️ Bot stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in loop: {str(e)}")
            time.sleep(60)


if __name__ == "__main__":
    # Initialize MT5
    if not initialize_mt5():
        exit(1)

    # Validate lot size (critical for demo accounts)
    if not validate_lot_size():
        mt5.shutdown()
        exit(1)

    # Run strategy
    try:
        main_strategy_loop()
    finally:
        mt5.shutdown()
        print("\n🔌 MT5 Connection Closed | Bot Terminated")