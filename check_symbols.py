import MetaTrader5 as mt5

# Initialize MT5
if not mt5.initialize():
    print(f"MT5 Initialization Failed! Error: {mt5.last_error()}")
else:
    # Get all symbols
    symbols = mt5.symbols_get()
    print(f"Total Symbols Found: {len(symbols)}")
    print("\n--- Symbols Containing 'XAU' or 'GOLD' ---")
    for s in symbols:
        if "XAU" in s.name or "GOLD" in s.name:
            print(f"Symbol Name: {s.name} | Description: {s.description}")
    mt5.shutdown()