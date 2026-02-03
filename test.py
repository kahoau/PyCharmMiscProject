import MetaTrader5 as mt5

if not mt5.initialize():
    print("init fail")
else:
    print("it is good")

print(f"mt5 version is {mt5.version()}")
print(f"account info is {mt5.account_info()}")