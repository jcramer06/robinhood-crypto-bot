from robinhood_api import CryptoAPITrading
import uuid
import threading
import time

timer = 0

#returns price of 1 BTC
def fetch_price():
    estimation = client.get_estimated_price("BTC-USD", "ask", "1")
    return float(estimation["results"][0]["price"])

#returns tuple of prices and time
xlist = []
ylist = []
def cost_over_time():
    global timer, xlist, ylist
    y = fetch_price()
    threading.Timer(0.5, cost_over_time).start()
    ylist.append(y)
    xlist.append(timer)
    timer += 1



client = CryptoAPITrading()
account = client.get_account()
buying_power = float(account["buying_power"])
print(buying_power)

cost_over_time()

