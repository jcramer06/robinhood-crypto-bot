from robinhood_api import CryptoAPITrading
import uuid
import threading
import time

timer = 0

#returns price of 1 BTC
def fetch_price():
    estimation = client.get_estimated_price("MOODENG-USD", "ask", "1")
    return float(estimation["results"][0]["price"])

def acct():
    return client.get_account()

def holdings():
    try:
        return float(client.get_holdings()["results"][0]["quantity_available_for_trading"])
    except:
        return 0



xlist = []
ylist = []
def cost_over_time():
    global timer, xlist, ylist
    y = fetch_price()
    threading.Timer(1, cost_over_time).start()
    ylist.append(y)
    xlist.append(timer)
    timer += 1

def get_buying_power():
    return float(account["buying_power"])



client = CryptoAPITrading()
account = client.get_account()

cost_over_time()
acct()
print(holdings())