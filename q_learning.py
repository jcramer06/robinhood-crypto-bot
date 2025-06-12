import numpy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import trading
import scraper
import models
import time
import uuid
import math
import traceback

trading.cost_over_time()
time.sleep(60)
orders = []
class Order:
    def get_order(self):
        try:
            orders = trading.client.get_orders()
            if "results" in orders and len(orders["results"]) > 0:
                return orders["results"][0]
            else:
                raise IndexError("No orders found.")
        except Exception as e:
            print("Error in get_order:", e)
            return {}

    def get_real_price(self):
        order = self.get_order()
        return float(order.get("effective_price", 0.0))

def update_normalized_state():
    state = numpy.array([sum(trading.ylist) / len(trading.ylist), scraper.sentiment(), trading.fetch_price(), (
    models.PolyRegression(trading.xlist, trading.ylist)[0]), trading.buying_power])
    return (state - np.min(state)) / (np.max(state) - np.min(state))

# 1. Define the Q-network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# 2. Replay memory for experience replay
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.memory)

# 3. Agent class to interact and learn
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory()
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.action_dim = action_dim

    def select_action(self, state):
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def optimize(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s,a)
        curr_q_values = self.policy_net(states).gather(1, actions)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)

        # Target for Q-learning
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss
        loss = nn.MSELoss()(curr_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

state_dim = 5  # 5 features in your state vector
action_dim = 3  # e.g., 0=hold, 1=buy, 2=sell

agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
state = update_normalized_state()

entry_price = None
holdings_at_entry = 0



while True:
    try:
        false_bought = 0
        cash = trading.buying_power
        cash_diff = cash
        crypto = trading.holdings()
        net_worth = cash + crypto
        SPREAD = 0.0061  # 0.61%
        FEE_RATE = 0.0085
        MIN_TRADE_USD = .1
        TRADE_INTERVAL = 0.1
        while len(trading.xlist) > 500:
            trading.xlist.pop(0)
        while len(trading.ylist) > 500:
            trading.ylist.pop(0)
        state = update_normalized_state()
        action = agent.select_action(state)

        if not isinstance(action, int):
            raise ValueError(f"Invalid action selected: {action}")
        info = {}

        current_price = trading.fetch_price()

        ask_price = current_price * (1 + SPREAD / 2)
        bid_price = current_price * (1 - SPREAD / 2)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.policy_net(state_tensor)
            probs = torch.softmax(q_values, dim=1)
            confidence = probs[0, action].item()
        #margin = (sorted_qs[0, 0] - sorted_qs[0, 1]).item()
        print(confidence)

        crypto = trading.holdings()
        # --- Step 4: Execute trade ---
        client_id = str(uuid.uuid4())
        if action == 1:  # Buy
            trade_unit = confidence * (cash / current_price)
            print("BUY")
            cost = ask_price * trade_unit
            fee = max(FEE_RATE * cost, 0.01)
            total_cost = cost + fee
            print(ask_price, trade_unit)
            print(cash, total_cost, cost, MIN_TRADE_USD)

            if cash >= total_cost and cost >= MIN_TRADE_USD:
                print("heyyyyyyyyyyyyyyyyy")
                try:
                    response = trading.client.place_order(
                        client_id, "buy", "market", "MOODENG-USD", {"asset_quantity": str(math.floor(trade_unit * 100)/100.0)}
                    )
                    print("Order response:", response)
                    #trade = Order
                    #orders.append(Order)
                    entry_price = ask_price  # Save the price you bought at
                    holdings_at_entry = trading.holdings() + trade_unit
                    if response["type"] == "validation_error":
                        false_bought = 10
                except Exception as e:
                        print("Order failed:", e)


        elif action == 2:  # Sell
            trade_unit = confidence * crypto
            print("SELL")
            revenue = bid_price * trade_unit
            fee = max(FEE_RATE * revenue, 0.01)
            net_revenue = revenue - fee

            if crypto >= trade_unit and revenue >= MIN_TRADE_USD:
                trading.client.place_order(client_id, "sell", "market", "MOODENG-USD", {"asset_quantity": str(math.floor(trade_unit * 100)/100.0)})
                #trade = Order
                #orders.append(Order)


        else:
            print("HOLD")
            fee = 0
            trade_unit = 0
        if entry_price is not None:
            unrealized_profit = (current_price - entry_price) * holdings_at_entry
            print("Unrealized profit:", unrealized_profit)
        else:
            unrealized_profit = 0
        # --- Step 5: Log performance ---
        time.sleep(1)
        cash = trading.buying_power
        cash_diff -= cash
        crypto = trading.holdings()
        cryptoHolding = crypto * trading.fetch_price()
        new_net_worth = cash + cryptoHolding
        net_worth_change = (new_net_worth - net_worth)
        reward = net_worth_change*10 + unrealized_profit*10 - cash_diff*100 - false_bought
        net_worth = new_net_worth
        info['net_worth'] = net_worth
        info['reward'] = reward

        print(f"{time.ctime()} | Net Worth: ${net_worth:.2f} | Reward: {reward:.5f}")

        # Optionally store in memory and train:
        next_state = update_normalized_state()
        done = False
        agent.memory.push(state, action, reward, next_state, done)
        agent.optimize()
        print(cash)
        print("trade unit:", trade_unit)
        # Sleep to match interval
        time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopped manually.")
        break
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        time.sleep(5)