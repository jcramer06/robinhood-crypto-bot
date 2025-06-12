import requests
from bs4 import BeautifulSoup
# Sentiment keywords
positive_keywords = ["bullish", "surge", "rise", "rally", "gain", "breakout", "soar", "pump", "above", "back online", "grow"]
negative_keywords = ["crash", "bearish", "fall", "dip", "plunge", "drop", "sell", "decline", "fight", "delisting", "lost"]

# Headers for requests
headers = {'User-Agent': 'Mozilla/5.0'}

# Scraper for CoinDesk
def scrape_coindesk():
    url = "https://www.coindesk.com/"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    return [h.text.strip().lower() for h in soup.find_all("h2") if h.text.strip()]

# Scraper for CoinTelegraph
def scrape_cointelegraph():
    url = "https://cointelegraph.com/"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []

    # Look for post card blocks
    for article in soup.select("div.post-card-inline__content"):
        a_tag = article.find("a")
        if a_tag and a_tag.text:
            headlines.append(a_tag.text.strip().lower())

    return headlines

# Combine all headlines
def gather_all_headlines():
    headlines = []
    try:
        headlines += scrape_coindesk()
    except Exception as e:
        print("Error scraping CoinDesk:", e)
    try:
        headlines += scrape_cointelegraph()
    except Exception as e:
        print("Error scraping CoinTelegraph:", e)
    return headlines

# Analyze sentiment
def analyze_sentiment(headlines):
    pos, neg = 0, 0
    for h in headlines:
        pos += sum(1 for word in positive_keywords if word in h)
        neg += sum(1 for word in negative_keywords if word in h)
    return pos, neg

def sentiment():
    headlines = gather_all_headlines()

    pos_count, neg_count = analyze_sentiment(headlines)
    if pos_count > neg_count:
        print("Overall Sentiment: Positive")
    elif neg_count > pos_count:
        print("Overall Sentiment: Negative")
    else:
        print("Overall Sentiment: Neutral")

    percent = (pos_count / (pos_count + neg_count)) * 100
    return percent

sentiment()