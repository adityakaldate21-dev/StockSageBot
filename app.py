import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os
import random
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------- PRELOAD ----------
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

# ---------- Q-Learning Setup ----------
actions = ['Buy', 'Sell', 'Hold']
q_table = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.2

def get_state(score, price_change):
    return (round(score, 1), round(price_change, 2))

def choose_action(state):
    if random.uniform(0, 1) < epsilon or state not in q_table:
        return random.choice(actions)
    return max(q_table[state], key=q_table[state].get)

def update_q(state, action, reward):
    if state not in q_table:
        q_table[state] = {a: 0 for a in actions}
    old = q_table[state][action]
    future = max(q_table[state].values())
    q_table[state][action] = old + alpha * (reward + gamma * future - old)

# ---------- LOAD DATA ----------
try:
    df_tweets = pd.read_csv("stock_tweets.csv")
    df_tweets.columns = df_tweets.columns.str.lower().str.strip()
except Exception as e:
    print(f"⚠️ Tweet data error: {e}")
    df_tweets = pd.DataFrame()

try:
    df_price = pd.read_csv("stock_yfinancedata.csv")
    df_price.columns = df_price.columns.str.lower().str.strip()
except:
    df_price = pd.DataFrame()

# ---------- SENTIMENT FUNCTION ----------
def get_sentiment_score(text):
    return analyzer.polarity_scores(text)['compound']

def classify_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# ---------- UI LOGIC ----------
def analyze_sentiment():
    text_input = entry_text.get().strip()
    stock_input = entry_stock.get().strip().upper()

    if not text_input:
        messagebox.showwarning("Missing Input", "Please enter stock-related text.")
        return

    sentiment_score = get_sentiment_score(text_input)
    sentiment_label = classify_sentiment(sentiment_score)
    result = f"🧠 Sentiment: {sentiment_label}\n"

    price_change = 0
    if not df_price.empty and "stock" in df_price.columns:
        match = df_price[df_price["stock"].str.upper() == stock_input]
        if not match.empty:
            info = match.iloc[0]
            price_change = (info['close'] - info['open']) / info['open']
            result += f"\n📊 {stock_input} Price Info:\nOpen: {info['open']}\nClose: {info['close']}\nVolume: {info['volume']}"
        else:
            result += f"\n⚠️ No price data found for {stock_input}"
    elif stock_input:
        result += "\n⚠️ Price data unavailable"

    current_state = get_state(sentiment_score, price_change)
    action = choose_action(current_state)
    reward = price_change if action == 'Buy' else -price_change if action == 'Sell' else 0
    update_q(current_state, action, reward)

    result += f"\n🤖 RL Suggestion: {action}"
    result_label.config(text=result)

# ---------- CHATBOT LOGIC ----------
def handle_chat():
    user_msg = chat_entry.get().strip()
    if not user_msg:
        messagebox.showinfo("Chatbot", "Please type a scenario or question.")
        return

    sentiment_score = get_sentiment_score(user_msg)
    sentiment_label = classify_sentiment(sentiment_score)
    price_change = random.uniform(-0.05, 0.05)
    state = get_state(sentiment_score, price_change)
    action = choose_action(state)
    reward = price_change if action == 'Buy' else -price_change if action == 'Sell' else 0
    update_q(state, action, reward)

    bot_reply = (
        f"💬 Scenario: {user_msg}\n"
        f"🧠 Sentiment: {sentiment_label} ({sentiment_score})\n"
        f"📉 Simulated Price Change: {round(price_change*100, 2)}%\n"
        f"🤖 RL Suggestion: {action.upper()}"
    )
    chat_output.config(text=bot_reply)

# ---------- TKINTER STYLED UI ----------
root = tk.Tk()
root.title("📈 Stock Sentiment Analyzer + RL Chatbot")
root.geometry("600x700")
root.configure(bg="#f0f4ff")

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Segoe UI", 10), padding=6)
style.configure("TEntry", font=("Segoe UI", 11))
style.configure("Header.TLabel", font=("Segoe UI Semibold", 14), background="#f0f4ff")
style.configure("Result.TLabel", font=("Consolas", 11), background="#fdfdfd", foreground="#003366", padding=10, borderwidth=2, relief="ridge")

# ---------- MAIN INPUT ----------
ttk.Label(root, text="📘 Stock Sentiment Analysis", style="Header.TLabel").pack(pady=10)
ttk.Label(root, text="📝 Enter tweet or headline:", style="Header.TLabel").pack(pady=5)
entry_text = ttk.Entry(root, width=60)
entry_text.pack(pady=5)

ttk.Label(root, text="🔎 Enter stock symbol (optional):", style="Header.TLabel").pack(pady=5)
entry_stock = ttk.Entry(root, width=30)
entry_stock.pack(pady=5)

ttk.Button(root, text="📊 Analyze Sentiment", command=analyze_sentiment).pack(pady=12)
result_label = ttk.Label(root, text="", style="Result.TLabel", justify="left", anchor="w")
result_label.pack(fill="x", padx=20, pady=10)

# ---------- CHATBOT SECTION ----------
ttk.Label(root, text="🤖 Chatbot — Ask a stock scenario", style="Header.TLabel").pack(pady=10)
chat_entry = ttk.Entry(root, width=60)
chat_entry.pack(pady=5)

ttk.Button(root, text="💬 Chat with Analyzer", command=handle_chat).pack(pady=8)
chat_output = ttk.Label(root, text="", style="Result.TLabel", justify="left", anchor="w")
chat_output.pack(fill="x", padx=20, pady=10)

# ---------- START APP ----------
root.mainloop()
