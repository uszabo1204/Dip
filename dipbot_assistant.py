# dipbot_assistant.py
# An intelligent investment assistant that tracks opportunities using the "Patient"
# strategy, provides daily briefings, and generates a strategic weekly AI report.

import pandas as pd
import datetime as dt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import config
import numpy as np
import time
import os
import pickle
import json
from alpha_vantage.timeseries import TimeSeries
import google.generativeai as genai
import markdown2
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import io
import re

# --- Configuration ---
# PRODUCTION SETTING: Always fetch fresh data from the API.
FORCE_REFRESH = True 
CACHE_FILE = 'market_data_cache.pkl' 
STATE_FILE = 'state.json'
PORTFOLIO_CSV_PATH = 'portfolio.csv'
MONTHLY_INVESTMENT = 1000

# Dip-buying criteria
DRAWDOWN_THRESHOLD = -0.07
RSI_THRESHOLD = 50
# Watchlist criteria (slightly looser)
WATCHLIST_DRAWDOWN = -0.05
WATCHLIST_RSI = 55

# Ticker lists
PORTFOLIO_TICKERS = [
    'NVDA', 'META', 'MSFT', 'AMZN', 'AAPL', 'PLTR', 'UBER', 'DASH', 'GOOGL',
    'CRWD', 'TSLA', 'VST', 'NRG', 'AMD', 'DELL', 'AVGO', 'ORCL', 'VRT'
]
MAG7_TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
BENCHMARK_TICKERS = ['SPY']

# --- Email & API Setup ---
SENDER_EMAIL = config.SENDER_EMAIL
SENDER_PASSWORD = config.SENDER_EMAIL_PASSWORD
RECEIVER_EMAIL = 'uszabo@googlemail.com'
ts = TimeSeries(key=config.ALPHA_VANTAGE_API_KEY, output_format='pandas')
genai.configure(api_key=config.GEMINI_API_KEY)
llm = genai.GenerativeModel('gemini-1.5-flash')

# --- State Management & Helpers ---
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: return json.load(f)
    return {"cash_on_hand": 0, "last_capital_injection_month": 0, "this_months_best_signal": None}

def save_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=4)

def is_market_open_today():
    nyse = mcal.get_calendar('NYSE')
    today = dt.datetime.now(dt.timezone.utc).date()
    schedule = nyse.schedule(start_date=today, end_date=today)
    return not schedule.empty

def calculate_drawdown_20(series):
    if len(series) < 20: return None
    rolling_high = series.rolling(window=20).max()
    return (series.iloc[-1] - rolling_high.iloc[-1]) / rolling_high.iloc[-1]

def calculate_rsi(series, window=14):
    if len(series) < window + 1: return None
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(com=window-1, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(com=window-1, adjust=False).mean()
    if loss.iloc[-1] == 0: return 100.0
    rs = gain / loss
    return 100 - (100 / (1 + rs.iloc[-1]))

# --- Data & Signal Analysis ---
def get_market_data(tickers):
    if not FORCE_REFRESH and os.path.exists(CACHE_FILE):
        print(f"Loading market data from cache: {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f: return pickle.load(f)
    else:
        print(f"--- Fetching fresh market data for {len(tickers)} tickers from API ---")
        all_data = {}
        for ticker in tickers:
            try:
                data, _ = ts.get_daily(symbol=ticker, outputsize='full')
                data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                data = data.astype(float).sort_index(ascending=True)
                all_data[ticker] = data
                print(f"Successfully fetched data for {ticker}.")
            except Exception as e:
                print(f"Could not fetch data for {ticker}: {e}")
                all_data[ticker] = pd.DataFrame()
            time.sleep(15)
        if not any(df.empty for df in all_data.values()):
            if not FORCE_REFRESH: # Only save cache if not in production mode
                print(f"Saving new market data to cache: {CACHE_FILE}")
                with open(CACHE_FILE, 'wb') as f: pickle.dump(all_data, f)
        return all_data

def find_todays_signals(market_data, is_watchlist=False):
    today = dt.datetime.now(dt.timezone.utc).date()
    signals = []
    dd_thresh = WATCHLIST_DRAWDOWN if is_watchlist else DRAWDOWN_THRESHOLD
    rsi_thresh = WATCHLIST_RSI if is_watchlist else RSI_THRESHOLD
    for ticker in PORTFOLIO_TICKERS:
        if ticker in market_data and not market_data[ticker].empty:
            if pd.to_datetime(today) in market_data[ticker].index.normalize():
                today_loc = market_data[ticker].index.get_loc(pd.to_datetime(today).normalize())
                historical_slice = market_data[ticker].iloc[:today_loc + 1]
                if len(historical_slice) >= 21:
                    drawdown = calculate_drawdown_20(historical_slice['Close'])
                    rsi = calculate_rsi(historical_slice['Close'])
                    if drawdown is not None and rsi is not None and drawdown <= dd_thresh and rsi <= rsi_thresh:
                        signals.append({'date': today.strftime('%Y-%m-%d'), 'ticker': ticker, 'price': historical_slice['Close'].iloc[-1], 'rsi': rsi, 'drawdown': drawdown})
    return signals

def read_portfolio_csv(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"ERROR: The file '{filepath}' was not found. Weekly report cannot be generated.")
        return None

# --- Chart Generation ---
def generate_portfolio_health_chart(performance_data):
    print("Generating Portfolio Health chart...")
    df = pd.DataFrame.from_dict(performance_data, orient='index', columns=['Change']).sort_values('Change', ascending=False)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in df['Change']]
    bars = ax.bar(df.index, df['Change'], color=colors)
    for bar in bars:
        height = bar.get_height()
        y_pos = height + (ax.get_ylim()[1] * 0.01) if height > 0 else height - (ax.get_ylim()[1] * 0.03)
        ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    ax.set_ylabel('Weekly % Change')
    ax.set_title('Portfolio Health: Stock Performance', fontsize=16)
    plt.xticks(rotation=45, ha="right")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.1, bottom=ax.get_ylim()[0] * 1.1 if ax.get_ylim()[0] < 0 else -1)
    fig.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer

def generate_performance_comparison_chart(all_market_data, portfolio_holdings):
    print("Generating Performance Comparison chart...")
    portfolio_daily_values = []
    for _, row in portfolio_holdings.iterrows():
        ticker, shares = row['Ticker'], row['Shares']
        if ticker in all_market_data and not all_market_data[ticker].empty:
            portfolio_daily_values.append(all_market_data[ticker]['Close'] * shares)
    total_portfolio_value = pd.concat(portfolio_daily_values, axis=1).sum(axis=1)
    weekly_values = {}
    weekly_values['AI Portfolio'] = total_portfolio_value.resample('W-FRI').last()
    spy_df = all_market_data.get('SPY', pd.DataFrame())
    if not spy_df.empty:
        weekly_values['S&P 500'] = spy_df['Close'].resample('W-FRI').last()
    df_weekly = pd.DataFrame(weekly_values).dropna().tail(12)
    if df_weekly.empty: return None
    normalized_returns = (df_weekly / df_weekly.iloc[0]) * 100
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(normalized_returns.index, normalized_returns['AI Portfolio'], 'o-', label='Your Portfolio', color='crimson', linewidth=2)
    ax.plot(normalized_returns.index, normalized_returns['S&P 500'], 'o-', label='S&P 500', color='royalblue', linewidth=2)
    ax.set_ylabel('Performance (Indexed to 100)')
    ax.set_title('Weekly Performance vs. S&P 500 (Last 12 Weeks)', fontsize=16)
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer

# --- Email Functions ---

def send_daily_briefing(state, new_best_signal_today, watchlist):
    subject = f"DipBot Daily Briefing: {dt.date.today():%B %d, %Y}"
    body_lines = [f"## DipBot Daily Briefing - {dt.date.today():%A, %B %d}\n"]
    if new_best_signal_today:
        body_lines.append("**Actionable Update:** A new best opportunity for the month was identified today!\n")
    else:
        body_lines.append("**Status:** Holding. No new *better* opportunities found today.\n")
    body_lines.append(f"**Capital Ready to Deploy:** €{state['cash_on_hand']:,.2f}\n")
    if state['this_months_best_signal']:
        best = state['this_months_best_signal']
        body_lines.append("### Current Best Opportunity for this Month:")
        body_lines.append(f"- **Ticker:** {best['ticker']}")
        body_lines.append(f"- **Entry Price:** €{best['price']:.2f} (as of {best['date']})")
    else:
        body_lines.append("\nNo valid dip-buying opportunities have been identified yet this month.")
    if watchlist:
        body_lines.append("\n### On the Watchlist:")
        body_lines.append("*These stocks are getting close to our dip-buying criteria and are worth monitoring.*\n")
        for stock in watchlist:
            body_lines.append(f"- **{stock['ticker']}:** RSI: {stock['rsi']:.2f}, Drawdown: {stock['drawdown']:.2%}")
    html_content = markdown2.markdown("\n".join(body_lines))
    msg = MIMEText(html_content, 'html')
    msg['Subject'], msg['From'], msg['To'] = subject, SENDER_EMAIL, RECEIVER_EMAIL
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        print("Daily briefing email sent successfully.")
    except Exception as e:
        print(f"Failed to send daily email. Error: {e}")

def send_weekly_report(state, market_data, weekly_signal_log, portfolio_holdings):
    report_text, health_data = generate_weekly_ai_report(state, market_data, weekly_signal_log)
    if health_data is None: 
        print("Could not generate health data for weekly report. Aborting send.")
        return

    health_chart_img = generate_portfolio_health_chart(health_data)
    comparison_chart_img = generate_performance_comparison_chart(market_data, portfolio_holdings)
    
    images_to_attach = {
        'portfolio_health_chart': health_chart_img,
        'performance_comparison_chart': comparison_chart_img
    }
    
    msg = MIMEMultipart('related')
    msg['Subject'] = f"DipBot Weekly Report: {dt.date.today():%B %d, %Y}"
    msg['From'], msg['To'] = SENDER_EMAIL, RECEIVER_EMAIL
    
    # --- THIS IS THE FIX ---
    # 1. Replace placeholders in the raw text BEFORE converting to HTML.
    if images_to_attach['portfolio_health_chart']:
        report_text = re.sub(r'\[CHART_?PORTFOLIO_?HEALTH\]', '<img src="cid:portfolio_health_chart">', report_text, flags=re.IGNORECASE)
    if images_to_attach['performance_comparison_chart']:
        report_text = re.sub(r'\[CHART_?PERFORMANCE_?COMPARISON\]', '<img src="cid:performance_comparison_chart">', report_text, flags=re.IGNORECASE)

    # 2. Now, convert the text (which now contains <img> tags) to HTML.
    report_html = markdown2.markdown(report_text, extras=["tables", "fenced-code-blocks"])

    html_body = f"""<html><head><style>body {{ font-family: sans-serif; line-height: 1.6; }} img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}</style></head><body>{report_html}</body></html>"""
    
    # 3. Attach the final HTML body first.
    msg.attach(MIMEText(html_body, 'html'))

    # 4. Then attach the images it refers to.
    for cid, img_buffer in images_to_attach.items():
        if img_buffer:
            mime_image = MIMEImage(img_buffer.getvalue(), 'png')
            mime_image.add_header('Content-ID', f'<{cid}>')
            msg.attach(mime_image)
            
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        print("Weekly report email sent successfully.")
    except Exception as e:
        print(f"Failed to send weekly email. Error: {e}")

# --- Weekly Report AI Logic ---

def calculate_weekly_performance(market_data):
    health_chart_data = {}
    for ticker in PORTFOLIO_TICKERS:
        if ticker in market_data and not market_data[ticker].empty and len(market_data[ticker]) >= 5:
            weekly_data = market_data[ticker].iloc[-5:]
            start_price, end_price = weekly_data['Close'].iloc[0], weekly_data['Close'].iloc[-1]
            pnl_pct = (end_price - start_price) / start_price * 100
            health_chart_data[ticker] = pnl_pct
    return health_chart_data

def generate_weekly_ai_report(state, market_data, weekly_signal_log):
    health_data = calculate_weekly_performance(market_data)
    weekly_log_text = "No official dip signals were identified this week."
    if weekly_signal_log:
        lines = ["The following dip signals were observed this week:"]
        for signal in weekly_signal_log:
            lines.append(f"- {signal['date']}: {signal['ticker']} at €{signal['price']:.2f} (RSI: {signal['rsi']:.2f})")
        weekly_log_text = "\n".join(lines)
    prompt_context = f"""
**Current State of "Patient" Strategy:**
- Cash on Hand to Deploy: €{state['cash_on_hand']:,.2f}
- Best Signal Found This Month: {json.dumps(state['this_months_best_signal'], indent=2) if state['this_months_best_signal'] else "None yet."}

**This Week's Dip Signal Log:**
```
{weekly_log_text}
```
"""
    best_signal = state.get('this_months_best_signal')
    if best_signal:
        plan_text = f"Continue to hold our €{state['cash_on_hand']:,.2f} cash, looking for an entry in {best_signal['ticker']} below €{best_signal['price']:.2f}."
    else:
        plan_text = f"Continue to hold our €{state['cash_on_hand']:,.2f} cash and wait for a compelling dip signal to emerge."
    dipbot_prompt = f"""
You are DipBot, an elite financial analyst.
**TASK:**
Generate a weekly report in clean MARKDOWN format. You MUST insert these exact placeholders on their own lines: `[CHART_PORTFOLIO_HEALTH]` and `[CHART_PERFORMANCE_COMPARISON]`.
... (rest of your prompt) ...
"""
    print("Generating weekly report with Gemini API...")
    try:
        response = llm.generate_content(dipbot_prompt)
        print("Successfully generated weekly report.")
        return response.text, health_data
    except Exception as e:
        print(f"ERROR generating weekly report from Gemini: {e}")
        return f"Error: Could not generate the weekly report due to an API error: {e}", None

# --- Main Orchestrator ---

if __name__ == "__main__":
    
    if not is_market_open_today() and not FORCE_REFRESH:
        print(f"Skipping analysis on {dt.datetime.now().strftime('%A')}, a non-trading day.")
        exit()

    state = load_state()
    today = dt.datetime.now(dt.timezone.utc)

    if today.month != state.get('last_capital_injection_month', 0):
        print(f"New month detected. Adding €{MONTHLY_INVESTMENT} to cash reserves.")
        state['cash_on_hand'] += MONTHLY_INVESTMENT
        state['last_capital_injection_month'] = today.month
        state['this_months_best_signal'] = None

    all_tickers = list(set(PORTFOLIO_TICKERS + BENCHMARK_TICKERS + MAG7_TICKERS))
    market_data = get_market_data(all_tickers)

    todays_signals = find_todays_signals(market_data)
    watchlist_signals = find_todays_signals(market_data, is_watchlist=True)
    print(f"Found {len(todays_signals)} official dip signals and {len(watchlist_signals)} on the watchlist today.")

    new_best_signal_found = False
    current_best = state['this_months_best_signal']
    if todays_signals:
        best_of_today = min(todays_signals, key=lambda x: x['price'])
        if current_best is None or best_of_today['price'] < current_best['price']:
            print(f"New best signal found for the month in {best_of_today['ticker']} at price {best_of_today['price']:.2f}")
            state['this_months_best_signal'] = best_of_today
            new_best_signal_found = True

    send_daily_briefing(state, new_best_signal_found, watchlist_signals)

    # PRODUCTION SETTING: Run weekly report only on Friday (4).
    if today.weekday() == 4 or FORCE_REFRESH: 
        print("\nIt's Friday! Generating weekly strategic report...")
        portfolio_holdings = read_portfolio_csv(PORTFOLIO_CSV_PATH)
        if portfolio_holdings is not None:
            start_of_week = today.date() - dt.timedelta(days=today.weekday())
            weekly_signal_log = []
            for day_offset in range(5):
                check_date = start_of_week + dt.timedelta(days=day_offset)
                if check_date <= today.date():
                    if check_date == today.date():
                        weekly_signal_log.extend(todays_signals)
            
            send_weekly_report(state, market_data, weekly_signal_log, portfolio_holdings)

    save_state(state)
    
    print(f"\n--- DipBot Assistant operations complete for {today:%Y-%m-%d} ---")
