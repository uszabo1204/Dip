# daily_dip_finder.py

import pandas as pd
import datetime as dt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import config
import numpy as np
import time
from alpha_vantage.timeseries import TimeSeries

import matplotlib.pyplot as plt
import io

# --- Configuration ---
PORTFOLIO_TICKERS = [
    'NVDA', 'META', 'MSFT', 'AMZN', 'AAPL', 'PLTR', 'UBER', 'DASH', 'GOOGL',
    'CRWD', 'TSLA', 'VST', 'NRG', 'AMD', 'DELL', 'AVGO', 'ORCL', 'VRT'
]

# Criteria for dip buying opportunity
DRAWDOWN_THRESHOLD = -0.07 # -7%
RSI_THRESHOLD = 50

# Email Configuration
SENDER_EMAIL = config.SENDER_EMAIL
SENDER_PASSWORD = config.SENDER_EMAIL_PASSWORD
RECEIVER_EMAIL = 'uszabo@googlemail.com'

# Alpha Vantage API setup
ALPHA_VANTAGE_API_KEY = config.ALPHA_VANTAGE_API_KEY
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# --- Helper Functions (for calculating metrics) ---

def calculate_drawdown_20(df_close_series):
    """
    Calculates the 20-day drawdown from a pandas Series of closing prices.
    """
    if len(df_close_series) < 20:
        return None
    rolling_high_20 = df_close_series.rolling(window=20).max()
    current_close = df_close_series.iloc[-1]
    max_val = rolling_high_20.iloc[-1]
    if pd.isna(max_val) or max_val == 0:
        return None
    drawdown = (current_close - max_val) / max_val
    return drawdown

def calculate_rsi(df_close_series, window=14):
    """
    Calculates the 14-day Relative Strength Index (RSI).
    """
    if len(df_close_series) < window + 1:
        return None
    delta = df_close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=window-1, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100).replace([float('inf'), -float('inf')], 100)
    return rsi.iloc[-1]

def get_single_metric_trend_indicator(current_val, previous_val, metric_type):
    """
    Determines if a single metric is trending towards or away from a dip opportunity.
    """
    if current_val is None or previous_val is None or pd.isna(current_val) or pd.isna(previous_val) or current_val == previous_val:
        return "Stable"
    if metric_type == 'drawdown' or metric_type == 'rsi':
        return "Towards Dip" if current_val < previous_val else "Away from Dip"
    return "Stable"

def get_overall_trend(dd_current, dd_previous, rsi_current, rsi_previous):
    """
    Combines individual metric trends into an overall trend.
    """
    dd_trend = get_single_metric_trend_indicator(dd_current, dd_previous, 'drawdown')
    rsi_trend = get_single_metric_trend_indicator(rsi_current, rsi_previous, 'rsi')
    if dd_trend == "Towards Dip" and rsi_trend == "Towards Dip":
        return "Towards Dip"
    elif dd_trend == "Away from Dip" and rsi_trend == "Away from Dip":
        return "Away from Dip"
    else:
        return "Stable/Mixed"

# --- Plotting Function ---
def generate_chart_image(ticker, historical_data_df):
    """
    Generates a chart of Drawdown and RSI over the last 120 trading days.
    """
    df_plot = historical_data_df.tail(120).copy()
    if len(df_plot) < 20:
        print(f"Not enough data for {ticker} to generate a meaningful chart.")
        return None

    df_plot['Drawdown_20_series'] = df_plot['Close'].rolling(window=20).apply(lambda x: (x.iloc[-1] - x.max()) / x.max() if x.max() != 0 else np.nan, raw=False)
    delta = df_plot['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=14-1, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(com=14-1, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df_plot['RSI_14_series'] = 100 - (100 / (1 + rs))
    df_plot['RSI_14_series'] = df_plot['RSI_14_series'].fillna(100).replace([float('inf'), -float('inf')], 100)

    df_plot.dropna(subset=['Drawdown_20_series', 'RSI_14_series'], inplace=True)
    if df_plot.empty:
        print(f"After calculating metrics, {ticker} has no valid data points for plotting.")
        return None

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)

    color_dd = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Drawdown_20', color=color_dd)
    ax1.plot(df_plot.index, df_plot['Drawdown_20_series'], color=color_dd, label='Drawdown_20')
    ax1.tick_params(axis='y', labelcolor=color_dd)
    ax1.axhline(DRAWDOWN_THRESHOLD, color=color_dd, linestyle='--', label='Drawdown Threshold')

    ax2 = ax1.twinx()
    color_rsi = 'gold'
    ax2.set_ylabel('RSI_14', color=color_rsi)
    ax2.plot(df_plot.index, df_plot['RSI_14_series'], color=color_rsi, label='RSI_14')
    ax2.tick_params(axis='y', labelcolor=color_rsi)
    ax2.axhline(RSI_THRESHOLD, color=color_rsi, linestyle='--', label='RSI Threshold')

    plt.title(f'{ticker}: DD_20d (blue) & RSI_14d (gold) over Last {len(df_plot)} Trading Days')
    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    if len(df_plot.index) > 5:
        locator = plt.MaxNLocator(nbins=6)
        ax1.xaxis.set_major_locator(locator)
        fig.autofmt_xdate()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer

# --- Main Analysis Logic ---
def find_and_analyze_portfolio():
    """
    Fetches daily stock data, calculates metrics, and determines opportunities.
    """
    portfolio_data = []
    all_historical_data = {}

    print(f"Starting Alpha Vantage data fetching for {len(PORTFOLIO_TICKERS)} tickers...")

    for ticker in PORTFOLIO_TICKERS:
        stock_info = {
            'Ticker': ticker, 'Drawdown_20': 'N/A', 'RSI_14': 'N/A',
            'Meet_DD_20d': 'No', 'Meet_RSI_14d': 'No', 'Trend': 'N/A',
            'Buying_Signal': 'No', 'Meets_Criteria': False
        }
        data = pd.DataFrame()
        try:
            print(f"Attempting to download data for {ticker} from Alpha Vantage...")
            data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data = data.astype(float).sort_index(ascending=True)
            print(f"DEBUG: {ticker} - Data fetched. Is empty: {data.empty}, Length: {len(data)}")
            if not data.empty:
                print(f"DEBUG: {ticker} - Last 5 rows:\n{data.tail()}")
                all_historical_data[ticker] = data
            else:
                print(f"Warning: Alpha Vantage download for {ticker} returned empty data.")
        except Exception as e:
            print(f"Error fetching data for {ticker} from Alpha Vantage: {e}")

        if data.empty or len(data) < 21:
            print(f"Warning: Not enough data for {ticker}. Skipping calculations.")
            portfolio_data.append(stock_info)
            time.sleep(15)
            continue

        try:
            today_close_series = data['Close']
            yesterday_close_series = data['Close'].iloc[:-1]
            dd_today = calculate_drawdown_20(today_close_series)
            rsi_today = calculate_rsi(today_close_series)
            dd_yesterday = calculate_drawdown_20(yesterday_close_series)
            rsi_yesterday = calculate_rsi(yesterday_close_series)
            
            if dd_today is not None and dd_today <= DRAWDOWN_THRESHOLD:
                stock_info['Meet_DD_20d'] = 'Yes'
            if rsi_today is not None and rsi_today <= RSI_THRESHOLD:
                stock_info['Meet_RSI_14d'] = 'Yes'
            
            stock_info.update({
                'Drawdown_20': f"{dd_today:.2%}" if dd_today is not None else 'N/A',
                'RSI_14': f"{rsi_today:.2f}" if rsi_today is not None else 'N/A',
                'Trend': get_overall_trend(dd_today, dd_yesterday, rsi_today, rsi_yesterday)
            })
            
            if stock_info['Meet_DD_20d'] == 'Yes' and stock_info['Meet_RSI_14d'] == 'Yes':
                stock_info['Buying_Signal'] = "Yes"
                stock_info['Meets_Criteria'] = True
                print(f"Found opportunity for {ticker}: DD={stock_info['Drawdown_20']}, RSI={stock_info['RSI_14']}")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
        
        portfolio_data.append(stock_info)
        time.sleep(15)
    return portfolio_data, all_historical_data

# --- Email Function (Corrected Version) ---
def send_email_notification(portfolio_analysis, all_historical_data):
    """
    Sends an email with a detailed HTML table of portfolio stock analysis,
    including charts for tickers with a buying signal.
    """
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("Email sender credentials not set. Skipping email.")
        return

    msg = MIMEMultipart('related')
    opportunities = [s for s in portfolio_analysis if s['Meets_Criteria']]
    
    if opportunities:
        subject = f"Daily Portfolio Dip Report: {dt.date.today():%Y-%m-%d} - {len(opportunities)} Opportunities Found!"
        summary_text = f"{len(opportunities)} opportunities found. Here are the stocks meeting your dip buying criteria today:"
    else:
        subject = f"Daily Portfolio Dip Report: {dt.date.today():%Y-%m-%d} - No immediate dip buying opportunities were identified today."
        summary_text = "No immediate dip buying opportunities were identified today. Here's a look at your portfolio's metrics:"

    html_table = """
    <table border="1" style="border-collapse: collapse; width: 100%; font-family: sans-serif;">
        <tr style="background-color: #f2f2f2;">
            <th>Ticker</th><th>DD_20d</th><th>RSI_14d</th><th>Meet DD_20d</th>
            <th>Meet RSI_14d</th><th>Trend</th><th>Buying Signal</th>
        </tr>
    """
    
    images_to_attach = []

    for stock in portfolio_analysis:
        row_style = ' style="background-color: #d4edda;"' if stock['Meets_Criteria'] else ''
        chart_html = ""
        if stock['Meets_Criteria'] and stock['Ticker'] in all_historical_data:
            img_buffer = generate_chart_image(stock['Ticker'], all_historical_data[stock['Ticker']])
            if img_buffer:
                img_data = img_buffer.read()
                img_cid = f"chart_{stock['Ticker']}"
                mime_image = MIMEImage(img_data, 'png')
                mime_image.add_header('Content-ID', f'<{img_cid}>')
                mime_image.add_header('Content-Disposition', 'inline', filename=f'{img_cid}.png')
                images_to_attach.append(mime_image)
                chart_html = f'<br><img src="cid:{img_cid}" alt="Chart for {stock["Ticker"]}" style="width:100%; max-width:800px; height:auto; margin-top: 10px;">'

        meet_dd_text = "✅ Yes" if stock['Meet_DD_20d'] == 'Yes' else "❌ No"
        dd_style = "color: green;" if stock['Meet_DD_20d'] == 'Yes' else "color: red;"
        meet_rsi_text = "✅ Yes" if stock['Meet_RSI_14d'] == 'Yes' else "❌ No"
        rsi_style = "color: green;" if stock['Meet_RSI_14d'] == 'Yes' else "color: red;"
        signal_style = "color: green; font-weight: bold;" if stock['Buying_Signal'] == 'Yes' else "color: red; font-weight: bold;"
        
        google_finance_url = f"https://www.google.com/finance/quote/{stock['Ticker']}:NASDAQ?window=6M"
        
        html_table += f"""
        <tr{row_style}>
            <td style="padding: 8px;"><a href="{google_finance_url}" target="_blank">{stock['Ticker']}</a></td>
            <td style="padding: 8px;">{stock['Drawdown_20']}</td><td style="padding: 8px;">{stock['RSI_14']}</td>
            <td style="padding: 8px; {dd_style}">{meet_dd_text}</td>
            <td style="padding: 8px; {rsi_style}">{meet_rsi_text}</td>
            <td style="padding: 8px;">{stock['Trend']}</td>
            <td style="padding: 8px; {signal_style}">{stock['Buying_Signal']}</td>
        </tr>
        {f'<tr><td colspan="7" style="padding: 10px; text-align: center;">{chart_html}</td></tr>' if chart_html else ''}
        """
    html_table += "</table>"

    body_plain_text = summary_text + "\n\n(HTML table not displayed. Please view in a compatible client.)"
    body_html = f"""
    <html><body>
    <p>{summary_text.replace('\\n', '<br>')}</p>{html_table}
    <p style="font-size: 0.8em; color: #777;">
        Meet DD_20d: '✅ Yes' if Drawdown (20-day) &lt;= {DRAWDOWN_THRESHOLD:.0%}.<br>
        Meet RSI_14d: '✅ Yes' if RSI (14-day) &lt;= {RSI_THRESHOLD}.<br>
        Click on Ticker symbols to view 6-month chart on Google Finance.
    </p>
    </body></html>"""
    
    alt_msg = MIMEMultipart('alternative')
    alt_msg.attach(MIMEText(body_plain_text, 'plain'))
    alt_msg.attach(MIMEText(body_html, 'html'))
    msg.attach(alt_msg)

    for image in images_to_attach:
        msg.attach(image)

    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        print("Email notification sent successfully.")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")

if __name__ == "__main__":
    today = dt.datetime.now()
    # Check if today is Saturday (5) or Sunday (6)
    if today.weekday() >= 5:
        print(f"Skipping analysis on {today.strftime('%A')}, a non-trading day.")
        exit() # Exit the script

    print(f"Starting daily portfolio analysis at {today.strftime('%Y-%m-%d %H:%M:%S')}")
    full_portfolio_analysis, all_historical_data = find_and_analyze_portfolio()
    
    print("\n--- Full Portfolio Analysis ---")
    for stock in full_portfolio_analysis:
        print(stock)

    send_email_notification(full_portfolio_analysis, all_historical_data)
    print(f"Daily portfolio analysis completed at {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")