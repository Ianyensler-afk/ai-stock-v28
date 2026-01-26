import logging

# [V31.2] 系統警示消音器
# 忽略 Streamlit 多執行緒的 Context 警告 (因為我們只做純運算，這是安全的)
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)

# ... (接著原本的 import streamlit as st 等等) ...
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings
import time
from datetime import datetime, timedelta, timezone 
import requests
import xml.etree.ElementTree as ET
import email.utils 
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import argrelextrema 
import json
import smtplib
import google.generativeai as genai
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 忽略警告
warnings.filterwarnings("ignore")

# --- 檢查必要套件 ---
try:
    import pygad
    HAS_PYGAD = True
except ImportError:
    HAS_PYGAD = False

try:
    from snownlp import SnowNLP
    HAS_SNOWNLP = True
except ImportError:
    HAS_SNOWNLP = False
# --- [V28.0 新增] 檢查 NLP 與 統計套件 ---
try:
    import jieba
    import jieba.analyse
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

from scipy.stats import pearsonr # 用於計算板塊相關性

# [V27.2] 自定義 JSON 編碼器，解決 int64 錯誤
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# ==========================================
# 0. 全域設定與 CSS (V27.10 兼容性修復版)
# ==========================================
st.set_page_config(page_title="AI 戰情室: V27.10 終極修復版", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 20px; }
    .stDataFrame { border: 1px solid #ddd; } 
    button[data-baseweb="tab"] { font-size: 1.2em; font-weight: bold; }
    /* [V28.1 新增] 緊湊型股票標籤樣式 */
    .stock-tag {
        display: inline-block; 
        padding: 2px 8px; 
        margin: 2px; 
        background-color: #e8eaed; 
        border-radius: 4px; 
        font-size: 0.85em; 
        color: #333; 
        font-family: monospace;
        border: 1px solid #ccc;
    }
    .stock-tag:hover { background-color: #d1d5db; color: #000; border-color: #999; }
    
    .link-btn {
        text-decoration: none; display: inline-block; padding: 8px 16px;
        border-radius: 5px; background-color: #f0f2f6; color: #31333F;
        border: 1px solid #d0d2d6; margin: 5px; font-size: 0.9em; font-weight: bold;
    }
    .link-btn:hover { background-color: #e0e2e6; border-color: #00adb5; color: #00adb5; }
            
    .link-btn {
        text-decoration: none; display: inline-block; padding: 8px 16px;
        border-radius: 5px; background-color: #f0f2f6; color: #31333F;
        border: 1px solid #d0d2d6; margin: 5px; font-size: 0.9em; font-weight: bold;
    }
    .link-btn:hover { background-color: #e0e2e6; border-color: #00adb5; color: #00adb5; }
    
    .news-card {
        padding: 12px; margin-bottom: 12px; border-left: 5px solid #ddd; 
        border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); transition: transform 0.2s;
    }
    .news-card:hover { transform: translateX(5px); }
    
    .news-title {
        text-decoration: none; color: inherit; font-weight: bold; 
        font-size: 1.0em; display: inline-block; margin-bottom: 5px;
    }
    .news-source { color: #00adb5; font-weight: bold; font-size: 0.85em; padding-right: 10px; }
    .news-time { color: gray; font-size: 0.85em; }
    
    .sentiment-tag {
        display: inline-block; padding: 2px 8px; border-radius: 12px; 
        font-size: 0.75em; font-weight: bold; color: white; margin-right: 8px; vertical-align: middle;
    }
    .sent-bull { background-color: #ff4b4b; } 
    .sent-bear { background-color: #21c354; } 
    .sent-neu { background-color: #808495; }
    
    .json-box {
        background-color: #f8f9fa; border: 1px solid #ddd; padding: 15px;
        border-radius: 5px; font-family: monospace; font-size: 0.9em;
        white-space: pre-wrap; overflow-x: auto; color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# 產業資料庫
# ==========================================
# 0.5 資料庫載入區 (V3.5 修復版 - 解決 NameError)
# ==========================================
import os

# 1. [絕對關鍵] 先定義全域變數，防止程式讀不到報錯
STOCK_NAMES = {} 

# 預設資料庫 (備用，防止 json 讀取失敗時全空)
DEFAULT_SECTOR_DB = {
    "💎 半導體 (範例)": {"1. 上游": ["2330.TW", "2454.TW"]}
}

def load_external_data():
    global STOCK_NAMES # 宣告我們要修改全域變數
    
    # 載入產業分類
    sector_data = DEFAULT_SECTOR_DB
    if os.path.exists("sector_db.json"):
        try:
            with open("sector_db.json", "r", encoding="utf-8") as f:
                sector_data = json.load(f)
        except: pass
    
    # 載入股票名稱
    if os.path.exists("stock_names.json"):
        try:
            with open("stock_names.json", "r", encoding="utf-8") as f:
                external_names = json.load(f)
                # 將載入的名稱更新到全域變數中
                STOCK_NAMES.update(external_names)
        except: pass
        
    return sector_data

# 執行載入 (這行會填滿 SECTOR_DB 和 STOCK_NAMES)
SECTOR_DB = load_external_data()


# ==========================================
# 1. 核心工具 (ETL)
# ==========================================

# [V31.5] 強化的數據獲取函數 (抗阻擋版)
import random

@st.cache_data(ttl=600)
def get_stock_data(ticker, period="2y"):
    # 偽裝成瀏覽器的 Header
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # 處理代號格式
    tickers_to_try = [ticker]
    if ticker.isdigit(): tickers_to_try = [f"{ticker}.TW", f"{ticker}.TWO"]
    elif not ticker.endswith(".TW") and not ticker.endswith(".TWO") and not ticker.isalpha(): 
        tickers_to_try = [f"{ticker}.TW"]
    
    # 開始嘗試
    for t in tickers_to_try:
        # [V31.5 新增] 重試迴圈 (Max 3次)
        for attempt in range(3):
            try:
                # 建立 Ticker 物件 (yfinance 內部會處理 session，但我們可以透過延遲來優化)
                stock = yf.Ticker(t)
                
                # 下載數據
                temp = stock.history(period=period)
                
                # 判定是否成功
                if not temp.empty and len(temp) > 30: 
                    df = temp
                    
                    # --- 資料清洗標準程序 ---
                    if df.index.tz is not None: df.index = df.index.tz_localize(None)
                    df = df[~df.index.duplicated(keep='first')] 
                    target_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    clean_df = pd.DataFrame(index=df.index)
                    col_map = {str(c).lower(): c for c in df.columns}
                    for target in target_cols:
                        target_lower = target.lower()
                        if target_lower in col_map: clean_df[target] = df[col_map[target_lower]]
                        else: clean_df[target] = 0.0
                    
                    clean_df = clean_df.ffill().bfill().fillna(0)
                    return clean_df.astype(float)
                
                else:
                    # 抓不到資料，休息一下再試 (Random Sleep 0.5 ~ 2.0s)
                    time.sleep(random.uniform(0.5, 2.0))
                    
            except Exception as e:
                # 發生錯誤，休息久一點再試
                time.sleep(random.uniform(1.0, 3.0))
                continue
                
    # 試了所有方法都失敗，回傳空
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_info(ticker):
    try:
        if ticker.isdigit(): ticker = f"{ticker}.TW"
        stock = yf.Ticker(ticker)
        return stock.info
    except: return {}

@st.cache_data(ttl=300)
@st.cache_data(ttl=300)
def get_special_news_v28(ticker, name):
    # 保留原本的爬蟲邏輯，但在最後加入 NLP 分析
    core_ticker = ticker.replace(".TW", "").replace(".TWO", "")
    target_sites = ["money.udn.com", "moneydj.com", "investor.com.tw", "sinotrade.com.tw", "ctee.com.tw"]
    site_query = " OR ".join([f"site:{site}" for site in target_sites])
    query = f"{name} {core_ticker} ({site_query})"
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant&tbs=qdr:m3"
    
    news_items = []
    all_titles = "" # 用於關鍵字分析
    
    try:
        response = requests.get(rss_url, timeout=5)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            tw_tz = timezone(timedelta(hours=8))
            
            for item in root.findall('./channel/item'):
                title_text = item.find('title').text
                all_titles += title_text + " "
                
                # Sentiment (維持 V27 邏輯，但增加權重)
                score = 0.5
                sentiment_label = "中性"; sentiment_color = "sent-neu"
                if HAS_SNOWNLP:
                    s = SnowNLP(title_text); score = s.sentiments
                
                # 關鍵字加權 (手動補強 SnowNLP 的不足)
                bull_tags = ['創新高', '漲停', '獲利', '優於', '三率三升', '擴產', '急單']
                bear_tags = ['跌停', '重挫', '不如', '衰退', '虧損', '裁員', '降評']
                for w in bull_tags: 
                    if w in title_text: score += 0.2
                for w in bear_tags: 
                    if w in title_text: score -= 0.2
                
                if score > 0.65: sentiment_label = "🔥 利多"; sentiment_color = "sent-bull"
                elif score < 0.35: sentiment_label = "❄️ 利空"; sentiment_color = "sent-bear"
                
                try:
                    dt = email.utils.parsedate_to_datetime(item.find('pubDate').text)
                    dt_tw = dt.astimezone(tw_tz)
                    pub_str = dt_tw.strftime('%Y-%m-%d %H:%M')
                    timestamp = dt_tw.timestamp()
                except: pub_str = ""; timestamp = 0

                news_items.append({
                    'title': title_text, 'link': item.find('link').text,
                    'publisher': item.find('source').text if item.find('source') is not None else "Google",
                    'pubDate': pub_str, 'timestamp': timestamp,
                    'sent_label': sentiment_label, 'sent_color': sentiment_color
                })
            
            # [V28.0 新增] NLP 關鍵字萃取
            top_keywords = []
            if HAS_JIEBA and all_titles:
                tags = jieba.analyse.extract_tags(all_titles, topK=5)
                top_keywords = tags
                
            return news_items, top_keywords
    except: return [], []
    return [], []

def get_sector_info(ticker):
    core_ticker = ticker.replace(".TW", "").replace(".TWO", "")
    found = []
    for sector, sub_dict in SECTOR_DB.items():
        for sub_sector, tickers in sub_dict.items():
            clean_tickers = [t.replace(".TW", "").replace(".TWO", "") for t in tickers]
            if core_ticker in clean_tickers: found.append(f"{sector} ➜ {sub_sector}")
    return found if found else ["未歸類 / 其他產業"]

def calculate_indicators(df):
    try:
        df = df.copy()
        if len(df) < 60: return df
        df['MA5'] = df['Close'].rolling(5).mean(); df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean(); df['MA200'] = df['Close'].rolling(200).mean()
        df['VolMA20'] = df['Volume'].rolling(20).mean(); df['MA60_Slope'] = df['MA60'].diff()
        delta = df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum(); df['OBV_MA'] = df['OBV'].rolling(20).mean()
        exp12 = df['Close'].ewm(span=12, adjust=False).mean(); exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26; df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean(); df['Hist'] = df['MACD'] - df['Signal']
        low_min = df['Low'].rolling(9).min(); high_max = df['High'].rolling(9).max()
        df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(com=2).mean(); df['D'] = df['K'].ewm(com=2).mean()
        std = df['Close'].rolling(20).std(); df['BBU'] = df['MA20'] + 2*std; df['BBL'] = df['MA20'] - 2*std
        df['BandWidth'] = (df['BBU'] - df['BBL']) / df['MA20'].replace(0, np.nan)
        vol_sum = df['Volume'].rolling(20).sum().replace(0, np.nan)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).rolling(20).sum() / vol_sum
        tr1 = df['High'] - df['Low']; tr2 = (df['High'] - df['Close'].shift(1)).abs(); tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1); df['ATR'] = df['TR'].rolling(14).mean().replace(0, 0.001) 
        plus_dm = np.where((df['High'].diff() > (df['Low'].shift(1) - df['Low'])) & (df['High'].diff() > 0), df['High'].diff(), 0.0)
        minus_dm = np.where(((df['Low'].shift(1) - df['Low']) > df['High'].diff()) & ((df['Low'].shift(1) - df['Low']) > 0), (df['Low'].shift(1) - df['Low']), 0.0)
        df['+DI'] = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / df['ATR'])
        df['-DI'] = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / df['ATR'])
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']).replace(0, 0.001); df['ADX'] = df['DX'].rolling(14).mean()
        
        df['Donchian_H20'] = df['High'].rolling(20).max()
        df['Donchian_L10'] = df['Low'].rolling(10).min()
        
        return df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    except: return df

@st.cache_data(ttl=1800)
def analyze_sector_linkage(ticker, period="6mo"):
    # 1. 找出同板塊的股票
    core_ticker = ticker.replace(".TW", "").replace(".TWO", "")
    my_sector = "未知"
    peers = []
    
    for sector, sub_dict in SECTOR_DB.items():
        for sub, tickers in sub_dict.items():
            clean_tickers = [t.replace(".TW", "").replace(".TWO", "") for t in tickers]
            if core_ticker in clean_tickers:
                my_sector = sub
                peers = [t for t in tickers if t.replace(".TW","").replace(".TWO","") != core_ticker][:4] # 取前4檔做比較
                break
    
    if not peers: return None
    
    # 2. 抓取資料並計算相關性
    main_df = get_stock_data(ticker, period=period)
    if main_df.empty: return None
    
    peer_corr = {}
    sector_trend = pd.DataFrame(index=main_df.index)
    sector_trend['Main'] = main_df['Close']
    
    for p in peers:
        p_df = get_stock_data(p, period=period)
        if not p_df.empty:
            # 對齊資料
            aligned_df = pd.DataFrame({'Main': main_df['Close'], 'Peer': p_df['Close']}).dropna()
            if len(aligned_df) > 30:
                corr, _ = pearsonr(aligned_df['Main'], aligned_df['Peer'])
                peer_name = STOCK_NAMES.get(p, p)
                peer_corr[peer_name] = corr
                sector_trend[peer_name] = p_df['Close']
    
    # 計算板塊平均走勢 (標準化後)
    normalized = sector_trend / sector_trend.iloc[0]
    avg_trend = normalized.mean(axis=1)
    
    return {"sector": my_sector, "correlations": peer_corr, "avg_trend": avg_trend, "normalized": normalized}

def find_patterns(df):
    highs = df['High'].values
    lows = df['Low'].values
    peaks = argrelextrema(highs, np.greater, order=5)[0]
    troughs = argrelextrema(lows, np.less, order=5)[0]
    return peaks, troughs

def generate_battle_report(top_stock, scan_results):
    report_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "champion": {
            "code": top_stock['代號'],
            "name": top_stock['名稱'],
            "score": top_stock['總分'],
            "price": top_stock['現價']
        },
        "top_3_list": scan_results[:3], # 取前三名
        "market_summary": f"本次掃描 {len(scan_results)} 檔股票，冠軍由 {top_stock['名稱']} 奪得，總分 {top_stock['總分']} 分。"
    }
    return json.dumps(report_data, ensure_ascii=False, indent=2, cls=NumpyEncoder)

def generate_app_report(ticker, df, res):
    strat_name = res['strat_name']
    total_ret = res['total_ret']
    mdd = res['mdd']
    pos = res['pos'].iloc[-1]
    
    last_date = df.index[-1].strftime("%Y-%m-%d")
    last_close = df['Close'].iloc[-1]
    last_signal = "買進/持有" if pos == 1 else "賣出/空手"
    
    trade_count = 0
    if 'pos' in res:
        trades = res['pos'].diff().fillna(0).abs()
        trade_count = trades.sum() / 2
    
    report_data = {
        "report_type": "GA_Strategy_Evolution",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": {
            "code": ticker,
            "last_price": last_close,
            "date": last_date
        },
        "strategy": {
            "name": strat_name,
            "signal": last_signal,
            "backtest_performance": {
                "total_return_pct": round(total_ret * 100, 2),
                "max_drawdown_pct": round(mdd * 100, 2),
                "estimated_trades": int(trade_count)
            }
        },
        "message": f"AI 演化完畢。最佳策略為 [{strat_name}]，回測報酬率 {total_ret:.1%}，目前建議：{last_signal}。"
    }
    return report_data

# ==========================================
# [V29.3] Email SMTP 模組 (永久免費穩定版)
# ==========================================
def send_email_report(subject, html_content):
    # 1. 檢查 Secrets
    if 'email_sender' not in st.secrets or 'email_password' not in st.secrets:
        return False, "❌ 未設定 Email 帳號或應用程式密碼"

    sender = st.secrets['email_sender']
    password = st.secrets['email_password']
    receiver = st.secrets.get('email_receiver', sender) # 若沒設接收者，預設寄給自己
    
    # 2. 建構郵件
    msg = MIMEMultipart()
    msg['From'] = f"AI 戰情室 <{sender}>"
    msg['To'] = receiver
    msg['Subject'] = subject
    
    # 支援 HTML 格式
    msg.attach(MIMEText(html_content, 'html'))
    
    try:
        # 3. 連接 Gmail SMTP Server (SSL Port 465)
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        return True, f"✅ 戰報已寄至 {receiver}！"
    except Exception as e:
        return False, f"❌ 發送失敗: {str(e)}"
    

def process_stock_task(ticker):
    try:
        # [V27.10] 隨機延遲，防止 IP 被鎖
        import random
        time.sleep(random.uniform(0.1, 0.5))
        
        name = STOCK_NAMES.get(ticker, ticker)
        df = get_stock_data(ticker)
        if df.empty or len(df) < 100: return None
        df = calculate_indicators(df)
        info = get_stock_info(ticker) 
        last = df.iloc[-1]
        t_score = 0
        if last['Close'] > last['MA20']: t_score += 2
        if last['MA60_Slope'] > 0: t_score += 3 
        if last['Close'] > last['MA60']: t_score += 1
        if last['MACD'] > last['Signal']: t_score += 2
        if last['RSI'] > 50: t_score += 2
        c_score = 0
        if last['OBV'] > df['OBV_MA'].iloc[-1]: c_score += 4 
        if last['Volume'] > df['VolMA20'].iloc[-1]: c_score += 3 
        if (last['Close'] - last['Open']) > 0: c_score += 3 
        m_score = 0
        ret_1m = (last['Close'] / df['Close'].iloc[-20]) - 1
        if ret_1m > 0: m_score += 5
        if ret_1m > 0.05: m_score += 5 
        f_score = 5 
        if info:
            try:
                pe = info.get('trailingPE', 0); pb = info.get('priceToBook', 0)
                if 0 < pe < 25: f_score += 2
                if 0 < pb < 4: f_score += 2
            except: pass
        total_score = t_score + c_score + m_score + f_score
        return {"代號": ticker, "名稱": name, "總分": total_score, "T-技術": t_score, "C-籌碼": c_score, "M-動能": m_score, "F-基本": f_score, "現價": last['Close'], "斜率": "⬆️" if last['MA60_Slope'] > 0 else "⬇️"}
    except: return None

# ==========================================
# 2. 策略核心
# ==========================================
def calculate_supertrend_core(high, low, close, atr, period, multiplier):
    n = len(close); final_upper = np.zeros(n); final_lower = np.zeros(n); supertrend = np.zeros(n); trend = np.ones(n, dtype=int)
    basic_upper = (high + low) / 2 + (multiplier * atr); basic_lower = (high + low) / 2 - (multiplier * atr)
    final_upper[0] = basic_upper[0]; final_lower[0] = basic_lower[0]; supertrend[0] = final_upper[0]
    for i in range(1, n):
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]: final_upper[i] = basic_upper[i]
        else: final_upper[i] = final_upper[i-1]
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]: final_lower[i] = basic_lower[i]
        else: final_lower[i] = final_lower[i-1]
        if trend[i-1] == 1:
            supertrend[i] = final_lower[i]
            if close[i] < final_lower[i]: trend[i] = -1; supertrend[i] = final_upper[i]
            else: trend[i] = 1
        else:
            supertrend[i] = final_upper[i]
            if close[i] > final_upper[i]: trend[i] = 1; supertrend[i] = final_lower[i]
            else: trend[i] = -1
    return trend, supertrend

# [V27.10] 核心策略執行函數 (補回遺失的部分)
# [V28.0] 核心策略執行函數 (包含夏普與勝率計算)
def run_strategy_multi(data_dict, strategy_type, p1, p2, p3, sl_atr, tp_atr, vol_factor, trend_filter_mode, risk_per_trade):
    # --- 0. 數據預處理 ---
    closes = data_dict['close']; highs = data_dict['high']; lows = data_dict['low']; opens = data_dict['open']
    volumes = data_dict['volume']; atrs = data_dict['atr']; adxs = data_dict['adx']
    vol_mas = data_dict['vol_ma']; ma60s = data_dict['ma60']; ma200s = data_dict['ma200']
    ma60_slopes = data_dict['ma60_slope']
    rsis = data_dict['rsi']; bb_ups = data_dict['bbu']; ma20s = data_dict['ma20']
    don_h = data_dict['don_h']; don_l = data_dict['don_l']
    
    # 預算 MACD
    exp12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
    exp26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
    macd_line = exp12 - exp26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    hist_np = hist.values 
    
    current_mode = st.session_state.get('current_running_mode', "一般")
    n = len(closes)
    strategy_mode = int(strategy_type) % 4
    
    # --- 1. 產生基礎訊號 ---
    raw_signal = np.zeros(n, dtype=bool)

    # 計算 SuperTrend
    atr_p_st = int(p1); mult_st = p2 / 10.0
    st_trends, st_line = calculate_supertrend_core(highs, lows, closes, atrs, atr_p_st, mult_st)

    if strategy_mode == 0: # SuperTrend
        adx_thresh = int(p3)
        raw_signal = (st_trends == 1) & (adxs > adx_thresh)
    elif strategy_mode == 1: # RSI
        buy_level = 30 + (p2/2)
        raw_signal = (rsis < buy_level)
    elif strategy_mode == 2: # BB Breakout
        raw_signal = (closes > bb_ups)
    elif strategy_mode == 3: # Turtle
        raw_signal = (closes > don_h)
        
    # --- 2. 智慧濾網 ---
    pass_vol = (volumes > vol_mas * vol_factor) | (vol_factor <= 0.3)
    
    is_volume_spike = volumes > (vol_mas * 1.5)
    is_big_candle = closes > (opens * 1.015) 
    is_macd_turn_up = (hist_np > 0) & (np.roll(hist_np, 1) <= 0)
    is_breakout = (is_volume_spike & is_big_candle) | is_macd_turn_up
    
    is_crashing = (ma60_slopes < -0.5)
    is_early_bull = (closes > ma20s) & (closes > np.roll(ma20s, 1))
    
    # --- 3. 核心迴圈 ---
    pos_list = np.zeros(n, dtype=int)
    entry_reasons = np.zeros(n, dtype=int) 
    
    current_pos = 0; entry_price = 0.0; dynamic_sl = 0.0
    warmup = 60
    
    for i in range(warmup, n):
        # A. 進場
        if current_pos == 0:
            can_trade = False
            reason_code = 0
            
            if ("激進" in current_mode) or ("狙擊" in current_mode):
                if is_crashing[i]: can_trade = False
                elif is_breakout[i]: can_trade = True; reason_code = 1 
                elif closes[i] > ma60s[i]: can_trade = True; reason_code = 3 
            elif "保守" in current_mode:
                std_condition = (closes[i] > ma60s[i]) and (ma60_slopes[i] > 0)
                if std_condition: can_trade = True; reason_code = 3
                elif is_early_bull[i]: can_trade = True; reason_code = 2 
            else:
                if closes[i] > ma60s[i]: can_trade = True; reason_code = 3 
            
            if can_trade and raw_signal[i] and pass_vol[i]:
                current_pos = 1
                entry_price = closes[i]
                dynamic_sl = entry_price - (atrs[i] * sl_atr)
                entry_reasons[i] = reason_code 
        
        # B. 出場
        elif current_pos == 1:
            hard_sl = entry_price - (atrs[i] * sl_atr)
            current_tp_dist = (atrs[i] * tp_atr)
            if adxs[i] > 25: current_tp_dist *= 1.5 
            
            trailing_sl = highs[i] - current_tp_dist
            dynamic_sl = max(dynamic_sl, hard_sl, trailing_sl)
            
            should_exit = False
            exit_price_check = closes[i] if "狙擊" in current_mode else lows[i]
            if exit_price_check <= dynamic_sl: should_exit = True
            
            trend_is_weak = (adxs[i] < 30)
            if strategy_mode == 1 and (rsis[i] > (70 - p3/2)) and trend_is_weak: should_exit = True
            elif strategy_mode == 0 and st_trends[i] == -1: should_exit = True 
            elif strategy_mode == 3 and closes[i] < don_l[i]: should_exit = True

            if should_exit:
                current_pos = 0; dynamic_sl = 0; entry_price = 0
                
        pos_list[i] = current_pos
        
    # --- 4. 績效結算 ---
    ret_arr = data_dict['raw_ret']
    strategy_ret = pos_list[:-1] * ret_arr[1:]
    trades = np.abs(np.diff(pos_list))
    costs = trades * 0.001
    if len(costs) > len(strategy_ret): costs = costs[:-1]
    final_ret_series = strategy_ret - costs
    cum_ret = np.cumprod(1 + final_ret_series)
    if len(cum_ret) == 0: return None
    total_ret = cum_ret[-1] - 1
    running_max = np.maximum.accumulate(cum_ret)
    mdd = np.min((cum_ret - running_max) / running_max)
    strat_names = {0:"SuperTrend", 1:"RSI逆勢", 2:"布林突破", 3:"海龜交易"}

    # [V28.0 新增] 計算 Sharpe 與 勝率
    daily_rets = pd.Series(strategy_ret).fillna(0)
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    
    sharpe_ratio = 0
    if std_daily_ret != 0:
        sharpe_ratio = (avg_daily_ret / std_daily_ret) * (252 ** 0.5)
        
    # 勝率計算
    trade_pnl = []
    curr_p = 0; entry_p = 0
    for idx, p in enumerate(pos_list):
        if curr_p == 0 and p == 1: entry_p = closes[idx]; curr_p = 1
        elif curr_p == 1 and p == 0: 
            pnl = (closes[idx] - entry_p) / entry_p
            trade_pnl.append(pnl)
            curr_p = 0
    win_rate = 0.0
    if len(trade_pnl) > 0:
        wins = sum(1 for x in trade_pnl if x > 0)
        win_rate = wins / len(trade_pnl)
    
    # 回傳 10 個值，解決錯誤
    return pos_list, np.concatenate(([1.0], cum_ret)), total_ret, mdd, strat_names[strategy_mode], st_line, st_trends, entry_reasons, sharpe_ratio, win_rate

# ==========================================
# [V27.10 補丁] 樣式小幫手 & 適應度函數
# 請將此區塊放在 run_strategy_multi 之後，page_ga 之前
# ==========================================

def highlight_trade_status(val):
    val_str = str(val)
    if '獲利' in val_str: return 'background-color: #155724; color: white' 
    elif '虧損' in val_str: return 'background-color: #721c24; color: white' 
    elif '建倉' in val_str: return 'color: #00ffff' 
    return ''

def fitness_func(ga_instance, sol, idx):
    # 讀取當前正在演化的模式
    current_mode = st.session_state.get('current_running_mode', "一般")
    
    # 1. 解碼基因
    strat_type = sol[0]
    p1 = sol[1]; p2 = sol[2]; p3 = sol[3]
    sl_atr = sol[4]/10.0; tp_atr = sol[5]/10.0
    vol_factor = sol[6]/10.0
    # 基因8: 趨勢濾網強度 (0=不看, 1=MA60, 2=MA200+斜率)
    trend_filter_mode = 1 if sol[7] > 5 else 0 
    risk = 0.01 
    
    data_dict = st.session_state.train_data_dict 
    
   # 呼叫策略 (接收新的回傳值)
    res = run_strategy_multi(data_dict, strat_type, p1, p2, p3, sl_atr, tp_atr, vol_factor, trend_filter_mode, risk)

    if res is None: return -9999
   # [V28.0] 接收 10 個回傳值
    pos, _, total_ret, mdd, _, _, _, _, sharpe, win_rate = res 
    
    trades = np.sum(np.abs(np.diff(pos))) / 2
    abs_mdd = abs(mdd)
    
    if trades < 3: return -5000 # 交易次數過少懲罰
    
    score = 0

# [V28.0] 全新評分公式
    if "保守" in current_mode:
        # 保守: 高權重在 MDD 與 夏普，要求勝率 > 50%
        if abs_mdd > 0.12: return -10000 * abs_mdd
        if win_rate < 0.4: score -= 2000
        score = (sharpe * 500) + (total_ret * 200) + (win_rate * 1000)
        
    elif "激進" in current_mode:
        # 激進: 追求總報酬，夏普其次，接受 MDD
        if abs_mdd > 0.45: return -5000
        score = (total_ret * 3000) + (sharpe * 100)
        
    elif "狙擊" in current_mode:
        # 狙擊: 極度要求勝率與盈虧比 (Sortino/Sharpe)
        if win_rate < 0.6: score -= 5000 # 狙擊失敗懲罰
        score = (sharpe * 1000) + (win_rate * 2000) + (total_ret * 500)
        
    return score

    
    # 防止過少交易 (倖存者偏差)
    if trades < 3: return -5000
    
    score = 0
    
    if "保守" in current_mode:
        # 🛡️ 保守型: 嚴禁大賠
        if abs_mdd > 0.15: return -10000 * abs_mdd
        score = (total_ret * 500) + (1 / (abs_mdd + 0.01) * 200)
        
    elif "激進" in current_mode:
        # ⚔️ 激進型: 容忍波動，追求獲利
        if abs_mdd > 0.40: return -5000
        score = (total_ret * 2000) - (abs_mdd * 500)
        
    elif "狙擊" in current_mode:
        # 🎯 狙擊型: 重視獲利回撤比 (Calmar)
        if trades > 20: score -= (trades - 20) * 50
        calmar = total_ret / (abs_mdd + 0.01)
        score = calmar * 1000
        
    return score

# ... (前段代碼不變)
    
    # [V28.0 修正] 計算進階績效指標
    # 計算每日報酬率 (用於夏普值)
    daily_rets = pd.Series(strategy_ret).fillna(0)
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    
    # 年化夏普比率 (假設無風險利率為0)
    sharpe_ratio = 0
    if std_daily_ret != 0:
        sharpe_ratio = (avg_daily_ret / std_daily_ret) * (252 ** 0.5)
        
    # 計算勝率
    winning_trades = np.sum(trades[1:] > 0) # 簡易估算，實際需紀錄每筆損益
    # 這裡用更精準的方式算勝率 (根據 pos 變化)
    trade_pnl = []
    curr_p = 0; entry_p = 0
    for idx, p in enumerate(pos_list):
        if curr_p == 0 and p == 1: entry_p = closes[idx]; curr_p = 1
        elif curr_p == 1 and p == 0: 
            pnl = (closes[idx] - entry_p) / entry_p
            trade_pnl.append(pnl)
            curr_p = 0
            
    win_rate = 0.0
    if len(trade_pnl) > 0:
        wins = sum(1 for x in trade_pnl if x > 0)
        win_rate = wins / len(trade_pnl)

    # 回傳增加 sharpe_ratio 和 win_rate
    return pos_list, np.concatenate(([1.0], cum_ret)), total_ret, mdd, strat_names[strategy_mode], st_line, st_trends, entry_reasons, sharpe_ratio, win_rate

# --- Page 1: AI 總司令選股 (V30.0 天網全域掃描版) ---
def page_ai_selector():
    st.header("🤖 AI 總司令：全自動選股戰情室 (V30.0)")
    
    # 初始化 Session State
    if 'scan_results_df' not in st.session_state: st.session_state.scan_results_df = None
    if 'scan_top_stock' not in st.session_state: st.session_state.scan_top_stock = None
    if 'scan_json_report' not in st.session_state: st.session_state.scan_json_report = None
    
    # [V30.0] 新增：掃描範圍選擇器
    c_mode, c_info = st.columns([1, 2])
    with c_mode:
        scan_scope = st.radio("📡 掃描雷達範圍", ["🎯 單一戰略板塊", "🌍 全球戰略 (全域掃描)"], horizontal=True)
    
    all_tickers = []
    selected_sector_name = "全域市場"
    
    # --- 邏輯分支 ---
    if scan_scope == "🎯 單一戰略板塊":
        # 原本的單一板塊邏輯
        selected_chain = st.selectbox("請選擇戰略板塊:", list(SECTOR_DB.keys()))
        selected_sector_name = selected_chain
        sub_sectors = SECTOR_DB[selected_chain]
        
        # 收集該板塊股票
        with st.expander(f"📂 檢視 {selected_chain} 成分股", expanded=True):
            for sub_name, tickers in sub_sectors.items():
                st.markdown(f"**📌 {sub_name}**")
                sorted_tickers = sorted(tickers)
                all_tickers.extend(sorted_tickers)
                # 顯示標籤
                html_tags = ""
                for t in sorted_tickers:
                    display_name = STOCK_NAMES.get(t, t.replace(".TW", "").replace(".TWO", ""))
                    clean_code = t.replace(".TW", "").replace(".TWO", "")
                    html_tags += f'<span class="stock-tag">{clean_code} {display_name}</span>'
                st.markdown(f'<div style="line-height: 1.8;">{html_tags}</div>', unsafe_allow_html=True)
                st.write("")
                
    else:
        # [V30.0] 全域掃描邏輯
        st.info("🌍 您已啟動「天網模式」，將掃描資料庫中 **所有板塊** 的股票。")
        
        # 收集所有股票 (去除重複)
        unique_tickers = set()
        total_sectors = 0
        for sector_name, sub_dict in SECTOR_DB.items():
            total_sectors += 1
            for t_list in sub_dict.values():
                for t in t_list:
                    unique_tickers.add(t)
        
        all_tickers = sorted(list(unique_tickers))
        
        # -------------------------------------------------------
        # [修正點] 這裡原本少了 # 號導致報錯，現在修復了
        # 統計各個板塊的數量 (用於核對資料一致性)
        sector_counts = {k: sum(len(v) for v in sub.values()) for k, sub in SECTOR_DB.items()}
        # -------------------------------------------------------
        
        with c_info:
            # [V31.3] 增加詳細核對資訊
            st.metric("掃描目標總數", f"{len(all_tickers)} 檔", f"涵蓋 {total_sectors} 大板塊")
            
            # 顯示前幾個板塊的數量，方便您核對 (只顯示前 3 個板塊當代表)
            # 這裡會把剛剛算出來的 sector_counts 轉成字串顯示
            check_str = " | ".join([f"{k}:{v}" for k,v in list(sector_counts.items())[:3]])
            st.caption(f"🛡️ 資料一致性核對: {check_str} ...")
            
        with st.expander("📂 檢視全域掃描清單 (已去重)", expanded=False):
            st.write(", ".join([t.replace(".TW","") for t in all_tickers]))

    st.markdown("---")
    
    # 啟動掃描按鈕
    btn_label = f"🚀 啟動{scan_scope}"
    if st.button(btn_label, type="primary"):
        if not all_tickers:
            st.error("❌ 掃描清單為空，請檢查 sector_db.json")
        else:
            results = []
            progress_bar = st.progress(0); status_text = st.empty(); 
            status_text.text(f"⚡ AI 部隊集結中，目標 {len(all_tickers)} 檔，正在平行掃描...")
            
            start_time = time.time()
            
            # 使用執行緒池 (雲端建議 max_workers 不要超過 5，避免記憶體爆掉)
            # 如果是在本機跑，可以改回 10 或 20
            # 使用執行緒池
            # [V31.5 建議] 雲端為了抗阻擋，降速求穩
            # 本機可以用 10，雲端建議改為 3 或 4
            workers = 4 
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = list(executor.map(process_stock_task, all_tickers))
                
            for i, res in enumerate(futures):
                if res: results.append(res)
                progress_bar.progress((i + 1) / len(all_tickers))
                
            end_time = time.time(); duration = end_time - start_time
            
            if results:
                # 處理結果
                res_df = pd.DataFrame(results).sort_values("總分", ascending=False)
                top_stock = res_df.iloc[0] # 找出全體總冠軍
                
                # [V31.4 新增] 資料品質健檢儀表板
                # 1. 計算成功率：實際抓到的數量 / 預計掃描的數量
                success_rate = len(res_df) / len(all_tickers)
                
                # 2. 檢查資料新鮮度：抓取冠軍股的最後一筆交易日期
                # 我們需要重新叫一次 get_stock_data 來確認日期，或者在 process_stock_task 回傳時就包含日期
                # 這裡用一個快速的方式：檢查 res_df 是否有包含日期欄位 (若之前沒存，這裡無法顯示，但可作為改善方向)
                # 替代方案：我們直接在畫面上顯示「本次掃描樣本數」
                
                with c_info:
                    # 覆蓋原本的 metric，顯示更詳細的品質數據
                    st.metric(
                        "掃描品質報告", 
                        f"{len(res_df)} / {len(all_tickers)} 檔",
                        f"成功率: {success_rate:.1%}"
                    )
                    
                    if success_rate < 0.95:
                        st.warning(f"⚠️ 警告：有 {len(all_tickers) - len(res_df)} 檔股票抓取失敗 (可能是雲端 IP 被擋)，結果可能失準。")
                    else:
                        st.caption("✅ 資料完整度良好 (Loss < 5%)")


                # 生成 JSON 報告
                scan_results_list = res_df.to_dict('records')
                json_report = generate_battle_report(top_stock, scan_results_list)
                
                # 存入 Session State
                st.session_state.scan_results_df = res_df
                st.session_state.scan_top_stock = top_stock
                st.session_state.scan_json_report = json_report
                
                status_text.success(f"✅ 全域掃描完成！耗時 {duration:.2f} 秒。")
            else:
                st.warning("無有效資料或連線失敗。")
            
    # --- 顯示結果與 Email 發送 (共用邏輯) ---
    # [修正重點] 下面這一行是第 943 行左右，注意看冒號 :
    if st.session_state.scan_results_df is not None:
        
        # [修正重點] 這裡必須縮排 (4個空白)，Python 才知道這些程式碼屬於上面的 if
        res_df = st.session_state.scan_results_df
        top_stock = st.session_state.scan_top_stock
        json_report = st.session_state.scan_json_report
        
        # 標題區分
        if scan_scope == "🎯 單一戰略板塊":
            st.success(f"🏆 【{selected_sector_name}】板塊冠軍：**{top_stock['名稱']}** 總分：{top_stock['總分']}")
        else:
            st.success(f"👑 **【全市場總冠軍】**：**{top_stock['名稱']} ({top_stock['代號']})** 總分：{top_stock['總分']}")
        
        # 顯示結果表格 (這一行原本報錯，現在縮排正確了)
        st.dataframe(res_df.head(50).style.background_gradient(subset=['總分'], cmap='RdYlGn'), use_container_width=True)
        st.caption(f"💡 僅顯示前 50 名 (共 {len(res_df)} 筆結果)")

        # ================= [V32.0 新增] 全市場熱力圖 (Market Treemap) =================
        st.markdown("---")
        with st.expander("🗺️ V32.0 戰略地圖：全市場資金流向熱力圖", expanded=True):
            if '板塊' not in res_df.columns:
                # 1. 建立反向索引 (Ticker -> Sector)
                ticker_to_sector = {}
                for main_sec, sub_dict in SECTOR_DB.items():
                    for sub_sec, t_list in sub_dict.items():
                        for t in t_list:
                            clean_t = t.replace(".TW", "").replace(".TWO", "")
                            # 格式: 主板塊 > 子板塊
                            ticker_to_sector[clean_t] = {"Main": main_sec, "Sub": sub_sec}
                
                # 2. 將板塊資訊 Map 回 res_df
                # 使用 apply 搭配 lambda 來查表
                def get_sector_info(row, key):
                    code = row['代號'].replace(".TW", "").replace(".TWO", "")
                    return ticker_to_sector.get(code, {}).get(key, "其他")

                # 為了不影響原始 df，建立一個繪圖專用 df
                plot_df = res_df.copy()
                plot_df['主板塊'] = plot_df.apply(lambda x: get_sector_info(x, "Main"), axis=1)
                plot_df['子板塊'] = plot_df.apply(lambda x: get_sector_info(x, "Sub"), axis=1)
                # 權重放大
                plot_df['權重'] = plot_df['總分'] ** 2 
                
                # 3. 繪製 Treemap
                import plotly.express as px
                
                # 定義顏色：分數越高越紅
                fig_tree = px.treemap(
                    plot_df, 
                    path=[px.Constant("台股全市場"), '主板塊', '子板塊', '名稱'], 
                    values='權重',
                    color='總分',
                    color_continuous_scale='RdYlGn_r', # 紅到綠
                    title=f"AI 戰力熱力圖 (總掃描: {len(plot_df)} 檔)"
                )
                fig_tree.update_traces(root_color="lightgrey")
                fig_tree.update_layout(margin=dict(t=30, l=10, r=10, b=10), height=500)
                
                st.plotly_chart(fig_tree, use_container_width=True)
        # =========================================================================

        target_code = top_stock['代號'].replace(".TW", "").replace(".TWO", "")
        st.info(f"建議將總冠軍 **{target_code}** 帶入 PyGAD 進行演化。")
        
        # ================= [V31.1] Email 發送區塊 =================
        st.markdown("---")
        
        # 準備 Email 標題
        if scan_scope == "🎯 單一戰略板塊":
            title_prefix = f"【{selected_sector_name}冠軍】"
        else:
            title_prefix = "【全域總冠軍】" if len(res_df) > 50 else "【掃描冠軍】"
            
        email_subject = f"AI戰報(V32)：{title_prefix} {top_stock['名稱']}({target_code}) 分析報告"
        
        # 生成 Top 10 HTML
        top_10_html = ""
        limit = min(10, len(res_df))
        for i in range(limit):
            row = res_df.iloc[i]
            price_val = row.get('現價', 0)
            icon = "🔹"
            if i == 0: icon = "🥇"
            elif i == 1: icon = "🥈"
            elif i == 2: icon = "🥉"
            top_10_html += f"<li>{icon} <b>{row['名稱']}</b> ({row['代號']}) - 總分: {row['總分']} | 現價: {price_val:.1f}</li>"

        # 組合 HTML
        email_html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #00adb5;">🤖 AI 戰情室 V32 每日晨報</h2>
            <hr>
            <p>早安！AI 系統已完成 V32 天眼掃描，今日決選結果如下：</p>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #f2f2f2;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><b>👑 總冠軍</b></td>
                    <td style="padding: 10px; border: 1px solid #ddd; color: red;"><b>{top_stock['名稱']} ({target_code})</b></td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><b>🔥 戰力總分</b></td>
                    <td style="padding: 10px; border: 1px solid #ddd;"><b>{top_stock['總分']} 分</b></td>
                </tr>
            </table>
            <br>
            <p><b>📊 今日強勢股 Top 10：</b></p>
            <ul style="line-height: 1.6;">{top_10_html}</ul>
            <br>
            <p style="color: gray; font-size: 0.8em;">本信件由 AI 戰情室 V32 自動發送。</p>
        </body>
        </html>
        """

        c_mail_1, c_mail_2 = st.columns([3, 1])
        with c_mail_1:
            st.info(f"📧 已準備好 HTML 戰報：**{email_subject}**")
        with c_mail_2:
            st.write(" ") 
            st.write(" ")
            if st.button("📧 發送 Email 戰報", type="primary"):
                success, status_msg = send_email_report(email_subject, email_html)
                if success:
                    st.toast(status_msg, icon="✅")
                    st.success(status_msg)
                else:
                    st.error(status_msg)
        # ============================================================
        
        st.markdown("---")
        with st.expander("📋 每日戰情通報 (JSON For App)", expanded=True):
            st.markdown(f'<div class="json-box">{json_report}</div>', unsafe_allow_html=True)
            
    st.markdown("---")
    with st.expander("📖 T.C.M.F. 戰力評分標準", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown("#### 📈 T - 技術"); st.write("MA60翻揚(+3), >MA20(+2), MACD(+2), RSI>50(+2), >MA60(+1)")
        with c2: st.markdown("#### 💸 C - 籌碼"); st.write("OBV多頭(+4), 爆量(+3), 收紅(+3)")
        with c3: st.markdown("#### 🚀 M - 動能"); st.write("月漲>0%(+5), 月漲>5%(+5)")
        with c4: st.markdown("#### 🏢 F - 基本"); st.write("基礎分(+5), PE<25(+2), PB<4(+2)")

# --- Page 2: 全能達人戰情室 (V32.0 Gemini 整合版) ---
def page_dashboard():
    # --- 除錯用 (測試完請刪除) ---
    st.write("目前 Secrets 裡有的鑰匙:", list(st.secrets.keys()))
    # ---------------------------
    st.header("⚡ 全能達人戰情室 (V32.0)")
    col_input, col_info = st.columns([1, 3])
    with col_input: 
        t = st.text_input("輸入個股代號", "2330", key="dash_t")
    
    if t:
        # 1. 抓取資料
        df = get_stock_data(t)
        if df.empty or len(df) < 30: 
            st.error("無資料或資料不足")
            return
        
        df = calculate_indicators(df)
        info = get_stock_info(t)
        # 嘗試取得名稱，若無則用代號
        name = STOCK_NAMES.get(t.upper() + ".TW", t)
        if name == t: name = STOCK_NAMES.get(t, t)
        
        last = df.iloc[-1]; prev = df.iloc[-2]
        change = last['Close'] - prev['Close']; pct = change / prev['Close']
        color = "red" if change > 0 else "green"
        
        with col_info: 
            st.markdown(f"### {name} ({t})")
            st.markdown(f"<h2 style='color:{color}'>{last['Close']:.2f} <small>({change:+.2f} / {pct:+.2%})</small></h2>", unsafe_allow_html=True)
            sectors = get_sector_info(t.upper() + ".TW") 
            for s in sectors: st.caption(f"📍 {s}")
            
        tab1, tab2, tab3 = st.tabs(["ℹ️ 資訊流 & AI", "💸 資金流", "📈 技術流"])
        
        # --- Tab 1: 資訊流 (含 V32.0 Gemini) ---
        with tab1:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("📰 特種搜查")
                # 呼叫新聞函數 (相容舊版名稱，若您有改名請自行調整)
                try:
                    news, keywords = get_special_news_v28(t, name)
                except:
                    # 相容性備案
                    news = get_special_news(t, name); keywords = []
                
                # 顯示關鍵字
                if keywords:
                    st.markdown("🔥 **AI 提取關鍵字:**")
                    kw_html = "".join([f"<span style='background:#333;color:#00adb5;padding:2px 6px;border-radius:4px;margin:2px;font-size:0.8em'>{k}</span>" for k in keywords])
                    st.markdown(kw_html, unsafe_allow_html=True)
                
                st.divider()

# ================= [V32.4] Gemini 分析師 (穩定額度版) =================
                if "AI_Studio_Key" in st.secrets:
                    if st.button("🤖 呼叫 Gemini 頂級分析師", type="primary"):
                        with st.spinner("Gemini 正在閱讀財報與新聞..."):
                            try:
                                # 設定 Key
                                genai.configure(api_key=st.secrets["AI_Studio_Key"])
                                
                                # [修正點] 改用 'gemini-flash-latest'
                                # 這會自動指向目前有免費額度的最新版本 (通常是 1.5 Flash)
                                model = genai.GenerativeModel('gemini-flash-latest')
                                
                                # 準備資料
                                last_close = df.iloc[-1]['Close']
                                ma60 = df.iloc[-1]['MA60']
                                trend = "多頭排列" if last_close > ma60 else "空頭/盤整"
                                news_titles = ", ".join([n['title'] for n in news[:5]]) if news else "無近期新聞"
                                
                                prompt = (
                                    f"你是一位華爾街頂級分析師。請分析台股 {name}({t})。\n"
                                    f"1. 技術面：現價 {last_close}，MA60為 {ma60:.2f}，目前呈現 {trend}。\n"
                                    f"2. 消息面：近期新聞標題包含「{news_titles}」。\n"
                                    f"3. 任務：請用繁體中文，綜合上述資訊，給出約 100 字的精簡點評，並指出潛在風險與機會。"
                                )
                                
                                response = model.generate_content(prompt)
                                st.success("🤖 Gemini 分析報告：")
                                st.markdown(f"> {response.text}")
                                
                            except Exception as e:
                                # 錯誤處理優化：如果還是 429，顯示更友善的訊息
                                if "429" in str(e):
                                    st.warning("⚠️ AI 分析師正在忙線中 (達到免費額度上限)，請稍等 1 分鐘後再試。")
                                else:
                                    st.error(f"Gemini 連線失敗: {e}")
                else:
                    st.caption("⚠️ 請在 Secrets 設定 AI_Studio_Key 以啟用 AI 分析")
                st.divider()
                # ===========================================================

                if news: 
                    for n in news: 
                        st.markdown(f'<div class="news-card"><a href="{n["link"]}" target="_blank" class="news-title"><span class="sentiment-tag {n.get("sent_color", "sent-neu")}">{n.get("sent_label", "中性")}</span> {n["title"]}</a><span class="news-source">{n["publisher"]}</span> <span class="news-time">{n["pubDate"]}</span></div>', unsafe_allow_html=True)
                else: 
                    st.info("無新聞")
                    st.markdown(f'<a href="https://www.google.com/search?q={t}+tw+stock+news&tbm=nws" target="_blank" class="link-btn">🔍 Google</a>', unsafe_allow_html=True)
            
            with c2: 
                # 板塊雷達 (V28 功能)
                st.subheader("🔗 板塊聯動雷達")
                try:
                    sec_data = analyze_sector_linkage(t)
                    if sec_data:
                        st.caption(f"所屬子板塊: **{sec_data['sector']}**")
                        if sec_data['correlations']:
                            corr_cols = st.columns(len(sec_data['correlations']))
                            for i, (p_name, corr_val) in enumerate(sec_data['correlations'].items()):
                                with corr_cols[i % 4]:
                                    st.metric(f"vs {p_name}", f"{corr_val:.2f}")
                        
                        norm_df = sec_data['normalized']
                        fig_sec = go.Figure()
                        fig_sec.add_trace(go.Scatter(x=norm_df.index, y=norm_df['Main'], name=name, line=dict(color='yellow', width=2)))
                        fig_sec.add_trace(go.Scatter(x=norm_df.index, y=sec_data['avg_trend'], name="同業平均", line=dict(color='gray', dash='dash')))
                        fig_sec.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", hovermode="x unified")
                        st.plotly_chart(fig_sec, use_container_width=True)
                    else:
                        st.warning("無法取得同業資料")
                except:
                    st.warning("板塊資料載入失敗")

                st.subheader("🏢 簡介")
                s = info.get('longBusinessSummary')
                st.write(s) if s else st.warning("無簡介")
                st.markdown(f'<a href="https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={t}" target="_blank" class="link-btn">Goodinfo</a>', unsafe_allow_html=True)
                
        # --- Tab 2: 資金流 ---
        with tab2:
            st.markdown("### 🏛️ 官方籌碼"); c_l = st.columns(3)
            with c_l[0]: st.markdown(f'<a href="https://goodinfo.tw/tw/ShowBuySaleChart.asp?STOCK_ID={t}&CHT_CAT=DATE" target="_blank" class="link-btn">Goodinfo</a>', unsafe_allow_html=True)
            with c_l[1]: st.markdown(f'<a href="https://www.tpex.org.tw/zh-tw/mainboard/trading/major-institutional/detail/day.html" target="_blank" class="link-btn">TPEx</a>', unsafe_allow_html=True)
            with c_l[2]: st.markdown(f'<a href="https://www.twse.com.tw/zh/trading/foreign/t86.html" target="_blank" class="link-btn">TWSE</a>', unsafe_allow_html=True)
            
            st.divider()
            m1, m2 = st.columns(2)
            obv_s = df['OBV'].iloc[-1] - df['OBV'].iloc[-20]
            m1.metric("OBV", "吸籌 🟢" if obv_s > 0 else "出貨 🔴")
            vr = last['Volume']/last['VolMA20'] if last['VolMA20']>0 else 0
            m2.metric("量能", f"{vr:.2f}x")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange'), name='VWAP'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], fill='tozeroy', line=dict(color='cyan'), name='OBV'), row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, key="fund")
            
        # --- Tab 3: 技術流 ---
        with tab3:
            st.write("📊 **進階技術 (含圖形識別)**")
            c1,c2,c3 = st.columns(3)
            c1.metric("ADX", f"{last.get('ADX',0):.1f}")
            c2.metric("KD", f"K={last['K']:.1f}")
            c3.metric("BW", f"{last.get('BandWidth',0):.2%}")
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
            
            peaks, troughs = find_patterns(df)
            if len(peaks) > 0: fig.add_trace(go.Scatter(x=df.index[peaks], y=df['High'].iloc[peaks], mode='markers', marker=dict(color='red', symbol='triangle-down', size=8), name='波峰'), row=1, col=1)
            if len(troughs) > 0: fig.add_trace(go.Scatter(x=df.index[troughs], y=df['Low'].iloc[troughs], mode='markers', marker=dict(color='green', symbol='triangle-up', size=8), name='波谷'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df.index, y=df['BBU'], line=dict(color='gray'), name='BBU'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BBL'], line=dict(color='gray'), fill='tonexty'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['K'], line=dict(color='yellow')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['D'], line=dict(color='purple')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color='white')), row=3, col=1)
            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, key="tech")
def page_ga():
    st.header("🧬 PyGAD 策略進化 (V28.2 儀表板修復版)")
    if not HAS_PYGAD: st.error("❌ 需安裝 pygad"); return
    
    # [V28.2 修改] 增加即時名稱顯示
    c1, c2 = st.columns([1, 2])
    with c1: 
        t = st.text_input("優化標的", "2330", key="ga_t")
        
        # 自動查找名稱邏輯
        stock_name = "未知 / 未載入"
        # 嘗試直接查找或加 .TW 查找
        if t in STOCK_NAMES: stock_name = STOCK_NAMES[t]
        elif f"{t}.TW" in STOCK_NAMES: stock_name = STOCK_NAMES[f"{t}.TW"]
        elif f"{t}.TWO" in STOCK_NAMES: stock_name = STOCK_NAMES[f"{t}.TWO"]
        
        # 使用 caption 顯示在輸入框正下方
        st.caption(f"📌 **{stock_name}**")
        
        cash = st.number_input("本金", value=1000000)

    with c2: 
        c2a, c2b = st.columns(2)
        period = c2a.selectbox("回測期間", ["1y", "2y", "3y", "5y"], index=1)
        split_pct = c2b.slider("訓練集佔比", 0.5, 0.9, 0.75, 0.05)
    
    with st.expander("⚙️ 進化參數"): 
        gens = st.slider("繁衍代數", 5, 100, 30)
        pop_size = st.slider("種群大小", 10, 50, 20)

    if st.button("🧬 啟動 AI 全方位進化 (一鍵三模)"):
        if 'ga_results' in st.session_state: del st.session_state.ga_results
        modes = ["🛡️ 保守型", "⚔️ 激進型", "🎯 狙擊型"]; results_store = {}
        
        # 1. 數據準備
        df = get_stock_data(t, period=period); 
        if df.empty: st.error("無資料"); return
        df = calculate_indicators(df).dropna()
        if len(df) < 50: st.error("資料不足"); return
        if 'MA60_Slope' not in df.columns: df['MA60_Slope'] = df['MA60'].diff().fillna(0)

        split_idx = int(len(df) * split_pct); train_df = df.iloc[:split_idx]; test_df = df.iloc[split_idx:]; 
        st.session_state.train_df = train_df; split_date = df.index[split_idx]
        
        data_dict = {
            'open': train_df['Open'].values, 'high': train_df['High'].values, 'low': train_df['Low'].values, 'close': train_df['Close'].values,
            'volume': train_df['Volume'].values, 'vol_ma': train_df['VolMA20'].fillna(0).values,
            'ma60': train_df['MA60'].fillna(0).values, 'ma60_slope': train_df['MA60_Slope'].fillna(0).values,
            'ma200': train_df['MA200'].fillna(0).values, 'adx': train_df['ADX'].fillna(0).values, 'atr': train_df['ATR'].fillna(0).values,
            'rsi': train_df['RSI'].fillna(50).values, 'bbu': train_df['BBU'].values, 'bbl': train_df['BBL'].values, 'ma20': train_df['MA20'].values,
            'don_h': train_df['Donchian_H20'].values, 'don_l': train_df['Donchian_L10'].values,
            'raw_ret': train_df['Close'].pct_change().fillna(0).values
        }
        st.session_state.train_data_dict = data_dict
        
        gene_space = [range(0, 4), range(5, 41), range(10, 61), range(15, 51), range(10, 51), range(10, 101), range(5, 21), range(0, 11), range(1, 11)]
        progress_bar = st.progress(0)
        
        # 2. 演化迴圈
        for i, m in enumerate(modes):
            st.session_state.current_running_mode = m 
            with st.spinner(f"正在演化 【{m}】..."):
                ga = pygad.GA(num_generations=gens, num_parents_mating=5, fitness_func=fitness_func, sol_per_pop=pop_size, num_genes=9, gene_space=gene_space, random_seed=42, suppress_warnings=True)
                ga.run(); best_sol, _, _ = ga.best_solution()
                
                # 全期間回測
                full_data_dict = {
                    'open': df['Open'].values, 'high': df['High'].values, 'low': df['Low'].values, 'close': df['Close'].values,
                    'volume': df['Volume'].values, 'vol_ma': df['VolMA20'].fillna(0).values,
                    'ma60': df['MA60'].fillna(0).values, 'ma60_slope': df['MA60_Slope'].fillna(0).values,
                    'ma200': df['MA200'].fillna(0).values, 'adx': df['ADX'].fillna(0).values, 'atr': df['ATR'].fillna(0).values,
                    'rsi': df['RSI'].fillna(50).values, 'bbu': df['BBU'].values, 'bbl': df['BBL'].values, 'ma20': df['MA20'].values,
                    'don_h': df['Donchian_H20'].values, 'don_l': df['Donchian_L10'].values,
                    'raw_ret': df['Close'].pct_change().fillna(0).values
                }
                
                strat_type = best_sol[0]; p1 = best_sol[1]; p2 = best_sol[2]; p3 = best_sol[3]
                sl_atr = best_sol[4]/10.0; tp_atr = best_sol[5]/10.0; vol_factor = best_sol[6]/10.0
                trend_filter_mode = 1 if best_sol[7]>5 else 0; risk = 0.01
                
                res_tuple = run_strategy_multi(full_data_dict, strat_type, p1, p2, p3, sl_atr, tp_atr, vol_factor, trend_filter_mode, risk)
                
                if res_tuple:
                    # [修改這裡] 這裡也要改成接收 10 個變數 (使用 _ 忽略最後兩個不需要畫圖的變數)
                    pos, cum_ret, total_ret, mdd, strat_name, st_line, trends, entry_reasons, _, _ = res_tuple
                    
                    results_store[m] = {
                        "params": (strat_type, p1, p2, p3, sl_atr, tp_atr, vol_factor, trend_filter_mode, risk), 
                        "pos": pd.Series(pos, index=df.index), 
                        "cum_ret": pd.Series(cum_ret, index=df.index), 
                        "mdd": mdd, 
                        "st_line": pd.Series(st_line, index=df.index), 
                        "trend": pd.Series(trends, index=df.index), 
                        "total_ret": total_ret, "df": df, "split_date": split_date, "strat_name": strat_name,
                        "entry_reasons": pd.Series(entry_reasons, index=df.index)
                    }
            progress_bar.progress((i + 1) / 3)
        st.session_state.ga_results = results_store; progress_bar.empty(); st.success("🏆 全方位戰略演化完成！")

    # 3. 顯示結果
    if 'ga_results' in st.session_state:
        results_store = st.session_state.ga_results; modes = list(results_store.keys())
        
        # 統計表
        summary_data = []
        for m in modes:
            res = results_store[m]; df_res = res['df']; cum_ret = res['cum_ret']; pos = res['pos']; strat_name = res['strat_name']
            split_date = res['split_date']
            train_mask = df_res.index < split_date; test_mask = df_res.index >= split_date
            
            # [V27.11] 補回 MDD 計算
            t_ret = 0; t_trades = 0; t_pnl = 0; t_mdd = 0.0
            if len(cum_ret[train_mask]) > 0:
                curve = cum_ret[train_mask] / cum_ret[train_mask].iloc[0]
                t_ret = curve.iloc[-1] - 1
                t_pnl = t_ret * cash 
                t_trades = (pos[train_mask].diff().abs().sum()) / 2
                t_mdd = ((curve - curve.cummax()) / curve.cummax()).min()
            
            v_ret = 0; v_trades = 0; v_pnl = 0; v_mdd = 0.0
            if len(cum_ret[test_mask]) > 0:
                curve = cum_ret[test_mask] / cum_ret[test_mask].iloc[0]
                v_ret = curve.iloc[-1] - 1
                v_pnl = v_ret * cash 
                v_trades = (pos[test_mask].diff().abs().sum()) / 2
                v_mdd = ((curve - curve.cummax()) / curve.cummax()).min()
                
            summary_data.append({
                "模式": m, "最佳策略": strat_name, 
                "訓練-報酬": f"{t_ret:.1%}", "訓練-損益": f"${t_pnl:,.0f}", "訓練-MDD": f"{t_mdd:.1%}", "訓練-次數": int(t_trades), 
                "驗證-報酬": f"{v_ret:.1%}", "驗證-損益": f"${v_pnl:,.0f}", "驗證-MDD": f"{v_mdd:.1%}", "驗證-次數": int(v_trades)
            })
        st.dataframe(pd.DataFrame(summary_data))
        
        if st.button("📱 生成 App 通報資料 (JSON)"):
            best_mode = modes[0]; best_res = results_store[best_mode]
            report = generate_app_report(t, df, best_res)
            st.json(report)

        # 4. 繪圖與儀表板區
        tabs = st.tabs(modes)
        for idx, tab in enumerate(tabs):
            m = modes[idx]; res = results_store[m]; df = res['df']; strat_name = res['strat_name']
            reasons = res['entry_reasons']; pos = res['pos']
            params = res['params'] # 取得參數
            
            with tab:
                # [V27.11] 戰情儀表板與參數顯示
                last_pos = pos.iloc[-1]
                last_close = df['Close'].iloc[-1]
                last_atr = df['ATR'].iloc[-1]
                
                # 計算操作數值
                strat_t, p1, p2, p3, sl_atr, tp_atr, vol_f, t_filt, _ = params
                
                # 目標價與停損價估算 (僅供參考)
                target_price = last_close + (last_atr * tp_atr)
                stop_price = last_close - (last_atr * sl_atr)
                
                # 狀態判斷
                status_color = "green" if last_pos == 1 else "gray"
                status_text = "🟢 持有中 (BULL)" if last_pos == 1 else "⚪ 空手觀望 (WAIT)"
                
                # 顯示儀表板
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid {status_color};">
                    <h3 style="margin:0; color: {status_color};">{status_text}</h3>
                    <p style="margin:5px 0 0 0;">
                    <b>現價:</b> {last_close:.2f} | 
                    <b>🎯 目標:</b> {target_price:.2f} | 
                    <b>🛡️ 停損:</b> {stop_price:.2f} (ATR={last_atr:.2f})
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # 顯示系統參數 (Best Genes)
                with st.expander("🧬 檢視 AI 演化之最佳系統參數", expanded=False):
                    st.write(f"**策略類型**: {strat_name} (Type {strat_t})")
                    st.write(f"**核心參數**: P1={p1}, P2={p2}, P3={p3}")
                    st.write(f"**風控參數**: 停損={sl_atr:.1f}x ATR, 停利={tp_atr:.1f}x ATR")
                    st.write(f"**濾網設定**: 量能係數={vol_f:.1f}, 趨勢濾網={'開啟' if t_filt else '關閉'}")

                # 繪圖
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
                
                # 買點標記
                buy_indices = df.index[pos.diff() == 1]
                idx_rocket = [ix for ix in buy_indices if reasons[ix] == 1]
                idx_shield = [ix for ix in buy_indices if reasons[ix] == 2]
                idx_std    = [ix for ix in buy_indices if reasons[ix] == 3]
                
                if idx_rocket: fig.add_trace(go.Scatter(x=idx_rocket, y=df.loc[idx_rocket, 'Low']*0.99, mode='text+markers', text='🚀', textposition='bottom center', marker=dict(symbol='star', size=14, color='#FF4B4B'), name='先鋒突擊'), row=1, col=1)
                if idx_shield: fig.add_trace(go.Scatter(x=idx_shield, y=df.loc[idx_shield, 'Low']*0.99, mode='text+markers', text='🛡️', textposition='bottom center', marker=dict(symbol='diamond', size=12, color='#21C354'), name='早鳥防禦'), row=1, col=1)
                if idx_std: fig.add_trace(go.Scatter(x=idx_std, y=df.loc[idx_std, 'Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00FFFF'), name='標準部隊'), row=1, col=1)
                
                # SuperTrend 線
                st_line = res['st_line']; trend = res['trend']
                st_bull = st_line.copy(); st_bull[trend == -1] = np.nan
                st_bear = st_line.copy(); st_bear[trend == 1] = np.nan
                fig.add_trace(go.Scatter(x=df.index, y=st_bull, mode='lines', line=dict(color='green', width=1), name='支撐'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=st_bear, mode='lines', line=dict(color='red', width=1), name='壓力'), row=1, col=1)

                # 賣點
                sp = df[(pos.diff() == -1)]; 
                fig.add_trace(go.Scatter(x=sp.index, y=sp['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', size=12, color='magenta'), name='賣出'), row=1, col=1)
                
                # 資產曲線
                fig.add_trace(go.Scatter(x=df.index, y=cash * res['cum_ret'], mode='lines', line=dict(color='orange'), name='總資產'), row=2, col=1)
                
                fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True, key=f"c_{idx}")
                
                # [V27.11] 圖表下方備註 (Legend)
                st.info("""
                **📝 戰術圖示說明：**
                * 🚀 **先鋒突擊**：偵測到爆量長紅或強勁動能，無視均線特權進場。
                * 🛡️ **早鳥防禦**：(僅保守型) 在均線未翻揚前，偵測到 W 底或強勢反彈提早佈局。
                * 🔵 **標準部隊**：符合均線多頭排列與技術指標的標準進場點。
                """)
                
                # 交易明細
                tl = []; cp = 0; ep = 0
                dates = df.index.strftime('%Y-%m-%d'); closes = df['Close'].values; positions = res['pos'].values
                for d, close, np_ in zip(dates, closes, positions):
                    if cp == 0 and np_ == 1: 
                        ep = close
                        reason_icon = "🔵"; r_code = reasons[df.index.get_loc(d)]
                        if r_code == 1: reason_icon = "🚀"
                        elif r_code == 2: reason_icon = "🛡️"
                        tl.append({"日期": d, "動作": f"買進 {reason_icon}", "價格": ep, "損益": "建倉"})
                    elif cp == 1 and np_ == 0: 
                        xp = close; pnl = (xp - ep) / ep
                        p_str = f"獲利 +{pnl:.2%}" if pnl > 0 else f"虧損 {pnl:.2%}"
                        tl.append({"日期": d, "動作": "賣出", "價格": xp, "損益": p_str})
                    cp = np_
                
                if tl: st.dataframe(pd.DataFrame(tl).style.applymap(highlight_trade_status, subset=['損益']), use_container_width=True, key=f"t_{idx}")

# ==========================================
# 4. 主程式入口
# ==========================================
PAGES = {"🤖 AI 總司令選股": page_ai_selector, "⚡ 全能達人戰情室": page_dashboard, "🧬 PyGAD 策略進化": page_ga}
st.sidebar.title("⚡ AI 戰情室 V32.0"); st.sidebar.caption("相容修復 | JSON完美")
sel = st.sidebar.radio("功能模組", list(PAGES.keys())); PAGES[sel]()
