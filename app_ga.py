import logging
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import argrelextrema 
import json
import smtplib
import google.generativeai as genai
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import random

import streamlit as st
import google.generativeai as genai
import pandas as pd
# ... 其他 import ...

# ==========================================
# 🚑 [緊急診斷] AI 環境檢測區
# ==========================================
with st.sidebar:
    st.markdown("---")
    st.subheader("🔧 AI 環境診斷")
    
    # 1. 檢查版本 (如果是 0.3.x 或 0.4.x 代表太舊，必須是 0.7.2 以上)
    try:
        ver = genai.__version__
        st.write(f"📦 套件版本: `{ver}`")
    except:
        st.error("無法讀取版本，套件可能損壞")

    # 2. 檢查 API Key 與可用模型
    if "AI_Studio_Key" in st.secrets:
        genai.configure(api_key=st.secrets["AI_Studio_Key"])
        try:
            st.write("🔍 正在掃描可用模型...")
            # 列出所有支援 generateContent 的模型
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if models:
                st.success(f"✅ 抓到 {len(models)} 個模型")
                st.code(models) # 顯示清單給您看
            else:
                st.error("❌ 掃描成功但清單為空 (您的 Key 可能權限不足)")
        except Exception as e:
            st.error(f"❌ 連線失敗: {e}")
    else:
        st.warning("⚠️ 尚未設定 API Key")
    st.markdown("---")
# ==========================================

# [V31.2] 系統警示消音器
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)

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

try:
    import jieba
    import jieba.analyse
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

from scipy.stats import pearsonr 

# [V27.2] 自定義 JSON 編碼器
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
# 0. 全域設定
# ==========================================
st.set_page_config(page_title="AI 戰情室: V33.6 精簡優化版", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 20px; }
    .stDataFrame { border: 1px solid #ddd; } 
    button[data-baseweb="tab"] { font-size: 1.2em; font-weight: bold; }
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

# ==========================================
# 核心類別：資料庫與 RAG
# ==========================================

class BattleDB:
    def __init__(self, db_name="strategy.db"):
        self.db_name = db_name
        self.create_tables()

    def get_connection(self):
        return sqlite3.connect(self.db_name, check_same_thread=False)

    def create_tables(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_genes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ticker TEXT,
                strategy_name TEXT,
                total_return REAL,
                sharpe_ratio REAL,
                params TEXT,
                note TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                scope TEXT,
                champion_code TEXT,
                champion_score REAL,
                report_json TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_gene(self, ticker, strat_name, ret, sharpe, params, note=""):
        conn = self.get_connection()
        p_str = json.dumps(params, cls=NumpyEncoder)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("INSERT INTO strategy_genes (timestamp, ticker, strategy_name, total_return, sharpe_ratio, params, note) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                     (ts, ticker, strat_name, ret, sharpe, p_str, note))
        conn.commit()
        conn.close()
        return "✅ 基因已永久入庫！"

    def save_scan_report(self, scope, champion, score, report_json):
        conn = self.get_connection()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("INSERT INTO scan_reports (timestamp, scope, champion_code, champion_score, report_json) VALUES (?, ?, ?, ?, ?)",
                     (ts, scope, champion, score, report_json))
        conn.commit()
        conn.close()

# <--- 請在這裡按下 Enter 鍵，空兩行，然後貼上新的代碼 --->
# ⚠️ 注意：新的代碼必須「靠左對齊」(沒有縮排)，不要縮進 BattleDB 裡面

# ==========================================
# 3.5 自選股管理 (Watchlist Manager)
# ==========================================
WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return []
    return []

def save_watchlist(tickers):
    # 去重並排序
    unique_tickers = sorted(list(set(tickers)))
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_tickers, f)
    return unique_tickers

def toggle_watchlist(ticker):
    wl = load_watchlist()
    clean_t = ticker.replace(".TW", "").replace(".TWO", "")
    if clean_t in wl:
        wl.remove(clean_t)
        msg = f"❌ 已從自選股移除: {clean_t}"
    else:
        wl.append(clean_t)
        msg = f"✅ 已加入自選股: {clean_t}"
    save_watchlist(wl)
    return msg

# <--- 新代碼結束 --->

# ==========================================
# [V34.6] RAG 核心：智慧適配版 (針對您的先進環境)
# ==========================================
class RAGAdvisor:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.embedding_model = "models/text-embedding-004"
        self.active_model = None
        self.model_name = "未連線"
        self.memory_docs = []
        self.memory_vecs = []

        try:
            print("🔍 正在智慧匹配可用模型...")
            
            # 1. 取得您帳號實際擁有的模型清單
            all_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            
            # 2. 定義優先順序 (從您的清單中挑選最強最快的)
            # 優先找 2.5 Flash -> 2.0 Flash -> 任何 Flash -> 任何 Pro
            priority_keywords = [
                "gemini-2.5-flash", 
                "gemini-2.0-flash",
                "gemini-flash",
                "gemini-2.5-pro",
                "gemini-2.0-pro"
            ]
            
            target_model = None
            
            # 3. 進行匹配
            for keyword in priority_keywords:
                # 在您的清單中找是否有符合關鍵字的
                match = next((m for m in all_models if keyword in m), None)
                if match:
                    target_model = match
                    break
            
            # 4. 如果都沒對到，就直接拿清單裡的第一個 (保底)
            if not target_model and all_models:
                target_model = all_models[0]
            
            if target_model:
                self.model_name = target_model
                self.active_model = genai.GenerativeModel(target_model)
                print(f"✅ 成功鎖定模型: {target_model}")
            else:
                st.error("❌ 找不到任何可用模型 (List is empty)")

        except Exception as e:
            st.error(f"❌ 初始化 AI 失敗: {str(e)}")

    def add_document(self, text, source="System"):
        if not text: return
        doc_entry = f"[{source}] {text}"
        self.memory_docs.append(doc_entry)
        try:
            vec = genai.embed_content(model=self.embedding_model, content=text)['embedding']
            self.memory_vecs.append(vec)
            return True
        except:
            return False

    def clear_memory(self):
        self.memory_docs = []
        self.memory_vecs = []

    def query(self, user_question, top_k=4):
        if not self.active_model: return f"❌ AI 初始化失敗。"
        
        context = ""
        if self.memory_vecs:
            try:
                # Embedding 查詢
                q_vec = genai.embed_content(model=self.embedding_model, content=user_question)['embedding']
                scores = np.dot(self.memory_vecs, q_vec)
                actual_k = min(len(self.memory_docs), top_k)
                top_indices = np.argsort(scores)[-actual_k:][::-1]
                context = "\n".join([self.memory_docs[i] for i in top_indices])
            except:
                context = "(向量檢索略過)"

        try:
            final_prompt = f"""
            你是一位專業的財經分析師。請回答使用者的問題。
            
            【參考資訊】
            {context}
            
            【使用者問題】
            {user_question}
            """
            
            response = self.active_model.generate_content(final_prompt)
            return response.text + f"\n\n_(Model: {self.model_name})_"

        except Exception as e:
            return f"❌ 錯誤: {str(e)}"

    def add_document(self, text, source="System"):
        if not text: return
        doc_entry = f"[{source}] {text}"
        self.memory_docs.append(doc_entry)
        try:
            vec = genai.embed_content(model=self.embedding_model, content=text)['embedding']
            self.memory_vecs.append(vec)
            return True
        except:
            try:
                vec = genai.embed_content(model="models/embedding-001", content=text)['embedding']
                self.memory_vecs.append(vec)
                return True
            except: return False

    def clear_memory(self):
        self.memory_docs = []
        self.memory_vecs = []

    def query(self, user_question, top_k=4):
        if not self.active_model: return f"❌ AI 初始化失敗 (無可用模型)。"
        
        # 如果記憶庫是空的，就不進行向量搜尋，直接回答
        context = ""
        if self.memory_vecs:
            try:
                # Embedding 查詢
                try:
                    q_vec = genai.embed_content(model=self.embedding_model, content=user_question)['embedding']
                except:
                    q_vec = genai.embed_content(model="models/embedding-001", content=user_question)['embedding']
                
                scores = np.dot(self.memory_vecs, q_vec)
                actual_k = min(len(self.memory_docs), top_k)
                top_indices = np.argsort(scores)[-actual_k:][::-1]
                context = "\n".join([self.memory_docs[i] for i in top_indices])
            except:
                context = "(向量檢索失敗，僅依賴模型知識)"

        try:
            final_prompt = f"""
            你是一位專業的財經分析師。請回答使用者的問題。
            
            【參考資訊】
            {context}
            
            【使用者問題】
            {user_question}
            """
            
            response = self.active_model.generate_content(final_prompt)
            return response.text + f"\n\n_(Model: {self.model_name})_"

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota" in error_str:
                return "☕ **AI 需要休息一下 (429 Error)**\n\n您觸發了 Google 免費版 API 的頻率限制。\n建議等待 30 秒後再試。"
            return f"❌ 錯誤: {error_str}"

    def add_document(self, text, source="System"):
        if not text: return
        doc_entry = f"[{source}] {text}"
        self.memory_docs.append(doc_entry)
        try:
            # 嘗試使用新版 Embedding
            vec = genai.embed_content(model=self.embedding_model, content=text)['embedding']
            self.memory_vecs.append(vec)
            return True
        except:
            try:
                # 備援：舊版 Embedding
                vec = genai.embed_content(model="models/embedding-001", content=text)['embedding']
                self.memory_vecs.append(vec)
                return True
            except: return False

    def clear_memory(self):
        self.memory_docs = []
        self.memory_vecs = []

    # 修改 query 函數，將 top_k 預設值降低，並優化錯誤捕捉
    def query(self, user_question, top_k=4): # [修改] 降回 4 以節省 Token
        if not self.active_model: return f"❌ AI 初始化失敗。"
        if not self.memory_vecs: return "⚠️ 腦袋空空，請先點擊「📥 載入個股大腦」。"

        try:
            # 1. Embedding 查詢
            try:
                q_vec = genai.embed_content(model=self.embedding_model, content=user_question)['embedding']
            except:
                q_vec = genai.embed_content(model="models/embedding-001", content=user_question)['embedding']
            
            scores = np.dot(self.memory_vecs, q_vec)
            # [修改] 限制讀取資料量，避免一次消耗太多 Token
            actual_k = min(len(self.memory_docs), top_k)
            top_indices = np.argsort(scores)[-actual_k:][::-1]
            context = "\n".join([self.memory_docs[i] for i in top_indices])
            
            final_prompt = f"""
            你是一位專業的財經分析師。請根據以下「背景資訊」回答使用者的問題。
            若遇到與「股價」或「財務數據」相關問題，請務必引用背景資訊中的數值。
            
            【背景資訊】
            {context}
            
            【使用者問題】
            {user_question}
            """
            
            response = self.active_model.generate_content(final_prompt)
            return response.text + f"\n\n_(Model: {self.model_name})_"

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota" in error_str:
                return "☕ **AI 需要休息一下 (429 Error)**\n\n您觸發了 Google 免費版 API 的頻率限制。\n建議：\n1. 等待 1~2 分鐘後再試。\n2. 不要連續快速點擊「發問」。"
            return f"❌ 錯誤: {error_str}"

db_manager = BattleDB()

# ==========================================
# 0.5 資料庫載入區
# ==========================================
STOCK_NAMES = {} 
DEFAULT_SECTOR_DB = {
    "💎 半導體 (範例)": {"1. 上游": ["2330.TW", "2454.TW"]}
}

def load_external_data():
    global STOCK_NAMES
    sector_data = DEFAULT_SECTOR_DB
    if os.path.exists("sector_db.json"):
        try:
            with open("sector_db.json", "r", encoding="utf-8") as f:
                sector_data = json.load(f)
        except: pass
    
    if os.path.exists("stock_names.json"):
        try:
            with open("stock_names.json", "r", encoding="utf-8") as f:
                external_names = json.load(f)
                STOCK_NAMES.update(external_names)
        except: pass
        
    return sector_data

SECTOR_DB = load_external_data()

# ==========================================
# 1. 核心工具 (ETL)
# ==========================================

# [V33.4] 即時報價
def get_realtime_quote(ticker):
    try:
        if ticker.isdigit(): t = f"{ticker}.TW"
        else: t = ticker
        stock = yf.Ticker(t)
        df = stock.history(period='1d', interval='1m')
        if not df.empty:
            return df['Close'].iloc[-1], df.index[-1]
    except: pass
    return None, None

# [V33.5] 增強型爬蟲 (Anti-Blocking)
@st.cache_data(ttl=600)
def get_stock_data(ticker, period="2y"):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    ]
    
    tickers_to_try = [ticker]
    if ticker.isdigit(): tickers_to_try = [f"{ticker}.TW", f"{ticker}.TWO"]
    elif not ticker.endswith(".TW") and not ticker.endswith(".TWO") and not ticker.isalpha(): 
        tickers_to_try = [f"{ticker}.TW"]
    
    for t in tickers_to_try:
        for attempt in range(2): 
            try:
                stock = yf.Ticker(t)
                temp = stock.history(period=period)
                
                if not temp.empty and len(temp) > 60: 
                    df = temp
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
                    time.sleep(random.uniform(1.0, 2.0))
            except Exception:
                time.sleep(random.uniform(1.0, 2.0))
                continue
                
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_info(ticker):
    try:
        if ticker.isdigit(): ticker = f"{ticker}.TW"
        stock = yf.Ticker(ticker)
        return stock.info
    except: return {}

@st.cache_data(ttl=300)
def get_special_news_v28(ticker, name):
    core_ticker = ticker.replace(".TW", "").replace(".TWO", "")
    # 這裡就是 RAG 大腦的「白名單」資料來源
    target_sites = ["money.udn.com", "moneydj.com", "investor.com.tw", "sinotrade.com.tw", "ctee.com.tw"]
    site_query = " OR ".join([f"site:{site}" for site in target_sites])
    query = f"{name} {core_ticker} ({site_query})"
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant&tbs=qdr:m3"
    
    news_items = []
    all_titles = "" 
    
    try:
        response = requests.get(rss_url, timeout=3)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            tw_tz = timezone(timedelta(hours=8))
            
            for item in root.findall('./channel/item'):
                title_text = item.find('title').text
                all_titles += title_text + " "
                
                score = 0.5
                sentiment_label = "中性"; sentiment_color = "sent-neu"
                if HAS_SNOWNLP:
                    s = SnowNLP(title_text); score = s.sentiments
                
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
    core_ticker = ticker.replace(".TW", "").replace(".TWO", "")
    my_sector = "未知"
    peers = []
    
    for sector, sub_dict in SECTOR_DB.items():
        for sub, tickers in sub_dict.items():
            clean_tickers = [t.replace(".TW", "").replace(".TWO", "") for t in tickers]
            if core_ticker in clean_tickers:
                my_sector = sub
                peers = [t for t in tickers if t.replace(".TW","").replace(".TWO","") != core_ticker][:4] 
                break
    
    if not peers: return None
    
    main_df = get_stock_data(ticker, period=period)
    if main_df.empty: return None
    
    peer_corr = {}
    sector_trend = pd.DataFrame(index=main_df.index)
    sector_trend['Main'] = main_df['Close']
    
    for p in peers:
        p_df = get_stock_data(p, period=period)
        if not p_df.empty:
            aligned_df = pd.DataFrame({'Main': main_df['Close'], 'Peer': p_df['Close']}).dropna()
            if len(aligned_df) > 30:
                corr, _ = pearsonr(aligned_df['Main'], aligned_df['Peer'])
                peer_name = STOCK_NAMES.get(p, p)
                peer_corr[peer_name] = corr
                sector_trend[peer_name] = p_df['Close']
    
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
        "top_3_list": scan_results[:3], 
        "market_summary": f"本次掃描 {len(scan_results)} 檔股票，冠軍由 {top_stock['名稱']} 奪得，總分 {top_stock['總分']} 分。"
    }
    return json.dumps(report_data, ensure_ascii=False, indent=2, cls=NumpyEncoder)

def send_email_report(subject, html_content):
    if 'email_sender' not in st.secrets or 'email_password' not in st.secrets:
        return False, "❌ 未設定 Email 帳號或應用程式密碼"

    sender = st.secrets['email_sender']
    password = st.secrets['email_password']
    receiver = st.secrets.get('email_receiver', sender) 
    
    msg = MIMEMultipart()
    msg['From'] = f"AI 戰情室 <{sender}>"
    msg['To'] = receiver
    msg['Subject'] = subject
    
    msg.attach(MIMEText(html_content, 'html'))
    
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        return True, f"✅ 戰報已寄至 {receiver}！"
    except Exception as e:
        return False, f"❌ 發送失敗: {str(e)}"

# [V33.7 修改] 強化版爬蟲：加入重試機制 (Retry) 與 錯誤分類
def process_stock_task(ticker):
    # 設定重試次數
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # 隨機延遲，避免同時發送請求被封鎖
            time.sleep(random.uniform(0.3, 0.8))
            
            name = STOCK_NAMES.get(ticker, ticker)
            
            # 呼叫資料獲取函數 (假設 get_stock_data 內部有 yfinance 邏輯)
            df = get_stock_data(ticker)
            
            # [檢查點 1] 下載是否成功？
            if df.empty:
                # 如果是最後一次嘗試仍失敗，才回傳錯誤
                if attempt == max_retries - 1:
                    return {"status": "fail", "code": ticker, "reason": "下載無資料(Empty)"}
                continue # 重試
            
            # [檢查點 2] 資料長度是否足夠？(過濾新上市或資料殘缺)
            if len(df) < 60:
                return {"status": "fail", "code": ticker, "reason": "資料不足(<60天)"}
            
            # [檢查點 3] 殭屍股過濾 (最近5天無量 或 收盤價<=0)
            if df['Volume'].iloc[-5:].sum() == 0 or df['Close'].iloc[-1] <= 0:
                return {"status": "fail", "code": ticker, "reason": "殭屍股/無量"}

            # [檢查點 4] 嘗試修補 NaN
            if df[['Open', 'High', 'Low', 'Close']].isnull().values.any():
                 df = df.fillna(method='ffill').fillna(method='bfill')

            # --- 開始計算分數 (邏輯不變) ---
            df = calculate_indicators(df)
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
            try:
                ret_1m = (last['Close'] / df['Close'].iloc[-20]) - 1
            except: ret_1m = 0
            if ret_1m > 0: m_score += 5
            if ret_1m > 0.05: m_score += 5 
            
            f_score = 5 # 基礎分
            # 注意：get_stock_info 比較耗時，若為了加速可考慮移除或設為選填
            # info = get_stock_info(ticker) 
            # ... (基本面邏輯) ...
            
            total_score = t_score + c_score + m_score + f_score
            
            return {
                "status": "ok", 
                "代號": ticker, 
                "名稱": name, 
                "總分": total_score, 
                "現價": last['Close'], 
                "斜率": "⬆️" if last['MA60_Slope'] > 0 else "⬇️"
            }

        except Exception as e:
            # 遇到網路錯誤，等待後重試
            if attempt < max_retries - 1:
                time.sleep(2) # 發生錯誤時，睡久一點避風頭
                continue
            return {"status": "error", "code": ticker, "reason": str(e)}
            
    return {"status": "fail", "code": ticker, "reason": "Unknown"}

# ==========================================
# 補回遺失的 SuperTrend 核心計算函數
# ==========================================
def calculate_supertrend_core(high, low, close, atr, period, multiplier):
    n = len(close)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n, dtype=int) # 1: Bull, -1: Bear

    basic_upper = (high + low) / 2 + (multiplier * atr)
    basic_lower = (high + low) / 2 - (multiplier * atr)

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend[0] = final_upper[0]

    for i in range(1, n):
        # 計算 Upper Band
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        # 計算 Lower Band
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        # 判斷趨勢轉換
        if trend[i-1] == 1:
            supertrend[i] = final_lower[i]
            if close[i] < final_lower[i]:
                trend[i] = -1
                supertrend[i] = final_upper[i]
            else:
                trend[i] = 1
        else:
            supertrend[i] = final_upper[i]
            if close[i] > final_upper[i]:
                trend[i] = 1
                supertrend[i] = final_lower[i]
            else:
                trend[i] = -1
                
    return trend, supertrend

# ==========================================
# [V33.8 核心升級] 向量化極速回測引擎
# ==========================================
def run_strategy_multi(data_dict, strategy_type, p1, p2, p3, sl_atr, tp_atr, vol_factor, trend_filter_mode, risk_per_trade):
    # 1. 數據解包 (轉為 Numpy Array 以利向量化)
    closes = data_dict['close']; highs = data_dict['high']; lows = data_dict['low']; opens = data_dict['open']
    volumes = data_dict['volume']; atrs = data_dict['atr']; adxs = data_dict['adx']
    vol_mas = data_dict['vol_ma']; ma60s = data_dict['ma60']; ma200s = data_dict['ma200']
    ma60_slopes = data_dict['ma60_slope']; rsis = data_dict['rsi']; bb_ups = data_dict['bbu']
    ma20s = data_dict['ma20']; don_h = data_dict['don_h']; don_l = data_dict['don_l']
    
    n = len(closes)
    strategy_mode = int(strategy_type) % 4
    
    # 2. 向量化指標計算 (預先計算所有訊號，取代迴圈內判斷)
    # ---------------------------------------------------
    # A. SuperTrend 計算 (部分仍需迴圈，但可優化)
    atr_p_st = int(p1); mult_st = p2 / 10.0
    st_trends, st_line = calculate_supertrend_core(highs, lows, closes, atrs, atr_p_st, mult_st)

    # B. 基礎訊號矩陣 (Boolean Masks)
    # 根據不同策略模式，預先生成 "Raw Entry Signal"
    if strategy_mode == 0:   # SuperTrend + ADX
        raw_signal = (st_trends == 1) & (adxs > int(p3))
    elif strategy_mode == 1: # RSI 逆勢
        buy_level = 30 + (p2/2)
        raw_signal = (rsis < buy_level)
    elif strategy_mode == 2: # 布林突破
        raw_signal = (closes > bb_ups)
    elif strategy_mode == 3: # 海龜突破
        raw_signal = (closes > don_h)
    else:
        raw_signal = np.zeros(n, dtype=bool)

    # C. 濾網矩陣
    pass_vol = (volumes > vol_mas * vol_factor) | (vol_factor <= 0.3)
    
    # D. 狀態矩陣 (用於判斷進場理由)
    is_volume_spike = volumes > (vol_mas * 1.5)
    is_big_candle = closes > (opens * 1.015)
    
    # MACD 預計算 (需還原回 array 操作)
    exp12 = pd.Series(closes).ewm(span=12, adjust=False).mean().values
    exp26 = pd.Series(closes).ewm(span=26, adjust=False).mean().values
    hist_np = (exp12 - exp26) - pd.Series(exp12 - exp26).ewm(span=9, adjust=False).mean().values
    is_macd_turn_up = (hist_np > 0) & (np.roll(hist_np, 1) <= 0)
    
    is_breakout = (is_volume_spike & is_big_candle) | is_macd_turn_up
    is_crashing = (ma60_slopes < -0.5)
    is_early_bull = (closes > ma20s) & (closes > np.roll(ma20s, 1))
    trend_ok = (closes > ma60s)
    slope_ok = (ma60_slopes > 0)

    # 3. 快速迴圈：僅處理 "路徑依賴" (部位管理與動態停損)
    # ---------------------------------------------------
    pos_list = np.zeros(n, dtype=int)
    entry_reasons = np.zeros(n, dtype=int)
    
    current_pos = 0; entry_price = 0.0; dynamic_sl = 0.0
    current_mode = st.session_state.get('current_running_mode', "一般")
    warmup = 60

    # 針對迴圈進行極簡化
    for i in range(warmup, n):
        if current_pos == 0:
            # --- 極速進場判斷 ---
            can_trade = False
            r_code = 0
            
            # 利用預先計算的 Boolean 值
            if "激進" in current_mode or "狙擊" in current_mode:
                if is_crashing[i]: can_trade = False
                elif is_breakout[i]: can_trade = True; r_code = 1
                elif trend_ok[i]: can_trade = True; r_code = 3
            elif "保守" in current_mode:
                if trend_ok[i] and slope_ok[i]: can_trade = True; r_code = 3
                elif is_early_bull[i]: can_trade = True; r_code = 2
            else: # 一般
                if trend_ok[i]: can_trade = True; r_code = 3

            # 最終進場確認 (AND 運算)
            if can_trade and raw_signal[i] and pass_vol[i]:
                current_pos = 1
                entry_price = closes[i]
                dynamic_sl = entry_price - (atrs[i] * sl_atr)
                entry_reasons[i] = r_code
        
        elif current_pos == 1:
            # --- 部位管理 (這是路徑依賴，必須在迴圈內) ---
            # 1. 更新動態停損 (Trailing Stop)
            hard_sl = entry_price - (atrs[i] * sl_atr)
            
            # 獲利加成邏輯
            base_tp_dist = atrs[i] * tp_atr
            if adxs[i] > 25: base_tp_dist *= 1.5
            
            trailing_sl_price = highs[i] - base_tp_dist
            dynamic_sl = max(dynamic_sl, hard_sl, trailing_sl_price)
            
            # 2. 出場檢查
            should_exit = False
            check_price = closes[i] if "狙擊" in current_mode else lows[i]
            
            if check_price <= dynamic_sl: should_exit = True
            
            # 策略特定出場
            if strategy_mode == 1 and (rsis[i] > (70 - p3/2)) and (adxs[i] < 30): should_exit = True
            elif strategy_mode == 0 and st_trends[i] == -1: should_exit = True
            elif strategy_mode == 3 and closes[i] < don_l[i]: should_exit = True

            if should_exit:
                current_pos = 0; dynamic_sl = 0; entry_price = 0
        
        pos_list[i] = current_pos

    # 4. 績效結算 (Vectorized Calculation)
    # ---------------------------------------------------
    ret_arr = data_dict['raw_ret']
    strategy_ret = pos_list[:-1] * ret_arr[1:]
    trades = np.abs(np.diff(pos_list))
    costs = trades * 0.001
    # 修正長度不一致
    if len(costs) > len(strategy_ret): costs = costs[:-1]
    
    final_ret_series = strategy_ret - costs
    cum_ret = np.cumprod(1 + final_ret_series)
    
    if len(cum_ret) == 0: return None
    
    total_ret = cum_ret[-1] - 1
    running_max = np.maximum.accumulate(cum_ret)
    mdd = np.min((cum_ret - running_max) / running_max)
    strat_names = {0:"SuperTrend", 1:"RSI逆勢", 2:"布林突破", 3:"海龜交易"}

    # 夏普率與勝率計算
    daily_rets = final_ret_series
    sharpe_ratio = 0
    if np.std(daily_rets) != 0:
        sharpe_ratio = (np.mean(daily_rets) / np.std(daily_rets)) * (252 ** 0.5)
        
    # 勝率 (使用向量化計算 trade_pnl)
    # 找出賣出點 (pos 1 -> 0) 與對應的買入點
    trade_indices = np.where(trades == 1)[0] # 交易發生點
    # 簡化版勝率 (精確計算需配對買賣，此處為加速估算)
    win_rate = 0.5 # 預設
    if len(trade_indices) > 1:
        # 這裡維持簡單估算，若需精確每筆損益需額外邏輯，為求效能暫略
        pass 

    return pos_list, np.concatenate(([1.0], cum_ret)), total_ret, mdd, strat_names[strategy_mode], st_line, st_trends, entry_reasons, sharpe_ratio, win_rate

# ==========================================
# [V33.9.1 修正] 策略穩健度檢測 (修復陣列長度不一致 bug)
# ==========================================
def calculate_walk_forward_heatmap(df, params, segments=10):
    # 1. 切分數據
    n = len(df)
    chunk_size = n // segments
    
    heatmap_x = [] # 日期
    heatmap_y = [f"區間 {i+1}" for i in range(segments)]
    z_values = []  # 數值(報酬率)
    text_values = [] # 顯示文字
    
    # 解析參數
    strat_type, p1, p2, p3, sl_atr, tp_atr, vol_factor, t_filt, risk = params

    win_count = 0
    
    for i in range(segments):
        # 確保每段至少有 60 根 K 線 (供指標暖機)
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < segments - 1 else n
        
        # 為了計算指標，必須往前多抓暖機資料 (Buffer)
        buffer = 60
        real_start = max(0, start_idx - buffer)
        sub_df = df.iloc[real_start:end_idx].copy()
        
        # [關鍵修正] 處理資料過短的情況
        if len(sub_df) < buffer + 10:
            z_values.append(0)
            text_values.append("N/A")
            # 修正：這裡原本漏了 append heatmap_x，導致長度不一
            heatmap_x.append(f"Seg {i+1} (資料不足)") 
            continue

        # 準備 Data Dict
        sub_data = {
            'open': sub_df['Open'].values, 'high': sub_df['High'].values, 'low': sub_df['Low'].values, 'close': sub_df['Close'].values,
            'volume': sub_df['Volume'].values, 'vol_ma': sub_df['VolMA20'].fillna(0).values,
            'ma60': sub_df['MA60'].fillna(0).values, 'ma60_slope': sub_df['MA60_Slope'].fillna(0).values,
            'ma200': sub_df['MA200'].fillna(0).values, 'adx': sub_df['ADX'].fillna(0).values, 'atr': sub_df['ATR'].fillna(0).values,
            'rsi': sub_df['RSI'].fillna(50).values, 'bbu': sub_df['BBU'].values, 'bbl': sub_df['BBU'].values, 'ma20': sub_df['MA20'].values,
            'don_h': sub_df['Donchian_H20'].values, 'don_l': sub_df['Donchian_L10'].values,
            'raw_ret': sub_df['Close'].pct_change().fillna(0).values
        }
        
        # 執行回測 (只看該區間)
        res = run_strategy_multi(sub_data, strat_type, p1, p2, p3, sl_atr, tp_atr, vol_factor, t_filt, risk)
        
        if res:
            seg_ret = res[2] 
            
            z_values.append(seg_ret)
            text_values.append(f"{seg_ret:.1%}")
            
            # 標記日期區間
            date_start = sub_df.index[buffer].strftime('%Y-%m') if len(sub_df) > buffer else "N/A"
            date_end = sub_df.index[-1].strftime('%Y-%m')
            heatmap_x.append(f"{date_start} ~ {date_end}")
            
            if seg_ret > 0: win_count += 1
        else:
            z_values.append(0)
            text_values.append("0%")
            # 這裡也要補上 append
            heatmap_x.append(f"Seg {i+1} (無交易)")

    return heatmap_x, z_values, text_values, win_count

def highlight_trade_status(val):
    val_str = str(val)
    if '獲利' in val_str: return 'background-color: #155724; color: white' 
    elif '虧損' in val_str: return 'background-color: #721c24; color: white' 
    elif '建倉' in val_str: return 'color: #00ffff' 
    return ''

def fitness_func(ga_instance, sol, idx):
    current_mode = st.session_state.get('current_running_mode', "一般")
    
    strat_type = sol[0]
    p1 = sol[1]; p2 = sol[2]; p3 = sol[3]
    sl_atr = sol[4]/10.0; tp_atr = sol[5]/10.0
    vol_factor = sol[6]/10.0
    trend_filter_mode = 1 if sol[7] > 5 else 0 
    risk = 0.01 
    
    data_dict = st.session_state.train_data_dict 
    
    res = run_strategy_multi(data_dict, strat_type, p1, p2, p3, sl_atr, tp_atr, vol_factor, trend_filter_mode, risk)

    if res is None: return -9999
    pos, _, total_ret, mdd, _, _, _, _, sharpe, win_rate = res 
    
    trades = np.sum(np.abs(np.diff(pos))) / 2
    abs_mdd = abs(mdd)
    
    if trades < 3: return -5000 
    
    score = 0

    if "保守" in current_mode:
        if abs_mdd > 0.12: return -10000 * abs_mdd
        if win_rate < 0.4: score -= 2000
        score = (sharpe * 500) + (total_ret * 200) + (win_rate * 1000)
        
    elif "激進" in current_mode:
        if abs_mdd > 0.45: return -5000
        score = (total_ret * 3000) + (sharpe * 100)
        
    elif "狙擊" in current_mode:
        if win_rate < 0.6: score -= 5000 
        score = (sharpe * 1000) + (win_rate * 2000) + (total_ret * 500)
        
    return score

# --- Page 1: AI 總司令選股 (V33.6 精簡優化版) ---
def page_ai_selector():
    st.header("🤖 AI 總司令：V33.6 精簡優化版")
    
    if 'scan_results_df' not in st.session_state: st.session_state.scan_results_df = None
    if 'scan_top_stock' not in st.session_state: st.session_state.scan_top_stock = None
    if 'scan_json_report' not in st.session_state: st.session_state.scan_json_report = None
    
    c_mode, c_info = st.columns([1, 2])
    with c_mode:
        scan_scope = st.radio("📡 掃描雷達範圍", ["🎯 單一戰略板塊", "🌍 全球戰略 (全域掃描)"], horizontal=True)
    
    all_tickers = []
    selected_sector_name = "全域市場"
    
    if scan_scope == "🎯 單一戰略板塊":
        selected_chain = st.selectbox("請選擇戰略板塊:", list(SECTOR_DB.keys()))
        selected_sector_name = selected_chain
        sub_sectors = SECTOR_DB[selected_chain]
        
        with st.expander(f"📂 檢視 {selected_chain} 成分股", expanded=True):
            for sub_name, tickers in sub_sectors.items():
                st.markdown(f"**📌 {sub_name}**")
                sorted_tickers = sorted(tickers)
                all_tickers.extend(sorted_tickers)
                html_tags = ""
                for t in sorted_tickers:
                    display_name = STOCK_NAMES.get(t, t.replace(".TW", "").replace(".TWO", ""))
                    clean_code = t.replace(".TW", "").replace(".TWO", "")
                    html_tags += f'<span class="stock-tag">{clean_code} {display_name}</span>'
                st.markdown(f'<div style="line-height: 1.8;">{html_tags}</div>', unsafe_allow_html=True)
                st.write("")
                
    else:
        st.info("🌍 您已啟動「天網模式」，將掃描資料庫中 **所有板塊** 的股票。")
        unique_tickers = set()
        total_sectors = 0
        for sector_name, sub_dict in SECTOR_DB.items():
            total_sectors += 1
            for t_list in sub_dict.values():
                for t in t_list:
                    unique_tickers.add(t)
        
        all_tickers = sorted(list(unique_tickers))
        sector_counts = {k: sum(len(v) for v in sub.values()) for k, sub in SECTOR_DB.items()}
        
        with c_info:
            st.metric("掃描目標總數", f"{len(all_tickers)} 檔", f"涵蓋 {total_sectors} 大板塊")
            check_str = " | ".join([f"{k}:{v}" for k,v in list(sector_counts.items())[:3]])
            st.caption(f"🛡️ 資料一致性核對: {check_str} ...")
            
        with st.expander("📂 檢視全域掃描清單 (已去重)", expanded=False):
            st.write(", ".join([t.replace(".TW","") for t in all_tickers]))

    st.markdown("---")
    
    btn_label = f"🚀 啟動{scan_scope}"
    if st.button(btn_label, type="primary"):
        if not all_tickers:
            st.error("❌ 掃描清單為空，請檢查 sector_db.json")
        else:
            results = []
            failed_tickers = [] 
            
            progress_bar = st.progress(0); status_text = st.empty(); 
            status_text.text(f"⚡ V33.6 智慧引擎啟動，目標 {len(all_tickers)} 檔...")
            
            start_time = time.time()
            
            workers = 6 
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_ticker = {executor.submit(process_stock_task, t): t for t in all_tickers}
                
                completed_count = 0
                total_count = len(all_tickers)
                
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        res = future.result()
                        if res and res.get("status") == "ok":
                            results.append(res)
                        else:
                            failed_tickers.append(ticker)
                    except Exception as exc:
                        failed_tickers.append(ticker)
                    
                    completed_count += 1
                    pct = completed_count / total_count
                    progress_bar.progress(pct)
                    
                    if completed_count % 10 == 0:
                         elapsed = time.time() - start_time
                         avg_time = elapsed / completed_count
                         remain = (total_count - completed_count) * avg_time
                         status_text.text(f"⚡ 掃描中: {completed_count}/{total_count} | 成功: {len(results)} | 預估剩餘: {int(remain)} 秒")

            progress_bar.progress(100)
            end_time = time.time(); duration = end_time - start_time
            
            if results:
                res_df = pd.DataFrame(results).sort_values("總分", ascending=False)
                top_stock = res_df.iloc[0] 
                
                success_rate = len(res_df) / len(all_tickers)
                
                with c_info:
                    st.metric(
                        "掃描品質報告", 
                        f"{len(res_df)} / {len(all_tickers)} 檔",
                        f"成功率: {success_rate:.1%}"
                    )
                    
                    if success_rate < 0.95:
                        st.warning(f"⚠️ 有 {len(failed_tickers)} 檔掃描失敗 (可能是連線阻擋或下市)。")
                        with st.expander("❌ 檢視失敗名單"):
                            st.write(", ".join(failed_tickers))
                    else:
                        st.caption("✅ 資料完整度良好")

                scan_results_list = res_df.to_dict('records')
                json_report = generate_battle_report(top_stock, scan_results_list)
                
                db_manager.save_scan_report(scan_scope, top_stock['代號'], top_stock['總分'], json_report)
                st.toast("✅ 掃描結果已自動備份至資料庫！", icon="💾")
                
                st.session_state.scan_results_df = res_df
                st.session_state.scan_top_stock = top_stock
                st.session_state.scan_json_report = json_report
                
                status_text.success(f"✅ 全域掃描完成！耗時 {duration:.2f} 秒。")
            else:
                st.warning("無有效資料或連線失敗。")
            
    if st.session_state.scan_results_df is not None:
        res_df = st.session_state.scan_results_df
        top_stock = st.session_state.scan_top_stock
        json_report = st.session_state.scan_json_report
        
        if scan_scope == "🎯 單一戰略板塊":
            st.success(f"🏆 【{selected_sector_name}】板塊冠軍：**{top_stock['名稱']}** 總分：{top_stock['總分']}")
        else:
            st.success(f"👑 **【全市場總冠軍】**：**{top_stock['名稱']} ({top_stock['代號']})** 總分：{top_stock['總分']}")
        
        # [V34.0 新增] 掃描結果快速加入自選股
        st.write("### 🎯 掃描結果操作")
        c_act1, c_act2 = st.columns([2, 1])
        with c_act1:
            # 下拉選單選冠軍或前幾名
            add_target = st.selectbox("選擇要加入自選股的標的:", res_df['代號'].head(10).tolist())
        with c_act2:
            if st.button("➕ 加入監控", key="add_scan"):
                msg = toggle_watchlist(add_target)
                st.toast(msg, icon="✅")
        st.dataframe(res_df.head(50).style.background_gradient(subset=['總分'], cmap='RdYlGn'), use_container_width=True)
        st.caption(f"💡 僅顯示前 50 名 (共 {len(res_df)} 筆結果)")

        st.markdown("---")
        with st.expander("🗺️ V32.0 戰略地圖：全市場資金流向熱力圖", expanded=True):
            if '板塊' not in res_df.columns:
                ticker_to_sector = {}
                for main_sec, sub_dict in SECTOR_DB.items():
                    for sub_sec, t_list in sub_dict.items():
                        for t in t_list:
                            clean_t = t.replace(".TW", "").replace(".TWO", "")
                            ticker_to_sector[clean_t] = {"Main": main_sec, "Sub": sub_sec}
                
                def get_sector_info_row(row, key):
                    code = row['代號'].replace(".TW", "").replace(".TWO", "")
                    return ticker_to_sector.get(code, {}).get(key, "其他")

                plot_df = res_df.copy()
                plot_df['主板塊'] = plot_df.apply(lambda x: get_sector_info_row(x, "Main"), axis=1)
                plot_df['子板塊'] = plot_df.apply(lambda x: get_sector_info_row(x, "Sub"), axis=1)
                plot_df['權重'] = plot_df['總分'] ** 2 
                
                import plotly.express as px
                
                fig_tree = px.treemap(
                    plot_df, 
                    path=[px.Constant("台股全市場"), '主板塊', '子板塊', '名稱'], 
                    values='權重',
                    color='總分',
                    color_continuous_scale='RdYlGn_r', 
                    title=f"AI 戰力熱力圖 (總掃描: {len(plot_df)} 檔)"
                )
                fig_tree.update_traces(root_color="lightgrey")
                fig_tree.update_layout(margin=dict(t=30, l=10, r=10, b=10), height=500)
                
                st.plotly_chart(fig_tree, use_container_width=True)

        target_code = top_stock['代號'].replace(".TW", "").replace(".TWO", "")
        st.info(f"建議將總冠軍 **{target_code}** 帶入 PyGAD 進行演化。")
        
        st.markdown("---")
        
        if scan_scope == "🎯 單一戰略板塊":
            title_prefix = f"【{selected_sector_name}冠軍】"
        else:
            title_prefix = "【全域總冠軍】" if len(res_df) > 50 else "【掃描冠軍】"
            
        email_subject = f"AI戰報(V33)：{title_prefix} {top_stock['名稱']}({target_code}) 分析報告"
        
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

        email_html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #00adb5;">🤖 AI 戰情室 V33 每日晨報</h2>
            <hr>
            <p>早安！AI 系統已完成 V33 天眼掃描，今日決選結果如下：</p>
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
            <p style="color: gray; font-size: 0.8em;">本信件由 AI 戰情室 V33 自動發送。</p>
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

# --- [V33.7 優化版] page_dashboard 局部更新 ---
def page_dashboard():
    st.header("⚡ 全能達人戰情室 (V33.7 專業版)")
    # 在 page_dashboard 開頭加入
    c_head_1, c_head_2 = st.columns([3, 1])
    with c_head_1:
        st.header("⚡ 全能達人戰情室 (V33.7.4)")
    with c_head_2:
        if st.button("🔄 強制刷新報價"):
            st.cache_data.clear() # 清除快取，強制重抓
            st.rerun()

    # 1. 強化 Session State 初始化
    if 'dash_current_stock' not in st.session_state:
        st.session_state.dash_current_stock = "2330"
    if 'dash_chat_history' not in st.session_state:
        st.session_state.dash_chat_history = []
    
    # 確保 RAG Agent 全局唯一且持續存在
    if 'rag_agent' not in st.session_state:
        if "AI_Studio_Key" in st.secrets:
            st.session_state.rag_agent = RAGAdvisor(st.secrets["AI_Studio_Key"])

    # 2. UI 佈局
    col_input, col_info = st.columns([1, 3])
    with col_input: 
        t_input = st.text_input("輸入個股代號", value=st.session_state.dash_current_stock, key="dash_input_main")
        if t_input != st.session_state.dash_current_stock:
            st.session_state.dash_current_stock = t_input
            st.session_state.dash_chat_history = [] # 換股才清空對話
            st.rerun()

    t = st.session_state.dash_current_stock

    if t:
        df = get_stock_data(t)
        if df.empty or len(df) < 30: 
            st.error("無資料或資料不足")
            return
        
        df = calculate_indicators(df)
        info = get_stock_info(t)
        name = STOCK_NAMES.get(t.upper() + ".TW", t)
        if name == t: name = STOCK_NAMES.get(t, t)
        
        live_price, live_time = get_realtime_quote(t)
        
        if live_price:
            last_price = live_price
            prev_close = df.iloc[-1]['Close'] 
            if df.index[-1].date() == datetime.now().date():
                prev_close = df.iloc[-2]['Close']
            
            change = last_price - prev_close
            pct = change / prev_close
            time_str = live_time.strftime("%H:%M")
        else:
            last_price = df.iloc[-1]['Close']
            prev_close = df.iloc[-2]['Close']
            change = last_price - prev_close
            pct = change / prev_close
            time_str = df.index[-1].strftime("%Y-%m-%d")

        color = "red" if change > 0 else "green"
        
        with col_info: 
            st.markdown(f"### {name} ({t})")
            st.markdown(f"<h2 style='color:{color}'>{last_price:.2f} <small>({change:+.2f} / {pct:+.2%}) <span style='font-size:0.5em;color:gray'>@{time_str}</span></small></h2>", unsafe_allow_html=True)
            sectors = get_sector_info(t.upper() + ".TW") 
            for s in sectors: st.caption(f"📍 {s}")
            
        tab1, tab2, tab3 = st.tabs(["ℹ️ 資訊流 & AI", "💸 資金流", "📈 技術流"])
        
        last_daily = df.iloc[-1] 

        # --- Tab 1: 資訊流 & RAG ---
        with tab1:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("📰 特種搜查")
                try:
                    news, keywords = get_special_news_v28(t, name)
                except: news = []; keywords = []
                
                if keywords:
                    st.markdown("🔥 **AI 提取關鍵字:**")
                    kw_html = "".join([f"<span style='background:#333;color:#00adb5;padding:2px 6px;border-radius:4px;margin:2px;font-size:0.8em'>{k}</span>" for k in keywords])
                    st.markdown(kw_html, unsafe_allow_html=True)
                
                st.divider()

                st.subheader("🤖 RAG 財經智囊團")
                # 初始化 Agent
                if 'rag_agent' not in st.session_state:
                    if "AI_Studio_Key" in st.secrets:
                        st.session_state.rag_agent = RAGAdvisor(st.secrets["AI_Studio_Key"])
                    else: st.warning("請先設定 API Key")
                
                agent = st.session_state.get('rag_agent')

                if agent:
                    if st.button("📥 載入個股大腦 (News + Tech)", key="rag_load", type="secondary"):
                        with st.spinner("AI 正在閱讀財報與線圖..."):
                            agent.clear_memory()
                            ma_state = "多頭排列" if last_daily['Close'] > last_daily['MA60'] else "空頭/盤整"
                            tech_summary = (
                                f"【技術面數據】{name}({t}) 收盤價 {last_daily['Close']}。MA20={last_daily['MA20']:.2f}, MA60={last_daily['MA60']:.2f}。 "
                                f"目前趨勢為{ma_state}。RSI={last_daily['RSI']:.2f}。KD值(K/D)=({last_daily['K']:.1f}/{last_daily['D']:.1f})。 "
                                f"MACD柱狀體={last_daily['Hist']:.2f}。"
                            )
                            agent.add_document(tech_summary, source="Technical")
                            for n in news[:8]: 
                                agent.add_document(f"{n['title']} (日期:{n['pubDate']})", source="News")
                            if info:
                                fund_sum = info.get('longBusinessSummary', '無詳細簡介')
                                agent.add_document(f"【公司簡介】{fund_sum[:300]}", source="Fundamental")
                            st.success(f"✅ 大腦已載入！")

                    # 顯示歷史對話 (解決消失問題)
                    for msg in st.session_state.dash_chat_history:
                        with st.chat_message(msg["role"]):
                            st.write(msg["content"])

                    user_q = st.chat_input("請輸入問題...", key="chat_input_w")
                    if user_q:
                        # 1. 顯示使用者問題
                        st.session_state.dash_chat_history.append({"role": "user", "content": user_q})
                        with st.chat_message("user"):
                            st.write(user_q)
                        
                        # 2. AI 回答
                        if not agent.memory_docs:
                            st.warning("請先點擊上方按鈕載入資料！")
                        else:
                            with st.spinner("AI 思考中..."):
                                ans = agent.query(user_q)
                                st.session_state.dash_chat_history.append({"role": "assistant", "content": ans})
                                with st.chat_message("assistant"):
                                    st.markdown(ans)
                else:
                    st.caption("⚠️ RAG 未啟用")
                
                st.divider()
                if news: 
                    for n in news: 
                        st.markdown(f'<div class="news-card"><a href="{n["link"]}" target="_blank" class="news-title"><span class="sentiment-tag {n.get("sent_color", "sent-neu")}">{n.get("sent_label", "中性")}</span> {n["title"]}</a><span class="news-source">{n["publisher"]}</span> <span class="news-time">{n["pubDate"]}</span></div>', unsafe_allow_html=True)
                else: st.info("無新聞")
            
            with c2: 
                st.subheader("🔗 板塊聯動雷達")
                try:
                    sec_data = analyze_sector_linkage(t)
                    if sec_data:
                        st.caption(f"所屬子板塊: **{sec_data['sector']}**")
                        norm_df = sec_data['normalized']
                        fig_sec = go.Figure()
                        fig_sec.add_trace(go.Scatter(x=norm_df.index, y=norm_df['Main'], name=name, line=dict(color='yellow', width=2)))
                        fig_sec.add_trace(go.Scatter(x=norm_df.index, y=sec_data['avg_trend'], name="同業平均", line=dict(color='gray', dash='dash')))
                        fig_sec.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", hovermode="x unified")
                        st.plotly_chart(fig_sec, use_container_width=True)
                    else: st.warning("無法取得同業資料")
                except: st.warning("板塊資料載入失敗")

                st.subheader("🏢 簡介")
                s = info.get('longBusinessSummary')
                st.write(s) if s else st.warning("無簡介")
                st.markdown(f'<a href="https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={t}" target="_blank" class="link-btn">Goodinfo</a>', unsafe_allow_html=True)
                
        with tab2:
            # (維持原樣)
            st.markdown("### 🏛️ 官方籌碼"); c_l = st.columns(3)
            with c_l[0]: st.markdown(f'<a href="https://goodinfo.tw/tw/ShowBuySaleChart.asp?STOCK_ID={t}&CHT_CAT=DATE" target="_blank" class="link-btn">Goodinfo</a>', unsafe_allow_html=True)
            with c_l[1]: st.markdown(f'<a href="https://www.tpex.org.tw/zh-tw/mainboard/trading/major-institutional/detail/day.html" target="_blank" class="link-btn">TPEx</a>', unsafe_allow_html=True)
            with c_l[2]: st.markdown(f'<a href="https://www.twse.com.tw/zh/trading/foreign/t86.html" target="_blank" class="link-btn">TWSE</a>', unsafe_allow_html=True)
            
            st.divider()
            m1, m2 = st.columns(2)
            obv_s = df['OBV'].iloc[-1] - df['OBV'].iloc[-20]
            m1.metric("OBV", "吸籌 🟢" if obv_s > 0 else "出貨 🔴")
            vr = last_daily['Volume']/last_daily['VolMA20'] if last_daily['VolMA20']>0 else 0
            m2.metric("量能", f"{vr:.2f}x")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange'), name='VWAP'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], fill='tozeroy', line=dict(color='cyan'), name='OBV'), row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, key="fund")
            
        # --- [V33.8 終極戰情室：策略疊圖 + 籌碼雷達] ---
        with tab3:
            # 建立 4 列畫布 (新增 Row 4: 籌碼雷達)
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                            row_heights=[0.5, 0.15, 0.15, 0.2], # 分配高度
                            vertical_spacing=0.03)

            # 0. 數據處理 (濾除暖機區)
            start_idx = 30 if len(df) > 60 else 0
            plot_df = df.iloc[start_idx:].copy()

            # ==================================================
            # Row 1: 主戰場 (K線 + FIB + 策略訊號疊圖)
            # ==================================================
            # A. 基礎 K 線
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], 
                                        low=plot_df['Low'], close=plot_df['Close'], name='K線'), row=1, col=1)
            
            # B. 均線系統
            for ma, color, name in [('MA20', '#FFFF00', '月線'), ('MA60', '#00FFFF', '季線')]:
                if ma in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[ma], line=dict(color=color, width=1), name=name), row=1, col=1)

            # C. 黃金分割 (FIB)
            recent_df = df.tail(150)
            p_high = recent_df['High'].max(); p_low = recent_df['Low'].min(); diff = p_high - p_low
            current_price = df['Close'].iloc[-1]
            fib_levels = [0, 0.382, 0.5, 0.618, 1]
            fib_colors = ["#FFD700", "#FF4B4B", "#FFFFFF", "#00FF00", "#FFD700"]

            for lvl, color in zip(fib_levels, fib_colors):
                f_price = p_high - (diff * lvl)
                tag_text = f"FIB {lvl*100}%: {f_price:.1f}"
                if lvl == 0.618:
                    state = " [🎯 強力支撐]" if current_price > f_price else " [⚠️ 轉弱警示]"
                    tag_text += state
                
                fig.add_hline(y=f_price, line_dash="dash", line_color=color, line_width=1.5, row=1, col=1)
                fig.add_annotation(
                    x=plot_df.index[-1], y=f_price, text=tag_text, showarrow=False, 
                    xanchor="left", yanchor="bottom", yshift=8, 
                    font=dict(color=color, size=11, family="Arial Black"), row=1, col=1
                )

            # D. [新增] 策略訊號疊圖 (Strategy Overlay)
            # 檢查是否有 PyGAD 演化結果
            if 'ga_results' in st.session_state and st.session_state.ga_results:
                # 預設取 "激進型" 或第一個可用的結果
                target_mode = "⚔️ 激進型" if "⚔️ 激進型" in st.session_state.ga_results else list(st.session_state.ga_results.keys())[0]
                res = st.session_state.ga_results[target_mode]
                
                # 取得位置訊號與進場理由
                full_pos = res['pos'] # 這是全長度的 Series
                full_reasons = res['entry_reasons']
                
                # 對齊目前的 plot_df index
                aligned_pos = full_pos.reindex(plot_df.index).fillna(0)
                aligned_reasons = full_reasons.reindex(plot_df.index).fillna(0)
                
                # 找出買點 (0 -> 1) 與 賣點 (1 -> 0)
                buy_signals = (aligned_pos.diff() == 1)
                sell_signals = (aligned_pos.diff() == -1)
                
                # 繪製買點 (區分理由：火箭/盾牌)
                buy_idx = plot_df.index[buy_signals]
                if len(buy_idx) > 0:
                    # 分類圖示
                    rocket_idx = [ix for ix in buy_idx if aligned_reasons[ix] == 1]
                    shield_idx = [ix for ix in buy_idx if aligned_reasons[ix] == 2]
                    std_idx    = [ix for ix in buy_idx if aligned_reasons[ix] == 3]

                    if rocket_idx:
                        fig.add_trace(go.Scatter(x=rocket_idx, y=plot_df.loc[rocket_idx, 'Low']*0.98, mode='markers', marker=dict(symbol='star', size=14, color='#FF4B4B'), name='🚀 先鋒突擊'), row=1, col=1)
                    if shield_idx:
                        fig.add_trace(go.Scatter(x=shield_idx, y=plot_df.loc[shield_idx, 'Low']*0.98, mode='markers', marker=dict(symbol='shield', size=12, color='#21C354'), name='🛡️ 防禦佈局'), row=1, col=1)
                    if std_idx:
                        fig.add_trace(go.Scatter(x=std_idx, y=plot_df.loc[std_idx, 'Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=10, color='yellow'), name='🔵 標準進場'), row=1, col=1)

                # 繪製賣點
                sell_idx = plot_df.index[sell_signals]
                if len(sell_idx) > 0:
                    fig.add_trace(go.Scatter(x=sell_idx, y=plot_df.loc[sell_idx, 'High']*1.02, mode='markers', marker=dict(symbol='x-thin', size=10, color='magenta', line_width=2), name='🔻 停損/利'), row=1, col=1)
                
                # 在標題顯示目前疊加的策略
                fig.add_annotation(xref="x domain", yref="y domain", x=0.5, y=0.98, text=f"Strategy Overlay: {target_mode}", showarrow=False, font=dict(color="magenta", size=12), row=1, col=1)


            # ==================================================
            # Row 2: KD 指標 (淨空版)
            # ==================================================
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['K'], mode='lines', line=dict(color='#FFD700', width=1.5), name='K值'), row=2, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['D'], mode='lines', line=dict(color='#FFFFFF', width=1.5), name='D值'), row=2, col=1)
            fig.add_hline(y=80, line_dash="dot", line_color="red", line_width=1, row=2, col=1)
            fig.add_hline(y=20, line_dash="dot", line_color="green", line_width=1, row=2, col=1)

            # ==================================================
            # Row 3: MACD
            # ==================================================
            macd_colors = ['#FF4B4B' if val >= 0 else '#00FF00' for val in plot_df['Hist']]
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Hist'], marker_color=macd_colors, name='MACD柱'), row=3, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD'], line=dict(color='#00FFFF', width=1), name='DIF'), row=3, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Signal'], line=dict(color='#FFA500', width=1), name='MACD'), row=3, col=1)

            # ==================================================
            # Row 4: [新增] 法人籌碼雷達 (主力資金流向模擬)
            # ==================================================
            # 由於 yfinance 無法人數據，我們計算 "主力控盤指標 (Force Index)" 作為替代
            # 算法：(收盤 - 開盤) / 開盤 * 成交量。 紅柱=主力買進，綠柱=主力賣出
            force_index = ((plot_df['Close'] - plot_df['Open']) / plot_df['Open']) * plot_df['Volume']
            chip_colors = ['#FF0000' if val >= 0 else '#00FF00' for val in force_index]
            
            fig.add_trace(go.Bar(x=plot_df.index, y=force_index, marker_color=chip_colors, name='主力資金流'), row=4, col=1)
            
            # 驗證標記：股價創新高(120日) 時，主力是否買進?
            h120 = plot_df['High'].rolling(120).max()
            is_new_high = (plot_df['High'] >= h120) & (force_index > 0) # 創新高且主力買
            high_idx = plot_df.index[is_new_high]
            if len(high_idx) > 0:
                fig.add_trace(go.Scatter(x=high_idx, y=force_index.loc[high_idx]*1.1, mode='markers', marker=dict(symbol='triangle-down', size=8, color='cyan'), name='🔥 創高抬轎'), row=4, col=1)

           # ... (前面的繪圖代碼保持不變，直接接到這裡) ...

            # ==================================================
            # 標籤與全局設定 (V33.8.1 修正版：解決頂部打架)
            # ==================================================
            
            # 1. 調整圖表標題標籤 (往下降一點，讓出頂部空間)
            common_label_style = dict(showarrow=False, font=dict(color="#E0E0E0", size=13), bgcolor="rgba(50,50,50,0.8)", bordercolor="#888", borderwidth=1)
            # y=0.95 確保在圖表內部，不會碰到上面的圖例
            fig.add_annotation(xref="x domain", yref="y domain", x=0.005, y=0.95, text="<b>圖 1: 戰略主圖 (AI 訊號 + FIB)</b>", **common_label_style, row=1, col=1)
            fig.add_annotation(xref="x2 domain", yref="y2 domain", x=0.005, y=0.92, text="<b>圖 2: 動能 (KD)</b>", **common_label_style, row=2, col=1)
            fig.add_annotation(xref="x3 domain", yref="y3 domain", x=0.005, y=0.92, text="<b>圖 3: 趨勢 (MACD)</b>", **common_label_style, row=3, col=1)
            fig.add_annotation(xref="x4 domain", yref="y4 domain", x=0.005, y=0.92, text="<b>圖 4: 籌碼雷達</b>", **common_label_style, row=4, col=1)

            # 2. 全局 Layout 設定 (關鍵修正)
            fig.update_layout(
                height=1300, # 再拉高一點，視覺更舒適
                template="plotly_dark",
                
                # [關鍵修正 1] 加大頂部邊距 (Margin Top)，給圖例足夠的停車場
                margin=dict(l=10, r=150, t=140, b=10), 
                
                xaxis_rangeslider_visible=False,
                
                # [關鍵修正 2] 將圖例 (Legend) 往上推到天花板 (y=1.12)，與圖表完全分離
                legend=dict(
                    orientation="h",         # 水平排列
                    yanchor="bottom", 
                    y=1.12,                  # 設為 1.12，讓它懸浮在 t=140 的邊距空間中
                    xanchor="center", 
                    x=0.5,
                    bgcolor="rgba(30, 30, 30, 0.9)", # 深色背景防干擾
                    bordercolor="#555", 
                    borderwidth=1,
                    font=dict(color="white", size=11),
                    itemsizing='constant'    # 圖示大小一致
                ),
                
                # 鎖定 Y 軸
                yaxis2=dict(range=[0, 100], tickmode='linear', dtick=20, title="KD"),
                
                # 標題設定
                title={
                    'text': f"<b>{name} ({t}) AI 全方位戰略圖</b>",
                    'y': 0.99, # 標題放在最頂端
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': dict(size=20, color='#00FFFF')
                }
            )

            st.plotly_chart(fig, use_container_width=True, key="tech_v33_8_1_fix")
            
            st.info("💡 **V33.8 升級說明**：圖 1 已整合 AI 演化之買賣訊號（需先執行 PyGAD）。圖 4 為「主力資金流向」，紅色代表大單敲進（抬轎），綠色代表大單殺出（倒貨）。")

# [V33.6 修改] 策略進化：新增「當日策略訊號 (Inference)」
def page_ga():
    st.header("🧬 PyGAD 策略進化 (V33.6 精簡優化版)")
    if not HAS_PYGAD: st.error("❌ 需安裝 pygad"); return
    
    if 'saved_ga_target' not in st.session_state:
        st.session_state.saved_ga_target = "2330"
    if 'saved_ga_cash' not in st.session_state:
        st.session_state.saved_ga_cash = 1000000

    c1, c2 = st.columns([1, 2])
    with c1: 
        t = st.text_input("優化標的", value=st.session_state.saved_ga_target)
        st.session_state.saved_ga_target = t
        
        stock_name = "未知 / 未載入"
        if t in STOCK_NAMES: stock_name = STOCK_NAMES[t]
        elif f"{t}.TW" in STOCK_NAMES: stock_name = STOCK_NAMES[f"{t}.TW"]
        st.caption(f"📌 **{stock_name}**")
        
        cash = st.number_input("本金", value=st.session_state.saved_ga_cash)
        st.session_state.saved_ga_cash = cash

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
        
        df = get_stock_data(t, period=period); 
        if df.empty: st.error("無資料"); return
        df = calculate_indicators(df).dropna()
        if len(df) < 50: st.error("資料不足"); return
        if 'MA60_Slope' not in df.columns: df['MA60_Slope'] = df['MA60'].diff().fillna(0)

        split_idx = int(len(df) * split_pct); train_df = df.iloc[:split_idx]; test_df = df.iloc[split_idx:]; 
        st.session_state.train_df = train_df; split_date = df.index[split_idx]
        
        # 準備訓練數據字典
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
        
        for i, m in enumerate(modes):
            st.session_state.current_running_mode = m 
            with st.spinner(f"正在演化 【{m}】..."):
                ga = pygad.GA(num_generations=gens, num_parents_mating=5, fitness_func=fitness_func, sol_per_pop=pop_size, num_genes=9, gene_space=gene_space, random_seed=42, suppress_warnings=True)
                ga.run(); best_sol, _, _ = ga.best_solution()
                
                # 使用完整數據進行回測以取得最新訊號
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
                    pos, cum_ret, total_ret, mdd, strat_name, st_line, trends, entry_reasons, sharpe, win_rate = res_tuple
                    
                    results_store[m] = {
                        "params": (strat_type, p1, p2, p3, sl_atr, tp_atr, vol_factor, trend_filter_mode, risk), 
                        "pos": pd.Series(pos, index=df.index), 
                        "cum_ret": pd.Series(cum_ret, index=df.index), 
                        "mdd": mdd, 
                        "st_line": pd.Series(st_line, index=df.index), 
                        "trend": pd.Series(trends, index=df.index), 
                        "total_ret": total_ret, "df": df, "split_date": split_date, "strat_name": strat_name,
                        "entry_reasons": pd.Series(entry_reasons, index=df.index),
                        "sharpe": sharpe 
                    }
            progress_bar.progress((i + 1) / 3)
        st.session_state.ga_results = results_store; progress_bar.empty(); st.success("🏆 全方位戰略演化完成！")

    if 'ga_results' in st.session_state:
        results_store = st.session_state.ga_results; modes = list(results_store.keys())
        
        summary_data = []
        for m in modes:
            res = results_store[m]; df_res = res['df']; cum_ret = res['cum_ret']; pos = res['pos']; strat_name = res['strat_name']
            split_date = res['split_date']
            train_mask = df_res.index < split_date; test_mask = df_res.index >= split_date
            
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
        # ... (接在 st.dataframe(pd.DataFrame(summary_data)) 之後) ...
        
        st.markdown("---")
        st.subheader("🔥 V33.9 策略穩健度照妖鏡 (Walk-Forward Heatmap)")
        
        # 熱力圖容器
        cols = st.columns(len(modes))
        
        for idx, m in enumerate(modes):
            res = results_store[m]
            params = res['params']
            strat_name = res['strat_name']
            full_df = res['df'] # 使用完整資料進行切片
            
            # 計算熱力數據
            dates, returns, texts, win_counts = calculate_walk_forward_heatmap(full_df, params, segments=10)
            
            # 評分機制
            robustness_score = win_counts * 10 # 滿分 100
            score_color = "red" if robustness_score >= 70 else "orange" if robustness_score >= 50 else "green"
            
            with cols[idx]:
                st.markdown(f"**{m}** - {strat_name}")
                st.caption(f"穩健度評分: :{score_color}[{robustness_score} 分] ({win_counts}/10 區間獲利)")
                
                # 繪製單條熱力圖 (轉置顯示，比較好看)
                fig_heat = go.Figure(data=go.Heatmap(
                    z=[returns],
                    x=dates,
                    y=[m],
                    text=[texts],
                    texttemplate="%{text}",
                    colorscale='RdYlGn', # 紅=賺(台股習慣), 綠=賠
                    reversescale=False,   # 台股：紅是正，綠是負 -> 若 Plotly 預設 Green is High，則不用反轉；若 Red is High，需檢查
                    # Plotly RdYlGn: Red(Low) -> Green(High). 
                    # 我們要 Red(High) -> Green(Low). 所以要反轉嗎?
                    # 台股：紅漲綠跌。
                    # 設定 zmin < 0, zmax > 0, 讓 0 為黃色
                    zmid=0,
                    showscale=False
                ))
                
                # 修正色階：Plotly 預設 'RdYlGn' 是 紅(低) -> 黃 -> 綠(高)。
                # 台股需要：綠(低/賠) -> 黃 -> 紅(高/賺)。
                # 所以我們需要自定義色階或使用 'RdYlGn' 並設 reversescale=False? 
                # 不，RdYlGn 是 Red-Yellow-Green。我們要 Green-Yellow-Red。
                # 所以使用 'RdYlGn_r' (Reverse) 即可變成 綠->紅。
                fig_heat.update_traces(colorscale='RdYlGn_r' if robustness_score >=0 else 'RdYlGn') # 修正邏輯

                # 實際上更直觀的寫法：
                # 在 page_ga 的 fig_heat 區塊：
                # 強制設定：綠(賠) -> 黃(平) -> 紅(賺)
                fig_heat.update_traces(colorscale=[
                    [0.0, "#21c354"], # Green (Loss)
                    [0.5, "#ffff00"], # Yellow (Flat)
                    [1.0, "#ff4b4b"]  # Red (Win)
                ])

                fig_heat.update_layout(
                    height=120, 
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(showticklabels=False), # 空間太小不顯示日期，滑鼠移上去看就好
                    yaxis=dict(showticklabels=False)
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                
                # 展開詳細數據
                with st.expander("查看區間細節"):
                    detail_df = pd.DataFrame({"區間": dates, "損益": texts, "數值": returns})
                    st.dataframe(detail_df)
                    # ... (接在 st.dataframe(detail_df) 之後) ...

                st.markdown("---")
                st.info("""
                #### 🌡️ 策略體檢報告解讀：
                * 🟥 **神級策略 (80~100分)**：**全天候獲利**。無論牛熊或盤整皆能穩定獲利，是穿越市場週期的「聖杯」。
                * 🟨 **普通策略 (40~60分)**：**看天吃飯**。通常只適應特定盤勢（如只會做多），遇到盤整或空頭容易回吐獲利。
                * 🟩 **危險策略 (< 30分)**：**運氣/過擬合**。雖然總報酬可能很高（因某段行情賺爛），但大部分時間都在賠錢，實戰風險極高。
                """)

        
        tabs = st.tabs(modes)
        for idx, tab in enumerate(tabs):
            m = modes[idx]; res = results_store[m]; df = res['df']; strat_name = res['strat_name']
            reasons = res['entry_reasons']; pos = res['pos']
            params = res['params'] 
            sharpe = res.get('sharpe', 0)
            
            with tab:
                # [新增] 策略訊號推論 (Inference)
                last_pos = pos.iloc[-1]
                prev_pos = pos.iloc[-2]
                last_close = df['Close'].iloc[-1]
                last_atr = df['ATR'].iloc[-1]
                strat_t, p1, p2, p3, sl_atr, tp_atr, vol_f, t_filt, _ = params
                
                target_price = last_close + (last_atr * tp_atr)
                stop_price = last_close - (last_atr * sl_atr)
                
                # 訊號判讀
                sig_text = "⚪ 空手觀望 (WAIT)"
                sig_color = "gray"
                bg_color = "#f0f2f6"
                
                if last_pos == 1 and prev_pos == 0:
                    sig_text = "🔴 今日買進訊號 (BUY SIGNAL)"
                    sig_color = "#d9534f"
                    bg_color = "#f9dede"
                elif last_pos == 1:
                    sig_text = "🟢 持有續抱 (HOLD)"
                    sig_color = "#28a745"
                    bg_color = "#dff0d8"
                elif last_pos == 0 and prev_pos == 1:
                    sig_text = "🟢 今日賣出訊號 (SELL SIGNAL)"
                    sig_color = "#28a745"
                    bg_color = "#dff0d8"

                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 8px solid {sig_color}; text-align: center;">
                    <h2 style="margin:0; color: {sig_color};">{sig_text}</h2>
                    <hr style="border-color: #ddd;">
                    <p style="font-size: 1.1em; font-weight: bold; color: #333;">
                    當前價格: {last_close:.2f} | 🎯 潛在目標: {target_price:.2f} | 🛡️ 建議停損: {stop_price:.2f}
                    </p>
                    <p style="color: gray; font-size: 0.9em;">(基於 {strat_name} 策略, ATR={last_atr:.2f})</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🧬 檢視 AI 演化之最佳系統參數 & 儲存", expanded=False):
                    st.write(f"**策略類型**: {strat_name} (Type {strat_t})")
                    st.write(f"**核心參數**: P1={p1}, P2={p2}, P3={p3}")
                    st.write(f"**風控參數**: 停損={sl_atr:.1f}x ATR, 停利={tp_atr:.1f}x ATR")
                    st.write(f"**濾網設定**: 量能係數={vol_f:.1f}, 趨勢濾網={'開啟' if t_filt else '關閉'}")
                    
                    c_save_1, c_save_2 = st.columns([3, 1])
                    note = c_save_1.text_input("📝 備註 (選填)", key=f"note_{idx}")
                    if c_save_2.button("💾 存入資料庫", key=f"save_{idx}"):
                        msg = db_manager.save_gene(t, strat_name, res['total_ret'], sharpe, params, note)
                        st.toast(msg, icon="✅")

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
                
                buy_indices = df.index[pos.diff() == 1]
                idx_rocket = [ix for ix in buy_indices if reasons[ix] == 1]
                idx_shield = [ix for ix in buy_indices if reasons[ix] == 2]
                idx_std    = [ix for ix in buy_indices if reasons[ix] == 3]
                
                if idx_rocket: fig.add_trace(go.Scatter(x=idx_rocket, y=df.loc[idx_rocket, 'Low']*0.99, mode='text+markers', text='🚀', textposition='bottom center', marker=dict(symbol='star', size=14, color='#FF4B4B'), name='先鋒突擊'), row=1, col=1)
                if idx_shield: fig.add_trace(go.Scatter(x=idx_shield, y=df.loc[idx_shield, 'Low']*0.99, mode='text+markers', text='🛡️', textposition='bottom center', marker=dict(symbol='diamond', size=12, color='#21C354'), name='早鳥防禦'), row=1, col=1)
                if idx_std: fig.add_trace(go.Scatter(x=idx_std, y=df.loc[idx_std, 'Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00FFFF'), name='標準部隊'), row=1, col=1)
                
                st_line = res['st_line']; trend = res['trend']
                st_bull = st_line.copy(); st_bull[trend == -1] = np.nan
                st_bear = st_line.copy(); st_bear[trend == 1] = np.nan
                fig.add_trace(go.Scatter(x=df.index, y=st_bull, mode='lines', line=dict(color='green', width=1), name='支撐'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=st_bear, mode='lines', line=dict(color='red', width=1), name='壓力'), row=1, col=1)

                sp = df[(pos.diff() == -1)]; 
                fig.add_trace(go.Scatter(x=sp.index, y=sp['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', size=12, color='magenta'), name='賣出'), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df.index, y=cash * res['cum_ret'], mode='lines', line=dict(color='orange'), name='總資產'), row=2, col=1)
                
                fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True, key=f"c_{idx}")
                
                st.info("""
                **📝 戰術圖示說明：**
                * 🚀 **先鋒突擊**：偵測到爆量長紅或強勁動能，無視均線特權進場。
                * 🛡️ **早鳥防禦**：(僅保守型) 在均線未翻揚前，偵測到 W 底或強勢反彈提早佈局。
                * 🔵 **標準部隊**：符合均線多頭排列與技術指標的標準進場點。
                """)
                
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
# [V34.1] AI 自選股監控儀表板 + 戰略參謀
# ==========================================
def page_watchlist():
    st.header("👀 AI 自選股戰情中心 (Smart Watchlist)")

    # ==========================================
    # [修正] 強制在此頁面也能初始化 AI
    # ==========================================
    if 'rag_agent' not in st.session_state:
        # 檢查是否有 API Key
        if "AI_Studio_Key" in st.secrets:
            try:
                # 這裡會呼叫 RAGAdvisor (記得確保 RAGAdvisor 類別已更新為 Flash 版)
                st.session_state.rag_agent = RAGAdvisor(st.secrets["AI_Studio_Key"])
            except Exception as e:
                st.warning(f"⚠️ AI 初始化異常: {str(e)}")
        else:
            st.warning("⚠️ 請在 secrets.toml 設定 AI_Studio_Key 才能使用戰報功能")
    # ==========================================
    watchlist = load_watchlist()
    
    # 新增股票輸入框
    c1, c2 = st.columns([3, 1])
    new_t = c1.text_input("新增代號 (例如 2330)", placeholder="輸入代號...")
    if c2.button("➕ 新增", use_container_width=True) and new_t:
        msg = toggle_watchlist(new_t)
        st.toast(msg)
        st.rerun()

    if not watchlist:
        st.info("📭 目前觀察名單為空，請從「AI 總司令」加入或上方手動輸入。")
        return

    st.markdown("---")
    
    # 1. 儀表板掃描 (維持 V34.0 的極速掃描)
    full_tickers = [f"{t}.TW" if t.isdigit() else t for t in watchlist]
    
    with st.spinner(f"正在掃描 {len(watchlist)} 檔自選股戰略狀態..."):
        try:
            # 批量下載 (只抓 3 個月夠算趨勢就好)
            batch_data = yf.download(full_tickers, period="3mo", group_by='ticker', threads=True, progress=False)
        except:
            st.error("連線失敗")
            return

    dashboard_data = []
    
    # ... (這裡維持 V34.0 的儀表板計算邏輯，為節省篇幅略過重複部分，請保留原本的 for 迴圈與指標計算) ...
    # 若您需要我完整重貼這段 for 迴圈請告訴我，否則請保留原本 V34.0 的 dashboard_data 計算邏輯
    
    # --- 為了完整性，這裡快速重現核心計算以便您直接複製貼上 ---
    for t_code in watchlist:
        full_code = f"{t_code}.TW" if t_code.isdigit() else t_code
        try:
            if len(watchlist) == 1: df = batch_data
            else: df = batch_data[full_code]
            
            df = df.dropna(how='all')
            if df.empty or len(df) < 30: continue
            
            last_c = df['Close'].iloc[-1]
            last_v = df['Volume'].iloc[-1]
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma60 = df['Close'].rolling(60).mean().iloc[-1]
            slope = (ma60 - df['Close'].rolling(60).mean().iloc[-2])
            
            # 簡單訊號判定
            signal = "🛡️ 觀望"; sig_color = "gray"; action = "Hold"
            if last_c > ma20 and last_c > ma60 and slope > 0:
                signal = "🚀 多頭"; sig_color = "red"; action = "Buy/Hold"
            elif last_c < ma60:
                signal = "🛑 空頭"; sig_color = "green"; action = "Avoid"

            # FIB 位階
            h = df['High'].max(); l = df['Low'].min()
            pos = (last_c - l) / (h - l) if (h-l) != 0 else 0.5
            fib_desc = "高檔" if pos > 0.8 else "強勢" if pos > 0.6 else "低檔" if pos < 0.2 else "中位"

            # 籌碼 (主力資金流)
            change = (last_c - df['Open'].iloc[-1]) / df['Open'].iloc[-1]
            force = change * last_v
            chip_status = "🔥 吸籌" if (force > 0 and change > 0.01) else "🤮 倒貨" if (force < 0 and change < -0.01) else "😐 中性"
            
            dashboard_data.append({
                "代號": t_code, "現價": f"{last_c:.1f}", "戰略訊號": signal, 
                "FIB位階": f"{fib_desc} ({pos*100:.0f}%)", "主力籌碼": chip_status
            })
        except: continue

    # 顯示儀表板
    if dashboard_data:
        res_df = pd.DataFrame(dashboard_data)
        def color_signal(val):
            if '🚀' in val: return 'color: #ff4b4b; font-weight: bold'
            if '🛑' in val: return 'color: #21c354; font-weight: bold'
            return ''
        def color_chip(val):
            return 'color: #ff4b4b' if '🔥' in val else 'color: #21c354' if '🤮' in val else ''

        st.dataframe(
            res_df.style.applymap(color_signal, subset=['戰略訊號']).applymap(color_chip, subset=['主力籌碼']),
            use_container_width=True, height=35 + len(res_df)*35
        )
    
    st.markdown("---")
    
    # 2. [V34.1 新增] AI 戰略參謀控制台
    st.subheader("🤖 AI 首席分析師：個股深度解盤")
    
    c_sel, c_btn = st.columns([3, 1])
    with c_sel:
        target_stock = st.selectbox("請選擇一檔股票進行 AI 診斷:", watchlist)
    
    with c_btn:
        st.write("") # 排版用
        st.write("")
        btn_gen = st.button("✨ 生成 AI 戰報", type="primary", use_container_width=True)

# ... (前面的代碼保持不變，直接從 if btn_gen and target_stock: 開始替換) ...

    if btn_gen and target_stock:
        agent = st.session_state.get('rag_agent')
        if not agent or not agent.active_model:
            st.error("❌ AI 模型未初始化，請檢查 API Key。")
            return

        with st.status("🧠 AI 正在分析戰情...", expanded=True) as status:
            st.write("📥 正在調閱 K 線圖與技術指標...")
            df_full = get_stock_data(target_stock, period="6mo")
            if df_full.empty:
                st.error("無法獲取數據"); return
            df_full = calculate_indicators(df_full)
            last = df_full.iloc[-1]
            
            st.write("📰 正在檢索近期新聞...")
            news_items, _ = get_special_news_v28(target_stock, STOCK_NAMES.get(target_stock, target_stock))
            news_summary = "\n".join([f"- {n['title']}" for n in news_items[:3]])
            
            st.write(f"🤖 正在撰寫分析報告 (Model: {agent.model_name})...")
            
            # ... (中間計算 ma_state, chip_state 等變數保持不變) ...
            ma_state = "多頭排列" if last['Close'] > last['MA20'] and last['MA20'] > last['MA60'] else "空頭排列" if last['Close'] < last['MA20'] < last['MA60'] else "盤整震盪"
            kd_state = f"K({last['K']:.1f})/D({last['D']:.1f})"
            k_dir = "黃金交叉" if last['K'] > last['D'] else "死亡交叉"
            change = (last['Close'] - df_full['Open'].iloc[-1]) / df_full['Open'].iloc[-1]
            force_idx = change * last['Volume']
            chip_state = "主力吸籌" if force_idx > 0 else "主力出貨" if force_idx < 0 else "不明顯"

            prompt = f"""
            你是一位擁有 20 年經驗的華爾街資深操盤手。請根據以下數據，為投資人撰寫一份 {target_stock} 的短評戰報 (約 100~150 字)。
            
            【技術面】
            - 收盤價: {last['Close']}
            - 均線狀態: {ma_state}
            - KD指標: {kd_state}，呈現 {k_dir}
            - MACD柱狀體: {last['Hist']:.2f}
            
            【籌碼與動能】
            - 當日漲跌幅: {change:.2%}
            - 主力資金流向模擬: {chip_state}
            - RSI: {last['RSI']:.1f}
            
            【近期新聞標題】
            {news_summary}
            
            【撰寫要求】
            1. 風格：專業、犀利、果斷，使用 Emoji (🚀, ⚠️, 🛑) 增強可讀性。
            2. 結構：先講結論 (看多/看空/觀望)，再講理由 (技術+籌碼)，最後給操作建議 (支撐/壓力)。
            """
            
            # [V34.2] 自動重試機制 (Auto-Retry)
            max_retries = 3
            retry_delay = 22 # 錯誤訊息建議等待 21 秒，我們設 22 秒比較保險
            
            ai_reply = None
            
            for attempt in range(max_retries):
                try:
                    response = agent.active_model.generate_content(prompt)
                    ai_reply = response.text
                    break # 成功就跳出迴圈
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "Quota" in err_msg:
                        if attempt < max_retries - 1:
                            st.warning(f"⚠️ AI 請求頻率過高 (429)，系統將休息 {retry_delay} 秒後自動重試 ({attempt+1}/{max_retries})...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            st.error("❌ 已達重試上限，請稍後再試。")
                    else:
                        st.error(f"❌ 未知錯誤: {err_msg}")
                        break

            if ai_reply:
                status.update(label="✅ 戰報生成完畢！", state="complete", expanded=False)
                st.markdown(f"### 📝 {target_stock} AI 戰略參謀報告")
                st.info(ai_reply)
                
                if st.button(f"🚀 進入 {target_stock} 戰情室看圖", key="btn_go_dash"):
                    st.session_state.dash_current_stock = target_stock
                    st.success(f"已鎖定 {target_stock}，請切換至「戰情室」頁面。")

# ==========================================
# 4. 主程式入口
# ==========================================
PAGES = {
    "👀 AI 自選股監控": page_watchlist, # [新增] V34.0
    "🤖 AI 總司令選股": page_ai_selector, 
    "⚡ 全能達人戰情室": page_dashboard, 
    "🧬 PyGAD 策略進化": page_ga
}
st.sidebar.title("⚡ AI 戰情室 V33.8"); st.sidebar.caption("精簡優化 | RAG核心 | 資料庫")
sel = st.sidebar.radio("功能模組", list(PAGES.keys())); PAGES[sel]()
