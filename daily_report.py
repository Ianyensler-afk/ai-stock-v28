import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import os
import json
import logging
import time
import io
import base64
import gspread  # [新增] Google Sheets 操作套件
from oauth2client.service_account import ServiceAccountCredentials # [新增] 驗證套件

# --- 設定 Matplotlib 後端為 Agg (非互動模式) ---
import matplotlib
matplotlib.use('Agg') 
# ---------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# 設定 Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 設定您的 Streamlit App 網址
APP_BASE_URL = "https://ai-stock-v28-izt7hannvryvbk5udoeq22.streamlit.app/" 

# ==========================================
# 0. 輔助功能：載入股票名稱
# ==========================================
STOCK_MAP = {}
if os.path.exists("stock_names.json"):
    try:
        with open("stock_names.json", "r", encoding="utf-8") as f:
            STOCK_MAP = json.load(f)
    except: pass

def get_name(ticker):
    # 移除 .TW/.TWO
    clean_t = str(ticker).replace(".TW", "").replace(".TWO", "")
    return STOCK_MAP.get(clean_t, clean_t)

# ==========================================
# 1. 核心指標計算 (改為 100 分制)
# ==========================================
def calculate_score_batch(df):
    try:
        if len(df) < 60: return None
        
        # 基礎計算
        close = df['Close']
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        vol = df['Volume']
        vol_ma20 = vol.rolling(20).mean()
        
        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
        rsi = 100 - (100 / (1 + gain/loss))

        # OBV
        obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        obv_ma = obv.rolling(20).mean()
        
        # 取最新值
        last_c = close.iloc[-1]
        last_ma20 = ma20.iloc[-1]
        last_ma60 = ma60.iloc[-1]
        last_vol = vol.iloc[-1]
        last_vol_ma = vol_ma20.iloc[-1]
        last_macd = macd.iloc[-1]
        last_sig = signal.iloc[-1]
        last_rsi = rsi.iloc[-1]
        last_obv = obv.iloc[-1]
        last_obv_ma = obv_ma.iloc[-1]
        
        # --- 評分邏輯 (滿分 100) ---
        score = 0
        
        # 1. 趨勢面 (40分)
        if last_c > last_ma20: score += 15
        if last_c > last_ma60: score += 15
        if last_ma20 > last_ma60: score += 10 # 均線多排
        
        # 2. 動能面 (30分)
        if last_macd > last_sig: score += 15
        if last_rsi > 50: score += 15
        
        # 3. 籌碼量能面 (30分)
        if last_obv > last_obv_ma: score += 15
        if last_vol > last_vol_ma: score += 15
        
        # 籌碼判斷 (簡易版：量增價漲=吸籌)
        change = (last_c - df['Open'].iloc[-1]) / df['Open'].iloc[-1]
        force_val = change * last_vol
        chip_status = "🔥吸籌" if force_val > 0 else "🤮倒貨" if force_val < 0 else "😐中性"

        return {
            "現價": round(last_c, 2),
            "總分": score,
            "RSI": round(last_rsi, 1),
            "籌碼": chip_status,
            "趨勢": "⬆️多" if score >= 60 else "⬇️空"
        }
    except Exception as e:
        return None

# ==========================================
# 2. 繪製靜態 K 線圖 (給 Email 用)
# ==========================================
def generate_chart_image(ticker, df):
    try:
        # 只取最後 60 天
        plot_df = df.tail(60).copy()
        
        plt.figure(figsize=(10, 5))
        plt.style.use('dark_background')
        
        # 繪製收盤價
        plt.plot(plot_df.index, plot_df['Close'], label='Price', color='cyan', linewidth=2)
        plt.plot(plot_df.index, plot_df['Close'].rolling(20).mean(), label='MA20', color='yellow', linestyle='--', alpha=0.7)
        plt.plot(plot_df.index, plot_df['Close'].rolling(60).mean(), label='MA60', color='magenta', linestyle='--', alpha=0.7)
        
        plt.title(f"{ticker} ({get_name(ticker)}) Daily Chart", fontsize=14, color='white')
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        # 轉為 Bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        logging.error(f"繪圖失敗: {e}")
        return None

# ==========================================
# 3. 批量掃描引擎
# ==========================================
def run_scan_turbo():
    logging.info("🚀 AI 總司令：V35.0 終極掃描 (Top 20 + Deep Link)")
    
    target_tickers = []
    if os.path.exists("sector_db.json"):
        with open("sector_db.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            for sector in data.values():
                for sub in sector.values():
                    target_tickers.extend(sub)
    else:
        target_tickers = ["2330", "2317", "2454", "2308", "2603", "2382", "3231", "3008"]

    clean_tickers = []
    for t in set(target_tickers):
        t_str = str(t).strip()
        if t_str.isdigit(): t_str += ".TW"
        clean_tickers.append(t_str)
        
    logging.info(f"📋 準備掃描: {len(clean_tickers)} 檔")

    chunk_size = 100
    results = []
    champion_df = None # 儲存冠軍的 dataframe 以便畫圖
    
    for i in range(0, len(clean_tickers), chunk_size):
        chunk = clean_tickers[i:i + chunk_size]
        try:
            data = yf.download(chunk, period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
            
            for ticker in chunk:
                try:
                    if len(chunk) == 1: df_t = data
                    else: 
                        if ticker not in data: continue
                        df_t = data[ticker]
                    
                    df_t = df_t.dropna(how='all')
                    if df_t.empty or len(df_t) < 60: continue
                    
                    res = calculate_score_batch(df_t)
                    if res:
                        res['代號'] = ticker.replace(".TW", "").replace(".TWO", "")
                        res['名稱'] = get_name(res['代號'])
                        results.append(res)
                        
                except: continue
            time.sleep(1)
        except: continue

    if not results: return None

    df_res = pd.DataFrame(results).sort_values("總分", ascending=False)
    
    # 抓取冠軍的完整資料來畫圖
    top_ticker = df_res.iloc[0]['代號']
    top_ticker_tw = f"{top_ticker}.TW"
    try:
        champion_df = yf.download(top_ticker_tw, period="6mo", progress=False)
    except: pass
    
    return df_res, champion_df, top_ticker

# ==========================================
# 4. Email 發送 (含圖表與連結)
# ==========================================
def send_email(df_res, champion_df, top_ticker):
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    receiver = os.environ.get("EMAIL_RECEIVER", sender)

    if not sender or not password: return

    # 取 Top 20
    top_20 = df_res.head(20)
    top_stock = top_20.iloc[0]
    
    # 生成表格
    table_html = ""
    for idx, row in top_20.iterrows():
        rank = idx + 1
        rank_icon = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank}."
        
        # Deep Link: 使用 query param ?stock=xxxx
        link = f"{APP_BASE_URL}/?stock={row['代號']}"
        
        # 分數顏色
        score_color = "#ff4b4b" if row['總分'] >= 80 else "#ffa500" if row['總分'] >= 60 else "#21c354"
        
        table_html += f"""
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding:6px;">{rank_icon}</td>
            <td style="padding:6px;">
                <a href="{link}" style="text-decoration:none; font-weight:bold; color:#007bff;">
                    {row['代號']} {row['名稱']}
                </a>
            </td>
            <td style="padding:6px; color:{score_color}; font-weight:bold;">{row['總分']}</td>
            <td style="padding:6px;">{row['現價']}</td>
            <td style="padding:6px;">{row['籌碼']}</td>
        </tr>
        """

    # 生成冠軍圖表
    chart_img = None
    if champion_df is not None:
        chart_buf = generate_chart_image(top_ticker, champion_df)
        if chart_buf:
            chart_img = MIMEImage(chart_buf.read())
            chart_img.add_header('Content-ID', '<champion_chart>')

    today_str = datetime.now().strftime("%Y-%m-%d")
    
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #333;">
        <div style="max-width: 600px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
            <h2 style="color: #00adb5; text-align: center;">🚀 V35.0 戰情日報 ({today_str})</h2>
            
            <div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 20px;">
                <h3>👑 本日冠軍: {top_stock['名稱']} ({top_stock['代號']})</h3>
                <h1 style="color: #d9534f; margin: 5px 0;">{top_stock['總分']} 分</h1>
                <p>收盤: {top_stock['現價']} | 籌碼: {top_stock['籌碼']}</p>
                <a href="{APP_BASE_URL}/?stock={top_stock['代號']}" 
                   style="display:inline-block; padding:10px 20px; background-color:#ff4b4b; color:white; text-decoration:none; border-radius:5px;">
                   🚀 進入 App 分析
                </a>
            </div>
            
            <div style="text-align:center; margin-bottom:20px;">
                <img src="cid:champion_chart" style="width:100%; max-width:500px; border-radius:5px;">
            </div>

            <h3>📊 強勢股 Top 20</h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <tr style="background-color: #eee;">
                    <th>#</th><th>股票</th><th>總分</th><th>現價</th><th>籌碼</th>
                </tr>
                {table_html}
            </table>
        </div>
    </body>
    </html>
    """

    msg = MIMEMultipart()
    msg['From'] = f"AI 戰情室 <{sender}>"
    msg['To'] = receiver
    msg['Subject'] = f"🚀 [V35] 冠軍: {top_stock['名稱']} ({top_stock['總分']}分)"
    
    msg.attach(MIMEText(html_content, 'html'))
    if chart_img:
        msg.attach(chart_img)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        logging.info("✅ Email 發送成功")
    except Exception as e:
        logging.error(f"❌ Email 發送失敗: {str(e)}")

# ==========================================
# 5. [新增] 寫入 Google Sheet (儲存 Top 20)
# ==========================================
def update_google_sheet(df_res):
    logging.info("📈 正在將數據寫入 Google Sheet...")
    
    # 讀取 Secret (請確保 GitHub Secret 名稱正確)
    json_creds = os.environ.get('GOOGLE_SHEETS_CREDENTIALS')
    sheet_url = os.environ.get('GOOGLE_SHEET_URL')
    
    if not json_creds or not sheet_url:
        logging.error("❌ 找不到 Google Sheet 設定，跳過寫入。")
        return

    try:
        # 1. 驗證與連線
        creds_dict = json.loads(json_creds)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # 2. 開啟 Sheet
        sheet = client.open_by_url(sheet_url).sheet1
        
        # 3. 準備資料 (取 Top 20)
        top_20 = df_res.head(20).copy()
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        rows_to_append = []
        for _, row in top_20.iterrows():
            # 轉換為 Python 原生型態，避免 numpy 錯誤
            rows_to_append.append([
                today_str,                      # Date
                str(row['代號']),               # Stock ID
                str(row['名稱']),               # Name
                float(row['現價']),             # Close Price
                int(row['總分']),               # Score
                float(row['RSI']),              # RSI
                str(row['籌碼']),               # Chip Status
                str(row['趨勢'])                # Trend
            ])
            
        # 4. 寫入資料
        if rows_to_append:
            sheet.append_rows(rows_to_append)
            logging.info(f"✅ 成功寫入 {len(rows_to_append)} 筆資料到 Google Sheet")
            
    except Exception as e:
        logging.error(f"❌ 寫入 Google Sheet 失敗: {str(e)}")

# ==========================================
# 主程式入口
# ==========================================
if __name__ == "__main__":
    res = run_scan_turbo()
    if res:
        # 1. 發送信件
        send_email(res[0], res[1], res[2])
        
        # 2. [新增] 同步寫入 Google Sheet
        update_google_sheet(res[0])
