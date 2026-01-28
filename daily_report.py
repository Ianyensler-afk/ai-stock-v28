import os
import time
import pandas as pd
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import json
import warnings
from datetime import datetime
import pytz # 處理時區
import gspread # Google Sheets 套件
from oauth2client.service_account import ServiceAccountCredentials

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 設定區 (偵錯模式)
# ==========================================
print("🔍 [偵錯] 開始檢查環境變數...")

EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", EMAIL_SENDER)

# 檢查 Email 設定
if not EMAIL_SENDER:
    print("❌ [嚴重錯誤] 找不到 EMAIL_SENDER！請檢查 GitHub Secrets。")
else:
    print(f"✅ [檢查] EMAIL_SENDER 設定為: {EMAIL_SENDER[:3]}***@***")

if not EMAIL_PASSWORD:
    print("❌ [嚴重錯誤] 找不到 EMAIL_PASSWORD！")
else:
    print("✅ [檢查] EMAIL_PASSWORD 已設定")

# 檢查 Google Sheet 設定
SHEET_CREDENTIALS = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
if not SHEET_CREDENTIALS:
    print("⚠️ [警告] 找不到 Google Sheet 憑證，將跳過存檔。")
else:
    print("✅ [檢查] Google Sheet 憑證已設定")

SHEET_URL = os.environ.get("GOOGLE_SHEET_URL")

# 設定台灣時區
TW_TZ = pytz.timezone('Asia/Taipei')

# ==========================================
# 1.2. 載入板塊資料 (偵錯模式)
# ==========================================
print("🔍 [偵錯] 準備載入 sector_db.json...")

SECTOR_DB = {}
if os.path.exists("sector_db.json"):
    print("✅ [檢查] 檔案存在：sector_db.json")
    try:
        with open("sector_db.json", "r", encoding="utf-8") as f:
            SECTOR_DB = json.load(f)
        print(f"✅ [成功] JSON 載入成功，共包含 {len(SECTOR_DB)} 個大板塊")
    except json.JSONDecodeError as e:
        print(f"❌ [嚴重錯誤] JSON 格式錯誤！請檢查檔案內容。錯誤訊息: {e}")
        # 這裡不給備用名單，直接讓它報錯，您才知道是格式錯了
    except Exception as e:
        print(f"❌ [未知錯誤] 讀取檔案失敗: {e}")
else:
    print("❌ [嚴重錯誤] 找不到 sector_db.json 檔案！請確認它在根目錄。")
    print(f"📂 目前目錄檔案列表: {os.listdir('.')}") # 印出現在有哪些檔案

# ==========================================

# ==========================================
# 2. 核心功能
# ==========================================
def get_stock_data(ticker):
    try:
        # 下載數據，多抓一點避免剛好跨日
        stock = yf.Ticker(ticker)
        df = stock.history(period="5d") 
        if df.empty: return pd.DataFrame()
        return df
    except: return pd.DataFrame()

def calculate_indicators(df):
    try:
        if len(df) < 60: return df
        df = df.copy()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['MA60_Slope'] = df['MA60'].diff()
        
        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        return df
    except: return df

def process_stock_task(ticker):
    try:
        df = get_stock_data(ticker)
        if df.empty or len(df) < 30: return None
        
        # --- [關鍵] 判斷是否為「今日」數據 ---
        # 轉換最後一筆資料的日期到台灣時間，或直接比較日期字串
        last_date = df.index[-1].date()
        today_date = datetime.now(TW_TZ).date()
        
        # 如果最後一筆資料不是今天，代表今天可能沒開盤或資料未更新
        # 但有些冷門股更新慢，這裡做個標記即可
        is_today = (last_date == today_date)

        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        score = 0
        if last['Close'] > last['MA20']: score += 2
        if last.get('MA60_Slope', 0) > 0: score += 3
        if last['Close'] > last.get('MA60', 0): score += 1
        if last.get('MACD', 0) > last.get('Signal', 0): score += 2
        if last.get('RSI', 50) > 50: score += 2
        
        return {
            "代號": ticker,
            "總分": score,
            "現價": round(last['Close'], 2),
            "日期": str(last_date),
            "資料狀態": "即時" if is_today else "延遲/休市",
            "斜率": "Up" if last.get('MA60_Slope', 0) > 0 else "Down"
        }
    except: return None

def save_to_google_sheet(data_list):
    if not SHEET_CREDENTIALS or not SHEET_URL:
        print("❌ 未設定 Google Sheet 憑證，跳過存檔。")
        return

    try:
        # 認證
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(SHEET_CREDENTIALS), scope)
        client = gspread.authorize(creds)
        
        # 開啟試算表
        sheet = client.open_by_url(SHEET_URL).sheet1
        
        # 準備要寫入的資料 (轉換為列表格式)
        # 格式: [掃描時間, 代號, 總分, 現價, 資料日期, 狀態]
        scan_time = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
        rows_to_append = []
        for item in data_list:
            rows_to_append.append([
                scan_time,
                item['代號'],
                item['總分'],
                item['現價'],
                item['日期'],
                item['資料狀態']
            ])
            
        # 批次寫入 (比一筆一筆寫快很多)
        sheet.append_rows(rows_to_append)
        print(f"✅ 已將 {len(rows_to_append)} 筆資料寫入 Google Sheet")
        
    except Exception as e:
        print(f"❌ Google Sheet 寫入失敗: {e}")

def send_email(subject, html_content):
    if not EMAIL_SENDER or not EMAIL_PASSWORD: return
    msg = MIMEMultipart()
    msg['From'] = f"AI 總司令 <{EMAIL_SENDER}>"
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.attach(MIMEText(html_content, 'html'))
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("✅ Email 發送成功")
    except Exception as e:
        print(f"❌ Email 發送失敗: {e}")

# ==========================================
# 3. 主執行區
# ==========================================
if __name__ == "__main__":
    # 檢查今天是不是週末 (GitHub Actions 排程雖然設了 Mon-Fri，但 UTC 轉換可能有誤差，多一層檢查)
    weekday = datetime.now(TW_TZ).weekday() # 0=Mon, 6=Sun
    if weekday > 4:
        print("😴 今天是週末，AI 休息中。")
        exit()

    print(f"🤖 AI 自動駕駛啟動 (台灣時間 {datetime.now(TW_TZ)})")
    
    # 1. 整理清單
    all_tickers = set()
    for sub in SECTOR_DB.values():
        for t_list in sub.values():
            for t in t_list: all_tickers.add(t)
    target_list = sorted(list(all_tickers))
    
    # 2. 掃描
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_ticker = {executor.submit(process_stock_task, t): t for t in target_list}
        for future in as_completed(future_to_ticker):
            res = future.result()
            if res: results.append(res)

    # 3. 處理結果
    if results:
        df_res = pd.DataFrame(results).sort_values("總分", ascending=False)
        
        # 檢查資料新鮮度：如果前 10 名的資料日期都不是今天，可能今天是大盤休市日
        top_10 = df_res.head(10)
        today_str = str(datetime.now(TW_TZ).date())
        fresh_data_count = df_res[df_res['日期'] == today_str].shape[0]
        
        if fresh_data_count < 10:
            print("⚠️ 警告：今日大部分資料未更新，可能是休市日。")
            subject_prefix = "【休市/延遲】"
        else:
            subject_prefix = "【最新戰報】"

        # 4. 存入 Google Sheet (這裡示範存全部)
        save_to_google_sheet(df_res.to_dict('records'))

        # 5. 寄信 (只寄 Top 10)
        champ = top_10.iloc[0]
        html_rows = ""
        for idx, row in top_10.iterrows():
            date_info = f"<small style='color:gray'>({row['日期']})</small>" if row['日期'] != today_str else ""
            html_rows += f"<li><b>{row['代號']}</b> {date_info} - 分: {row['總分']} | 價: {row['現價']}</li>"

        email_html = f"""
        <html><body>
            <h2>🤖 AI 全球戰略日報 ({today_str})</h2>
            <p>資料狀態：{fresh_data_count}/{len(df_res)} 檔已更新</p>
            <hr>
            <p><b>👑 今日總冠軍：{champ['代號']} (總分 {champ['總分']})</b></p>
            <h3>📊 強勢股 Top 10</h3>
            <ul>{html_rows}</ul>
            <p><a href="{SHEET_URL}">🔗 點此查看完整 Google Sheet 報表</a></p>
        </body></html>
        """
        send_email(f"AI {subject_prefix} ({today_str}): 冠軍 {champ['代號']}", email_html)
    else:
        print("❌ 無掃描結果")

