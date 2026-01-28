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
import pytz
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests # 新增 requests 來處理 Session

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 設定區 & 載入區
# ==========================================
print("🔍 [系統] 初始化設定 (V33.9 忍者潛行版)...")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", EMAIL_SENDER)
SHEET_CREDENTIALS = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
SHEET_URL = os.environ.get("GOOGLE_SHEET_URL")
TW_TZ = pytz.timezone('Asia/Taipei')

SECTOR_DB = {}
if os.path.exists("sector_db.json"):
    try:
        with open("sector_db.json", "r", encoding="utf-8") as f:
            SECTOR_DB = json.load(f)
        print(f"✅ [成功] JSON 載入成功，共包含 {len(SECTOR_DB)} 個大板塊")
    except Exception as e:
        print(f"❌ [錯誤] JSON 讀取失敗: {e}")

# ==========================================
# 2. 核心功能 (偽裝瀏覽器)
# ==========================================

# [新增] 建立一個偽裝成 Chrome 瀏覽器的 Session
def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://finance.yahoo.com/"
    })
    return session

def get_stock_data(ticker):
    # 重試機制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 隨機延遲，模仿人類行為
            time.sleep(random.uniform(1.0, 2.5)) 
            
            # 使用偽裝 Session
            session = get_session()
            stock = yf.Ticker(ticker, session=session)
            
            # 抓取數據
            df = stock.history(period="5d")
            
            if df.empty:
                return pd.DataFrame()
            return df
            
        except Exception as e:
            err_msg = str(e)
            if "Too Many Requests" in err_msg or "429" in err_msg:
                # 如果被擋，休息久一點 (15秒) 再重試
                print(f"⚠️ [流量管制] {ticker} 被擋，休息 15 秒後重試... ({attempt+1}/{max_retries})")
                time.sleep(15)
            else:
                # 其他錯誤，稍微休息
                time.sleep(2)
    
    return pd.DataFrame()

def calculate_indicators(df):
    try:
        if len(df) < 30: return df
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
        
        if df.empty or len(df) < 20: 
            return {"status": "fail", "code": ticker, "reason": "Empty/No Data"}
        
        last_date = df.index[-1].date()
        today_date = datetime.now(TW_TZ).date()
        is_today = (last_date == today_date)

        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        score = 0
        if last['Close'] > last.get('MA20', 0): score += 2
        if last.get('MA60_Slope', 0) > 0: score += 3
        if last['Close'] > last.get('MA60', 0): score += 1
        if last.get('MACD', 0) > last.get('Signal', 0): score += 2
        if last.get('RSI', 50) > 50: score += 2
        
        return {
            "status": "ok",
            "代號": ticker,
            "總分": score,
            "現價": round(last['Close'], 2),
            "日期": str(last_date),
            "資料狀態": "即時" if is_today else "延遲",
            "斜率": "Up" if last.get('MA60_Slope', 0) > 0 else "Down"
        }
    except Exception as e:
        return {"status": "fail", "code": ticker, "reason": str(e)}

def save_to_google_sheet(data_list):
    if not SHEET_CREDENTIALS or not SHEET_URL: return
    try:
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(SHEET_CREDENTIALS), scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(SHEET_URL).sheet1
        scan_time = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
        
        # [修改] 為了避免 Sheet 寫入過慢，這裡只寫入前 100 名
        top_100 = data_list[:100]
        rows = [[scan_time, i['代號'], i['總分'], i['現價'], i['日期'], i['資料狀態']] for i in top_100]
        sheet.append_rows(rows)
        print(f"✅ Google Sheet 寫入 Top {len(rows)} 筆")
    except Exception as e: print(f"❌ Sheet Error: {e}")

def send_email(subject, html_content):
    if not EMAIL_SENDER or not EMAIL_PASSWORD: return
    try:
        msg = MIMEMultipart()
        msg['From'] = f"AI 總司令 <{EMAIL_SENDER}>"
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject
        msg.attach(MIMEText(html_content, 'html'))
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("✅ Email 發送成功")
    except Exception as e: print(f"❌ Email Error: {e}")

# ==========================================
# 3. 主執行區
# ==========================================
if __name__ == "__main__":
    print(f"🤖 AI 自動駕駛啟動 (台灣時間 {datetime.now(TW_TZ)})")
    
    # 1. 整理清單
    all_tickers = set()
    for sub in SECTOR_DB.values():
        for t_list in sub.values():
            for t in t_list: all_tickers.add(t)
    target_list = sorted(list(all_tickers))
    
    print(f"📋 準備掃描清單，共 {len(target_list)} 檔...")
    if len(target_list) == 0: exit()

    # 2. 掃描
    results = []
    
    # [關鍵修改] 將 Workers 降到 2，雖然慢但不會被擋
    # 如果還是失敗，請改成 max_workers=1 (完全單線程)
    workers = 2 
    print(f"🚀 開始執行 ThreadPool (Max Workers={workers}, 慢速穩定模式)...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_ticker = {executor.submit(process_stock_task, t): t for t in target_list}
        for i, future in enumerate(as_completed(future_to_ticker)):
            res = future.result()
            
            # 每 20 檔回報一次
            if i % 20 == 0:
                print(f"⏳ 進度: {i}/{len(target_list)} (目前成功: {len(results)})")

            if res and res['status'] == 'ok':
                results.append(res)

    print(f"🛑 掃描結束。成功: {len(results)}")
    
    # 3. 處理結果
    if results:
        df_res = pd.DataFrame(results).sort_values("總分", ascending=False)
        top_10 = df_res.head(10)
        
        # 存檔與寄信
        save_to_google_sheet(df_res.to_dict('records'))
        
        champ = top_10.iloc[0]
        html_rows = ""
        for idx, row in top_10.iterrows():
            html_rows += f"<li><b>{row['代號']}</b> - 分: {row['總分']} | 價: {row['現價']}</li>"

        email_html = f"""
        <html><body>
            <h2>🤖 AI 全球戰略日報</h2>
            <p>成功掃描：{len(results)} / {len(target_list)}</p>
            <hr>
            <p><b>👑 總冠軍：{champ['代號']}</b></p>
            <ul>{html_rows}</ul>
        </body></html>
        """
        send_email(f"AI 戰報: 冠軍 {champ['代號']}", email_html)
    else:
        print("❌ 無有效掃描結果")
