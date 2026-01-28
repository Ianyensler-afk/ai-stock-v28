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

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 設定區 & 載入區 (維持您已成功的設定)
# ==========================================
print("🔍 [系統] 初始化設定...")
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
# 2. 核心功能 (加入除錯訊息)
# ==========================================
def get_stock_data(ticker):
    try:
        # [修改] 使用 Ticker 但不做任何處理，直接抓歷史資料
        # 有時候 Yahoo 會擋特定 User-Agent，這裡依賴 yfinance 的自動處理
        stock = yf.Ticker(ticker)
        df = stock.history(period="5d")
        
        if df.empty:
            # 回傳空值前，印出一個失敗標記 (僅印出前幾個避免洗版，這裡簡化處理)
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"❌ [下載錯誤] {ticker}: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    try:
        if len(df) < 30: return df # 放寬標準到 30 天
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
        # 增加隨機延遲，避免太快被擋
        time.sleep(random.uniform(0.1, 0.5))
        
        df = get_stock_data(ticker)
        
        # [偵錯] 如果是空的，這裡會被跳過
        if df.empty or len(df) < 20: 
            # 這裡不 return None，而是回傳一個錯誤標記，讓我們知道它是因為沒資料
            return {"status": "fail", "code": ticker, "reason": "Empty/No Data"}
        
        # 判斷日期
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

# ... (save_to_google_sheet 和 send_email 維持不變，省略以節省空間) ...
def save_to_google_sheet(data_list):
    if not SHEET_CREDENTIALS or not SHEET_URL: return
    try:
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(SHEET_CREDENTIALS), scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(SHEET_URL).sheet1
        scan_time = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
        rows = [[scan_time, i['代號'], i['總分'], i['現價'], i['日期'], i['資料狀態']] for i in data_list]
        sheet.append_rows(rows)
        print(f"✅ Google Sheet 寫入 {len(rows)} 筆")
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
    
    # [偵錯] 印出清單數量，確認 JSON 解析是否有問題
    print(f"📋 準備掃描清單，共 {len(target_list)} 檔...")
    if len(target_list) == 0:
        print("❌ 錯誤：目標清單為空！請檢查 sector_db.json 的結構。")
        exit()

    # 2. 掃描 (降低併發數，避免瞬間被封鎖)
    results = []
    fail_count = 0
    fail_reasons = []

    print("🚀 開始執行 ThreadPool (Max Workers=4)...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_ticker = {executor.submit(process_stock_task, t): t for t in target_list}
        for i, future in enumerate(as_completed(future_to_ticker)):
            res = future.result()
            
            # 每 50 檔回報一次進度，確認程式有在跑
            if i % 50 == 0:
                print(f"⏳ 進度: {i}/{len(target_list)} (目前成功: {len(results)})")

            if res and res['status'] == 'ok':
                results.append(res)
            else:
                fail_count += 1
                if res and len(fail_reasons) < 5: # 只記錄前 5 個錯誤原因
                    fail_reasons.append(f"{res['code']}: {res['reason']}")

    print(f"🛑 掃描結束。成功: {len(results)} | 失敗: {fail_count}")
    
    if fail_reasons:
        print("🔍 部分失敗原因範例:", fail_reasons)

    # 3. 處理結果
    if results:
        df_res = pd.DataFrame(results).sort_values("總分", ascending=False)
        top_10 = df_res.head(10)
        
        # 存檔與寄信 (維持不變)
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
        print("❌ 無有效掃描結果，不寄信。")
