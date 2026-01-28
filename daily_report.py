import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import os
import json
import logging
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 設定 Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 1. 核心指標計算 (向量化極速版)
# ==========================================
def calculate_score_batch(df):
    """
    針對批量下載的 DataFrame (Single Ticker) 進行計算
    """
    try:
        # 確保長度足夠
        if len(df) < 60: return None
        
        # 1. 基礎指標
        close = df['Close']
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        vol = df['Volume']
        vol_ma20 = vol.rolling(20).mean()
        
        # 2. MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # 3. RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
        rsi = 100 - (100 / (1 + gain/loss))

        # 4. OBV
        obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        obv_ma = obv.rolling(20).mean()
        
        # --- 取最後一天數據 ---
        last_c = close.iloc[-1]
        last_ma20 = ma20.iloc[-1]
        last_ma60 = ma60.iloc[-1]
        last_slope = ma60.diff().iloc[-1]
        last_vol = vol.iloc[-1]
        last_vol_ma = vol_ma20.iloc[-1]
        last_macd = macd.iloc[-1]
        last_sig = signal.iloc[-1]
        last_rsi = rsi.iloc[-1]
        last_obv = obv.iloc[-1]
        last_obv_ma = obv_ma.iloc[-1]
        
        # --- 評分邏輯 ---
        score = 0
        if last_c > last_ma20: score += 2
        if last_slope > 0: score += 3
        if last_c > last_ma60: score += 1
        if last_macd > last_sig: score += 2
        if last_rsi > 50: score += 2
        if last_obv > last_obv_ma: score += 4
        if last_vol > last_vol_ma: score += 3
        
        return {
            "現價": round(last_c, 2),
            "總分": score,
            "RSI": round(last_rsi, 1),
            "趨勢": "⬆️" if last_slope > 0 else "⬇️"
        }
    except Exception as e:
        return None

# ==========================================
# 2. 批量掃描引擎
# ==========================================
def run_scan_turbo():
    logging.info("🚀 AI 總司令：Turbo 極速掃描啟動 (Batch Mode)")
    
    # 1. 準備清單
    target_tickers = []
    if os.path.exists("sector_db.json"):
        with open("sector_db.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            for sector in data.values():
                for sub in sector.values():
                    target_tickers.extend(sub)
    else:
        # 測試用預設清單
        target_tickers = ["2330", "2317", "2454", "2308", "2603", "2382", "3231", "3008"]

    # 去重並標準化代號 (加上 .TW)
    clean_tickers = []
    for t in set(target_tickers):
        t_str = str(t).strip()
        if t_str.isdigit(): t_str += ".TW"
        clean_tickers.append(t_str)
        
    logging.info(f"📋 準備掃描: {len(clean_tickers)} 檔")

    # 2. 分批下載 (避免 URL 過長，每批 100 檔)
    chunk_size = 100
    results = []
    
    for i in range(0, len(clean_tickers), chunk_size):
        chunk = clean_tickers[i:i + chunk_size]
        logging.info(f"⚡ 下載批次 {i}-{i+len(chunk)}...")
        
        try:
            # 關鍵優化：一次下載 100 檔
            # group_by='ticker' 讓回傳格式為 Dict-like: data['2330.TW'] = DataFrame
            data = yf.download(chunk, period="6mo", interval="1d", group_by='ticker', threads=True, progress=False)
            
            # 3. 處理資料
            for ticker in chunk:
                try:
                    # 處理單一或多檔回傳結構差異
                    if len(chunk) == 1:
                        df_t = data
                    else:
                        df_t = data[ticker]
                    
                    # 移除空值行
                    df_t = df_t.dropna(how='all')
                    
                    if df_t.empty or df_t['Volume'].sum() == 0: continue
                    
                    res = calculate_score_batch(df_t)
                    if res:
                        res['代號'] = ticker
                        results.append(res)
                        
                except KeyError:
                    continue # 該股票可能下市或無資料
                except Exception as e:
                    continue
                    
        except Exception as e:
            logging.error(f"❌ 批次下載失敗: {str(e)}")
            time.sleep(5) # 稍微冷卻

    if not results:
        logging.error("❌ 掃描無結果")
        return None

    df_res = pd.DataFrame(results).sort_values("總分", ascending=False)
    
    # 移除 .TW 以美化顯示
    df_res['代號'] = df_res['代號'].astype(str).str.replace('.TW', '').str.replace('.TWO', '')
    
    top_stock = df_res.iloc[0]
    logging.info(f"👑 冠軍出爐: {top_stock['代號']} (分: {top_stock['總分']})")
    
    return df_res

# ==========================================
# 3. Email 發送 (維持原樣)
# ==========================================
def send_email(df_res):
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    receiver = os.environ.get("EMAIL_RECEIVER", sender)

    if not sender or not password:
        logging.error("❌ 未設定 Email Secrets")
        return

    top_stock = df_res.iloc[0]
    top_10 = df_res.head(10)
    
    table_html = ""
    for idx, row in top_10.iterrows():
        rank_icon = "🔹"
        if idx == top_10.index[0]: rank_icon = "🥇"
        elif idx == top_10.index[1]: rank_icon = "🥈"
        elif idx == top_10.index[2]: rank_icon = "🥉"
        
        table_html += f"""
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding:8px;">{rank_icon} <b>{row['代號']}</b></td>
            <td style="padding:8px; color:red;"><b>{row['總分']}</b></td>
            <td style="padding:8px;">{row['現價']}</td>
            <td style="padding:8px;">{row['趨勢']}</td>
        </tr>
        """

    today_str = datetime.now().strftime("%Y-%m-%d")
    html_content = f"""
    <html>
    <body style="font-family: Helvetica, Arial, sans-serif; color: #333;">
        <div style="max-width: 600px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
            <h2 style="color: #00adb5; text-align: center;">🚀 V34.0 極速版戰情日報 ({today_str})</h2>
            <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center;">
                <h3>👑 本日全域總冠軍</h3>
                <h1 style="color: #d9534f; margin: 10px 0;">{top_stock['代號']}</h1>
                <p>戰力總分: <b>{top_stock['總分']}</b> | 收盤價: <b>{top_stock['現價']}</b></p>
            </div>
            <h3>📊 強勢股 Top 10</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #eee;">
                    <th>代號</th><th>分數</th><th>現價</th><th>趨勢</th>
                </tr>
                {table_html}
            </table>
            <p style="text-align: center; color: gray; font-size: 12px;">GitHub Turbo Mode: 掃描 {len(df_res)} 檔完成。</p>
        </div>
    </body>
    </html>
    """

    msg = MIMEMultipart()
    msg['From'] = f"AI 戰情室 <{sender}>"
    msg['To'] = receiver
    msg['Subject'] = f"🚀 [V34.0] 冠軍: {top_stock['代號']} (分: {top_stock['總分']})"
    msg.attach(MIMEText(html_content, 'html'))

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        logging.info("✅ Email 發送成功")
    except Exception as e:
        logging.error(f"❌ Email 發送失敗: {str(e)}")

if __name__ == "__main__":
    start_time = time.time()
    result_df = run_scan_turbo()
    if result_df is not None:
        send_email(result_df)
    logging.info(f"🏁 全部完成，耗時: {time.time() - start_time:.2f} 秒")
