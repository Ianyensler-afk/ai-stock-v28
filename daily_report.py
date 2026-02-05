import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import os
import json
import logging
import time
import io
import gspread
import twstock  # [新增] 用於抓取台股名稱
from oauth2client.service_account import ServiceAccountCredentials

# --- 設定 Matplotlib 後端為 Agg (非互動模式) ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# 設定 Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 設定您的 Streamlit App 網址
APP_BASE_URL = "https://ai-stock-v28-izt7hannvryvbk5udoeq22.streamlit.app/" 

# ==========================================
# 0. 輔助功能：載入股票名稱 (升級版)
# ==========================================
# [修正 A] 更強的名稱清洗函式
def get_stock_name(stock_id):
    """
    優先使用 twstock 庫查詢即時名稱，
    自動去除 .TW, .TWO 以及異常的 'O' 後綴以提高辨識率
    """
    try:
        # 強制轉字串並轉大寫
        s_id = str(stock_id).upper()
        
        # 關鍵修正：使用 rstrip('O') 去除尾部多餘的 O
        clean_id = s_id.replace(".TW", "").replace(".TWO", "").rstrip('O')
        
        # 查詢 twstock
        if clean_id in twstock.codes:
            return twstock.codes[clean_id].name
            
        return clean_id # 真的查不到才回傳代號
    except:
        return stock_id

# ==========================================
# 1. 核心指標計算 (含技術指標與籌碼)
# ==========================================
def calculate_score_batch(df):
    try:
        if len(df) < 60: return None
        
        # --- 基礎數據 ---
        close = df['Close']
        open_price = df['Open']
        vol = df['Volume']
        
        # --- 技術指標計算 ---
        # 1. 均線
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        vol_ma20 = vol.rolling(20).mean()
        
        # 2. MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal # 柱狀圖
        
        # 3. RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
        rsi = 100 - (100 / (1 + gain/loss))

        # 4. OBV (能量潮)
        obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        obv_ma = obv.rolling(20).mean()
        
        # --- 取最新一筆數值 ---
        last_c = close.iloc[-1]
        last_o = open_price.iloc[-1]
        last_ma20 = ma20.iloc[-1]
        last_ma60 = ma60.iloc[-1]
        last_vol = vol.iloc[-1]
        last_vol_ma = vol_ma20.iloc[-1]
        last_macd = macd.iloc[-1]
        last_sig = signal.iloc[-1]
        last_hist = macd_hist.iloc[-1]
        last_rsi = rsi.iloc[-1]
        last_obv = obv.iloc[-1]
        last_obv_ma = obv_ma.iloc[-1]
        
        # 計算漲跌幅 (%)
        pct_change = ((last_c - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100

        # --- 評分邏輯 (滿分 100) ---
        score = 0
        
        # 1. 趨勢面 (40分)
        if last_c > last_ma20: score += 15
        if last_c > last_ma60: score += 15
        if last_ma20 > last_ma60: score += 10 
        
        # 2. 動能面 (30分)
        if last_macd > last_sig: score += 15
        if last_rsi > 50: score += 15
        
        # 3. 籌碼量能面 (30分)
        if last_obv > last_obv_ma: score += 15
        if last_vol > last_vol_ma: score += 15
        
        # 籌碼判斷 (簡易版：量增價漲=吸籌)
        # 註：因 yfinance 無法直接取得分點與法人資料，此處維持以量價關係模擬
        intra_change = (last_c - last_o) / last_o
        force_val = intra_change * last_vol
        chip_status = "🔥吸籌" if force_val > 0 else "🤮倒貨" if force_val < 0 else "😐中性"

        return {
            "現價": round(last_c, 2),
            "漲跌幅": round(pct_change, 2),
            "成交量": int(last_vol),
            "總分": score,
            "RSI": round(last_rsi, 1),
            "MACD_Hist": round(last_hist, 2), # 新增 MACD 柱狀值
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
        plot_df = df.tail(60).copy()
        
        plt.figure(figsize=(10, 5))
        plt.style.use('dark_background')
        
        plt.plot(plot_df.index, plot_df['Close'], label='Price', color='cyan', linewidth=2)
        plt.plot(plot_df.index, plot_df['Close'].rolling(20).mean(), label='MA20', color='yellow', linestyle='--', alpha=0.7)
        plt.plot(plot_df.index, plot_df['Close'].rolling(60).mean(), label='MA60', color='magenta', linestyle='--', alpha=0.7)
        
        plt.title(f"{ticker} Daily Chart", fontsize=14, color='white')
        plt.legend()
        plt.grid(True, alpha=0.2)
        
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
    logging.info("🚀 AI 總司令：V36.0 終極掃描 (整合版)")
    
    # 這裡可以替換成你自己的股票清單邏輯
    target_tickers = ["2330", "2317", "2454", "2603", "2609", "2615", "3231", "2382", "3008", "3037"]
    
    # 嘗試讀取 sector_db (如果有的話)
    if os.path.exists("sector_db.json"):
        try:
            with open("sector_db.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                target_tickers = []
                for sector in data.values():
                    for sub in sector.values():
                        target_tickers.extend(sub)
        except: pass

    clean_tickers = []
    for t in set(target_tickers):
        t_str = str(t).strip()
        if t_str.isdigit(): t_str += ".TW"
        clean_tickers.append(t_str)
        
    logging.info(f"📋 準備掃描: {len(clean_tickers)} 檔")

    chunk_size = 100
    results = []
    champion_df = None
    
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
                        # 處理代號與名稱
                        stock_id = ticker.replace(".TW", "").replace(".TWO", "")
                        res['代號'] = stock_id
                        res['名稱'] = get_stock_name(stock_id) # 使用 twstock 抓名稱
                        results.append(res)
                        
                except: continue
            time.sleep(1)
        except: continue

# ... (前面的程式碼不變) ...

    if not results: return None

    # 轉為 DataFrame
    df_res = pd.DataFrame(results)

    # [修正 B] 超級排序邏輯
    # 1. 總分 (降冪)
    # 2. 漲跌幅 (降冪) -> 這樣 100 分俱樂部裡，漲最多的會排第一
    # 3. 成交量 (降冪) -> 如果漲幅也一樣，量大的贏
    df_res = df_res.sort_values(
        by=["總分", "漲跌幅", "成交量"], 
        ascending=[False, False, False]
    )
    
    # [修正 C] 確保寫入 Excel 的代號也不會有 'O'
    # 這樣你的 Google Sheet 就不會出現 '4542O' 這種怪代號
    df_res['代號'] = df_res['代號'].astype(str).apply(lambda x: x.replace(".TW", "").replace(".TWO", "").rstrip('O'))
    
    # 抓取冠軍 (現在這個冠軍會非常穩定了)
    top_ticker = df_res.iloc[0]['代號']
    top_ticker_tw = f"{top_ticker}.TW"

    # ... (後面的程式碼不變) ...
    try:
        champion_df = yf.download(top_ticker_tw, period="6mo", progress=False)
    except: pass
    
    return df_res, champion_df, top_ticker

# ==========================================
# 4. Email 發送
# ==========================================
def send_email(df_res, champion_df, top_ticker):
    sender = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    receiver = os.environ.get("EMAIL_RECEIVER", sender)

    if not sender or not password: return

    top_20 = df_res.head(20)
    top_stock = top_20.iloc[0]
    
    table_html = ""
    for idx, row in top_20.iterrows():
        rank = idx + 1
        link = f"{APP_BASE_URL}/?stock={row['代號']}"
        score_color = "#ff4b4b" if row['總分'] >= 80 else "#ffa500" if row['總分'] >= 60 else "#21c354"
        
        # 漲跌幅顏色 (紅漲綠跌)
        pct_color = "red" if row['漲跌幅'] > 0 else "green" if row['漲跌幅'] < 0 else "black"
        
        table_html += f"""
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding:6px;">{rank}</td>
            <td style="padding:6px;">
                <a href="{link}" style="text-decoration:none; font-weight:bold; color:#007bff;">
                    {row['代號']} {row['名稱']}
                </a>
            </td>
            <td style="padding:6px; color:{score_color}; font-weight:bold;">{row['總分']}</td>
            <td style="padding:6px;">{row['現價']}</td>
            <td style="padding:6px; color:{pct_color};">{row['漲跌幅']}%</td>
            <td style="padding:6px;">{row['籌碼']}</td>
        </tr>
        """

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
            <h2 style="color: #00adb5; text-align: center;">🚀 每日戰報 ({today_str})</h2>
            <div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 20px;">
                <h3>👑 冠軍: {top_stock['名稱']} ({top_stock['代號']})</h3>
                <h1 style="color: #d9534f; margin: 5px 0;">{top_stock['總分']} 分</h1>
            </div>
            <div style="text-align:center; margin-bottom:20px;">
                <img src="cid:champion_chart" style="width:100%; max-width:500px; border-radius:5px;">
            </div>
            <h3>📊 強勢股 Top 20</h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <tr style="background-color: #eee;">
                    <th>#</th><th>股票</th><th>總分</th><th>現價</th><th>漲幅</th><th>籌碼</th>
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
    msg['Subject'] = f"🚀 [V36] 冠軍: {top_stock['名稱']} ({top_stock['總分']}分)"
    msg.attach(MIMEText(html_content, 'html'))
    if chart_img: msg.attach(chart_img)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        logging.info("✅ Email 發送成功")
    except Exception as e:
        logging.error(f"❌ Email 發送失敗: {str(e)}")

# ==========================================
# 5. 寫入 Google Sheet (含標題與新指標)
# ==========================================
def update_google_sheet(df_res):
    logging.info("📈 正在將數據寫入 Google Sheet...")
    
    json_creds = os.environ.get('GOOGLE_SHEETS_CREDENTIALS')
    sheet_url = os.environ.get('GOOGLE_SHEET_URL')
    
    if not json_creds or not sheet_url:
        logging.error("❌ 找不到 Google Sheet 設定")
        return

    try:
        creds_dict = json.loads(json_creds)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        sheet = client.open_by_url(sheet_url).sheet1
        
        # 檢查是否需要寫入標題 (如果目前是空表)
        current_data = sheet.get_all_values()
        if not current_data:
            headers = [
                "日期", "代號", "名稱", "收盤價", "漲跌幅(%)", 
                "總分", "成交量", "RSI(14)", "MACD柱狀", "籌碼狀態", "訊號"
            ]
            sheet.append_row(headers)
            logging.info("📝 已新增標題列")

        # 準備 Top 20 資料
        top_20 = df_res.head(20).copy()
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        rows_to_append = []
        for _, row in top_20.iterrows():
            rows_to_append.append([
                today_str,
                str(row['代號']),
                str(row['名稱']),
                float(row['現價']),
                float(row['漲跌幅']), # 新增
                int(row['總分']),
                int(row['成交量']),   # 新增
                float(row['RSI']),
                float(row['MACD_Hist']), # 新增
                str(row['籌碼']),
                str(row['趨勢'])
            ])
            
        if rows_to_append:
            sheet.append_rows(rows_to_append)
            logging.info(f"✅ 成功寫入 {len(rows_to_append)} 筆資料")
            
    except Exception as e:
        logging.error(f"❌ 寫入 Google Sheet 失敗: {str(e)}")

# ==========================================
# 主程式入口
# ==========================================
if __name__ == "__main__":
    res = run_scan_turbo()
    if res:
        # 解包回傳值
        df_results, champion_data, top_stock_id = res
        
        # 1. 發送 Email
        send_email(df_results, champion_data, top_stock_id)
        
        # 2. 更新 Google Sheet
        update_google_sheet(df_results)



