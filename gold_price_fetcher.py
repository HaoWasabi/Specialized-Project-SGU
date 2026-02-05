import yfinance as yf
import pandas as pd

def download_gold_data():
    symbol = "GC=F"
    print(f"Dang tai toan bo du lieu lich su {symbol} tu Yahoo Finance...")
    
    try:
        # 1. Tải dữ liệu
        df = yf.download(symbol, start="2000-08-30")
        
        if df.empty:
            print("Khong co du lieu.")
            return

        # 2. XỬ LÝ LỖI DÍNH CỘT: San phẳng MultiIndex
        # Nếu tiêu đề có 2 tầng (Price, Ticker), ta chỉ lấy tầng 1 (Price)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 3. Đưa cột Date ra khỏi index để nó thành một cột bình thường
        df = df.reset_index()

        # 4. Định dạng lại cột Date để chắc chắn không bị lỗi hiển thị
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        # 5. Lưu file CSV với định dạng chuẩn nhất
        file_name = "gold_gc_f_lich_su_day_du.csv"
        df.to_csv(file_name, index=False, sep=';', encoding='utf-8') 

        print(f"---")
        print(f"Da tai va luu thanh cong {len(df)} dong vao file: {file_name}")
        print(f"Khoang thoi gian: tu {df['Date'].iloc[0]} den {df['Date'].iloc[-1]}")
        print(f"Cac cot da fix: {', '.join(df.columns)}")

    except Exception as e:
        print(f"Loi phat sinh: {e}")

if __name__ == "__main__":
    download_gold_data()