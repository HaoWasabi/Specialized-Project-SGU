# Specialized-Project-SGU

## Giới thiệu

Đây là đồ án học phần Chuyên đề chuyên ngành tại Trường Đại học Sài Gòn (SGU), học kỳ 2 năm học 2025-2026. Đề tài tập trung vào bài toán dự báo giá vàng (XAU/USD) bằng mô hình LSTM và tối ưu siêu tham số bằng PSO (Particle Swarm Optimization).

Dự án được triển khai chủ yếu dưới dạng Jupyter Notebook, bao gồm các phiên bản thử nghiệm và phiên bản cải tiến với tập dữ liệu benchmark lấy từ yfinance.

## Thông tin học phần

- Giảng viên hướng dẫn: TS. Phan Tấn Quốc

## Thành viên nhóm

| STT | MSSV | Họ và tên |
|---|---|---|
| 1 | 3122410100 | Trương Gia Hào |
| 2 | 3122410004 | Nguyễn Văn An |

## Mục tiêu dự án

- Thu thập dữ liệu giá vàng và các biến ngoại sinh (DXY, dầu, S&P500, lợi suất trái phiếu) từ yfinance.
- Xây dựng quy trình tiền xử lý và tạo tập chuỗi thời gian cho mô hình LSTM.
- Sử dụng PSO để tối ưu siêu tham số cho mô hình LSTM.
- Đánh giá và so sánh PSO-LSTM với mô hình LSTM baseline trên tập benchmark.
- Trình bày kết quả thông qua biểu đồ và các chỉ số đánh giá (MAE, RMSE, MAPE, R2, Directional Accuracy).

## Cấu trúc thư mục

```text
.
|-- README.md
|-- requirements.txt
|-- docs/
|   `-- Report_GPP_PSO_LSTM_GiaHao_VanAn_2026.pdf
|-- slides/
|   `-- Slide_GPP_PSO_LSTM_GiaHao_VanAn_2026.pptx
`-- notebooks/
    |-- 260407_v01_ex01.ipynb
    |-- 260412_v02_ex01.ipynb
    |-- 260412_v02_ex02.ipynb
    |-- 260412_v02_ex03.ipynb
    |-- 260412_v02_ex04.ipynb
    `-- gia_vang_benchmark.csv (dataset)
```

## Công nghệ và thư viện

- Python 3.10+ (khuyến nghị 3.10 hoặc 3.11)
- Jupyter Notebook
- Các thư viện chính:
  - yfinance
  - pandas
  - numpy
  - scikit-learn
  - tensorflow
  - pyswarms
  - matplotlib

## Cài đặt môi trường

1. Tạo và kích hoạt môi trường ảo.
2. Cài đặt các gói cơ bản trong requirements:

```bash
pip install -r requirements.txt
```

Nếu bạn dùng Google Colab, các notebook đã có cell cài đặt thư viện bằng pip.

## Cách chạy nhanh

1. Mở Jupyter Notebook hoặc Jupyter Lab.
2. Chạy lần lượt các notebook trong thư mục notebooks.
3. Để tái tạo kết quả đầy đủ, ưu tiên chạy notebook:
   - notebooks/260412_v02_ex01.ipynb
4. Đảm bảo kernel đã cài đầy đủ thư viện trước khi chạy các cell huấn luyện PSO-LSTM.

## Đầu ra chính mong đợi

- Tệp dữ liệu benchmark:
  - notebooks/gia_vang_benchmark.csv
- Mô hình và các artifact có thể được tạo trong quá trình chạy notebook:
  - model_PSO_LSTM_30.h5
  - scaler_gold_30.pkl
  - best_params_30.json
  - so_sanh_PSO_vs_Baseline.xlsx

## Tài liệu tham khảo trong dự án

- Báo cáo: [Project Report](docs/Report_GPP_PSO_LSTM_GiaHao_VanAn_2026.pdf)
- Slide: [Presentation Slides](slides/Slide_GPP_PSO_LSTM_GiaHao_VanAn_2026.pptx)

## Ghi chú

- Một số notebook có thể mất nhiều thời gian khi tối ưu PSO (phụ thuộc vào số particle và số iteration).
- Kết quả có thể dao động theo thời điểm tải dữ liệu mới từ yfinance.
- Nên chạy trên máy có RAM/GPU đủ tốt để rút ngắn thời gian huấn luyện.

