# Dự báo giá nhà TP. Hồ Chí Minh

Đồ án tham khảo quy trình khám phá tin rao, trích xuất đặc trưng, so sánh mô hình
và dự báo giá bất động sản bằng Streamlit. Giá dự đoán được biểu thị bằng tỷ VND.

## Chạy dự án

1. Tạo môi trường Python và cài `requirements.txt`.
2. Đặt CSV có quyền sử dụng tại `data/raw/TPHCM.csv`.
3. Chạy `python build_project.py` để làm sạch dữ liệu, lấy mẫu, so sánh và lưu model.
4. Chạy lần lượt bốn notebook trong `notebooks/` để xem báo cáo từng bước.
5. Chạy `python -m streamlit run app/app.py` để mở ứng dụng.

## Nguồn tham khảo công khai

Dataset Kaggle do người dùng cung cấp: [Apartment prices in the city Ho Chi Minh City](https://www.kaggle.com/datasets/hoandan/apartment-prices-in-the-city-ho-chi-minh-city). Theo phần giới thiệu trên Kaggle, dữ liệu gồm gần 2.000 căn hộ và được thu thập từ Chotot.vn.

Dataset này khác với CSV 51.304 tin đang được xử lý trong project; pipeline hiện dùng `data/raw/TPHCM.csv` và không tự tải Kaggle dataset. Hãy kiểm tra giấy phép/điều khoản của cả Kaggle và nguồn gốc dữ liệu trước khi tái phân phối.

## Dữ liệu và quyền riêng tư

Toàn bộ tin hợp lệ được giữ trong máy để khám phá và làm sạch. Huấn luyện dùng mẫu
ngẫu nhiên cố định tối đa 5.000 tin (`seed=42`), giúp chạy nhanh và lặp lại được.
`data/demo/tphcm_sample_5000.csv` chỉ gồm giá, diện tích, phòng ngủ, số tầng,
mặt tiền, quận và loại bất động sản; không gồm mô tả, phường, tọa độ, ID hay
thông tin môi giới. CSV gốc và dữ liệu làm sạch đầy đủ được loại khỏi Git.

Mẫu demo vẫn được rút trích từ nguồn dữ liệu. Trước khi công khai, hãy xác minh
điều khoản của website nguồn và quyền tái phân phối; không tải số điện thoại,
tên người đăng, mô tả gốc hoặc dữ liệu định vị chính xác lên GitHub.

## Phương pháp và giới hạn

So sánh Ridge Regression, Random Forest và XGBoost bằng MAE, RMSE, R² trên tập
kiểm tra 20% của mẫu. Tiền xử lý gồm trích xuất đặc trưng, chuẩn hóa một phần địa
chỉ, TF-IDF, imputation trong pipeline và log-transform giá mục tiêu. XGBoost
thử CUDA khi khả dụng và tự chuyển CPU nếu không.

Tin rao có phân phối giá lệch mạnh và nhiều trường thiếu. Kết quả là tham khảo
học thuật, không phải thẩm định giá. Hãy xem `outputs/model_comparison.csv` trước
khi diễn giải chất lượng mô hình. Repository chưa gán giấy phép; chọn giấy phép
mã nguồn và dữ liệu riêng trước khi cho phép tái sử dụng.
