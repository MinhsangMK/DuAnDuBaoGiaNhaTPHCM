# Đề tài dự báo giá nhà TP. Hồ Chí Minh — Nhóm 7

Đồ án tham khảo quy trình khám phá tin rao, trích xuất đặc trưng, so sánh mô hình
và dự báo giá bất động sản bằng Streamlit. Giá dự đoán được biểu thị bằng tỷ VND.

Chạy trực tiếp Demo: https://minhsangmk.streamlit.app/

## Chạy dự án

1. Tạo môi trường Python và cài `requirements.txt`.
2. Đặt CSV có quyền sử dụng tại `data/raw/TPHCM.csv`.
3. Chạy `python build_project.py` để làm sạch dữ liệu, lấy mẫu, so sánh và lưu model.
4. Chạy lần lượt bốn notebook trong `notebooks/` để xem báo cáo từng bước.
5. Chạy `python -m streamlit run app/app.py` để mở ứng dụng.
6. Chạy `python scripts/create_presentation_charts.py` để xuất bốn ảnh PNG 16:9 tại `outputs/presentation/` cho PowerPoint.

Giao diện Streamlit dùng nền tối xanh navy, card màu lam đậm và điểm nhấn xanh
ngọc/tím/cam theo bộ biểu đồ trình bày. Chủ đề cơ sở được cấu hình trong
`.streamlit/config.toml`; hero, form, metric và bảng kết quả được tạo kiểu trong
`app/app.py`. Không dùng ảnh chart làm background trực tiếp để giữ độ tương phản
và khả năng đọc trên màn hình nhỏ.

## Lưu dữ liệu đóng góp lên GitHub

Ứng dụng có biểu mẫu quản trị để lưu nối thêm các quan sát có **giá thực tế**
vào `data/contributions/verified_listings.json` trên GitHub. Dữ liệu mới được
đánh dấu `pending_review`, không tự đưa vào tập kiểm thử và không làm model tự
huấn luyện lại. Chỉ sau khi kiểm tra chất lượng và quyền sử dụng dữ liệu, quản
trị viên mới nên tích hợp chúng vào một lần huấn luyện có phiên bản.

Để bật tính năng trên Streamlit Community Cloud, thêm secret sau trong phần
**App settings → Secrets**. Dùng fine-grained GitHub token chỉ cấp quyền
`Contents: Read and write` cho đúng repository; không commit token hoặc mật
khẩu quản trị vào Git:

```toml
[github]
owner = "MinhsangMK"
repo = "DuAnDuBaoGiaNhaTPHCM"
branch = "main"
path = "data/contributions/verified_listings.json"
token = "GITHUB_FINE_GRAINED_TOKEN"
admin_password = "MAT_KHAU_QUAN_TRI_DAI_VA_NGAU_NHIEN"
```

Mỗi lần lưu tạo một commit lên nhánh đã cấu hình. Nếu repository công khai thì
các trường tổng quát đã gửi và lịch sử commit cũng công khai; biểu mẫu yêu cầu
xác nhận trước khi ghi. Không nhập địa chỉ chi tiết, tên, số điện thoại, mô tả
tin hoặc dữ liệu cá nhân. GitHub Contents API giới hạn kích thước tệp; khi tập
đóng góp tăng lớn, hãy chuyển sang cơ sở dữ liệu thay vì tiếp tục lưu trong Git.

## Nguồn tham khảo công khai

Dataset Kaggle do người dùng cung cấp: [Apartment prices in the city Ho Chi Minh City](https://www.kaggle.com/datasets/hoandan/apartment-prices-in-the-city-ho-chi-minh-city). Theo phần giới thiệu trên Kaggle, dữ liệu gồm gần 2.000 căn hộ và được thu thập từ Chotot.vn.

Dataset này khác với CSV 51.304 tin đang được xử lý trong project; pipeline hiện dùng `data/raw/TPHCM.csv` và không tự tải Kaggle dataset. Hãy kiểm tra giấy phép/điều khoản của cả Kaggle và nguồn gốc dữ liệu trước khi tái phân phối.

## Dữ liệu và quyền riêng tư

Toàn bộ tin hợp lệ được giữ trong máy để khám phá, làm sạch và huấn luyện.
Pipeline huấn luyện/đánh giá dùng toàn bộ dữ liệu có giá hợp lệ; chia 80/20
train/holdout với `random_state=42`, rồi fit model cuối trên toàn bộ dữ liệu.
Riêng `data/demo/tphcm_sample_5000.csv` vẫn chỉ là mẫu công khai tối đa 5.000
dòng, gồm giá, diện tích, phòng ngủ, số tầng,
mặt tiền, quận và loại bất động sản; không gồm mô tả, phường, tọa độ, ID hay
thông tin môi giới. CSV gốc và dữ liệu làm sạch đầy đủ được loại khỏi Git.

Mẫu demo vẫn được rút trích từ nguồn dữ liệu. Trước khi công khai, hãy xác minh
điều khoản của website nguồn và quyền tái phân phối; không tải số điện thoại,
tên người đăng, mô tả gốc hoặc dữ liệu định vị chính xác lên GitHub.

## Phương pháp và giới hạn

So sánh Ridge Regression, Random Forest và XGBoost bằng MAE, RMSE, R² trên tập
holdout 20% của toàn bộ dữ liệu hợp lệ. Tiền xử lý gồm trích xuất đặc trưng, chuẩn hóa một phần địa
chỉ, TF-IDF, imputation trong pipeline và log-transform giá mục tiêu. XGBoost
thử CUDA khi khả dụng và tự chuyển CPU nếu không.

### Kết quả lần chạy toàn bộ CSV

CSV có 51.304 dòng; sau làm sạch còn 51.132 tin có giá hợp lệ. Lượt đánh giá
dùng 40.905 dòng train và 10.227 dòng holdout. Model triển khai được fit lại
trên cả 51.132 dòng.

| Mô hình | MAE (tỷ VND) | RMSE (tỷ VND) | R² | Thiết bị |
| --- | ---: | ---: | ---: | --- |
| XGBoost | 25,30 | 84,05 | 0,0395 | GPU |
| Ridge Regression | 26,02 | 84,89 | 0,0202 | CPU |
| Random Forest | 27,54 | 86,82 | -0,0249 | CPU |

Tin rao có phân phối giá lệch mạnh và nhiều trường thiếu. Kết quả là tham khảo
học thuật, không phải thẩm định giá. Hãy xem `outputs/model_comparison.csv` trước
khi diễn giải chất lượng mô hình. Repository chưa gán giấy phép; chọn giấy phép
mã nguồn và dữ liệu riêng trước khi cho phép tái sử dụng.
