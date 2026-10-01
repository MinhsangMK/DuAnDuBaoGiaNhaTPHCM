import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import make_prediction_frame


MODEL_PATH = PROJECT_ROOT / "models" / "gia_nha_tphcm.joblib"
REFERENCE_PATH = PROJECT_ROOT / "data" / "demo" / "tphcm_sample_5000.csv"


@st.cache_resource
def load_bundle():
    bundle = joblib.load(MODEL_PATH)
    bundle["model"].regressor_.named_steps["model"].get_booster().set_param(
        {"device": "cpu"}
    )
    return bundle


@st.cache_data
def load_reference():
    if REFERENCE_PATH.is_file():
        return pd.read_csv(REFERENCE_PATH, encoding="utf-8-sig")
    return pd.DataFrame()


def district_label(code):
    if code.startswith("Q") and code[1:].isdigit():
        return f"Quận {code[1:]}"
    names = {
        "TP_Thu_Duc": "TP. Thủ Đức",
        "Binh_Thanh": "Bình Thạnh",
        "Binh_Tan": "Bình Tân",
        "Go_Vap": "Gò Vấp",
        "Phu_Nhuan": "Phú Nhuận",
        "Tan_Binh": "Tân Bình",
        "Tan_Phu": "Tân Phú",
        "Binh_Chanh": "Bình Chánh",
        "Can_Gio": "Cần Giờ",
        "Cu_Chi": "Củ Chi",
        "Hoc_Mon": "Hóc Môn",
        "Nha_Be": "Nhà Bè",
    }
    return names.get(code) or code.replace("_", " ").title()


def find_similar(reference, district, property_type, area):
    required = {"price_bil", "area_m2", "district", "property_type"}
    if reference.empty or not required.issubset(reference.columns):
        return pd.DataFrame()

    data = reference.copy()
    data["price_bil"] = pd.to_numeric(data["price_bil"], errors="coerce")
    data["area_m2"] = pd.to_numeric(data["area_m2"], errors="coerce")
    data = data.dropna(subset=["price_bil", "area_m2"])

    same_district = data[data["district"].astype(str) == str(district)]
    if not same_district.empty:
        data = same_district

    same_type = data[data["property_type"].astype(str) == str(property_type)]
    if not same_type.empty:
        data = same_type

    data["area_difference"] = (data["area_m2"] - area).abs()
    return data.sort_values("area_difference").head(10)


st.set_page_config(page_title="Dự báo giá nhà TP.HCM", page_icon="🏠", layout="wide")
st.title("🏠 Dự báo giá nhà tại TP. Hồ Chí Minh")
st.caption("Giá tham khảo từ tin rao; không thay thế thẩm định giá chuyên môn.")

if not MODEL_PATH.is_file():
    st.error(f"Chưa có model: {MODEL_PATH}. Hãy chạy build_project.py trước.")
    st.stop()

bundle = load_bundle()
reference = load_reference()

if not reference.empty and "district" in reference:
    districts = sorted(
        x for x in reference["district"].dropna().astype(str).unique()
        if x != "__missing__"
    )
else:
    districts = ["TP_Thu_Duc", "Q1", "Q3", "Binh_Thanh"]

if not reference.empty and "property_type" in reference:
    property_types = sorted(
        x for x in reference["property_type"].dropna().astype(str).unique()
        if x != "__missing__"
    )
else:
    property_types = ["nha rieng", "can ho chung cu", "dat nen"]

if not districts:
    districts = ["TP_Thu_Duc"]
if not property_types:
    property_types = ["nha rieng"]

default_district = "TP_Thu_Duc" if "TP_Thu_Duc" in districts else districts[0]

with st.form("price_form"):
    left, right = st.columns(2)

    with left:
        area = st.number_input("Diện tích (m²)", min_value=8.0, value=60.0, step=1.0)
        bedrooms = st.number_input("Phòng ngủ", min_value=0, value=2, step=1)
        floors = st.number_input("Số tầng", min_value=1, value=2, step=1)
        frontage = st.number_input(
            "Mặt tiền (m)", min_value=0.5, value=4.0, step=0.5
        )

    with right:
        district = st.selectbox(
            "Quận/Huyện",
            districts,
            index=districts.index(default_district),
            format_func=district_label,
        )
        ward = st.text_input("Phường/Xã", placeholder="Ví dụ: Phường 5")
        property_type = st.selectbox(
            "Loại bất động sản",
            property_types,
            format_func=lambda value: value.replace("_", " ").title(),
        )

    description = st.text_area(
        "Mô tả thêm",
        placeholder="Ví dụ: hẻm xe hơi, gần trường học, sổ hồng riêng..."
    )
    submitted = st.form_submit_button("Dự báo giá")

if submitted:
    row = make_prediction_frame(
        area, bedrooms, floors, frontage,
        district, ward, property_type, description
    )
    prediction = max(float(bundle["model"].predict(row)[0]), 0)

    col1, col2 = st.columns(2)
    col1.metric("Giá dự báo", f"{prediction:,.2f} tỷ đồng")
    col2.metric("Giá trên m²", f"{prediction * 1000 / area:,.1f} triệu/m²")

    mae = bundle.get("validation_mae_ty")
    if mae is not None and np.isfinite(mae):
        st.caption(f"MAE trên tập kiểm tra: khoảng {mae:,.2f} tỷ đồng.")
    if bundle.get("train_rows", 0) < 30:
        st.warning("Dữ liệu huấn luyện còn ít; kết quả chỉ để minh họa.")

    similar = find_similar(reference, district, property_type, area)
    if not similar.empty:
        st.subheader("Tin rao gần giống để tham khảo")
        st.caption(
            f"Trung vị giá các tin tìm được: "
            f"{similar['price_bil'].median():,.2f} tỷ đồng."
        )

        columns = [
            c for c in [
                "price_bil", "area_m2", "bedrooms", "floors",
                "frontage", "district", "ward", "property_type",
            ]
            if c in similar.columns
        ]
        table = similar[columns].rename(columns={
            "price_bil": "Giá rao (tỷ)",
            "area_m2": "Diện tích (m²)",
            "bedrooms": "Phòng ngủ",
            "floors": "Số tầng",
            "frontage": "Mặt tiền (m)",
            "district": "Quận/Huyện",
            "ward": "Phường/Xã",
            "property_type": "Loại nhà",
        })
        st.dataframe(table, width="stretch", hide_index=True)
    else:
        st.info("Không tìm thấy tin tương tự trong dữ liệu tham chiếu.")
