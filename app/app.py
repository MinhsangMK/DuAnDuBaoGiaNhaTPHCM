import base64
from datetime import datetime, timezone
import hmac
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import uuid

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
MAX_CONTRIBUTIONS_FILE_BYTES = 900_000


@st.cache_resource
def load_bundle():
    bundle = joblib.load(MODEL_PATH)
    estimator = bundle["model"].regressor_.named_steps["model"]
    get_booster = getattr(estimator, "get_booster", None)
    if callable(get_booster):
        get_booster().set_param({"device": "cpu"})
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


def find_similar(
    reference: pd.DataFrame,
    district: str,
    property_type: str,
    area: float,
) -> tuple[pd.DataFrame, str]:
    required = {"price_bil", "area_m2", "district", "property_type"}
    if reference.empty or not required.issubset(reference.columns):
        return pd.DataFrame(), "không có dữ liệu tham chiếu phù hợp"

    data = reference.copy()
    data["price_bil"] = pd.to_numeric(data["price_bil"], errors="coerce")
    data["area_m2"] = pd.to_numeric(data["area_m2"], errors="coerce")
    data = data.dropna(subset=["price_bil", "area_m2"])
    data = data[
        np.isfinite(data["price_bil"]) & np.isfinite(data["area_m2"])
    ]
    if data.empty:
        return data, "không có tin tham chiếu hợp lệ"

    same_district = data[data["district"].astype(str) == str(district)]
    if not same_district.empty:
        data = same_district
        same_type = data[data["property_type"].astype(str) == str(property_type)]
        if not same_type.empty:
            data = same_type
            match_scope = "cùng quận và cùng loại bất động sản"
        else:
            match_scope = "cùng quận; không có tin cùng loại bất động sản"
    else:
        same_type = data[data["property_type"].astype(str) == str(property_type)]
        if not same_type.empty:
            data = same_type
            match_scope = "cùng loại bất động sản, mở rộng sang quận khác"
        else:
            match_scope = "mở rộng sang mọi quận và loại bất động sản"

    data["area_difference"] = (data["area_m2"] - area).abs()
    return data.sort_values("area_difference").head(10), match_scope


def github_storage_config() -> dict[str, str] | None:
    try:
        settings = st.secrets["github"]
    except (KeyError, FileNotFoundError):
        return None

    required = ("token", "owner", "repo", "branch", "path", "admin_password")
    config = {key: str(settings.get(key, "")).strip() for key in required}
    if not all(config.values()):
        return None
    if ".." in Path(config["path"]).parts:
        raise ValueError("Đường dẫn lưu dữ liệu GitHub không hợp lệ.")
    return config


def github_api_request(
    config: dict[str, str],
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = (
        f"https://api.github.com/repos/{quote(config['owner'], safe='')}/"
        f"{quote(config['repo'], safe='')}/contents/"
        f"{quote(path, safe='/')}"
    )
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {config['token']}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "DuAnDuBaoGiaNhaTPHCM",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def load_contributions(
    config: dict[str, str],
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        file_info = github_api_request(config, "GET", config["path"])
    except HTTPError as error:
        if error.code == 404:
            return [], None
        raise RuntimeError(
            f"Không đọc được dữ liệu GitHub (HTTP {error.code})."
        ) from error

    if file_info.get("encoding") != "base64" or "content" not in file_info:
        raise ValueError("Tệp dữ liệu GitHub không có định dạng base64 hợp lệ.")
    raw = base64.b64decode(file_info["content"], validate=True)
    if len(raw) > MAX_CONTRIBUTIONS_FILE_BYTES:
        raise ValueError(
            "Tệp dữ liệu đóng góp đã gần giới hạn GitHub Contents API; "
            "cần chuyển sang kho dữ liệu phù hợp hơn."
        )
    records = json.loads(raw.decode("utf-8"))
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ValueError("Tệp dữ liệu đóng góp GitHub không đúng cấu trúc.")
    return records, file_info["sha"]


def append_contribution(
    config: dict[str, str], record: dict[str, Any]
) -> None:
    records, sha = load_contributions(config)
    updated = [*records, record]
    raw = json.dumps(updated, ensure_ascii=False, indent=2).encode("utf-8")
    if len(raw) > MAX_CONTRIBUTIONS_FILE_BYTES:
        raise ValueError(
            "Không thể lưu thêm: tệp đóng góp đã gần giới hạn GitHub "
            "Contents API."
        )

    payload = {
        "message": "Add a pending property-price data contribution",
        "content": base64.b64encode(raw).decode("ascii"),
        "branch": config["branch"],
    }
    if sha:
        payload["sha"] = sha

    try:
        github_api_request(config, "PUT", config["path"], payload)
    except HTTPError as error:
        if error.code in (409, 422):
            raise RuntimeError(
                "Tệp vừa được cập nhật đồng thời hoặc GitHub từ chối ghi. "
                "Hãy tải lại trang rồi thử lại."
            ) from error
        raise RuntimeError(
            f"Không lưu được dữ liệu lên GitHub (HTTP {error.code}); "
            "kiểm tra quyền Contents: write của token."
        ) from error


st.set_page_config(
    page_title="Đề tài dự báo giá nhà TP.HCM · Nhóm 7",
    page_icon="🏠",
    layout="wide",
)
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --canvas: #0B1020;
        --surface: #151F35;
        --surface-raised: #1B2740;
        --text-main: #F7F8FC;
        --text-muted: #B7C1D6;
        --accent-teal: #31D6C5;
        --accent-purple: #9B7BFF;
        --accent-orange: #FFB454;
        --stroke: #2B3957;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 97% 0%, rgba(155, 123, 255, .15), transparent 27rem),
            radial-gradient(ellipse at 2% 88%, rgba(49, 214, 197, .10), transparent 24rem),
            var(--canvas);
        color: var(--text-main);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, .72);
        backdrop-filter: blur(12px);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1160px;
        padding-top: 2.25rem;
        padding-bottom: 4rem;
    }

    [data-testid="stAppViewContainer"] :is(h1, h2, h3, p, label, li) {
        color: var(--text-main);
    }

    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {
        color: var(--text-muted);
    }

    .app-hero {
        position: relative;
        overflow: hidden;
        margin: .25rem 0 1.25rem;
        padding: clamp(1.4rem, 4vw, 2.5rem);
        border: 1px solid rgba(117, 139, 184, .28);
        border-radius: 26px;
        background:
            radial-gradient(ellipse at 100% 0%, rgba(155, 123, 255, .20), transparent 42%),
            linear-gradient(130deg, rgba(21, 31, 53, .98), rgba(16, 27, 47, .94));
        box-shadow: 0 22px 60px rgba(0, 0, 0, .24);
    }

    .app-hero::after {
        position: absolute;
        right: -3.6rem;
        bottom: -7.2rem;
        width: 15rem;
        height: 15rem;
        border: 1px solid rgba(49, 214, 197, .25);
        border-radius: 50%;
        background: rgba(49, 214, 197, .06);
        content: "";
        pointer-events: none;
    }

    .app-hero-eyebrow {
        margin: 0 0 .7rem;
        color: var(--accent-teal);
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .14em;
        text-transform: uppercase;
    }

    .app-hero h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        color: var(--text-main) !important;
        font-size: clamp(2rem, 4.5vw, 3.15rem);
        font-weight: 760;
        letter-spacing: -.045em;
        line-height: 1.12;
    }

    .app-hero-copy {
        position: relative;
        z-index: 1;
        max-width: 46rem;
        margin: .85rem 0 0;
        color: var(--text-muted) !important;
        font-size: 1rem;
        line-height: 1.65;
    }

    [data-testid="stForm"] {
        padding: clamp(1rem, 3vw, 1.8rem);
        border: 1px solid var(--stroke);
        border-radius: 22px;
        background: linear-gradient(155deg, rgba(21, 31, 53, .97), rgba(16, 25, 43, .96));
        box-shadow: 0 20px 55px rgba(0, 0, 0, .20);
    }

    [data-testid="stForm"] label,
    [data-testid="stExpander"] label {
        color: #DCE4F3 !important;
        font-weight: 600;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input,
    [data-baseweb="select"] > div {
        border-color: #354563;
        border-radius: 11px;
        background-color: var(--surface-raised);
        color: var(--text-main);
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: var(--accent-teal);
        box-shadow: 0 0 0 1px var(--accent-teal);
    }

    [data-testid="stFormSubmitButton"] button,
    [data-testid="stButton"] button {
        min-height: 2.8rem;
        border: 0;
        border-radius: 12px;
        background: linear-gradient(110deg, #19AFA8, var(--accent-teal));
        color: #071522;
        font-weight: 750;
        box-shadow: 0 8px 22px rgba(49, 214, 197, .19);
        transition: transform .16s ease, filter .16s ease, box-shadow .16s ease;
    }

    [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stButton"] button:hover {
        border: 0;
        filter: brightness(1.08);
        transform: translateY(-1px);
        box-shadow: 0 12px 28px rgba(49, 214, 197, .25);
    }

    [data-testid="stMetric"] {
        padding: 1rem 1.15rem;
        border: 1px solid var(--stroke);
        border-radius: 17px;
        background: linear-gradient(145deg, #18243B, #131D31);
        box-shadow: 0 12px 32px rgba(0, 0, 0, .16);
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--accent-teal) !important;
        font-weight: 750;
    }

    [data-testid="stAlert"] {
        border: 1px solid rgba(49, 214, 197, .24);
        border-radius: 14px;
        background: rgba(21, 31, 53, .9);
        color: var(--text-main);
    }

    [data-testid="stExpander"] {
        overflow: hidden;
        border: 1px solid var(--stroke);
        border-radius: 16px;
        background: rgba(21, 31, 53, .78);
    }

    [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--stroke);
        border-radius: 15px;
    }

    [data-testid="stMarkdown"] hr {
        border-color: var(--stroke);
    }

    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.15rem 1rem 2.5rem;
        }

        .app-hero {
            border-radius: 19px;
        }

        [data-testid="stForm"] {
            padding: 1rem;
            border-radius: 18px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    <header class="app-hero">
        <p class="app-hero-eyebrow">Đề tài dự báo nhà · Nhóm 7</p>
        <h1>Ước tính giá nhà<br>TP. Hồ Chí Minh</h1>
        <p class="app-hero-copy">
            Điền thông tin bất động sản để xem mức giá tham khảo và những tin rao gần giống.
            Kết quả chỉ hỗ trợ tham khảo, không thay thế thẩm định chuyên môn.
        </p>
    </header>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error(f"Chưa có model: {MODEL_PATH}. Hãy chạy build_project.py trước.")
    st.stop()

bundle = load_bundle()
reference = load_reference()

model_details = []
model_name = bundle.get("model_name")
if model_name:
    model_details.append(f"Mô hình: {model_name}")
train_rows = bundle.get("train_rows")
if isinstance(train_rows, (int, np.integer)):
    model_details.append(f"Dữ liệu huấn luyện: {train_rows:,} tin")
if model_details:
    st.caption(" | ".join(model_details))

mae = bundle.get("validation_mae_ty")
if mae is not None and np.isfinite(mae):
    st.info(
        f"MAE trên tập kiểm tra: {mae:,.2f} tỷ đồng. Đây là sai số tuyệt đối "
        "trung bình, không phải giới hạn sai số cho từng dự đoán."
    )
if isinstance(train_rows, (int, np.integer)) and train_rows < 30:
    st.warning("Dữ liệu huấn luyện còn ít; kết quả chỉ để minh họa.")

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
    st.markdown("#### Thông tin dự báo giá nhà")
    left, right = st.columns(2)

    with left:
        area = st.number_input(
            "Diện tích (m²)",
            min_value=8.0,
            value=60.0,
            step=1.0,
            help="Nhập diện tích ghi trong tin rao.",
        )
        bedrooms = st.number_input(
            "Phòng ngủ", min_value=0, value=2, step=1
        )
        floors = st.number_input("Số tầng", min_value=1, value=2, step=1)
        frontage = st.number_input(
            "Mặt tiền (m)",
            min_value=0.5,
            value=4.0,
            step=0.5,
            help="Mặt tiền nhỏ nhất được nhập là 0,5 m.",
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
        "Đặc điểm nổi bật",
        placeholder="Ví dụ: hẻm xe hơi, gần trường học, sổ hồng riêng...",
        help="Có thể thêm đặc điểm vị trí, pháp lý và tiện ích nếu có.",
    )
    submitted = st.form_submit_button("Ước tính giá")

if submitted:
    row = make_prediction_frame(
        area, bedrooms, floors, frontage,
        district, ward, property_type, description
    )
    prediction = max(float(bundle["model"].predict(row)[0]), 0)

    col1, col2 = st.columns(2)
    col1.metric("Giá dự báo", f"{prediction:,.2f} tỷ đồng")
    col2.metric("Giá trên m²", f"{prediction * 1000 / area:,.1f} triệu/m²")

    similar, match_scope = find_similar(
        reference, district, property_type, area
    )
    if not similar.empty:
        st.subheader("Tin rao gần giống để tham khảo")
        st.caption(
            f"Phạm vi đối chiếu: {match_scope}. "
            f"Hiển thị {len(similar)} tin gần nhất theo diện tích."
        )
        st.caption(
            f"Trung vị giá trong các tin hiển thị: "
            f"{similar['price_bil'].median():,.2f} tỷ đồng."
        )

        columns = [
            c for c in [
                "price_bil", "area_m2", "area_difference", "bedrooms", "floors",
                "frontage", "district", "ward", "property_type",
            ]
            if c in similar.columns
        ]
        table = similar[columns].rename(columns={
            "price_bil": "Giá rao (tỷ)",
            "area_m2": "Diện tích (m²)",
            "area_difference": "Lệch diện tích (m²)",
            "bedrooms": "Phòng ngủ",
            "floors": "Số tầng",
            "frontage": "Mặt tiền (m)",
            "district": "Quận/Huyện",
            "ward": "Phường/Xã",
            "property_type": "Loại nhà",
        })
        st.dataframe(table, width="stretch", hide_index=True)
    else:
        st.info(match_scope)

st.divider()
st.subheader("Đóng góp giá thực tế để cập nhật dữ liệu")
st.caption(
    "Chỉ lưu thông tin tổng quát cùng giá thực tế do quản trị viên xác nhận. "
    "Không lưu mô tả, địa chỉ chi tiết, tên hoặc số điện thoại."
)

try:
    contribution_config = github_storage_config()
except ValueError as error:
    contribution_config = None
    st.error(str(error))

if contribution_config is None:
    st.info(
        "Chưa cấu hình GitHub để lưu dữ liệu. Xem hướng dẫn cấu hình "
        "Streamlit secrets trong README.md."
    )
else:
    with st.expander("Mở biểu mẫu quản trị"):
        admin_password = st.text_input(
            "Mật khẩu quản trị",
            type="password",
            key="contribution_admin_password",
        )
        if not hmac.compare_digest(
            admin_password, contribution_config["admin_password"]
        ):
            st.caption("Nhập mật khẩu quản trị để mở biểu mẫu.")
        else:
            with st.form("contribution_form"):
                contribution_left, contribution_right = st.columns(2)
                with contribution_left:
                    actual_price = st.number_input(
                        "Giá thực tế (tỷ đồng)",
                        min_value=0.01,
                        value=5.0,
                        step=0.1,
                    )
                    actual_area = st.number_input(
                        "Diện tích (m²)",
                        min_value=8.0,
                        value=60.0,
                        step=1.0,
                    )
                    actual_bedrooms = st.number_input(
                        "Phòng ngủ",
                        min_value=0,
                        value=2,
                        step=1,
                    )
                with contribution_right:
                    actual_floors = st.number_input(
                        "Số tầng",
                        min_value=1,
                        value=2,
                        step=1,
                    )
                    actual_frontage = st.number_input(
                        "Mặt tiền (m)",
                        min_value=0.0,
                        value=4.0,
                        step=0.5,
                    )
                    actual_district = st.selectbox(
                        "Quận/Huyện",
                        districts,
                        format_func=district_label,
                        key="contribution_district",
                    )
                    actual_property_type = st.selectbox(
                        "Loại bất động sản",
                        property_types,
                        format_func=lambda value: value.replace(
                            "_", " "
                        ).title(),
                        key="contribution_property_type",
                    )

                confirmed_actual_price = st.checkbox(
                    "Tôi xác nhận đây là giá thực tế, không phải giá dự đoán."
                )
                consent_to_publish = st.checkbox(
                    "Tôi đồng ý lưu các trường tổng quát này trong GitHub "
                    "repository; nếu repository công khai, dữ liệu và lịch sử "
                    "commit sẽ công khai."
                )
                save_contribution = st.form_submit_button(
                    "Lưu dữ liệu chờ duyệt"
                )

            if save_contribution:
                if not confirmed_actual_price or not consent_to_publish:
                    st.error(
                        "Cần xác nhận giá thực tế và đồng ý lưu dữ liệu trên "
                        "GitHub trước khi gửi."
                    )
                else:
                    contribution = {
                        "id": str(uuid.uuid4()),
                        "submitted_at": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        "status": "pending_review",
                        "price_bil": float(actual_price),
                        "area_m2": float(actual_area),
                        "bedrooms": int(actual_bedrooms),
                        "floors": int(actual_floors),
                        "frontage": float(actual_frontage),
                        "district": actual_district,
                        "property_type": actual_property_type,
                    }
                    try:
                        append_contribution(contribution_config, contribution)
                    except (HTTPError, OSError, RuntimeError, ValueError) as error:
                        st.error(f"Không lưu được dữ liệu: {error}")
                    else:
                        st.success(
                            "Đã lưu dữ liệu chờ duyệt vào GitHub. "
                            "Dữ liệu chưa được dùng để huấn luyện model."
                        )
