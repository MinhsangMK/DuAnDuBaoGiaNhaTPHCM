from pathlib import Path
import json
import sys
import textwrap
import uuid

ROOT = Path(__file__).resolve().parent

PIPELINE_SOURCE = r'''
from pathlib import Path
import math
import re

import joblib
import numpy as np
import pandas as pd
from unidecode import unidecode

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "TPHCM.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "tphcm_cleaned.csv"
REFERENCE_PATH = PROJECT_ROOT / "data" / "demo" / "tphcm_sample_5000.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "gia_nha_tphcm.joblib"
SCORES_PATH = PROJECT_ROOT / "outputs" / "model_comparison.csv"
TRAIN_SAMPLE_SIZE = None
PUBLIC_SAMPLE_SIZE = 5000
TRAIN_SAMPLE_SEED = 42
PUBLIC_SAMPLE_COLUMNS = [
    "price_bil", "area_m2", "bedrooms", "floors", "frontage",
    "district", "property_type",
]

NUMERIC_COLUMNS = ["area_m2", "bedrooms", "floors", "frontage"]
CATEGORICAL_COLUMNS = ["district", "ward", "property_type"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ["text"]

ALIASES = {
    "price": ["price", "gia", "gia_ban", "muc_gia"],
    "title": ["title", "tieu_de", "ten_tin"],
    "description": ["description", "mo_ta", "noi_dung", "mo_ta_chi_tiet"],
    "agent_name": ["agent_name", "agent", "ten_moi_gioi", "ten_nguoi_dang"],
    "location": ["location", "address", "dia_chi", "vi_tri"],
    "area": ["area", "dien_tich", "acreage", "dt"],
    "bedrooms": ["bedrooms", "bedroom", "phong_ngu", "so_pn", "so_phong_ngu"],
    "floors": ["floors", "floor", "so_tang", "tang"],
    "width": ["width", "frontage", "mat_tien", "ngang", "chieu_rong"],
    "property_type": ["property_type", "loai_bds", "loai_bat_dong_san", "type"],
}


def normalize_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = unidecode(str(value).lower().replace("²", "2"))
    return re.sub(r"\s+", " ", text).strip()


def normalize_column(value):
    return re.sub(r"[^a-z0-9]+", "_", unidecode(str(value)).lower()).strip("_")


def find_column(data, role):
    for name in ALIASES.get(role, [role]):
        if name in data.columns:
            return name
    return None


def load_raw_data(path=RAW_DATA_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy CSV: {path}")

    last_error = None
    for encoding in ("utf-8-sig", "cp1258"):
        try:
            data = pd.read_csv(path, sep=",", engine="python", encoding=encoding)
            data.columns = [normalize_column(c) for c in data.columns]
            return data
        except UnicodeDecodeError as error:
            last_error = error

    raise last_error


def to_number(value):
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass

    if isinstance(value, (int, float, np.number)):
        return float(value)

    match = re.search(r"-?\d+(?:[.,]\d+)?", normalize_text(value))
    if not match:
        return np.nan
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return np.nan


def scale_numeric_price(value):
    if not np.isfinite(value):
        return np.nan
    if value >= 100_000_000:   # giá nhập bằng VND
        return value / 1_000_000_000
    if value >= 100:           # giá nhập bằng triệu VND; ví dụ 7500 = 7,5 tỷ
        return value / 1000
    return value               # giá đã là tỷ VND


def parse_price(value):
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass

    if isinstance(value, (int, float, np.number)):
        return scale_numeric_price(float(value))

    text = normalize_text(value)

    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ty|ti)\b", text)
    if match:
        total = to_number(match.group(1))
        remainder = text[match.end():]
        extra = re.match(r"\s*(\d+(?:[.,]\d+)?)\s*(?:trieu|tr)\b", remainder)
        if extra:
            total += to_number(extra.group(1)) / 1000
        return total

    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:trieu|tr)\b", text)
    if match:
        return to_number(match.group(1)) / 1000

    plain = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*", text)
    if plain:
        return scale_numeric_price(to_number(plain.group(1)))

    digits = re.sub(r"\D", "", text)
    return scale_numeric_price(float(digits)) if digits else np.nan


def _series(data, role):
    column = find_column(data, role)
    if column is None:
        return pd.Series("", index=data.index, dtype=object)
    return data[column]


def _text_series(data, role):
    return _series(data, role).fillna("").astype(str)


def extract_area(text):
    text = normalize_text(text)

    match = re.search(
        r"(?:dien\s*tich|d\.?\s*t\.?)\D{0,12}(\d+(?:[.,]\d+)?)\s*(?:m2|m\^2)?",
        text,
    )
    if match:
        return to_number(match.group(1))

    match = re.search(r"\b(\d+(?:[.,]\d+)?)\s*m(?:2|\^2)\b", text)
    if match:
        return to_number(match.group(1))

    match = re.search(r"\b(\d+(?:[.,]\d+)?)\s*[x*]\s*(\d+(?:[.,]\d+)?)\s*m?\b", text)
    if match:
        return to_number(match.group(1)) * to_number(match.group(2))

    return np.nan


def extract_bedrooms(text):
    match = re.search(r"\b(\d{1,2})\s*(?:pn|phong\s*ngu)\b", normalize_text(text))
    return float(match.group(1)) if match else np.nan


def extract_floors(text):
    text = normalize_text(text)
    match = re.search(r"\b(\d{1,2})\s*(tang|lau)\b", text)
    if match:
        count = int(match.group(1))
        return float(count + (1 if "tret" in text and match.group(2) == "lau" else 0))
    return 1.0 if "tret" in text else np.nan


def extract_frontage(text):
    text = normalize_text(text)
    match = re.search(
        r"\b(?:mat\s*tien|ngang|rong)\D{0,8}(\d+(?:[.,]\d+)?)\s*m?\b",
        text,
    )
    if match:
        return to_number(match.group(1))

    match = re.search(r"\b(\d+(?:[.,]\d+)?)\s*[x*]\s*(\d+(?:[.,]\d+)?)\s*m?\b", text)
    return to_number(match.group(1)) if match else np.nan


DISTRICT_NAMES = [
    ("thanh pho thu duc", "TP_Thu_Duc"),
    ("thu duc", "TP_Thu_Duc"),
    ("binh thanh", "Binh_Thanh"),
    ("binh tan", "Binh_Tan"),
    ("go vap", "Go_Vap"),
    ("phu nhuan", "Phu_Nhuan"),
    ("tan binh", "Tan_Binh"),
    ("tan phu", "Tan_Phu"),
    ("binh chanh", "Binh_Chanh"),
    ("can gio", "Can_Gio"),
    ("cu chi", "Cu_Chi"),
    ("hoc mon", "Hoc_Mon"),
    ("nha be", "Nha_Be"),
]


def extract_district(value):
    text = normalize_text(value).replace("_", " ")
    match = re.search(r"\b(?:quan|q|district)\.?\s*0?(\d{1,2})\b", text)
    if match:
        return f"Q{int(match.group(1))}"

    for name, code in DISTRICT_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", text):
            return code

    return "__missing__"


def extract_ward(value):
    text = normalize_text(value)
    match = re.search(
        r"\b(?:phuong|p)\.?\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,3})"
        r"(?=\s*(?:,|;|\||\bquan\b|\bq\.?\s*\d|\btp\.?|$))",
        text,
    )
    if match:
        return "P_" + re.sub(r"\s+", "_", match.group(1).strip())

    match = re.search(
        r"\bxa\.?\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,3})"
        r"(?=\s*(?:,|;|\||\bquan\b|\btp\.?|$))",
        text,
    )
    if match:
        return "X_" + re.sub(r"\s+", "_", match.group(1).strip())

    return "__missing__"


def clean_description(value):
    text = normalize_text(value)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"(?<!\d)0\d[\d .-]{7,}\d(?!\d)", " ", text)
    text = re.sub(
        r"\b\d+(?:[.,]\d+)?\s*(?:ty|ti|trieu|tr)\b"
        r"(?:\s*\d+(?:[.,]\d+)?\s*(?:trieu|tr))?",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def clean_data(raw):
    if find_column(raw, "price") is None:
        raise ValueError(f"Không tìm thấy cột giá. Các cột đọc được: {list(raw.columns)}")

    title = _text_series(raw, "title")
    description = _text_series(raw, "description")
    location = _text_series(raw, "location")
    agent_names = _text_series(raw, "agent_name").map(normalize_text)
    full_text = pd.Series(
        [
            re.sub(re.escape(agent), " ", f"{listing_title} {listing_description}")
            if agent else f"{listing_title} {listing_description}"
            for listing_title, listing_description, agent
            in zip(title, description, agent_names)
        ],
        index=raw.index,
    )
    context = full_text + " " + location

    price_column = find_column(raw, "price")
    price = raw[price_column].map(parse_price)

    area_column = find_column(raw, "area")
    area = (
        raw[area_column].map(to_number)
        if area_column else pd.Series(np.nan, index=raw.index)
    )
    area = area.where(area.notna(), full_text.map(extract_area))

    bedrooms_column = find_column(raw, "bedrooms")
    bedrooms = (
        raw[bedrooms_column].map(to_number)
        if bedrooms_column else pd.Series(np.nan, index=raw.index)
    )
    bedrooms = bedrooms.where(bedrooms.notna(), full_text.map(extract_bedrooms))

    floors_column = find_column(raw, "floors")
    floors = (
        raw[floors_column].map(to_number)
        if floors_column else pd.Series(np.nan, index=raw.index)
    )
    floors = floors.where(floors.notna(), full_text.map(extract_floors))

    width_column = find_column(raw, "width")
    frontage = (
        raw[width_column].map(to_number)
        if width_column else pd.Series(np.nan, index=raw.index)
    )
    frontage = frontage.where(frontage.notna(), full_text.map(extract_frontage))

    district = location.map(extract_district)
    district_fallback = context.map(extract_district)
    district = district.where(district.ne("__missing__"), district_fallback)

    ward = location.map(extract_ward)
    ward_fallback = context.map(extract_ward)
    ward = ward.where(ward.ne("__missing__"), ward_fallback)

    property_type = _series(raw, "property_type").map(normalize_text)
    property_type = property_type.replace("", "__missing__")

    cleaned = pd.DataFrame({
        "price_bil": price,
        "area_m2": area,
        "bedrooms": bedrooms,
        "floors": floors,
        "frontage": frontage,
        "district": district,
        "ward": ward,
        "property_type": property_type,
        "text": full_text.map(clean_description),
    }, index=raw.index)

    cleaned["text"] = cleaned["text"].replace("", "khong mo ta")

    duplicate_mask = raw.duplicated(keep="last")
    duplicate_rows = int(duplicate_mask.sum())
    cleaned = cleaned.loc[~duplicate_mask].copy()

    invalid_price = (
        cleaned["price_bil"].isna()
        | ~cleaned["price_bil"].between(0.1, 1000)
    )
    invalid_price_rows = int(invalid_price.sum())
    cleaned = cleaned.loc[~invalid_price].copy()

    invalid_feature_values = 0
    bounds = {
        "area_m2": (8, 5000),
        "bedrooms": (0, 30),
        "floors": (1, 100),
        "frontage": (0.5, 100),
    }
    for column, (low, high) in bounds.items():
        invalid = cleaned[column].notna() & ~cleaned[column].between(low, high)
        invalid_feature_values += int(invalid.sum())
        cleaned.loc[invalid, column] = np.nan

    # Gắn cờ giá ngoại lai trên log-giá để báo cáo; không xóa các giao dịch hợp lệ.
    price_outlier_flags = 0
    if len(cleaned) >= 20:
        log_price = np.log1p(cleaned["price_bil"])
        q1, q3 = log_price.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            price_outlier_flags = int(
                ((log_price < q1 - 3 * iqr) | (log_price > q3 + 3 * iqr)).sum()
            )

    cleaned = cleaned.reset_index(drop=True)

    report = pd.DataFrame([
        {"metric": "raw_rows", "value": len(raw)},
        {"metric": "duplicate_rows_removed", "value": duplicate_rows},
        {"metric": "missing_or_invalid_price_removed", "value": invalid_price_rows},
        {"metric": "impossible_numeric_values_set_missing", "value": invalid_feature_values},
        {"metric": "price_outlier_flags_iqr_log", "value": price_outlier_flags},
        {"metric": "final_rows", "value": len(cleaned)},
    ])
    return cleaned, report


def select_training_sample(
    data, sample_size=TRAIN_SAMPLE_SIZE, random_state=TRAIN_SAMPLE_SEED
):
    if sample_size is None or len(data) <= sample_size:
        return data.reset_index(drop=True).copy()
    if sample_size < 1:
        raise ValueError("sample_size phải là số nguyên dương hoặc None.")
    return data.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def save_processed_data(data, report=None):
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    public_sample = select_training_sample(
        data, sample_size=PUBLIC_SAMPLE_SIZE
    )
    public_sample[PUBLIC_SAMPLE_COLUMNS].to_csv(
        REFERENCE_PATH, index=False, encoding="utf-8-sig"
    )

    if report is not None:
        report.to_csv(
            PROJECT_ROOT / "outputs" / "cleaning_report.csv",
            index=False,
            encoding="utf-8-sig",
        )


def load_processed_data():
    if PROCESSED_PATH.is_file():
        return pd.read_csv(PROCESSED_PATH, encoding="utf-8-sig")

    raw = load_raw_data()
    data, report = clean_data(raw)
    save_processed_data(data, report)
    return data


def make_preprocessor():
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ("scaler", RobustScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", encoder),
    ])
    text_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=8000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_COLUMNS),
        ("categorical", categorical_pipe, CATEGORICAL_COLUMNS),
        ("text", text_vectorizer, "text"),
    ])


MODEL_NAMES = ["Linear Regression (Ridge)", "Random Forest"]
if XGBRegressor is not None:
    MODEL_NAMES.append("XGBoost")


def make_model_pipeline(name, use_gpu=False):
    if name == "Linear Regression (Ridge)":
        estimator = Ridge(alpha=10.0, solver="lsqr")
    elif name == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=40,
            min_samples_leaf=4,
            max_features="sqrt",
            max_depth=20,
            n_jobs=-1,
            random_state=42,
        )
    elif name == "Extra Trees":
        estimator = ExtraTreesRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            max_features=0.7,
            n_jobs=-1,
            random_state=42,
        )
    elif name == "XGBoost" and XGBRegressor is not None:
        estimator = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=180,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Mô hình không được hỗ trợ: {name}")

    return Pipeline([
        ("features", make_preprocessor()),
        ("model", estimator),
    ])


def make_target_model(name, use_gpu=False):
    return TransformedTargetRegressor(
        regressor=make_model_pipeline(name, use_gpu=use_gpu),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def fit_model_with_fallback(name, X, y):
    if name == "XGBoost":
        try:
            model = make_target_model(name, use_gpu=True)
            model.fit(X, y)
            model.regressor_.named_steps["model"].get_booster().set_param(
                {"device": "cpu"}
            )
            print("XGBoost đang chạy bằng GPU CUDA.")
            return model, "GPU"
        except Exception as error:
            print(f"Không chạy được XGBoost GPU ({error}); chuyển sang CPU.")

    model = make_target_model(name, use_gpu=False)
    model.fit(X, y)
    return model, "CPU"


def save_metric_plot(scores):
    valid = scores.dropna(subset=["RMSE_ty"])
    if valid.empty:
        return

    import matplotlib.pyplot as plt

    metrics = [("MAE_ty", "MAE (tỷ đồng)"),
               ("RMSE_ty", "RMSE (tỷ đồng)"),
               ("R2", "R²")]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))

    for axis, (column, label) in zip(axes, metrics):
        axis.bar(valid["Model"], valid[column])
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=35)

    figure.tight_layout()
    output = PROJECT_ROOT / "outputs" / "model_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def train_and_save(data=None):
    data = load_processed_data() if data is None else data.copy()
    if data.empty:
        raise ValueError("Không còn dòng dữ liệu hợp lệ sau tiền xử lý.")

    for column in FEATURE_COLUMNS:
        if column not in data:
            data[column] = np.nan

    data["price_bil"] = pd.to_numeric(data["price_bil"], errors="coerce")
    data = data.dropna(subset=["price_bil"]).copy()
    if data.empty:
        raise ValueError("Không có giá hợp lệ để huấn luyện.")

    full_data_rows = len(data)
    data = select_training_sample(data)
    print(f"Huấn luyện và đánh giá trên toàn bộ {len(data):,} tin hợp lệ.")

    X = data[FEATURE_COLUMNS].copy()
    y = data["price_bil"].astype(float)

    for column in NUMERIC_COLUMNS:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        X[column] = X[column].fillna("__missing__").astype(str)
    X["text"] = X["text"].fillna("khong mo ta").astype(str)

    results = []
    best_name = MODEL_NAMES[0]
    validation_mae = None

    if len(data) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for name in MODEL_NAMES:
            print(f"Đang đánh giá: {name}...")
            try:
                model, device_used = fit_model_with_fallback(name, X_train, y_train)
                prediction = np.maximum(model.predict(X_test), 0)

                results.append({
                    "Model": name,
                    "MAE_ty": mean_absolute_error(y_test, prediction),
                    "RMSE_ty": math.sqrt(mean_squared_error(y_test, prediction)),
                    "R2": r2_score(y_test, prediction),
                    "Device": device_used,
                    "Status": "OK",
                })
            except Exception as error:
                print(f"Bỏ qua {name}: {error}")
                results.append({
                    "Model": name,
                    "MAE_ty": np.nan,
                    "RMSE_ty": np.nan,
                    "R2": np.nan,
                    "Device": "failed",
                    "Status": str(error),
                })

        scores = pd.DataFrame(results)
        valid = scores.dropna(subset=["RMSE_ty"])
        if valid.empty:
            raise RuntimeError("Không mô hình nào huấn luyện thành công.")

        best_name = valid.sort_values("RMSE_ty").iloc[0]["Model"]
        validation_mae = float(
            valid.loc[valid["Model"] == best_name, "MAE_ty"].iloc[0]
        )
    else:
        scores = pd.DataFrame([
            {
                "Model": name,
                "MAE_ty": np.nan,
                "RMSE_ty": np.nan,
                "R2": np.nan,
                "Device": "not_evaluated",
                "Status": "Cần ít nhất 10 tin để đánh giá holdout đáng tin cậy",
            }
            for name in MODEL_NAMES
        ])
        print("Ít hơn 10 tin: chưa thể đánh giá mô hình độc lập; sẽ lưu Ridge.")

    candidate_names = [best_name] + [
        name for name in MODEL_NAMES if name != best_name
    ]
    final_model = None
    final_device = None
    for name in candidate_names:
        try:
            candidate, final_device = fit_model_with_fallback(name, X, y)
            final_model = candidate
            best_name = name
            break
        except Exception as error:
            print(f"Không thể huấn luyện model cuối {name}: {error}")

    if final_model is None:
        raise RuntimeError("Không tạo được model cuối.")

    scores = scores.sort_values("RMSE_ty", na_position="last").reset_index(drop=True)
    PROJECT_ROOT.joinpath("outputs").mkdir(parents=True, exist_ok=True)
    PROJECT_ROOT.joinpath("models").mkdir(parents=True, exist_ok=True)

    scores.to_csv(SCORES_PATH, index=False, encoding="utf-8-sig")
    save_metric_plot(scores)

    joblib.dump({
        "model": final_model,
        "model_name": best_name,
        "features": FEATURE_COLUMNS,
        "train_rows": len(data),
        "available_rows": full_data_rows,
        "validation_mae_ty": validation_mae,
        "device": final_device,
        "price_unit": "tỷ VND",
    }, MODEL_PATH, compress=3)

    print(f"Model tốt nhất: {best_name}")
    print(f"Đã lưu model: {MODEL_PATH}")
    return scores, best_name


def make_prediction_frame(
    area, bedrooms, floors, frontage,
    district, ward, property_type, description=""
):
    district_code = extract_district(district)
    if district_code == "__missing__" and normalize_text(district):
        district_code = normalize_text(district).replace(" ", "_")

    ward_code = extract_ward(ward)
    if ward_code == "__missing__" and normalize_text(ward):
        ward_code = normalize_text(ward).replace(" ", "_")

    row = {
        "area_m2": to_number(area),
        "bedrooms": to_number(bedrooms),
        "floors": to_number(floors),
        "frontage": to_number(frontage),
        "district": district_code,
        "ward": ward_code,
        "property_type": normalize_text(property_type) or "__missing__",
        "text": clean_description(description) or "khong mo ta",
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
'''

APP_SOURCE = r'''
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


st.set_page_config(
    page_title="Đề tài dự báo giá nhà TP.HCM · Nhóm 7",
    page_icon="🏠",
    layout="wide",
)
st.title("🏠 Đề tài dự báo giá nhà TP.HCM — Nhóm 7")
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
    st.markdown("#### 🏡 Thông tin dự báo giá nhà")
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
'''

REQUIREMENTS = """\
numpy
pandas
scikit-learn
xgboost
Unidecode
joblib
streamlit
matplotlib
jupyter
ipykernel
"""

GITIGNORE = """\
.venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
data/raw/*.csv
data/processed/*.csv
.streamlit/secrets.toml
"""

README = """\
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
"""


def write_file(relative_path, content):
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return path


def make_cell(kind, source):
    language = "markdown" if kind == "markdown" else "python"
    metadata = {"id": uuid.uuid4().hex[:8], "language": language}
    if kind == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": metadata,
            "source": source.splitlines(keepends=True),
        }
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write_notebook(relative_path, cells):
    notebook = {
        "cells": [make_cell(kind, textwrap.dedent(source).lstrip())
                  for kind, source in cells],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def create_notebooks():
    setup = """
            from pathlib import Path
            import sys

            PROJECT_ROOT = Path.cwd().resolve()
            for candidate in (PROJECT_ROOT, *PROJECT_ROOT.parents):
                if (candidate / "src" / "pipeline.py").is_file():
                    PROJECT_ROOT = candidate
                    break
            else:
                raise FileNotFoundError("Không tìm thấy thư mục dự án có src/pipeline.py")
            sys.path.insert(0, str(PROJECT_ROOT))
    """

    write_notebook("notebooks/01_kham_pha_du_lieu.ipynb", [
        ("markdown", """
            # 01 — Khám phá dữ liệu

            **Mục tiêu:** hiểu quy mô, độ thiếu, phân bố giá và phạm vi địa lý trước khi chọn mô hình.

            Giá được quy về **tỷ VND**. Để tránh lộ dữ liệu cá nhân khi lưu notebook, các bảng chỉ hiển thị đặc trưng đã làm sạch, không hiển thị mô tả hoặc địa chỉ chi tiết.
            """),
        ("code", setup + """
            from IPython.display import display
            import matplotlib.pyplot as plt
            import pandas as pd
            from src.pipeline import (
                clean_data,
                find_column,
                load_raw_data,
                parse_price,
            )

            raw = load_raw_data()
            cleaned, cleaning_report = clean_data(raw)
            print(f"Tin đọc được: {len(raw):,}")
            print(f"Tin còn lại sau làm sạch: {len(cleaned):,}")
            print(f"Số trường trong CSV: {len(raw.columns)}")

            schema = pd.DataFrame({
                "Vai trò": ["Giá", "Diện tích", "Phòng ngủ", "Số tầng", "Mặt tiền", "Địa chỉ"],
                "Cột CSV": [
                    find_column(raw, role)
                    for role in ["price", "area", "bedrooms", "floors", "width", "location"]
                ],
            })
            display(schema)
            display(cleaning_report)
        """),
        ("markdown", "## Chất lượng và độ phủ dữ liệu\n\nBảng dưới đây chỉ chứa các cột cần cho phân tích; dữ liệu gốc không được in ra notebook."),
        ("code", """
            safe_columns = [
                "price_bil", "area_m2", "bedrooms", "floors", "frontage",
                "district", "property_type",
            ]
            display(cleaned[safe_columns].head(10))

            missing = cleaned[safe_columns].isna().mean().sort_values(ascending=False)
            display(missing.rename("Tỷ lệ thiếu").to_frame())

            print("Số Quận/Huyện chuẩn hóa:", cleaned["district"].nunique())
            print("Số tin chưa chuẩn hóa được Quận/Huyện:",
                  int(cleaned["district"].eq("__missing__").sum()))
            print("Các loại bất động sản phổ biến:")
            display(cleaned["property_type"].value_counts().head(10).to_frame("Số tin"))
        """),
        ("markdown", "## Phân phối giá và quan hệ với diện tích\n\nGiá có đuôi phải dài; biểu đồ giới hạn ở phân vị 99 để phần lớn tin dễ quan sát hơn. Không xóa các tin đắt chỉ vì chúng hiếm."),
        ("code", """
            prices = cleaned["price_bil"].dropna()
            display(prices.describe(percentiles=[.5, .9, .95, .99]).to_frame("Giá (tỷ VND)"))

            upper_price = prices.quantile(.99)
            price_view = cleaned.loc[cleaned["price_bil"] <= upper_price]
            scatter = price_view.dropna(subset=["area_m2"]).sample(
                n=min(5000, price_view["area_m2"].notna().sum()),
                random_state=42,
            )

            figure, axes = plt.subplots(1, 2, figsize=(13, 4))
            axes[0].hist(price_view["price_bil"], bins=40, color="#2878B5")
            axes[0].set(title="Giá rao đến phân vị 99", xlabel="Tỷ VND", ylabel="Số tin")
            axes[1].scatter(scatter["area_m2"], scatter["price_bil"], s=10, alpha=.3)
            axes[1].set(title="Diện tích và giá rao", xlabel="Diện tích (m²)", ylabel="Tỷ VND")
            figure.tight_layout()
            plt.show()

            print("Kết luận: cần xem riêng sai số theo phân khúc; MAE/RMSE gốc chịu ảnh hưởng mạnh từ tin giá rất cao.")
        """),
    ])

    write_notebook("notebooks/02_tien_xu_ly.ipynb", [
        ("markdown", """
            # 02 — Tiền xử lý

            Pipeline trích xuất giá/diện tích/phòng/tầng/mặt tiền, chuẩn hóa một phần Quận/Huyện/Phường/Xã và đánh dấu giá trị số không hợp lý.

            **Nguyên tắc:** giá thiếu hoặc không đọc được bị loại; đặc trưng thiếu được giữ lại để impute bên trong pipeline huấn luyện, tránh rò rỉ thống kê từ tập kiểm tra.
            """),
        ("code", setup + """
            from IPython.display import display
            import matplotlib.pyplot as plt
            from src.pipeline import load_raw_data, clean_data, save_processed_data

            raw = load_raw_data()
            cleaned, report = clean_data(raw)
            save_processed_data(cleaned, report)
            display(report)
            print(f"Dữ liệu đầy đủ đã lưu cục bộ: {len(cleaned):,} dòng")
            print("Tệp làm sạch đầy đủ nằm trong data/processed/ và được loại khỏi Git.")
            print("Bản demo công khai chỉ gồm 7 đặc trưng tổng quát, tối đa 5.000 dòng.")
        """),
        ("markdown", "## Kiểm tra kết quả làm sạch\n\nCác giá trị thiếu trong đặc trưng không bị điền bằng toàn bộ dữ liệu tại bước này; model sẽ học median/category imputation chỉ từ tập train."),
        ("code", """
            safe_columns = [
                "price_bil", "area_m2", "bedrooms", "floors", "frontage",
                "district", "property_type",
            ]
            missing = cleaned[safe_columns].isna().mean().sort_values(ascending=False)
            display(missing.rename("Tỷ lệ thiếu").to_frame())

            numeric = cleaned[["price_bil", "area_m2", "bedrooms", "floors", "frontage"]]
            display(numeric.describe(percentiles=[.5, .9, .99]).T)

            figure, axis = plt.subplots(figsize=(9, 3.5))
            missing.sort_values().plot.barh(ax=axis, color="#2A9D8F")
            axis.set(title="Tỷ lệ thiếu theo đặc trưng", xlabel="Tỷ lệ", ylabel="")
            figure.tight_layout()
            plt.show()
        """),
    ])

    write_notebook("notebooks/03_feature_engineering.ipynb", [
        ("markdown", """
            # 03 — Feature engineering

            Dùng đặc trưng số đã trích xuất, category hành chính/loại bất động sản và TF-IDF cho mô tả. `ColumnTransformer` giữ toàn bộ phép impute, one-hot và vector hóa bên trong pipeline để không fit trước trên tập kiểm tra.

            Notebook khám phá toàn bộ dữ liệu hợp lệ; chỉ bản demo xuất ra mới giới hạn 5.000 dòng.
            """),
        ("code", setup + """
            from IPython.display import display
            from sklearn.feature_extraction.text import TfidfVectorizer
            from src.pipeline import (
                FEATURE_COLUMNS,
                PUBLIC_SAMPLE_COLUMNS,
                PUBLIC_SAMPLE_SIZE,
                TRAIN_SAMPLE_SIZE,
                load_processed_data,
                select_training_sample,
            )

            full_data = load_processed_data()
            data = select_training_sample(full_data)
            print(f"Dữ liệu dùng để huấn luyện: {len(data):,}/{len(full_data):,} dòng")
            print(f"Kích thước mẫu demo công khai: tối đa {PUBLIC_SAMPLE_SIZE:,} dòng")
            print("Đặc trưng dùng trong model:", FEATURE_COLUMNS)
            print("Cột được phép xuất trong bản demo:", PUBLIC_SAMPLE_COLUMNS)
            display(data[[
                "price_bil", "area_m2", "bedrooms", "floors", "frontage",
                "district", "property_type",
            ]].head(10))

            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, max_features=8000,
                sublinear_tf=True, token_pattern=r"(?u)\\b\\w+\\b",
            )
            tfidf = vectorizer.fit_transform(
                data["text"].fillna("khong mo ta").astype(str)
            )
            print(f"Ma trận TF-IDF: {tfidf.shape[0]:,} tin × {tfidf.shape[1]:,} từ/cụm từ")
            print(f"Dữ liệu demo xuất ra: {PROJECT_ROOT / 'data' / 'demo'}")
        """),
        ("markdown", "## Tách tập train/test\n\nGiữ 20% mẫu làm tập kiểm tra; mọi phép biến đổi học tham số trong pipeline trên phần train."),
        ("code", """
            from sklearn.model_selection import train_test_split

            sample_train, sample_test = train_test_split(
                data, test_size=0.2, random_state=42
            )
            print(f"Train: {len(sample_train):,} dòng")
            print(f"Test:  {len(sample_test):,} dòng")
            print("Không dùng giá, ID tin, tọa độ hoặc nội dung liên hệ làm đầu vào trực tiếp.")
        """),
    ])

    write_notebook("notebooks/04_train_model.ipynb", [
        ("markdown", """
            # 04 — Huấn luyện và đánh giá mô hình

            So sánh ba thuật toán trên toàn bộ dữ liệu hợp lệ: Ridge Regression, Random Forest và XGBoost. Tập holdout chiếm 20%; model cuối được fit lại trên toàn bộ dữ liệu. XGBoost thử CUDA rồi tự fallback CPU nếu GPU không dùng được.

            **Chỉ số:** MAE và RMSE tính theo tỷ VND; R² càng gần 1 càng tốt. Kết quả thấp/âm là dấu hiệu cần cải thiện dữ liệu hoặc phân khúc, không nên diễn giải thành giá thẩm định.
            """),
        ("code", setup + """
            from IPython.display import display
            from src.pipeline import MODEL_PATH, SCORES_PATH, train_and_save

            scores, best_model = train_and_save()
            display(scores.style.format({
                "MAE_ty": "{:.2f}",
                "RMSE_ty": "{:.2f}",
                "R2": "{:.3f}",
            }, na_rep="—"))

            valid = scores.dropna(subset=["RMSE_ty"])
            best = valid.loc[valid["Model"] == best_model].iloc[0]
            print(f"Mô hình tốt nhất theo RMSE: {best_model}")
            print(f"Thiết bị: {best['Device']}")
            print(f"MAE: {best['MAE_ty']:.2f} tỷ | RMSE: {best['RMSE_ty']:.2f} tỷ | R²: {best['R2']:.3f}")
            print(f"Bảng điểm: {SCORES_PATH}")
            print(f"Model: {MODEL_PATH}")

            if best["R2"] < 0.5:
                print("Lưu ý: R² còn thấp; kết quả chỉ dùng minh họa học thuật.")
        """),
        ("markdown", "## Đọc kết quả\n\nSo sánh sai số tuyệt đối với phân phối giá ở Notebook 01. Một vài bất động sản rất đắt có thể chi phối RMSE và R²; nên báo cáo thêm sai số theo loại tài sản/khu vực khi cải tiến đồ án."),
        ("code", """
            from IPython.display import Image, display

            plot_path = PROJECT_ROOT / "outputs" / "model_comparison.png"
            if plot_path.is_file():
                display(Image(filename=str(plot_path)))
        """),
    ])


def main():
    for directory in (
        "app", "data/raw", "data/processed", "models",
        "notebooks", "outputs", "src",
    ):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    write_file("src/pipeline.py", PIPELINE_SOURCE)
    write_file("src/__init__.py", "")
    if not (ROOT / "app/app.py").exists():
        write_file("app/app.py", APP_SOURCE)
    write_file("requirements.txt", REQUIREMENTS)
    write_file(".gitignore", GITIGNORE)
    write_file("README.md", README)
    create_notebooks()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.pipeline import (
        clean_data,
        load_raw_data,
        save_processed_data,
        train_and_save,
    )

    raw = load_raw_data()
    cleaned, report = clean_data(raw)
    save_processed_data(cleaned, report)

    print(f"Đọc được {len(raw):,} dòng; giữ lại {len(cleaned):,} dòng.")
    print(report.to_string(index=False))

    scores, best_model = train_and_save(cleaned)
    print("\nSo sánh mô hình:")
    print(scores.to_string(index=False))
    print(f"\nHoàn tất. Model tốt nhất: {best_model}")
    print(f"Web app: {ROOT / 'app' / 'app.py'}")


if __name__ == "__main__":
    main()