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
TRAIN_SAMPLE_SIZE = 5000
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


def select_training_sample(data, sample_size=TRAIN_SAMPLE_SIZE, random_state=TRAIN_SAMPLE_SEED):
    if len(data) <= sample_size:
        return data.reset_index(drop=True).copy()
    return data.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def save_processed_data(data, report=None):
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    public_sample = select_training_sample(data)
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
    print(
        f"Huấn luyện trên {len(data):,}/{full_data_rows:,} tin "
        f"(seed={TRAIN_SAMPLE_SEED})."
    )

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
