from pathlib import Path
import math
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Patch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "presentation"
DATA_PATH = ROOT / "data" / "processed" / "tphcm_cleaned.csv"
METRICS_PATH = ROOT / "outputs" / "model_comparison.csv"
CLEANING_PATH = ROOT / "outputs" / "cleaning_report.csv"

WIDTH, HEIGHT, DPI = 16, 9, 120
NAVY = "#0B1020"
CARD = "#151F35"
WHITE = "#F7F8FC"
MUTED = "#B7C1D6"
GRID = "#34415D"
CYAN = "#31D6C5"
PURPLE = "#9B7BFF"
ORANGE = "#FFB454"
PINK = "#FF6B91"
GREEN = "#75E0A7"
MODEL_COLORS = {
    "XGBoost": CYAN,
    "Linear Regression (Ridge)": PURPLE,
    "Random Forest": ORANGE,
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.unicode_minus": False,
    "savefig.facecolor": NAVY,
})


def make_slide(kicker: str, title: str, subtitle: str, number: int):
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI, facecolor=NAVY)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis("off")
    bg.add_patch(Circle((0.96, 0.98), 0.25, color=PURPLE, alpha=0.10))
    bg.add_patch(Circle((0.04, 0.05), 0.22, color=CYAN, alpha=0.08))
    bg.text(
        0.07, 0.944, kicker.upper(), color=CYAN, fontsize=10,
        fontweight="bold", va="top",
    )
    bg.text(
        0.07, 0.885, title, color=WHITE, fontsize=27,
        fontweight="bold", va="top",
    )
    bg.text(
        0.07, 0.812, subtitle, color=MUTED, fontsize=11, va="top",
    )
    bg.text(
        0.94, 0.045, f"{number:02d}  /  04", color=MUTED,
        fontsize=10, ha="right",
    )
    return fig, bg


def add_card(ax, x: float, y: float, width: float, height: float):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=CARD, edgecolor="#27344F", linewidth=1,
    ))


def save_slide(fig, filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / filename
    fig.savefig(target, dpi=DPI, facecolor=NAVY)
    plt.close(fig)
    print(f"saved {target} ({WIDTH * DPI}x{HEIGHT * DPI})")


def read_inputs():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu đã làm sạch: {DATA_PATH}"
        )
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy bảng model: {METRICS_PATH}")
    if not CLEANING_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy báo cáo làm sạch: {CLEANING_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH,
        usecols=["price_bil", "property_type"],
        encoding="utf-8-sig",
    )
    metrics = pd.read_csv(METRICS_PATH, encoding="utf-8-sig")
    cleaning = pd.read_csv(CLEANING_PATH, encoding="utf-8-sig")

    required_metrics = {"Model", "MAE_ty", "RMSE_ty", "R2", "Status"}
    if not required_metrics.issubset(metrics.columns):
        raise ValueError("Bảng so sánh model thiếu các cột chỉ số cần thiết.")
    if not (metrics["Status"] == "OK").all():
        raise ValueError(
            "Có model chưa đánh giá thành công; kiểm tra model_comparison.csv."
        )
    if not {"metric", "value"}.issubset(cleaning.columns):
        raise ValueError("Báo cáo làm sạch thiếu cột metric/value.")
    if data.empty or data["price_bil"].isna().any():
        raise ValueError("Dữ liệu giá rao trống hoặc có giá trị thiếu.")

    return data, metrics, cleaning


def draw_overview(cleaning: pd.DataFrame):
    values = cleaning.set_index("metric")["value"].to_dict()
    raw_rows = int(values["raw_rows"])
    clean_rows = int(values["final_rows"])
    invalid_price = int(values["missing_or_invalid_price_removed"])
    retained = clean_rows / raw_rows * 100

    fig, bg = make_slide(
        "Đồ án dự báo giá nhà  |  TP. Hồ Chí Minh",
        "Từ tin rao đến dự báo",
        "Toàn bộ CSV được làm sạch; model cuối huấn luyện trên mọi dòng có giá hợp lệ.",
        1,
    )
    cards = [
        (0.08, "DỮ LIỆU GỐC", f"{raw_rows:,}", "dòng CSV", PURPLE),
        (0.375, "SAU LÀM SẠCH", f"{clean_rows:,}", "tin có giá hợp lệ", CYAN),
        (0.67, "TỶ LỆ GIỮ LẠI", f"{retained:.1f}%", "sau kiểm tra giá", ORANGE),
    ]
    for x, label, value, note, color in cards:
        add_card(bg, x, 0.47, 0.25, 0.235)
        bg.add_patch(FancyBboxPatch(
            (x, 0.47), 0.012, 0.235,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor=color, edgecolor="none",
        ))
        bg.text(x + 0.03, 0.655, label, color=MUTED, fontsize=10, va="top")
        bg.text(
            x + 0.03, 0.595, value, color=color, fontsize=28,
            fontweight="bold", va="top",
        )
        bg.text(x + 0.03, 0.515, note, color=WHITE, fontsize=11, va="top")

    bg.text(0.09, 0.355, "QUY TRÌNH", color=CYAN, fontsize=10, fontweight="bold")
    steps = [
        ("01", "CSV", "51.304 tin"),
        ("02", "LÀM SẠCH", f"Loại {invalid_price} tin thiếu/giá lỗi"),
        ("03", "ĐÁNH GIÁ", "Holdout 20% • random_state=42"),
        ("04", "MÔ HÌNH", "Fit cuối trên 51.132 tin"),
    ]
    positions = [0.09, 0.31, 0.55, 0.78]
    for idx, (number, label, note) in enumerate(steps):
        x = positions[idx]
        bg.text(x, 0.275, number, color=PURPLE, fontsize=11, fontweight="bold")
        bg.text(x + 0.04, 0.275, label, color=WHITE, fontsize=12, fontweight="bold")
        bg.text(x, 0.22, note, color=MUTED, fontsize=9)
        if idx < len(steps) - 1:
            bg.annotate(
                "", xy=(positions[idx + 1] - 0.025, 0.285), xytext=(x + 0.15, 0.285),
                arrowprops={"arrowstyle": "->", "color": GRID, "lw": 1.5},
            )
    bg.text(
        0.09, 0.105,
        "Lưu ý: 2.778 giá trị đặc trưng bất hợp lý được chuyển thành thiếu, không đồng nghĩa với xóa 2.778 dòng.",
        color=MUTED, fontsize=9,
    )
    save_slide(fig, "01_tong_quan_du_lieu.png")


def draw_model_comparison(metrics: pd.DataFrame, test_rows: int):
    order = ["XGBoost", "Linear Regression (Ridge)", "Random Forest"]
    table = metrics.set_index("Model").reindex(order).dropna(
        subset=["MAE_ty", "RMSE_ty", "R2"]
    )
    if table.empty:
        raise ValueError("Không có kết quả model hợp lệ để vẽ.")

    labels = ["XGB", "Ridge", "RF"]
    colors = [MODEL_COLORS[name] for name in table.index]
    legend_handles = [
        Patch(facecolor=MODEL_COLORS[name], edgecolor="none", label=label)
        for name, label in zip(
            table.index,
            ["XGBoost", "Linear Regression (Ridge)", "Random Forest"],
        )
    ]

    fig, bg = make_slide(
        "Kết quả mô hình",
        "XGBoost đứng đầu — nhưng độ chính xác còn hạn chế",
        f"So sánh trên cùng {test_rows:,} tin holdout · MAE/RMSE tính bằng tỷ VND · thấp hơn là tốt hơn.",
        2,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.775),
        ncol=3,
        frameon=True,
        facecolor=CARD,
        edgecolor="#27344F",
        labelcolor=WHITE,
        fontsize=9,
        handlelength=1.2,
        columnspacing=1.8,
    )

    panels = [
        (0.07, "MAE  ·  SAI SỐ TUYỆT ĐỐI TB", "MAE_ty", 2, "Tỷ VND"),
        (0.365, "RMSE  ·  NHẠY VỚI SAI SỐ LỚN", "RMSE_ty", 2, "Tỷ VND"),
        (0.66, "R²  ·  ĐỘ GIẢI THÍCH", "R2", 3, "Điểm R²"),
    ]
    positions = np.arange(len(table))
    short_labels = labels[:len(table)]

    for x, heading, metric_name, decimals, unit in panels:
        add_card(bg, x, 0.285, 0.27, 0.42)
        bg.text(
            x + 0.02, 0.665, heading, color=MUTED,
            fontsize=8, fontweight="bold",
        )
        axis = fig.add_axes([x + 0.035, 0.345, 0.20, 0.27], facecolor="none")
        values = table[metric_name].to_numpy(dtype=float)
        axis.bar(
            positions, values, color=colors, width=0.58, edgecolor="none",
        )
        value_span = max(float(np.max(np.abs(values))), 0.1)
        lower = min(0.0, float(values.min()) - value_span * 0.32)
        upper = max(0.0, float(values.max()) + value_span * 0.62)
        axis.set_ylim(lower, upper)
        axis.axhline(0, color=GRID, linewidth=1)
        axis.set_xticks(positions, labels=short_labels, color=WHITE, fontsize=8)
        axis.set_ylabel(unit, color=MUTED, fontsize=8, labelpad=5)
        axis.tick_params(axis="y", colors=MUTED, labelsize=7, length=0)
        axis.tick_params(axis="x", length=0, pad=6)
        axis.grid(axis="y", color=GRID, alpha=0.65, linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)

        for index, value in enumerate(values):
            label_y = value + value_span * 0.055 if value >= 0 else value - value_span * 0.055
            vertical_alignment = "bottom" if value >= 0 else "top"
            axis.text(
                index,
                label_y,
                f"{value:.{decimals}f}\nn={test_rows:,}",
                color=WHITE,
                fontsize=7,
                ha="center",
                va=vertical_alignment,
                fontweight="bold",
            )

    bg.text(
        0.09, 0.105,
        "N trên cột là số tin holdout dùng để chấm từng model; R² gần 0 cho thấy độ giải thích còn thấp.",
        color=ORANGE, fontsize=9,
    )
    save_slide(fig, "02_so_sanh_mo_hinh.png")


def draw_price_distribution(data: pd.DataFrame):
    prices = pd.to_numeric(data["price_bil"], errors="coerce").dropna()
    prices = prices[prices > 0]
    median = float(prices.median())
    p90 = float(prices.quantile(0.9))
    p99 = float(prices.quantile(0.99))
    minimum = max(float(prices.min()), 0.01)
    maximum = float(prices.max())
    edges = np.geomspace(minimum, maximum, 38)

    fig, bg = make_slide(
        "Khám phá dữ liệu",
        "Giá rao trải rộng qua nhiều phân khúc",
        f"Phân bố của {len(prices):,} tin có giá; trục ngang logarit để nhìn rõ cả phân khúc thấp và cao.",
        3,
    )
    add_card(bg, 0.08, 0.19, 0.59, 0.53)
    axis = fig.add_axes([0.125, 0.265, 0.49, 0.37], facecolor="none")
    weights = np.full(len(prices), 100 / len(prices))
    _, _, patches = axis.hist(prices, bins=edges, weights=weights, edgecolor="none")
    palette = [CYAN, "#42C7D5", PURPLE, "#B16CFF", PINK, ORANGE]
    for index, patch in enumerate(patches):
        patch.set_facecolor(palette[min(index * len(palette) // len(patches), len(palette) - 1)])
        patch.set_alpha(0.88)
    axis.set_xscale("log")
    axis.set_xlim(minimum, maximum)
    axis.set_ylabel("% tin rao", color=MUTED, fontsize=9)
    axis.set_xlabel("Giá rao (tỷ VND) · thang log", color=MUTED, fontsize=9, labelpad=8)
    axis.tick_params(colors=MUTED, labelsize=8, length=0)
    axis.grid(axis="y", color=GRID, alpha=0.7, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.axvline(median, color=WHITE, linestyle="--", linewidth=1.8)
    axis.axvline(p90, color=ORANGE, linestyle="--", linewidth=1.8)
    axis.text(
        0.03, 0.94, f"Trung vị  {median:,.1f} tỷ",
        transform=axis.transAxes, color=WHITE, fontsize=9, va="top",
    )
    axis.text(
        0.03, 0.85, f"P90  {p90:,.1f} tỷ",
        transform=axis.transAxes, color=ORANGE, fontsize=9, va="top",
    )

    stats = [
        ("SỐ TIN", f"{len(prices):,}", CYAN),
        ("TRUNG VỊ", f"{median:,.1f} tỷ", PURPLE),
        ("P90", f"{p90:,.1f} tỷ", ORANGE),
        ("P99", f"{p99:,.1f} tỷ", PINK),
    ]
    for index, (label, value, color) in enumerate(stats):
        y = 0.59 - index * 0.115
        add_card(bg, 0.72, y, 0.20, 0.085)
        bg.text(0.745, y + 0.055, label, color=MUTED, fontsize=8, va="center")
        bg.text(
            0.745, y + 0.022, value, color=color,
            fontsize=14, fontweight="bold", va="center",
        )
    bg.text(
        0.09, 0.105,
        "Đây là giá đăng trong tin rao, không phải giá giao dịch đã xác nhận.",
        color=ORANGE, fontsize=10,
    )
    save_slide(fig, "03_phan_bo_gia_rao.png")


def draw_property_medians(data: pd.DataFrame):
    summary = (
        data.groupby("property_type")["price_bil"]
        .agg(count="count", median="median")
        .query("count >= 100")
        .sort_values("median", ascending=True)
    )
    if summary.empty:
        raise ValueError("Không có nhóm loại bất động sản đủ dữ liệu để vẽ.")
    labels_map = {
        "dat": "Đất",
        "can ho chung cu": "Căn hộ chung cư",
        "nha rieng": "Nhà riêng",
        "kho, nha xuong": "Kho / nhà xưởng",
        "khach san": "Khách sạn",
        "nha tro": "Nhà trọ",
    }
    labels = [labels_map.get(value, value.replace("_", " ").title()) for value in summary.index]
    values = summary["median"].to_numpy()
    colors = [CYAN, PURPLE, PINK, ORANGE, GREEN, "#6DB5FF"]
    colors = colors[-len(summary):]

    fig, bg = make_slide(
        "Phân khúc bất động sản",
        "Giá trung vị khác nhau theo loại tin",
        "Chỉ hiển thị nhóm có từ 100 tin; các phân khúc không đồng nhất nên không so sánh như cùng một loại tài sản.",
        4,
    )
    add_card(bg, 0.08, 0.18, 0.84, 0.55)
    axis = fig.add_axes([0.19, 0.255, 0.57, 0.39], facecolor="none")
    positions = np.arange(len(summary))
    axis.barh(positions, values, color=colors, height=0.58, edgecolor="none")
    axis.set_yticks(positions, labels=labels, color=WHITE, fontsize=10)
    axis.set_xlim(0, float(values.max()) * 1.38)
    axis.set_xlabel("Trung vị giá rao (tỷ VND)", color=MUTED, fontsize=9, labelpad=8)
    axis.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    axis.grid(axis="x", color=GRID, alpha=0.7, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for index, (value, count) in enumerate(zip(values, summary["count"])):
        axis.text(
            value + float(values.max()) * 0.025, index,
            f"{value:,.1f} tỷ   ·   n={int(count):,}",
            color=WHITE, fontsize=9, va="center", fontweight="bold",
        )
    bg.text(
        0.105, 0.105,
        "Trung vị hạn chế ảnh hưởng của các tin giá cực cao; vẫn cần kiểm tra chất lượng dữ liệu và nguồn tin.",
        color=ORANGE, fontsize=9,
    )
    save_slide(fig, "04_gia_trung_vi_theo_loai.png")


def main():
    data, metrics, cleaning = read_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_overview(cleaning)
    test_rows = math.ceil(len(data) * 0.20)
    draw_model_comparison(metrics, test_rows)
    draw_price_distribution(data)
    draw_property_medians(data)

    slide_names = [
        "01_tong_quan_du_lieu.png",
        "02_so_sanh_mo_hinh.png",
        "03_phan_bo_gia_rao.png",
        "04_gia_trung_vi_theo_loai.png",
    ]
    archive_path = OUTPUT_DIR / "bo_bieu_do_powerpoint.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for slide_name in slide_names:
            archive.write(OUTPUT_DIR / slide_name, arcname=slide_name)
    print(f"saved {archive_path}")


if __name__ == "__main__":
    main()
