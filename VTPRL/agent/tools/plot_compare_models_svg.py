import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "logs" / "compare_models" / "compare_models.csv"
SVG_PATH = ROOT / "logs" / "compare_models" / "compare_models_bars_latex.svg"
PDF_PATH = ROOT / "logs" / "compare_models" / "compare_models_bars_latex.pdf"


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: float, digits: int) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def clean_label(text: str) -> str:
    return text.replace("_", " ")


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_text(x: float, y: float, size: int, text: str, font: str = "/F1") -> str:
    return f"BT {font} {size} Tf 1 0 0 1 {x:.2f} {y:.2f} Tm ({pdf_escape(text)}) Tj ET"


def pdf_text_center(x: float, y: float, size: int, text: str, font: str = "/F1") -> str:
    width = len(text) * size * 0.28
    return pdf_text(x - width / 2, y, size, text, font)


def pdf_rotated_text(x: float, y: float, size: int, angle_deg: float, text: str, font: str = "/F1") -> str:
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return (
        f"BT {font} {size} Tf "
        f"{cos_a:.5f} {sin_a:.5f} {-sin_a:.5f} {cos_a:.5f} {x:.2f} {y:.2f} Tm "
        f"({pdf_escape(text)}) Tj ET"
    )


def pdf_line(x1: float, y1: float, x2: float, y2: float, width: float = 1.0, dash: str = "") -> str:
    dash_cmd = f"{dash} " if dash else ""
    return f"{width:.2f} w {dash_cmd}{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S"


def pdf_rect(x: float, y: float, w: float, h: float, fill_rgb: Tuple[float, float, float]) -> str:
    r, g, b = fill_rgb
    return (
        f"{r:.3f} {g:.3f} {b:.3f} rg "
        f"0 0 0 RG 1.00 w "
        f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re B"
    )


def build_panel_svg(
    title: str,
    ylabel: str,
    values: List[float],
    labels: List[str],
    x0: int,
    y0: int,
    width: int,
    height: int,
    digits: int,
    ymin: float,
    ymax: float,
) -> str:
    left = x0 + 68
    right = x0 + width - 20
    top = y0 + 36
    bottom = y0 + height - 56
    plot_w = right - left
    plot_h = bottom - top
    bar_w = plot_w / max(len(values) * 2.0, 1)
    gap = bar_w
    start_x = left + gap / 2
    grid_values = [ymin + (ymax - ymin) * i / 4 for i in range(5)]
    colors = ["#2d3748", "#718096"]

    parts: List[str] = []
    parts.append(f'<text x="{x0 + width / 2:.1f}" y="{y0 + 18}" text-anchor="middle" class="title">{title}</text>')
    parts.append(f'<text x="{x0 + 8}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 + 8},{y0 + height / 2:.1f})" text-anchor="middle" class="label">{ylabel}</text>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')

    for gv in grid_values:
        ratio = 0.0 if ymax == ymin else (gv - ymin) / (ymax - ymin)
        y = bottom - ratio * plot_h
        tick_digits = digits if ymax <= 2 else 0
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{fmt(gv, tick_digits)}</text>')

    for idx, value in enumerate(values):
        ratio = 0.0 if ymax == ymin else (value - ymin) / (ymax - ymin)
        ratio = max(0.0, min(1.0, ratio))
        h = ratio * plot_h
        x = start_x + idx * (bar_w + gap)
        y = bottom - h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[idx % len(colors)]}" stroke="#000000" stroke-width="1.1"/>'
        )
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="value">{fmt(value, digits)}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{bottom + 18:.1f}" text-anchor="middle" class="tick">{labels[idx]}</text>')

    return "\n".join(parts)


def build_panel_pdf(
    title: str,
    ylabel: str,
    values: List[float],
    labels: List[str],
    x0: int,
    y0_top: int,
    width: int,
    height: int,
    digits: int,
    ymin: float,
    ymax: float,
    page_height: int,
) -> List[str]:
    left = x0 + 68
    right = x0 + width - 20
    top = y0_top + 36
    bottom = y0_top + height - 56
    plot_w = right - left
    plot_h = bottom - top
    bar_w = plot_w / max(len(values) * 2.0, 1)
    gap = bar_w
    start_x = left + gap / 2
    grid_values = [ymin + (ymax - ymin) * i / 4 for i in range(5)]
    colors = [
        (0x2D / 255.0, 0x37 / 255.0, 0x48 / 255.0),
        (0x71 / 255.0, 0x80 / 255.0, 0x96 / 255.0),
    ]

    def fy(y_svg: float) -> float:
        return page_height - y_svg

    parts: List[str] = []
    parts.append(pdf_text_center(x0 + width / 2, fy(y0_top + 18), 15, title, "/F2"))
    parts.append(pdf_rotated_text(x0 + 16, fy(y0_top + height / 2), 13, 90, ylabel))
    parts.append(pdf_line(left, fy(bottom), right, fy(bottom), 1.2))
    parts.append(pdf_line(left, fy(top), left, fy(bottom), 1.2))

    for gv in grid_values:
        ratio = 0.0 if ymax == ymin else (gv - ymin) / (ymax - ymin)
        y = bottom - ratio * plot_h
        tick_digits = digits if ymax <= 2 else 0
        parts.append(pdf_line(left, fy(y), right, fy(y), 0.8))
        parts.append(pdf_text(left - 38, fy(y + 4), 10, fmt(gv, tick_digits)))

    for idx, value in enumerate(values):
        ratio = 0.0 if ymax == ymin else (value - ymin) / (ymax - ymin)
        ratio = max(0.0, min(1.0, ratio))
        h = ratio * plot_h
        x = start_x + idx * (bar_w + gap)
        y = bottom - h
        parts.append(pdf_rect(x, fy(y) - h, bar_w, h, colors[idx % len(colors)]))
        parts.append(pdf_text_center(x + bar_w / 2, fy(y - 8), 10, fmt(value, digits)))
        parts.append(pdf_text_center(x + bar_w / 2, fy(bottom + 16), 10, labels[idx]))

    return parts


def write_pdf(labels: List[str], success: List[float], collision: List[float], reward: List[float], steps: List[float]) -> None:
    width = 1760
    height = 470
    content: List[str] = ["0 0 0 rg", "0 0 0 RG"]
    content.extend(build_panel_pdf("Success Rate", "Rate", success, labels, 10, 20, 420, 410, 2, 0.0, 1.0, height))
    content.extend(build_panel_pdf("Collision Rate", "Rate", collision, labels, 450, 20, 420, 410, 2, 0.0, 0.1, height))
    content.extend(
        build_panel_pdf(
            "Mean Success Reward",
            "Reward",
            reward,
            labels,
            890,
            20,
            420,
            410,
            2,
            min(reward) - 0.5,
            max(reward) + 0.5,
            height,
        )
    )
    content.extend(
        build_panel_pdf(
            "Mean Steps",
            "Steps",
            steps,
            labels,
            1330,
            20,
            420,
            410,
            1,
            math.floor(min(steps) / 100) * 100,
            math.ceil(max(steps) / 100) * 100,
            height,
        )
    )
    stream = "\n".join(content).encode("latin-1", errors="replace")

    objects: List[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>".encode(
            "latin-1"
        )
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold >>")
    objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream")

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    )
    PDF_PATH.write_bytes(pdf)


def main() -> None:
    rows = read_rows(CSV_PATH)
    if not rows:
        raise SystemExit(f"No rows found in {CSV_PATH}")

    labels = [clean_label(row["name"]) for row in rows]
    success = [float(row["success_rate"]) for row in rows]
    collision = [float(row["collision_rate"]) for row in rows]
    reward = [float(row["mean_reward_success"]) for row in rows]
    steps = [float(row["mean_steps"]) for row in rows]

    width = 1760
    height = 470
    panels = [
        build_panel_svg("Success Rate", "Rate", success, labels, 10, 20, 420, 410, 2, 0.0, 1.0),
        build_panel_svg("Collision Rate", "Rate", collision, labels, 450, 20, 420, 410, 2, 0.0, 0.1),
        build_panel_svg("Mean Success Reward", "Reward", reward, labels, 890, 20, 420, 410, 2, min(reward) - 0.5, max(reward) + 0.5),
        build_panel_svg("Mean Steps", "Steps", steps, labels, 1330, 20, 420, 410, 1, math.floor(min(steps) / 100) * 100, math.ceil(max(steps) / 100) * 100),
    ]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .bg {{ fill: #ffffff; }}
  .title {{ font-family: 'CMU Serif', 'Computer Modern Serif', 'Latin Modern Roman', 'Times New Roman', serif; font-size: 20px; font-weight: 700; }}
  .label {{ font-family: 'CMU Serif', 'Computer Modern Serif', 'Latin Modern Roman', 'Times New Roman', serif; font-size: 18px; }}
  .tick {{ font-family: 'CMU Serif', 'Computer Modern Serif', 'Latin Modern Roman', 'Times New Roman', serif; font-size: 14px; }}
  .value {{ font-family: 'CMU Serif', 'Computer Modern Serif', 'Latin Modern Roman', 'Times New Roman', serif; font-size: 14px; }}
  .axis {{ stroke: #000000; stroke-width: 1.2; }}
  .grid {{ stroke: #000000; stroke-width: 0.8; opacity: 0.22; }}
</style>
<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>
{''.join(panels)}
</svg>
"""
    SVG_PATH.write_text(svg, encoding="utf-8")
    write_pdf(labels, success, collision, reward, steps)
    print(f"Saved {SVG_PATH}")
    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    main()
