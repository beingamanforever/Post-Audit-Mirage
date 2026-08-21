from __future__ import annotations

import html
import math
from typing import Iterable

METHODS = (
    "always_hold",
    "greedy",
    "fixed_threshold",
    "shrinking_budget",
    "addis_spending",
    "online_closed_e",
    "pace_reset",
    "reused_holdout",
    "sgm_transferred",
    "monitor",
    "oracle",
)
LABELS = {
    "always_hold": "Always hold",
    "greedy": "Greedy",
    "fixed_threshold": "Fixed threshold",
    "shrinking_budget": "Shrinking budget",
    "addis_spending": "ADDIS spending",
    "online_closed_e": "Online closed e-test",
    "pace_reset": "PACE reset",
    "reused_holdout": "Reusable holdout",
    "sgm_transferred": "Transferred SGM",
    "monitor": "Monitor unavailable",
    "oracle": "True answer",
    "correct_width": "Correct width",
    "too_narrow": "Too narrow",
}
FAMILY_LABELS = {
    "authgate_v0": "AuthGate",
    "constraint_plan_v0": "ConstraintPlan",
}
NAVY = "#183153"
BLUE = "#2F6FED"
TEAL = "#0F9D8A"
PURPLE = "#7C5CE7"
RED = "#D9485F"
ORANGE = "#E98335"
INK = "#243247"
MUTED = "#68768A"
GRID = "#DDE3EC"
PAPER = "#F7F9FC"
WHITE = "#FFFFFF"


def landscape_svg(rows: list[dict[str, object]], summary: dict[str, object]) -> str:
    width, height = 1440, 940
    svg = _start(
        width,
        height,
        "Experiment 2: lifecycle landscape",
        "Lifecycle harm and genuine-improvement acceptance for every decision method on two environment families.",
    )
    status = _status(summary, "experiment_2")
    svg += _title_block(
        "Experiment 2  |  The lifecycle landscape",
        "500 paired runs per cell  •  T = 50  •  color and printed value encode lifecycle rate",
        status,
    )
    columns = (
        ("null_only", "harmful_lifecycle", "Null-only\nharm"),
        ("mixed", "harmful_lifecycle", "Mixed-stream\nharm"),
        ("all_good", "genuine_acceptance", "All-good\nacceptance"),
        ("mixed", "genuine_acceptance", "Mixed-stream\nacceptance"),
    )
    for panel_index, family in enumerate(FAMILY_LABELS):
        panel_x = 54 + panel_index * 704
        panel_y = 154
        svg += _text(panel_x, panel_y, FAMILY_LABELS[family], 22, NAVY, 500)
        svg += _text(
            panel_x, panel_y + 25, "controlled lifecycle methods marked ●", 13, MUTED
        )
        cell_x = panel_x + 238
        for column_index, (_, _, label) in enumerate(columns):
            for line_index, line in enumerate(label.split("\n")):
                svg += _text(
                    cell_x + column_index * 104 + 46,
                    panel_y + line_index * 17,
                    line,
                    12,
                    MUTED,
                    500,
                    "middle",
                )
        for row_index, method in enumerate(METHODS):
            y = panel_y + 50 + row_index * 59
            if method == "sgm_transferred":
                svg += _rect(panel_x - 10, y - 23, 668, 50, "#FFF2E8", 10)
            elif method in {"shrinking_budget", "addis_spending", "online_closed_e"}:
                svg += _circle(panel_x + 4, y, 5, BLUE)
            svg += _text(
                panel_x + 18,
                y + 5,
                LABELS[method],
                13,
                INK,
                500 if method in {"sgm_transferred", "oracle"} else 400,
            )
            for column_index, (scenario, metric, _) in enumerate(columns):
                result = _find(rows, family, scenario, method, metric)
                value = float(result["estimate"])
                harm = metric == "harmful_lifecycle"
                color = _rate_color(value, harm)
                x = cell_x + column_index * 104
                svg += _rect(x, y - 20, 92, 40, color, 8)
                svg += _text(
                    x + 46,
                    y + 5,
                    _percent(value),
                    13,
                    WHITE if value > 0.52 else INK,
                    500,
                    "middle",
                )
        svg += _text(panel_x, 827, "Lower harm is better", 12, RED, 500)
        svg += _text(
            panel_x + 505, 827, "Higher acceptance is better", 12, TEAL, 500, "middle"
        )
    svg += _footer(
        "Transferred SGM carries evidence between different update hypotheses; blue dots identify methods evaluated for lifecycle control."
    )
    return svg + "</svg>\n"


def impossibility_svg(rows: list[dict[str, object]], summary: dict[str, object]) -> str:
    width, height = 1440, 940
    svg = _start(
        width,
        height,
        "Experiment 3: matched-world impossibility",
        "Safe and harmful worlds emit exactly the same permitted observations, causing every offline decision to match.",
    )
    status = _status(summary, "experiment_3")
    svg += _title_block(
        "Experiment 3  |  Identical evidence, identical decisions",
        "Each ring overlays safe-world and harmful-world acceptance  •  separation is impossible before deployment",
        status,
    )
    axis_x, axis_width = 288, 370
    for panel_index, family in enumerate(FAMILY_LABELS):
        panel_x = 52 + panel_index * 704
        panel_y = 164
        svg += _text(panel_x, panel_y, FAMILY_LABELS[family], 22, NAVY, 500)
        svg += _line(
            panel_x + axis_x,
            panel_y + 28,
            panel_x + axis_x + axis_width,
            panel_y + 28,
            GRID,
        )
        for tick in (0, 0.25, 0.5, 0.75, 1):
            x = panel_x + axis_x + tick * axis_width
            svg += _line(x, panel_y + 24, x, panel_y + 35, GRID)
            svg += _text(x, panel_y + 52, _percent(tick), 11, MUTED, 400, "middle")
        for row_index, method in enumerate(METHODS[:9]):
            y = panel_y + 87 + row_index * 55
            safe = _find(rows, family, "safe", method, "acceptance")
            harmful = _find(rows, family, "harmful", method, "acceptance")
            safe_value = float(safe["estimate"])
            harmful_value = float(harmful["estimate"])
            x_safe = panel_x + axis_x + safe_value * axis_width
            x_harmful = panel_x + axis_x + harmful_value * axis_width
            svg += _text(panel_x, y + 5, LABELS[method], 13, INK)
            svg += _line(
                panel_x + axis_x, y, panel_x + axis_x + axis_width, y, "#EBEEF4"
            )
            svg += _line(x_safe, y, x_harmful, y, PURPLE, 3)
            svg += _circle(x_safe, y, 8, WHITE, BLUE, 3)
            svg += _circle(x_harmful, y, 3, RED)
            label_x = x_safe - 15 if safe_value > 0.80 else x_safe + 15
            anchor = "end" if safe_value > 0.80 else "start"
            svg += _text(
                label_x,
                y + 5,
                _percent(safe_value),
                12,
                INK,
                500,
                anchor,
            )
        box_y = 724
        svg += _rect(panel_x, box_y, 655, 100, "#EEF6F4", 14)
        svg += _text(
            panel_x + 20, box_y + 29, "Identified-range decision", 14, NAVY, 500
        )
        svg += _text(panel_x + 20, box_y + 57, "safe world", 12, TEAL, 500)
        svg += _text(panel_x + 128, box_y + 57, "+", 14, MUTED, 500)
        svg += _text(panel_x + 150, box_y + 57, "harmful world", 12, RED, 500)
        svg += _text(panel_x + 280, box_y + 57, "both compatible", 12, MUTED)
        svg += _text(
            panel_x + 507, box_y + 61, "CANNOT DETERMINE", 12, NAVY, 500, "middle"
        )
        svg += _line(panel_x + 410, box_y + 51, panel_x + 452, box_y + 51, MUTED, 2)
        svg += _polygon(
            f"{panel_x + 452},{box_y + 51} {panel_x + 442},{box_y + 46} {panel_x + 442},{box_y + 56}",
            MUTED,
        )
    svg += _footer(
        "A nontrivial offline gate accepts the safe world and therefore accepts its observationally identical harmful twin at exactly the same rate."
    )
    return svg + "</svg>\n"


def restoration_svg(rows: list[dict[str, object]], summary: dict[str, object]) -> str:
    width, height = 1440, 940
    svg = _start(
        width,
        height,
        "Experiment 4: monitoring restores separation",
        "Classification rates across monitor sample sizes for correct-width and deliberately too-narrow rules.",
    )
    status = _status(summary, "experiment_4")
    svg += _title_block(
        "Experiment 4  |  Live monitoring restores identifiability",
        "Log-scaled monitor size  •  lines show classification rates across 500 deployment-like streams",
        status,
    )
    sizes = sorted(
        {int(row["monitor_n"]) for row in rows if int(row.get("monitor_n", 0)) > 0}
    )
    left, top, panel_width, panel_height = 84, 202, 600, 500
    for panel_index, family in enumerate(FAMILY_LABELS):
        panel_x = left + panel_index * 690
        svg += _text(panel_x, 165, FAMILY_LABELS[family], 22, NAVY, 500)
        svg += _text(
            panel_x + panel_width, 165, "correct monitor", 12, TEAL, 500, "end"
        )
        for tick in (0, 0.25, 0.5, 0.75, 1):
            y = top + panel_height * (1 - tick)
            svg += _line(panel_x, y, panel_x + panel_width, y, GRID)
            svg += _text(panel_x - 14, y + 4, _percent(tick), 11, MUTED, 400, "end")
        tick_sizes = (5, 20, 100, 500, 2500, 10000, 20000)
        for sample_size in tick_sizes:
            x = panel_x + _log_position(sample_size, sizes[0], sizes[-1]) * panel_width
            svg += _line(x, top + panel_height, x, top + panel_height + 7, GRID)
            svg += _text(
                x,
                top + panel_height + 26,
                _compact(sample_size),
                11,
                MUTED,
                400,
                "middle",
            )
        series = (
            (
                "safe",
                "correct_width",
                "classified_safe",
                TEAL,
                "Safe world: classified safe",
                None,
                "circle",
            ),
            (
                "harmful",
                "correct_width",
                "classified_harmful",
                PURPLE,
                "Harmful world: classified harmful",
                "7 4",
                "square",
            ),
            (
                "harmful",
                "too_narrow",
                "classified_safe",
                RED,
                "Harmful world: false safe",
                "2 4",
                "triangle",
            ),
        )
        for series_index, (
            world,
            method,
            metric,
            color,
            label,
            dash,
            marker,
        ) in enumerate(series):
            points = []
            for sample_size in sizes:
                result = _find(rows, family, world, method, metric, sample_size)
                x = (
                    panel_x
                    + _log_position(sample_size, sizes[0], sizes[-1]) * panel_width
                )
                y = top + panel_height * (1 - float(result["estimate"]))
                points.append((x, y))
            svg += _polyline(points, color, 3, dash)
            for x, y in points:
                svg += _marker(x, y, marker, color)
            legend_y = 756 + series_index * 28
            svg += _line(panel_x, legend_y, panel_x + 28, legend_y, color, 3, dash)
            svg += _marker(panel_x + 14, legend_y, marker, color)
            svg += _text(panel_x + 40, legend_y + 5, label, 12, INK)
        first = summary["experiments"]["experiment_4"]["first_separation_monitor_n"][
            family
        ]
        if first is not None:
            x = panel_x + _log_position(int(first), sizes[0], sizes[-1]) * panel_width
            svg += _line(x, top, x, top + panel_height, ORANGE, 2, "5 5")
            svg += _text(
                x - 8,
                top + 20,
                f"first ≥80%: {_compact(int(first))}",
                11,
                ORANGE,
                500,
                "end",
            )
        svg += _text(
            panel_x + panel_width / 2,
            top + panel_height + 52,
            "monitor observations (log scale)",
            12,
            MUTED,
            500,
            "middle",
        )
    svg += _footer(
        "Correct-width uncertainty abstains until evidence separates the worlds; the too-narrow rule can call a harmful AuthGate update safe after sparse early observations."
    )
    return svg + "</svg>\n"


def _start(width: int, height: int, title: str, description: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">'
        f'<title id="title">{html.escape(title)}</title><desc id="description">{html.escape(description)}</desc>'
        f'<rect width="{width}" height="{height}" fill="{PAPER}"/>'
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}</style>'
    )


def _title_block(title: str, subtitle: str, status: str) -> str:
    color = TEAL if status == "PASSED" else RED
    return (
        _text(52, 60, title, 30, NAVY, 500)
        + _text(52, 91, subtitle, 15, MUTED)
        + _rect(1275, 41, 113, 38, color, 19)
        + _text(1331, 66, status, 12, WHITE, 500, "middle")
        + _line(52, 119, 1388, 119, GRID)
    )


def _footer(text: str) -> str:
    return _line(52, 840, 1388, 840, GRID) + _text(52, 869, text, 13, MUTED)


def _status(summary: dict[str, object], experiment: str) -> str:
    return str(summary["experiments"][experiment]["status"]).upper()


def _find(
    rows: Iterable[dict[str, object]],
    family: str,
    scenario: str,
    method: str,
    metric: str,
    monitor_n: int | None = None,
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["family"] == family
        and row["scenario"] == scenario
        and row["method"] == method
        and row["metric"] == metric
        and (monitor_n is None or row.get("monitor_n") == monitor_n)
    )


def _rate_color(value: float, harm: bool) -> str:
    low = (244, 247, 251)
    high = (217, 72, 95) if harm else (15, 157, 138)
    amount = min(1.0, max(0.0, value))
    return "#" + "".join(
        f"{round(start + (end - start) * amount):02X}"
        for start, end in zip(low, high, strict=True)
    )


def _text(
    x: float,
    y: float,
    value: str,
    size: int,
    color: str,
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{html.escape(str(value))}</text>'


def _rect(
    x: float, y: float, width: float, height: float, fill: str, radius: float = 0
) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="{radius:.1f}" fill="{fill}"/>'


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    width: float = 1,
    dash: str | None = None,
) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dashed}/>'


def _circle(
    cx: float,
    cy: float,
    radius: float,
    fill: str,
    stroke: str = "none",
    width: float = 0,
) -> str:
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def _polygon(points: str, fill: str) -> str:
    return f'<polygon points="{points}" fill="{fill}"/>'


def _marker(cx: float, cy: float, shape: str, color: str) -> str:
    if shape == "square":
        return _rect(cx - 4, cy - 4, 8, 8, color, 1)
    if shape == "triangle":
        return _polygon(
            f"{cx:.1f},{cy - 4.5:.1f} {cx - 4.5:.1f},{cy + 3.5:.1f} {cx + 4.5:.1f},{cy + 3.5:.1f}",
            color,
        )
    return _circle(cx, cy, 3.5, color, WHITE, 1.5)


def _polyline(
    points: list[tuple[float, float]],
    color: str,
    width: float,
    dash: str | None = None,
) -> str:
    values = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{values}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dashed}/>'


def _log_position(value: int, minimum: int, maximum: int) -> float:
    return (math.log(value) - math.log(minimum)) / (
        math.log(maximum) - math.log(minimum)
    )


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _compact(value: int) -> str:
    return f"{value // 1000}k" if value >= 1000 and value % 1000 == 0 else str(value)
