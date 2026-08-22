from __future__ import annotations

import html
import math

FAMILY_LABELS = {
    "authgate_v0": "AuthGate",
    "constraint_plan_v0": "ConstraintPlan",
    "batch_triage_v0": "BatchTriage",
}
NAVY = "#183153"
BLUE = "#2F6FED"
TEAL = "#0F8F7F"
PURPLE = "#7157D9"
RED = "#C83E55"
ORANGE = "#D9732F"
INK = "#243247"
MUTED = "#68768A"
GRID = "#DDE3EC"
PAPER = "#F7F9FC"
WHITE = "#FFFFFF"


def identifiability_restoration_svg(summary: dict[str, object]) -> str:
    width, height = 1500, 900
    rows = list(summary["exact_family"])
    sizes = tuple(summary["config"]["monitor_sizes"])
    svg = _start(
        width,
        height,
        "Identifiability restoration by exact family",
        "Correct classification and false-safe risk over live monitor observations for three exact environment families.",
    )
    svg += _text(
        52, 58, "Live observations restore identified decisions", 30, NAVY, 600
    )
    svg += _text(
        52,
        89,
        "Matched offline worlds, group-aware confidence sequences, semantic lifecycle risk units",
        15,
        MUTED,
    )
    svg += _line(52, 116, 1448, 116, GRID)
    left, top, panel_width, panel_height = 65, 190, 420, 470
    for panel_index, family in enumerate(FAMILY_LABELS):
        panel_x = left + panel_index * 480
        svg += _text(panel_x, 157, FAMILY_LABELS[family], 21, NAVY, 600)
        _axes(svg_parts := [], panel_x, top, panel_width, panel_height, sizes)
        svg += "".join(svg_parts)
        series = (
            ("safe", "correct", TEAL, None, "Safe: correct deploy"),
            ("harmful", "correct", PURPLE, "7 4", "Harmful: correct hold"),
            (
                "harmful",
                "false_safe",
                RED,
                "2 4",
                "Harmful: any false safe",
            ),
        )
        for series_index, (world, metric, color, dash, label) in enumerate(series):
            points = []
            for monitor_n in sizes:
                row = _find_exact(rows, family, world, metric, monitor_n)
                x = (
                    panel_x
                    + _log_position(monitor_n, sizes[0], sizes[-1]) * panel_width
                )
                y = top + panel_height * (1 - float(row["estimate"]))
                points.append((x, y))
            svg += _polyline(points, color, dash)
            for x, y in points:
                svg += _circle(x, y, 3.5, color)
            legend_y = 718 + series_index * 28
            svg += _line(panel_x, legend_y, panel_x + 28, legend_y, color, 3, dash)
            svg += _text(panel_x + 38, legend_y + 5, label, 12, INK)
        svg += _text(
            panel_x + panel_width / 2,
            top + panel_height + 47,
            "attempted monitor observations (log scale)",
            11,
            MUTED,
            500,
            "middle",
        )
    svg += _line(52, 828, 1448, 828, GRID)
    svg += _text(
        52,
        858,
        "Correct means at least 80% of repeated streams classify the semantic lifecycle correctly at that look. False safe means any harmful stream deployed by that look.",
        12,
        MUTED,
    )
    return svg + "</svg>\n"


def monitor_sample_complexity_svg(summary: dict[str, object]) -> str:
    width, height = 1500, 900
    cells = list(summary["controlled_panel"])
    prevalences = (0.02, 0.10, 0.50)
    gaps = (0.025, 0.05, 0.10)
    sizes = tuple(summary["config"]["monitor_sizes"])
    svg = _start(
        width,
        height,
        "Controlled monitor sample complexity",
        "First qualifying live monitor look by rare-group prevalence, effect gap, and outcome-independent retention.",
    )
    svg += _text(52, 58, "Controlled rare-group sample complexity", 30, NAVY, 600)
    svg += _text(
        52,
        89,
        "Qualification requires both correct-decision lower bounds at least 80% and the harmful false-safe upper bound at most 5%",
        15,
        MUTED,
    )
    svg += _line(52, 116, 1448, 116, GRID)
    colors = {0.025: BLUE, 0.05: PURPLE, 0.10: TEAL}
    left, top, panel_width, panel_height = 80, 205, 400, 440
    for panel_index, prevalence in enumerate(prevalences):
        panel_x = left + panel_index * 475
        svg += _text(
            panel_x,
            158,
            f"Rare-group prevalence {_percent(prevalence)}",
            20,
            NAVY,
            600,
        )
        for tick in sizes:
            if tick not in (sizes[0], sizes[len(sizes) // 2], sizes[-1]):
                continue
            y = top + (1 - _log_position(tick, sizes[0], sizes[-1])) * panel_height
            svg += _line(panel_x, y, panel_x + panel_width, y, GRID)
            svg += _text(panel_x - 12, y + 4, _compact(tick), 11, MUTED, 400, "end")
        for retention_index, retention in enumerate((1.0, 0.5)):
            x = panel_x + 115 + retention_index * 190
            svg += _text(
                x,
                top + panel_height + 34,
                f"retention {_percent(retention)}",
                12,
                MUTED,
                500,
                "middle",
            )
            for gap_index, gap in enumerate(gaps):
                cell = _find_cell(cells, gap, prevalence, retention)
                first = cell["first_qualifying_monitor_n"]
                color = colors[gap]
                offset = (gap_index - 1) * 28
                if first is None:
                    y = top + panel_height
                    svg += _line(x + offset - 7, y - 7, x + offset + 7, y + 7, color, 2)
                    svg += _line(x + offset - 7, y + 7, x + offset + 7, y - 7, color, 2)
                else:
                    y = (
                        top
                        + (1 - _log_position(int(first), sizes[0], sizes[-1]))
                        * panel_height
                    )
                    svg += _circle(x + offset, y, 7, color)
        legend_y = 730
        for gap_index, gap in enumerate(gaps):
            x = panel_x + gap_index * 130
            svg += _circle(x, legend_y, 5, colors[gap])
            svg += _text(x + 12, legend_y + 4, f"gap {gap:.3f}", 11, INK)
        svg += _text(
            panel_x + panel_width / 2,
            780,
            "Cross means no scheduled look qualified",
            11,
            MUTED,
            400,
            "middle",
        )
    svg += _line(52, 828, 1448, 828, GRID)
    svg += _text(
        52,
        858,
        "Monitor size counts attempted live observations. Retention reduces observed group counts and can delay or prevent qualification.",
        12,
        MUTED,
    )
    return svg + "</svg>\n"


def _axes(
    svg: list[str],
    left: float,
    top: float,
    width: float,
    height: float,
    sizes: tuple[int, ...],
) -> None:
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + height * (1 - tick)
        svg.append(_line(left, y, left + width, y, GRID))
        svg.append(_text(left - 10, y + 4, _percent(tick), 10, MUTED, 400, "end"))
    for sample_size in (sizes[0], sizes[len(sizes) // 2], sizes[-1]):
        x = left + _log_position(sample_size, sizes[0], sizes[-1]) * width
        svg.append(_line(x, top + height, x, top + height + 6, GRID))
        svg.append(
            _text(x, top + height + 23, _compact(sample_size), 10, MUTED, 400, "middle")
        )


def _find_exact(
    rows: list[object], family: str, world: str, metric: str, monitor_n: int
) -> dict[str, object]:
    return next(
        row
        for item in rows
        if isinstance(item, dict)
        and (row := item)["family"] == family
        and row["world"] == world
        and row["metric"] == metric
        and row["monitor_n"] == monitor_n
    )


def _find_cell(
    cells: list[object], gap: float, prevalence: float, retention: float
) -> dict[str, object]:
    return next(
        cell
        for item in cells
        if isinstance(item, dict)
        and (cell := item)["gap"] == gap
        and cell["rare_prevalence"] == prevalence
        and cell["retention"] == retention
    )


def _start(width: int, height: int, title: str, description: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">'
        f'<title id="title">{html.escape(title)}</title>'
        f'<desc id="description">{html.escape(description)}</desc>'
        f'<rect width="{width}" height="{height}" fill="{PAPER}"/>'
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}</style>'
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
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(value)}</text>'


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    width: int = 1,
    dash: str | None = None,
) -> str:
    pattern = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{pattern}/>'


def _circle(x: float, y: float, radius: float, color: str) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="{WHITE}" stroke-width="1.5"/>'


def _polyline(points: list[tuple[float, float]], color: str, dash: str | None) -> str:
    values = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    pattern = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{values}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"{pattern}/>'


def _log_position(value: int, minimum: int, maximum: int) -> float:
    if minimum == maximum:
        return 0.5
    return (math.log(value) - math.log(minimum)) / (
        math.log(maximum) - math.log(minimum)
    )


def _percent(value: float) -> str:
    return f"{100 * value:.0f}%"


def _compact(value: int) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    return str(value)
