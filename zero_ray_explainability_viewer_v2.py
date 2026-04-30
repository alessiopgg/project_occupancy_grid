from __future__ import annotations

"""
Viewer di explainability per i raggi a zero del LaserScan.

Obiettivo:
- leggere il bag tramite preprocess_bag.py
- ricostruire, per ogni scan allineato, le categorie dei raggi grezzi:
  1) validi: range finito, > range_min e <= effective_range_max
  2) zero: range == 0, proiettati artificialmente a zero_ray_range metri
  3) troppo lontani: range > effective_range_max, disegnati come raggi tagliati a effective_range_max
  4) altri invalidi: NaN/Inf/<=range_min
- costruire mappe step-by-step usando SOLO i raggi validi
- sovrapporre i raggi zero come underlay grafico, senza usarli come occupied
- generare una viewer HTML con slider sincronizzato.

Uso tipico:
    python zero_ray_explainability_viewer_v2.py "D:\\tesi\\acquisizioni\\testInAula\\rosbag2_2024_11_15-16_36_27" --start-step 8

Note importanti:
- I raggi a zero NON aggiornano la occupancy grid.
- Il valore 12 m è usato solo per visualizzazione/explainability.
- Lo script è separato dai file principali, quindi non modifica la pipeline di mapping.
"""

import argparse
import inspect
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from preprocess_bag import (
    DEFAULT_KEEP_MAX_RANGE_RETURNS,
    DEFAULT_ODOM_TOPIC,
    DEFAULT_SCAN_TOPIC,
    DEFAULT_TIME_TOLERANCE_NS,
    DEFAULT_ZED_TO_LIDAR_DX,
    DEFAULT_ZED_TO_LIDAR_DY,
    DEFAULT_ZED_TO_LIDAR_DYAW,
    DEFAULT_ZED_YAW_CORRECTION,
    AlignedScan,
    Pose2D,
    PreprocessedBagData,
    ScanSample,
    process_bag_for_mapping,
)

# I progetti aggiornati hanno questi default; se stai usando una versione più vecchia
# di preprocess_bag.py, lo script resta comunque compatibile.
try:
    from preprocess_bag import DEFAULT_TIMESTAMP_SOURCE  # type: ignore
except Exception:
    DEFAULT_TIMESTAMP_SOURCE = "header_time"

try:
    from preprocess_bag import DEFAULT_EFFECTIVE_RANGE_MAX  # type: ignore
except Exception:
    DEFAULT_EFFECTIVE_RANGE_MAX = 12.0

from build_occupancy_grid import (
    GridBounds,
    OccupancyGridConfig,
    bresenham_line,
    classify_probability_grid,
    create_experiment_output_dir,
    log_odds_to_probability,
    probability_to_log_odds,
    update_cells_from_ray,
    world_to_grid,
)


# =========================
# Configurazione rapida
# =========================

BAG_PATH = r"D:\tesi\acquisizioni\rosbag2_2023_11_21-22_15_47"
RESULTS_ROOT = Path("results")
EXPERIMENT_TAG = "zero_ray_explainability"

# Primo step da usare per costruire la mappa.
# Con START_STEP = 8, gli step 1-7 vengono ignorati completamente:
# la griglia parte vuota dall\'ottavo scan visualizzato.
START_STEP = 8

CONFIG = OccupancyGridConfig(
    resolution=0.05,
    margin=1.0,
    p_occ=0.75,
    p_free=0.40,
    log_odds_min=-4.0,
    log_odds_max=4.0,
    free_threshold=0.35,
    occ_threshold=0.75,
    scan_stride=1,
    exclude_origin_cell_from_free=False,
)

# Palette volutamente esplicita: serve proprio a distinguere le categorie.
VALID_RAY_COLOR = "#1f77b4"      # blu: misura valida
VALID_HIT_COLOR = "#d62728"      # rosso: endpoint occupato valido
ZERO_RAY_COLOR = "#7b2cbf"       # viola: range == 0 proiettato a 12 m
TOO_FAR_RAY_COLOR = "#ff7f0e"    # arancione: oltre range operativo, tagliato
INVALID_RAY_COLOR = "#8c8c8c"    # grigio: altri invalidi
LIDAR_COLOR = "#111111"
POSE_POINT_COLOR = "#2ca02c"       # verde: campioni pose raw da odom, non interpolati
POSE_SMOOTH_COLOR = "#006d2c"      # verde scuro: traiettoria pose raw smussata


# =========================
# Dataclass interne
# =========================

@dataclass(frozen=True)
class RaySegment:
    """Segmento di raggio, in coordinate locali e globali."""

    ray_index: int
    angle: float
    original_range: float
    plotted_range: float
    local_start: Tuple[float, float]
    local_end: Tuple[float, float]
    global_start: Tuple[float, float]
    global_end: Tuple[float, float]


@dataclass(frozen=True)
class PreparedScan:
    """Scan arricchito con categorie dei raggi per explainability."""

    scan_index: int
    aligned_scan: AlignedScan
    raw_scan: Optional[ScanSample]
    valid_segments: Tuple[RaySegment, ...]
    zero_segments: Tuple[RaySegment, ...]
    too_far_segments: Tuple[RaySegment, ...]
    invalid_count: int

    @property
    def bag_time_ns(self) -> Optional[int]:
        return getattr(self.aligned_scan, "bag_time_ns", None)

    @property
    def time_ns(self) -> Optional[int]:
        return getattr(self.aligned_scan, "time_ns", self.bag_time_ns)


# =========================
# Utility generali
# =========================

def _filter_supported_kwargs(func: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Passa a una funzione solo gli argomenti supportati dalla sua signature."""
    signature = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in signature.parameters}


def _get_time_key(obj: Any) -> Optional[int]:
    """Chiave temporale robusta: usa time_ns se esiste, altrimenti bag_time_ns."""
    if hasattr(obj, "time_ns"):
        return int(getattr(obj, "time_ns"))
    if hasattr(obj, "bag_time_ns"):
        return int(getattr(obj, "bag_time_ns"))
    return None


def _safe_timestamp_for_label(scan: PreparedScan) -> str:
    if scan.time_ns is not None:
        return str(scan.time_ns)
    if scan.bag_time_ns is not None:
        return str(scan.bag_time_ns)
    return "n/a"


def _sanitize_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("._-") or "unnamed"


def transform_local_to_global(
    local_xy: Tuple[float, float],
    pose: Pose2D,
) -> Tuple[float, float]:
    """Trasformazione rigida 2D: punto locale LiDAR -> frame globale odom."""
    x_local, y_local = local_xy
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    x_global = pose.x + c * x_local - s * y_local
    y_global = pose.y + s * x_local + c * y_local
    return float(x_global), float(y_global)


def transform_global_to_local(
    global_xy: Tuple[float, float],
    pose: Pose2D,
) -> Tuple[float, float]:
    """Trasformazione inversa 2D: punto globale odom -> frame locale della posa data."""
    x_global, y_global = global_xy
    dx = x_global - pose.x
    dy = x_global * 0.0 + (y_global - pose.y)
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    return float(x_local), float(y_local)


def _get_sample_time_ns(obj: Any) -> Optional[int]:
    """Tempo del campione raw: usa time_ns se esiste, altrimenti bag_time_ns."""
    if hasattr(obj, "time_ns"):
        return int(getattr(obj, "time_ns"))
    if hasattr(obj, "bag_time_ns"):
        return int(getattr(obj, "bag_time_ns"))
    return None


def extract_raw_pose_trajectory(
    data: PreprocessedBagData,
    start_time_ns: Optional[int] = None,
) -> Tuple[Tuple[float, float], ...]:
    """
    Estrae la traiettoria delle pose raw dell'odometria.

    Importante: questi sono i campioni originali letti da /odom,
    quindi NON sono le pose interpolate associate agli scan.
    Se start_time_ns è valorizzato, tiene solo le pose raw da quel tempo in poi.
    """
    trajectory: List[Tuple[float, float]] = []
    for sample in data.odom_samples:
        sample_time_ns = _get_sample_time_ns(sample)
        if start_time_ns is not None and sample_time_ns is not None and sample_time_ns < start_time_ns:
            continue
        trajectory.append((float(sample.x), float(sample.y)))
    return tuple(trajectory)


def smooth_trajectory(
    trajectory: Sequence[Tuple[float, float]],
    window_size: int = 5,
) -> Tuple[Tuple[float, float], ...]:
    """Smussa una traiettoria 2D con media mobile semplice."""
    if not trajectory:
        return tuple()
    if len(trajectory) < 3:
        return tuple((float(x), float(y)) for x, y in trajectory)

    window_size = max(1, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2

    smoothed: List[Tuple[float, float]] = []
    for i in range(len(trajectory)):
        j0 = max(0, i - half_window)
        j1 = min(len(trajectory), i + half_window + 1)
        xs = [trajectory[j][0] for j in range(j0, j1)]
        ys = [trajectory[j][1] for j in range(j0, j1)]
        smoothed.append((float(sum(xs) / len(xs)), float(sum(ys) / len(ys))))
    return tuple(smoothed)


def _add_pose_trajectory_global(
    ax: Any,
    raw_pose_trajectory: Optional[Sequence[Tuple[float, float]]] = None,
    smooth_pose_trajectory: Optional[Sequence[Tuple[float, float]]] = None,
) -> None:
    """Disegna punti raw e traiettoria smussata in coordinate globali."""
    if smooth_pose_trajectory:
        xs = [p[0] for p in smooth_pose_trajectory]
        ys = [p[1] for p in smooth_pose_trajectory]
        ax.plot(xs, ys, color=POSE_SMOOTH_COLOR, linewidth=1.8, alpha=0.95, zorder=6)

    if raw_pose_trajectory:
        xs = [p[0] for p in raw_pose_trajectory]
        ys = [p[1] for p in raw_pose_trajectory]
        ax.scatter(xs, ys, s=13, color=POSE_POINT_COLOR, alpha=0.75, zorder=7)


def _add_pose_trajectory_local(
    ax: Any,
    reference_pose: Pose2D,
    raw_pose_trajectory: Optional[Sequence[Tuple[float, float]]] = None,
    smooth_pose_trajectory: Optional[Sequence[Tuple[float, float]]] = None,
) -> None:
    """Disegna la traiettoria raw nel frame locale dello scan corrente."""
    if smooth_pose_trajectory:
        local_smooth = [transform_global_to_local(p, reference_pose) for p in smooth_pose_trajectory]
        xs = [p[0] for p in local_smooth]
        ys = [p[1] for p in local_smooth]
        ax.plot(xs, ys, color=POSE_SMOOTH_COLOR, linewidth=1.8, alpha=0.95, zorder=6)

    if raw_pose_trajectory:
        local_raw = [transform_global_to_local(p, reference_pose) for p in raw_pose_trajectory]
        xs = [p[0] for p in local_raw]
        ys = [p[1] for p in local_raw]
        ax.scatter(xs, ys, s=13, color=POSE_POINT_COLOR, alpha=0.75, zorder=7)


def make_ray_segment(
    ray_index: int,
    angle: float,
    original_range: float,
    plotted_range: float,
    lidar_pose: Pose2D,
) -> RaySegment:
    """Crea un segmento locale/globale partendo da angolo e range di disegno."""
    local_start = (0.0, 0.0)
    local_end = (
        plotted_range * math.cos(angle),
        plotted_range * math.sin(angle),
    )
    global_start = (float(lidar_pose.x), float(lidar_pose.y))
    global_end = transform_local_to_global(local_end, lidar_pose)

    return RaySegment(
        ray_index=ray_index,
        angle=float(angle),
        original_range=float(original_range),
        plotted_range=float(plotted_range),
        local_start=local_start,
        local_end=(float(local_end[0]), float(local_end[1])),
        global_start=global_start,
        global_end=global_end,
    )


def build_scan_lookup(data: PreprocessedBagData) -> Dict[int, ScanSample]:
    """Mappa time_ns/bag_time_ns -> ScanSample grezzo."""
    lookup: Dict[int, ScanSample] = {}
    for scan in data.scan_samples:
        key = _get_time_key(scan)
        if key is not None:
            lookup[key] = scan
        # Fallback aggiuntivo: se time_ns e bag_time_ns sono entrambi presenti e diversi,
        # accetta entrambi per evitare mismatch tra versioni del preprocess.
        if hasattr(scan, "bag_time_ns"):
            lookup[int(getattr(scan, "bag_time_ns"))] = scan
    return lookup


# =========================
# Preparazione categorie raggi
# =========================

def prepare_scan_rays(
    data: PreprocessedBagData,
    zero_ray_range: float,
    effective_range_max: float,
    zero_epsilon: float = 1e-12,
) -> List[PreparedScan]:
    """Associa ogni aligned_scan al LaserScan grezzo e classifica i raggi."""
    scan_lookup = build_scan_lookup(data)
    prepared: List[PreparedScan] = []
    laser = data.laser_config

    for scan_index, aligned in enumerate(data.aligned_scans):
        key = _get_time_key(aligned)
        raw_scan = scan_lookup.get(key) if key is not None else None
        if raw_scan is None and hasattr(aligned, "bag_time_ns"):
            raw_scan = scan_lookup.get(int(getattr(aligned, "bag_time_ns")))

        valid_segments: List[RaySegment] = []
        zero_segments: List[RaySegment] = []
        too_far_segments: List[RaySegment] = []
        invalid_count = 0

        if raw_scan is None:
            # Fallback: se non troviamo il LaserScan grezzo, possiamo ancora usare
            # i global_points già preprocessati come validi, ma non possiamo ricostruire gli zero.
            for p in aligned.global_points:
                range_value = float(getattr(p, "range_value", 0.0))
                if 0.0 < range_value <= effective_range_max:
                    valid_segments.append(
                        make_ray_segment(
                            ray_index=int(getattr(p, "ray_index", -1)),
                            angle=float(getattr(p, "angle_local", 0.0)),
                            original_range=range_value,
                            plotted_range=range_value,
                            lidar_pose=aligned.lidar_pose,
                        )
                    )
        else:
            for ray_index, raw_range in enumerate(raw_scan.ranges):
                r = float(raw_range)
                angle = laser.angle_min + ray_index * laser.angle_increment

                if math.isfinite(r) and abs(r) <= zero_epsilon:
                    zero_segments.append(
                        make_ray_segment(
                            ray_index=ray_index,
                            angle=angle,
                            original_range=r,
                            plotted_range=zero_ray_range,
                            lidar_pose=aligned.lidar_pose,
                        )
                    )
                    continue

                if not math.isfinite(r) or r <= laser.range_min:
                    invalid_count += 1
                    continue

                if r > laser.range_max:
                    invalid_count += 1
                    continue

                if r > effective_range_max:
                    too_far_segments.append(
                        make_ray_segment(
                            ray_index=ray_index,
                            angle=angle,
                            original_range=r,
                            plotted_range=effective_range_max,
                            lidar_pose=aligned.lidar_pose,
                        )
                    )
                    continue

                valid_segments.append(
                    make_ray_segment(
                        ray_index=ray_index,
                        angle=angle,
                        original_range=r,
                        plotted_range=r,
                        lidar_pose=aligned.lidar_pose,
                    )
                )

        prepared.append(
            PreparedScan(
                scan_index=scan_index,
                aligned_scan=aligned,
                raw_scan=raw_scan,
                valid_segments=tuple(valid_segments),
                zero_segments=tuple(zero_segments),
                too_far_segments=tuple(too_far_segments),
                invalid_count=invalid_count,
            )
        )

    return prepared


def iter_bounds_points(prepared_scans: Sequence[PreparedScan]) -> Iterable[Tuple[float, float]]:
    """Punti da considerare per i bound globali della viewer."""
    for scan in prepared_scans:
        pose = scan.aligned_scan.lidar_pose
        yield pose.x, pose.y
        for segment in scan.valid_segments:
            yield segment.global_end
        for segment in scan.zero_segments:
            yield segment.global_end
        for segment in scan.too_far_segments:
            yield segment.global_end


def compute_viewer_bounds(
    prepared_scans: Sequence[PreparedScan],
    config: OccupancyGridConfig,
) -> GridBounds:
    """Calcola bound comuni includendo anche i raggi zero proiettati a 12 m."""
    points = list(iter_bounds_points(prepared_scans))
    if not points:
        raise ValueError("Nessun punto disponibile per calcolare i bound della viewer.")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = min(xs) - config.margin
    x_max = max(xs) + config.margin
    y_min = min(ys) - config.margin
    y_max = max(ys) + config.margin

    num_cols = int(math.ceil((x_max - x_min) / config.resolution))
    num_rows = int(math.ceil((y_max - y_min) / config.resolution))
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError("Dimensioni griglia non valide.")

    return GridBounds(
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        num_rows=num_rows,
        num_cols=num_cols,
    )


# =========================
# Update occupancy grid
# =========================

def update_grid_with_prepared_scan(
    log_odds_grid: np.ndarray,
    bounds: GridBounds,
    scan: PreparedScan,
    config: OccupancyGridConfig,
    l_free: float,
    l_occ: float,
) -> Dict[str, int]:
    """
    Aggiorna la griglia usando solo i raggi validi.

    I raggi zero sono volutamente esclusi dall'update: compaiono solo nei plot.
    """
    pose = scan.aligned_scan.lidar_pose
    start_cell = world_to_grid(pose.x, pose.y, bounds, config.resolution)
    if start_cell is None:
        return {
            "rays_used": 0,
            "rays_skipped_out_of_bounds": len(scan.valid_segments),
            "free_updates": 0,
            "occ_updates": 0,
        }

    rays_used = 0
    rays_skipped = 0
    free_updates_total = 0
    occ_updates_total = 0

    for segment in scan.valid_segments:
        end_x, end_y = segment.global_end
        end_cell = world_to_grid(end_x, end_y, bounds, config.resolution)
        if end_cell is None:
            rays_skipped += 1
            continue

        cells = bresenham_line(start_cell[0], start_cell[1], end_cell[0], end_cell[1])
        if not cells:
            continue

        free_updates, occ_updates = update_cells_from_ray(
            log_odds_grid=log_odds_grid,
            cells=cells,
            l_free=l_free,
            l_occ=l_occ,
            log_odds_min=config.log_odds_min,
            log_odds_max=config.log_odds_max,
            exclude_origin_cell_from_free=config.exclude_origin_cell_from_free,
        )
        rays_used += 1
        free_updates_total += free_updates
        occ_updates_total += occ_updates

    return {
        "rays_used": rays_used,
        "rays_skipped_out_of_bounds": rays_skipped,
        "free_updates": free_updates_total,
        "occ_updates": occ_updates_total,
    }


# =========================
# Plot
# =========================

def _segments_for_line_collection(
    segments: Sequence[RaySegment],
    coordinate: str,
) -> List[List[Tuple[float, float]]]:
    if coordinate not in {"local", "global"}:
        raise ValueError("coordinate deve essere 'local' o 'global'.")
    if coordinate == "local":
        return [[seg.local_start, seg.local_end] for seg in segments]
    return [[seg.global_start, seg.global_end] for seg in segments]


def _add_segments(
    ax: Any,
    segments: Sequence[RaySegment],
    coordinate: str,
    color: str,
    linewidth: float,
    alpha: float,
    label: Optional[str] = None,
    zorder: int = 2,
) -> None:
    if not segments:
        return
    from matplotlib.collections import LineCollection

    lines = _segments_for_line_collection(segments, coordinate)
    collection = LineCollection(
        lines,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )
    ax.add_collection(collection)


def save_ray_explainability_png(
    scan: PreparedScan,
    output_file: str | Path,
    zero_ray_range: float,
    raw_pose_trajectory: Optional[Sequence[Tuple[float, float]]] = None,
    smooth_pose_trajectory: Optional[Sequence[Tuple[float, float]]] = None,
) -> None:
    """Plot locale dello scan: validi, zero proiettati a 12 m, troppo lontani."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    _add_segments(ax, scan.zero_segments, "local", ZERO_RAY_COLOR, 0.45, 0.22, zorder=1)
    _add_segments(ax, scan.too_far_segments, "local", TOO_FAR_RAY_COLOR, 0.50, 0.28, zorder=2)
    _add_segments(ax, scan.valid_segments, "local", VALID_RAY_COLOR, 0.55, 0.42, zorder=3)

    if scan.valid_segments:
        xs = [s.local_end[0] for s in scan.valid_segments]
        ys = [s.local_end[1] for s in scan.valid_segments]
        ax.scatter(xs, ys, s=5, color=VALID_HIT_COLOR, alpha=0.75, zorder=4)

    _add_pose_trajectory_local(
        ax,
        reference_pose=scan.aligned_scan.lidar_pose,
        raw_pose_trajectory=raw_pose_trajectory,
        smooth_pose_trajectory=smooth_pose_trajectory,
    )

    ax.scatter([0.0], [0.0], s=30, color=LIDAR_COLOR, zorder=8)
    ax.set_xlim(-zero_ray_range * 1.08, zero_ray_range * 1.08)
    ax.set_ylim(-zero_ray_range * 1.08, zero_ray_range * 1.08)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_xlabel("x locale LiDAR [m]")
    ax.set_ylabel("y locale LiDAR [m]")
    ax.set_title(
        "Scan explainability - "
        f"step {scan.scan_index} | valid={len(scan.valid_segments)}, "
        f"zero={len(scan.zero_segments)}, over={len(scan.too_far_segments)}, "
        f"invalid={scan.invalid_count}"
    )

    legend_items = [
        Line2D([0], [0], color=VALID_RAY_COLOR, lw=2, label="raggio valido"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VALID_HIT_COLOR, markersize=6, label="endpoint valido"),
        Line2D([0], [0], color=ZERO_RAY_COLOR, lw=2, alpha=0.55, label=f"range = 0 -> {zero_ray_range:g} m"),
        Line2D([0], [0], color=TOO_FAR_RAY_COLOR, lw=2, alpha=0.55, label="oltre range operativo"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=POSE_POINT_COLOR, markersize=6, label="pose raw /odom"),
        Line2D([0], [0], color=POSE_SMOOTH_COLOR, lw=2, label="traiettoria pose smussata"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LIDAR_COLOR, markersize=7, label="LiDAR"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_map_png_with_zero_underlay(
    probability_grid: np.ndarray,
    bounds: GridBounds,
    scan: PreparedScan,
    output_file: str | Path,
    title: str,
    show_valid_rays: bool = True,
    raw_pose_trajectory: Optional[Sequence[Tuple[float, float]]] = None,
    smooth_pose_trajectory: Optional[Sequence[Tuple[float, float]]] = None,
) -> None:
    """Salva una mappa con i raggi zero sovrapposti come underlay leggero."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        probability_grid,
        origin="lower",
        extent=[bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max],
        interpolation="nearest",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        alpha=0.92,
        zorder=0,
    )

    # Underlay: volutamente sottile e trasparente.
    _add_segments(ax, scan.zero_segments, "global", ZERO_RAY_COLOR, 0.35, 0.20, zorder=1)
    _add_segments(ax, scan.too_far_segments, "global", TOO_FAR_RAY_COLOR, 0.35, 0.16, zorder=1)

    if show_valid_rays:
        _add_segments(ax, scan.valid_segments, "global", VALID_RAY_COLOR, 0.25, 0.10, zorder=2)
        if scan.valid_segments:
            xs = [s.global_end[0] for s in scan.valid_segments]
            ys = [s.global_end[1] for s in scan.valid_segments]
            ax.scatter(xs, ys, s=4, color=VALID_HIT_COLOR, alpha=0.35, zorder=3)

    _add_pose_trajectory_global(
        ax,
        raw_pose_trajectory=raw_pose_trajectory,
        smooth_pose_trajectory=smooth_pose_trajectory,
    )

    pose = scan.aligned_scan.lidar_pose
    ax.scatter([pose.x], [pose.y], s=22, color=LIDAR_COLOR, zorder=8)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")

    legend_items = [
        Line2D([0], [0], color=ZERO_RAY_COLOR, lw=2, alpha=0.55, label="raggi zero proiettati"),
        Line2D([0], [0], color=TOO_FAR_RAY_COLOR, lw=2, alpha=0.45, label="oltre 12 m tagliati"),
        Line2D([0], [0], color=VALID_RAY_COLOR, lw=2, alpha=0.35, label="raggi validi"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=POSE_POINT_COLOR, markersize=6, label="pose raw /odom"),
        Line2D([0], [0], color=POSE_SMOOTH_COLOR, lw=2, label="traiettoria pose smussata"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LIDAR_COLOR, markersize=7, label="LiDAR"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =========================
# Viewer HTML
# =========================

def build_viewer_html(
    frames: Sequence[Dict[str, Any]],
    output_file: str | Path,
) -> None:
    """Crea una viewer HTML a tre pannelli sincronizzati."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames_json = json.dumps(list(frames), ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Zero Ray Explainability Viewer</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; color: #222; }}
    .wrap {{ max-width: 1700px; margin: 0 auto; padding: 20px; }}
    .panel {{ background: white; border-radius: 12px; padding: 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
    .muted {{ color: #666; font-size: 14px; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 16px 0; padding: 12px; background: #fafafa; border: 1px solid #ececec; border-radius: 10px; }}
    button {{ padding: 10px 14px; border: 0; border-radius: 8px; cursor: pointer; }}
    input[type=range] {{ flex: 1; min-width: 260px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
    .viewer {{ background: #fff; border: 1px solid #ececec; border-radius: 12px; padding: 12px; }}
    img {{ width: 100%; max-height: 70vh; object-fit: contain; background: #fff; border-radius: 10px; border: 1px solid #ddd; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin-top: 14px; }}
    .box {{ background: #fafafa; border: 1px solid #ececec; border-radius: 8px; padding: 8px; overflow-wrap: anywhere; }}
    @media (max-width: 1200px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h2 style="margin:0 0 6px 0;">Zero Ray Explainability Viewer</h2>
      <div class="muted">
        I raggi a zero sono proiettati graficamente a 12 m e mostrati in viola a bassa opacità. Non aggiornano la occupancy grid.
      </div>

      <div class="controls">
        <button id="prevBtn">⟨ Prev</button>
        <button id="playBtn">Play</button>
        <button id="nextBtn">Next ⟩</button>
        <label>Velocità <input id="speedInput" type="number" min="50" step="50" value="180" style="width:90px; margin-left:6px;" /> ms</label>
        <input id="slider" type="range" min="0" max="0" value="0" />
      </div>

      <div style="margin-bottom: 14px;">
        <strong id="stepLabel"></strong>
        <div class="muted" id="stepSubtitle"></div>
      </div>

      <div class="grid">
        <div class="viewer">
          <h3 style="margin:0 0 8px 0;">Mappa cumulativa</h3>
          <img id="cumImage" alt="Cumulative map" src="" />
        </div>
        <div class="viewer">
          <h3 style="margin:0 0 8px 0;">Mappa singolo scan</h3>
          <img id="singleImage" alt="Single map" src="" />
        </div>
        <div class="viewer">
          <h3 style="margin:0 0 8px 0;">Raggi dello scan</h3>
          <img id="rayImage" alt="Ray explainability" src="" />
        </div>
      </div>

      <div class="meta">
        <div class="box"><strong>scan_index</strong><div id="scanIndex"></div></div>
        <div class="box"><strong>time_ns</strong><div id="timeNs"></div></div>
        <div class="box"><strong>valid</strong><div id="validCount"></div></div>
        <div class="box"><strong>zero</strong><div id="zeroCount"></div></div>
        <div class="box"><strong>over_range</strong><div id="overCount"></div></div>
        <div class="box"><strong>invalid</strong><div id="invalidCount"></div></div>
        <div class="box"><strong>free updates</strong><div id="freeUpdates"></div></div>
        <div class="box"><strong>occ updates</strong><div id="occUpdates"></div></div>
      </div>
    </div>
  </div>

  <script>
    const frames = {frames_json};
    const slider = document.getElementById('slider');
    const playBtn = document.getElementById('playBtn');
    const speedInput = document.getElementById('speedInput');
    let current = 0;
    let timer = null;

    slider.max = Math.max(frames.length - 1, 0);

    function clampIndex(index) {{
      if (!frames.length) return 0;
      return Math.max(0, Math.min(index, frames.length - 1));
    }}

    function render(index) {{
      if (!frames.length) return;
      current = clampIndex(index);
      const f = frames[current];
      slider.value = current;
      document.getElementById('cumImage').src = f.cumulative_png;
      document.getElementById('singleImage').src = f.single_png;
      document.getElementById('rayImage').src = f.ray_png;
      document.getElementById('stepLabel').textContent = `Step ${{current + 1}} / ${{frames.length}}`;
      document.getElementById('stepSubtitle').textContent = `Scan index ${{f.scan_index}} — i raggi zero sono visualizzati fino a ${{f.zero_ray_range_m}} m`;
      document.getElementById('scanIndex').textContent = String(f.scan_index);
      document.getElementById('timeNs').textContent = String(f.time_ns);
      document.getElementById('validCount').textContent = String(f.valid_count);
      document.getElementById('zeroCount').textContent = String(f.zero_count);
      document.getElementById('overCount').textContent = String(f.too_far_count);
      document.getElementById('invalidCount').textContent = String(f.invalid_count);
      document.getElementById('freeUpdates').textContent = String(f.free_updates);
      document.getElementById('occUpdates').textContent = String(f.occ_updates);
    }}

    function stopPlayback() {{
      if (timer) {{ clearInterval(timer); timer = null; }}
      playBtn.textContent = 'Play';
    }}

    function startPlayback() {{
      stopPlayback();
      playBtn.textContent = 'Pause';
      const delay = Math.max(50, Number(speedInput.value) || 180);
      timer = setInterval(() => {{
        if (current >= frames.length - 1) {{ stopPlayback(); return; }}
        render(current + 1);
      }}, delay);
    }}

    slider.addEventListener('input', () => render(Number(slider.value)));
    document.getElementById('prevBtn').addEventListener('click', () => render(current - 1));
    document.getElementById('nextBtn').addEventListener('click', () => render(current + 1));
    playBtn.addEventListener('click', () => timer ? stopPlayback() : startPlayback());
    window.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowLeft') render(current - 1);
      if (e.key === 'ArrowRight') render(current + 1);
      if (e.key === ' ') {{ e.preventDefault(); timer ? stopPlayback() : startPlayback(); }}
    }});
    render(0);
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


# =========================
# Pipeline principale
# =========================

def run_zero_ray_explainability_pipeline(
    bag_path: str | Path,
    results_root: str | Path,
    experiment_tag: str,
    config: OccupancyGridConfig,
    zero_ray_range: float = 12.0,
    effective_range_max: float = 12.0,
    scan_topic: str = DEFAULT_SCAN_TOPIC,
    odom_topic: str = DEFAULT_ODOM_TOPIC,
    timestamp_source: str = DEFAULT_TIMESTAMP_SOURCE,
    zed_to_lidar_dx: float = DEFAULT_ZED_TO_LIDAR_DX,
    zed_to_lidar_dy: float = DEFAULT_ZED_TO_LIDAR_DY,
    zed_to_lidar_dyaw: float = DEFAULT_ZED_TO_LIDAR_DYAW,
    zed_yaw_correction: float = DEFAULT_ZED_YAW_CORRECTION,
    keep_max_range_returns: bool = DEFAULT_KEEP_MAX_RANGE_RETURNS,
    time_tolerance_ns: int = DEFAULT_TIME_TOLERANCE_NS,
    start_step: int = START_STEP,
) -> Path:
    """Esegue preprocess, costruzione stepwise e viewer HTML."""
    preprocess_kwargs = _filter_supported_kwargs(
        process_bag_for_mapping,
        dict(
            bag_path=bag_path,
            scan_topic=scan_topic,
            odom_topic=odom_topic,
            timestamp_source=timestamp_source,
            zed_to_lidar_dx=zed_to_lidar_dx,
            zed_to_lidar_dy=zed_to_lidar_dy,
            zed_to_lidar_dyaw=zed_to_lidar_dyaw,
            zed_yaw_correction=zed_yaw_correction,
            keep_max_range_returns=keep_max_range_returns,
            time_tolerance_ns=time_tolerance_ns,
            effective_range_max=effective_range_max,
        ),
    )
    data = process_bag_for_mapping(**preprocess_kwargs)

    prepared_scans_all = prepare_scan_rays(
        data=data,
        zero_ray_range=zero_ray_range,
        effective_range_max=effective_range_max,
    )
    if start_step < 1:
        raise ValueError("start_step deve essere >= 1. Usa 1 per partire dall'inizio.")

    prepared_scans_after_stride = [
        scan for scan in prepared_scans_all
        if scan.scan_index % config.scan_stride == 0
    ]
    if not prepared_scans_after_stride:
        raise ValueError("Nessuno scan disponibile dopo l'applicazione di scan_stride.")

    if start_step > len(prepared_scans_after_stride):
        raise ValueError(
            f"start_step={start_step} non valido: dopo scan_stride sono disponibili "
            f"solo {len(prepared_scans_after_stride)} step."
        )

    # Scelta sperimentale: gli step precedenti vengono ignorati completamente.
    # La griglia cumulativa viene inizializzata vuota e aggiornata solo da questo punto in poi.
    prepared_scans = prepared_scans_after_stride[start_step - 1:]
    first_selected_time_ns = prepared_scans[0].time_ns
    raw_pose_trajectory = extract_raw_pose_trajectory(
        data,
        start_time_ns=first_selected_time_ns,
    )
    smooth_pose_trajectory = smooth_trajectory(raw_pose_trajectory, window_size=5)

    output_dir = create_experiment_output_dir(
        bag_path=bag_path,
        results_root=results_root,
        experiment_tag=experiment_tag,
    )
    cumulative_dir = output_dir / "cumulative_maps"
    single_dir = output_dir / "single_scan_maps"
    rays_dir = output_dir / "ray_explainability"
    arrays_dir = output_dir / "arrays"
    for directory in (cumulative_dir, single_dir, rays_dir, arrays_dir):
        directory.mkdir(parents=True, exist_ok=True)

    bounds = compute_viewer_bounds(prepared_scans, config)
    l_free = probability_to_log_odds(config.p_free)
    l_occ = probability_to_log_odds(config.p_occ)

    cumulative_grid = np.zeros((bounds.num_rows, bounds.num_cols), dtype=np.float32)
    frames: List[Dict[str, Any]] = []

    total_rays_used = 0
    total_free_updates = 0
    total_occ_updates = 0
    total_rays_skipped = 0

    for visible_step, scan in enumerate(prepared_scans, start=1):
        # Update cumulativo: solo raggi validi.
        cumulative_stats = update_grid_with_prepared_scan(
            log_odds_grid=cumulative_grid,
            bounds=bounds,
            scan=scan,
            config=config,
            l_free=l_free,
            l_occ=l_occ,
        )
        total_rays_used += cumulative_stats["rays_used"]
        total_free_updates += cumulative_stats["free_updates"]
        total_occ_updates += cumulative_stats["occ_updates"]
        total_rays_skipped += cumulative_stats["rays_skipped_out_of_bounds"]

        cumulative_probability = log_odds_to_probability(cumulative_grid).astype(np.float32)

        # Mappa singola: griglia vuota aggiornata solo con lo scan corrente.
        single_grid = np.zeros_like(cumulative_grid)
        single_stats = update_grid_with_prepared_scan(
            log_odds_grid=single_grid,
            bounds=bounds,
            scan=scan,
            config=config,
            l_free=l_free,
            l_occ=l_occ,
        )
        single_probability = log_odds_to_probability(single_grid).astype(np.float32)

        source_step = start_step + visible_step - 1
        stem = f"step_{source_step:04d}_scan_{scan.scan_index:04d}"
        cumulative_png = cumulative_dir / f"{stem}_cumulative.png"
        single_png = single_dir / f"{stem}_single.png"
        ray_png = rays_dir / f"{stem}_rays.png"

        save_map_png_with_zero_underlay(
            probability_grid=cumulative_probability,
            bounds=bounds,
            scan=scan,
            output_file=cumulative_png,
            title=f"Cumulative map + zero-ray underlay - experiment step {visible_step} (source step {source_step})",
            show_valid_rays=False,
            raw_pose_trajectory=raw_pose_trajectory,
            smooth_pose_trajectory=smooth_pose_trajectory,
        )
        save_map_png_with_zero_underlay(
            probability_grid=single_probability,
            bounds=bounds,
            scan=scan,
            output_file=single_png,
            title=f"Single scan map + zero-ray underlay - source step {source_step} / scan {scan.scan_index}",
            show_valid_rays=True,
            raw_pose_trajectory=raw_pose_trajectory,
            smooth_pose_trajectory=smooth_pose_trajectory,
        )
        save_ray_explainability_png(
            scan=scan,
            output_file=ray_png,
            zero_ray_range=zero_ray_range,
            raw_pose_trajectory=raw_pose_trajectory,
            smooth_pose_trajectory=smooth_pose_trajectory,
        )

        frames.append(
            {
                "step": visible_step,
                "source_step": source_step,
                "scan_index": scan.scan_index,
                "time_ns": _safe_timestamp_for_label(scan),
                "bag_time_ns": scan.bag_time_ns,
                "zero_ray_range_m": zero_ray_range,
                "effective_range_max_m": effective_range_max,
                "valid_count": len(scan.valid_segments),
                "zero_count": len(scan.zero_segments),
                "too_far_count": len(scan.too_far_segments),
                "invalid_count": scan.invalid_count,
                "rays_used": cumulative_stats["rays_used"],
                "rays_skipped_out_of_bounds": cumulative_stats["rays_skipped_out_of_bounds"],
                "free_updates": cumulative_stats["free_updates"],
                "occ_updates": cumulative_stats["occ_updates"],
                "single_free_updates": single_stats["free_updates"],
                "single_occ_updates": single_stats["occ_updates"],
                "cumulative_png": Path(os.path.relpath(cumulative_png, start=output_dir)).as_posix(),
                "single_png": Path(os.path.relpath(single_png, start=output_dir)).as_posix(),
                "ray_png": Path(os.path.relpath(ray_png, start=output_dir)).as_posix(),
            }
        )

    final_probability = log_odds_to_probability(cumulative_grid).astype(np.float32)
    final_classified = classify_probability_grid(
        final_probability,
        free_threshold=config.free_threshold,
        occ_threshold=config.occ_threshold,
    )

    np.save(arrays_dir / "final_log_odds_grid.npy", cumulative_grid)
    np.save(arrays_dir / "final_probability_grid.npy", final_probability)
    np.save(arrays_dir / "final_classified_grid.npy", final_classified)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bag_path": str(bag_path),
        "output_dir": str(output_dir),
        "viewer_html": str(output_dir / "viewer_zero_rays.html"),
        "timestamp_source_requested": timestamp_source,
        "start_step": start_step,
        "num_scans_available_after_stride": len(prepared_scans_after_stride),
        "first_visualized_scan_index": prepared_scans[0].scan_index if prepared_scans else None,
        "zero_ray_range_m": zero_ray_range,
        "effective_range_max_m": effective_range_max,
        "config": asdict(config),
        "bounds": asdict(bounds),
        "num_scans_total_after_preprocess": len(data.aligned_scans),
        "num_scans_visualized": len(prepared_scans),
        "total_valid_rays_used": total_rays_used,
        "total_rays_skipped_out_of_bounds": total_rays_skipped,
        "total_free_updates": total_free_updates,
        "total_occ_updates": total_occ_updates,
        "total_zero_rays_visualized": int(sum(len(scan.zero_segments) for scan in prepared_scans)),
        "total_too_far_rays_visualized": int(sum(len(scan.too_far_segments) for scan in prepared_scans)),
        "frames": frames,
    }
    (output_dir / "zero_ray_explainability_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    build_viewer_html(frames, output_dir / "viewer_zero_rays.html")

    print("\n=== ZERO RAY EXPLAINABILITY COMPLETED ===")
    print(f"Bag: {bag_path}")
    print(f"Output root: {output_dir.resolve()}")
    print(f"Viewer HTML: {(output_dir / 'viewer_zero_rays.html').resolve()}")
    print(f"Start step esperimento: {start_step}")
    print(f"Scan visualizzati: {len(prepared_scans)}")
    print(f"Raggi validi usati nel mapping: {total_rays_used}")
    print(f"Raggi zero visualizzati: {manifest['total_zero_rays_visualized']}")
    print("Nota: i raggi zero sono solo overlay grafico, non aggiornano la mappa.")

    return output_dir


# =========================
# CLI
# =========================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Viewer stepwise di explainability per raggi LaserScan a zero"
    )
    parser.add_argument("bag_path", nargs="?", default=BAG_PATH, type=str)
    parser.add_argument("--results-root", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--experiment-tag", type=str, default=EXPERIMENT_TAG)

    parser.add_argument("--scan-topic", type=str, default=DEFAULT_SCAN_TOPIC)
    parser.add_argument("--odom-topic", type=str, default=DEFAULT_ODOM_TOPIC)
    parser.add_argument(
        "--timestamp-source",
        type=str,
        default=DEFAULT_TIMESTAMP_SOURCE,
        choices=["bag_time", "header_time"],
        help="Usato solo se il tuo preprocess_bag.py supporta timestamp_source.",
    )

    parser.add_argument("--zed-to-lidar-dx", type=float, default=DEFAULT_ZED_TO_LIDAR_DX)
    parser.add_argument("--zed-to-lidar-dy", type=float, default=DEFAULT_ZED_TO_LIDAR_DY)
    parser.add_argument("--zed-to-lidar-dyaw", type=float, default=DEFAULT_ZED_TO_LIDAR_DYAW)
    parser.add_argument("--zed-yaw-correction", type=float, default=DEFAULT_ZED_YAW_CORRECTION)
    parser.add_argument("--keep-max-range-returns", action="store_true", default=DEFAULT_KEEP_MAX_RANGE_RETURNS)
    parser.add_argument("--time-tolerance-ns", type=int, default=DEFAULT_TIME_TOLERANCE_NS)

    parser.add_argument("--zero-ray-range", type=float, default=12.0)
    parser.add_argument("--effective-range-max", type=float, default=float(DEFAULT_EFFECTIVE_RANGE_MAX))
    parser.add_argument(
        "--start-step",
        type=int,
        default=START_STEP,
        help="Primo step da usare per l'esperimento. Con 8, gli step 1-7 vengono ignorati.",
    )

    parser.add_argument("--resolution", type=float, default=CONFIG.resolution)
    parser.add_argument("--margin", type=float, default=CONFIG.margin)
    parser.add_argument("--p-occ", type=float, default=CONFIG.p_occ)
    parser.add_argument("--p-free", type=float, default=CONFIG.p_free)
    parser.add_argument("--log-odds-min", type=float, default=CONFIG.log_odds_min)
    parser.add_argument("--log-odds-max", type=float, default=CONFIG.log_odds_max)
    parser.add_argument("--free-threshold", type=float, default=CONFIG.free_threshold)
    parser.add_argument("--occ-threshold", type=float, default=CONFIG.occ_threshold)
    parser.add_argument("--scan-stride", type=int, default=CONFIG.scan_stride)
    parser.add_argument("--exclude-origin-cell-from-free", action="store_true", default=CONFIG.exclude_origin_cell_from_free)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = OccupancyGridConfig(
        resolution=args.resolution,
        margin=args.margin,
        p_occ=args.p_occ,
        p_free=args.p_free,
        log_odds_min=args.log_odds_min,
        log_odds_max=args.log_odds_max,
        free_threshold=args.free_threshold,
        occ_threshold=args.occ_threshold,
        scan_stride=args.scan_stride,
        exclude_origin_cell_from_free=args.exclude_origin_cell_from_free,
    )

    run_zero_ray_explainability_pipeline(
        bag_path=args.bag_path,
        results_root=args.results_root,
        experiment_tag=args.experiment_tag,
        config=config,
        zero_ray_range=args.zero_ray_range,
        effective_range_max=args.effective_range_max,
        scan_topic=args.scan_topic,
        odom_topic=args.odom_topic,
        timestamp_source=args.timestamp_source,
        zed_to_lidar_dx=args.zed_to_lidar_dx,
        zed_to_lidar_dy=args.zed_to_lidar_dy,
        zed_to_lidar_dyaw=args.zed_to_lidar_dyaw,
        zed_yaw_correction=args.zed_yaw_correction,
        keep_max_range_returns=args.keep_max_range_returns,
        time_tolerance_ns=args.time_tolerance_ns,
        start_step=args.start_step,
    )


if __name__ == "__main__":
    main()
