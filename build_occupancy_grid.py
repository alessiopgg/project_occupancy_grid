from __future__ import annotations

"""
Costruzione di una Occupancy Grid 2D a partire dall'output di preprocess_bag.py.

Obiettivo:
- prendere in input un oggetto PreprocessedBagData già costruito dal preprocess
- calcolare i bound spaziali della mappa
- creare una griglia 2D in log-odds
- aggiornare la griglia raggio per raggio con Bresenham
- produrre mappe in log-odds, probabilità e classificazione finale
- salvare ogni test in una sottocartella dedicata, senza sovrascrivere run precedenti

Uso tipico da import:
    from preprocess_bag import process_bag_for_mapping
    from build_occupancy_grid import (
        build_occupancy_grid,
        OccupancyGridConfig,
        create_experiment_output_dir,
        save_all_outputs,
    )

    data = process_bag_for_mapping("/percorso/al/bag")
    result = build_occupancy_grid(data, OccupancyGridConfig())
    out_dir = create_experiment_output_dir(data.bag_path)
    save_all_outputs(result, out_dir)

Uso tipico da terminale:
    python build_occupancy_grid.py /percorso/al/bag --results-root ./results
"""

import argparse
import json
import math
import re
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
    Pose2D,
    PreprocessedBagData,
    process_bag_for_mapping,
)


# =========================
# Configurazione
# =========================


@dataclass(frozen=True)
class OccupancyGridConfig:
    """Parametri principali del mapping."""

    resolution: float = 0.05
    margin: float = 1.0
    p_occ: float = 0.70
    p_free: float = 0.35
    log_odds_min: float = -4.0
    log_odds_max: float = 4.0

    # Classificazione finale della probabilità.
    free_threshold: float = 0.35
    occ_threshold: float = 0.65

    # Se > 1 usa solo una scansione ogni scan_stride.
    scan_stride: int = 1

    # Se True non aggiorna come free la cella di origine del LiDAR.
    exclude_origin_cell_from_free: bool = False


@dataclass(frozen=True)
class GridBounds:
    """Bound metrici della mappa nel frame globale odom."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    num_rows: int
    num_cols: int


@dataclass(frozen=True)
class OccupancyGridSummary:
    """Riepilogo utile per debug rapido."""

    num_scans_total: int
    num_scans_used: int
    num_rays_used: int
    num_rays_skipped_out_of_bounds: int
    num_free_updates: int
    num_occ_updates: int
    num_rows: int
    num_cols: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float


@dataclass(frozen=True)
class OccupancyGridResult:
    """Risultato completo della costruzione della mappa."""

    config: OccupancyGridConfig
    bounds: GridBounds
    log_odds_grid: np.ndarray
    probability_grid: np.ndarray
    classified_grid: np.ndarray
    summary: OccupancyGridSummary


# =========================
# Utility log-odds
# =========================


def probability_to_log_odds(p: float) -> float:
    """Converte una probabilità in log-odds."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"La probabilità deve stare in (0, 1), ricevuto: {p}")
    return float(math.log(p / (1.0 - p)))



def log_odds_to_probability(log_odds: np.ndarray) -> np.ndarray:
    """Converte una griglia log-odds in probabilità."""
    return 1.0 / (1.0 + np.exp(-log_odds))


# =========================
# Bound e discretizzazione
# =========================


def _iter_all_world_points(data: PreprocessedBagData) -> Iterable[Tuple[float, float]]:
    """Itera su tutte le pose LiDAR e su tutti gli endpoint globali."""
    for scan in data.aligned_scans:
        yield scan.lidar_pose.x, scan.lidar_pose.y
        for p in scan.global_points:
            yield p.x, p.y



def compute_grid_bounds(
    data: PreprocessedBagData,
    config: OccupancyGridConfig,
) -> GridBounds:
    """
    Calcola i bound metrici della mappa e le sue dimensioni discrete.

    La mappa viene costruita a partire da:
    - tutte le pose del LiDAR
    - tutti gli endpoint globali dei raggi validi
    """
    points = list(_iter_all_world_points(data))
    if not points:
        raise ValueError("Impossibile costruire la mappa: nessun punto disponibile.")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min_raw = min(xs)
    x_max_raw = max(xs)
    y_min_raw = min(ys)
    y_max_raw = max(ys)

    x_min = x_min_raw - config.margin
    x_max = x_max_raw + config.margin
    y_min = y_min_raw - config.margin
    y_max = y_max_raw + config.margin

    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        raise ValueError("Bound della mappa non validi: larghezza o altezza non positive.")

    num_cols = int(math.ceil(width / config.resolution))
    num_rows = int(math.ceil(height / config.resolution))

    if num_cols <= 0 or num_rows <= 0:
        raise ValueError("Numero di righe o colonne non valido per la griglia.")

    return GridBounds(
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        num_rows=num_rows,
        num_cols=num_cols,
    )



def world_to_grid(
    x: float,
    y: float,
    bounds: GridBounds,
    resolution: float,
) -> Optional[Tuple[int, int]]:
    """
    Converte coordinate metriche globali in indici (row, col) della griglia.

    Restituisce None se il punto cade fuori dalla mappa.
    """
    col = int((x - bounds.x_min) / resolution)
    row = int((y - bounds.y_min) / resolution)

    if row < 0 or row >= bounds.num_rows or col < 0 or col >= bounds.num_cols:
        return None
    return row, col



def grid_to_world_center(
    row: int,
    col: int,
    bounds: GridBounds,
    resolution: float,
) -> Tuple[float, float]:
    """Restituisce il centro metrico della cella (row, col)."""
    x = bounds.x_min + (col + 0.5) * resolution
    y = bounds.y_min + (row + 0.5) * resolution
    return x, y


# =========================
# Bresenham
# =========================


def bresenham_line(
    row0: int,
    col0: int,
    row1: int,
    col1: int,
) -> List[Tuple[int, int]]:
    """
    Restituisce le celle percorse dalla linea discreta tra due celle.

    Implementazione standard di Bresenham adattata a indici (row, col).
    """
    cells: List[Tuple[int, int]] = []

    x0, y0 = col0, row0
    x1, y1 = col1, row1

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return cells


# =========================
# Aggiornamento griglia
# =========================


def clamp_log_odds(value: float, min_value: float, max_value: float) -> float:
    """Satura un singolo valore log-odds nell'intervallo desiderato."""
    return max(min_value, min(max_value, value))



def update_cells_from_ray(
    log_odds_grid: np.ndarray,
    cells: Sequence[Tuple[int, int]],
    l_free: float,
    l_occ: float,
    log_odds_min: float,
    log_odds_max: float,
    exclude_origin_cell_from_free: bool = False,
) -> Tuple[int, int]:
    """
    Aggiorna la griglia lungo un singolo raggio discreto.

    Restituisce:
    - numero di aggiornamenti free eseguiti
    - numero di aggiornamenti occupied eseguiti
    """
    if not cells:
        return 0, 0

    free_cells = list(cells[:-1])
    occ_cell = cells[-1]

    if exclude_origin_cell_from_free and free_cells:
        free_cells = free_cells[1:]

    free_updates = 0
    for row, col in free_cells:
        log_odds_grid[row, col] = clamp_log_odds(
            float(log_odds_grid[row, col]) + l_free,
            log_odds_min,
            log_odds_max,
        )
        free_updates += 1

    occ_updates = 0
    row, col = occ_cell
    log_odds_grid[row, col] = clamp_log_odds(
        float(log_odds_grid[row, col]) + l_occ,
        log_odds_min,
        log_odds_max,
    )
    occ_updates += 1

    return free_updates, occ_updates



def classify_probability_grid(
    probability_grid: np.ndarray,
    free_threshold: float,
    occ_threshold: float,
) -> np.ndarray:
    """
    Classifica la griglia in:
    - 0 = unknown
    - 1 = free
    - 2 = occupied
    """
    if not 0.0 <= free_threshold < occ_threshold <= 1.0:
        raise ValueError(
            "Soglie non valide: deve valere 0 <= free_threshold < occ_threshold <= 1"
        )

    classified = np.zeros(probability_grid.shape, dtype=np.uint8)
    classified[probability_grid < free_threshold] = 1
    classified[probability_grid > occ_threshold] = 2
    return classified


# =========================
# Funzione principale
# =========================


def build_occupancy_grid(
    data: PreprocessedBagData,
    config: Optional[OccupancyGridConfig] = None,
) -> OccupancyGridResult:
    """
    Costruisce la occupancy grid a partire dai dati preprocessati.

    Pipeline:
    1. calcolo bound e dimensioni della griglia
    2. inizializzazione log-odds a zero (prior ignoto)
    3. per ogni scan usato:
       - origine = posa LiDAR
       - endpoint = ciascun global_point
       - Bresenham tra origine ed endpoint
       - aggiornamento free / occupied
    4. conversione finale in probabilità e classificazione
    """
    if config is None:
        config = OccupancyGridConfig()

    if config.scan_stride <= 0:
        raise ValueError("scan_stride deve essere >= 1")

    if not data.aligned_scans:
        raise ValueError("Nessuna scansione allineata disponibile nel preprocess.")

    bounds = compute_grid_bounds(data, config)
    log_odds_grid = np.zeros((bounds.num_rows, bounds.num_cols), dtype=np.float32)

    l_occ = probability_to_log_odds(config.p_occ)
    l_free = probability_to_log_odds(config.p_free)

    num_scans_total = len(data.aligned_scans)
    num_scans_used = 0
    num_rays_used = 0
    num_rays_skipped_out_of_bounds = 0
    num_free_updates = 0
    num_occ_updates = 0

    for scan_index, scan in enumerate(data.aligned_scans):
        if scan_index % config.scan_stride != 0:
            continue

        start_cell = world_to_grid(
            scan.lidar_pose.x,
            scan.lidar_pose.y,
            bounds,
            config.resolution,
        )
        if start_cell is None:
            continue

        num_scans_used += 1

        for endpoint in scan.global_points:
            end_cell = world_to_grid(
                endpoint.x,
                endpoint.y,
                bounds,
                config.resolution,
            )
            if end_cell is None:
                num_rays_skipped_out_of_bounds += 1
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
            num_rays_used += 1
            num_free_updates += free_updates
            num_occ_updates += occ_updates

    probability_grid = log_odds_to_probability(log_odds_grid).astype(np.float32)
    classified_grid = classify_probability_grid(
        probability_grid=probability_grid,
        free_threshold=config.free_threshold,
        occ_threshold=config.occ_threshold,
    )

    summary = OccupancyGridSummary(
        num_scans_total=num_scans_total,
        num_scans_used=num_scans_used,
        num_rays_used=num_rays_used,
        num_rays_skipped_out_of_bounds=num_rays_skipped_out_of_bounds,
        num_free_updates=num_free_updates,
        num_occ_updates=num_occ_updates,
        num_rows=bounds.num_rows,
        num_cols=bounds.num_cols,
        x_min=bounds.x_min,
        x_max=bounds.x_max,
        y_min=bounds.y_min,
        y_max=bounds.y_max,
        resolution=config.resolution,
    )

    return OccupancyGridResult(
        config=config,
        bounds=bounds,
        log_odds_grid=log_odds_grid,
        probability_grid=probability_grid,
        classified_grid=classified_grid,
        summary=summary,
    )


# =========================
# Salvataggio e plotting
# =========================


def save_grid_arrays(result: OccupancyGridResult, output_dir: str | Path) -> None:
    """Salva le griglie numeriche principali in formato .npy."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "log_odds.npy", result.log_odds_grid)
    np.save(output_path / "probability_grid.npy", result.probability_grid)
    np.save(output_path / "classified_grid.npy", result.classified_grid)



def save_grid_metadata(result: OccupancyGridResult, output_dir: str | Path) -> None:
    """Salva metadata e summary in JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "config": asdict(result.config),
        "bounds": asdict(result.bounds),
        "summary": asdict(result.summary),
    }

    with (output_path / "occupancy_grid_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



def _plot_base_map(
    image: np.ndarray,
    bounds: GridBounds,
    title: str,
    output_file: str | Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    lidar_trajectory: Optional[Sequence[Pose2D]] = None,
) -> None:
    """Helper interno per salvare una mappa come immagine."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        image,
        origin="lower",
        extent=[bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max],
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    if lidar_trajectory:
        xs = [pose.x for pose in lidar_trajectory]
        ys = [pose.y for pose in lidar_trajectory]
        ax.plot(xs, ys, linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)



def save_probability_map_png(
    result: OccupancyGridResult,
    output_dir: str | Path,
    lidar_trajectory: Optional[Sequence[Pose2D]] = None,
) -> None:
    """Salva la mappa probabilistica come PNG."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _plot_base_map(
        image=result.probability_grid,
        bounds=result.bounds,
        title="Occupancy Probability Map",
        output_file=output_path / "occupancy_probability.png",
        vmin=0.0,
        vmax=1.0,
        lidar_trajectory=lidar_trajectory,
    )



def save_classified_map_png(
    result: OccupancyGridResult,
    output_dir: str | Path,
    lidar_trajectory: Optional[Sequence[Pose2D]] = None,
) -> None:
    """Salva la mappa classificata come PNG."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _plot_base_map(
        image=result.classified_grid,
        bounds=result.bounds,
        title="Occupancy Classified Map (0=unknown, 1=free, 2=occupied)",
        output_file=output_path / "occupancy_classified.png",
        vmin=0.0,
        vmax=2.0,
        lidar_trajectory=lidar_trajectory,
    )



def save_all_outputs(
    result: OccupancyGridResult,
    output_dir: str | Path,
    lidar_trajectory: Optional[Sequence[Pose2D]] = None,
) -> None:
    """Salva arrays, metadata e immagini principali."""
    save_grid_arrays(result, output_dir)
    save_grid_metadata(result, output_dir)
    save_probability_map_png(result, output_dir, lidar_trajectory=lidar_trajectory)
    save_classified_map_png(result, output_dir, lidar_trajectory=lidar_trajectory)


# =========================
# Cartella risultati esperimento
# =========================


def _sanitize_name(value: str) -> str:
    """Pulisce un nome per usarlo in modo sicuro come directory."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("._-") or "unnamed"



def create_experiment_output_dir(
    bag_path: str | Path,
    results_root: str | Path = "results",
    experiment_tag: str = "experiment",
    timestamp: Optional[str] = None,
) -> Path:
    """
    Crea una directory risultati del tipo:
    results/occupancy_grid_<bag_name>_<experiment_tag>_<timestamp>/

    Esempio:
    results/occupancy_grid_rosbag2_2024_12_03-14_22_49_experiment_20260421_153005/
    """
    bag_name = _sanitize_name(Path(bag_path).name)
    experiment_name = _sanitize_name(experiment_tag)
    run_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_timestamp = _sanitize_name(run_timestamp)

    dir_name = f"occupancy_grid_{bag_name}_{experiment_name}_{run_timestamp}"
    output_dir = Path(results_root) / dir_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


# =========================
# Funzioni di supporto/debug
# =========================


def extract_lidar_trajectory(data: PreprocessedBagData) -> List[Pose2D]:
    """Restituisce la traiettoria LiDAR come lista ordinata di pose 2D."""
    return [scan.lidar_pose for scan in data.aligned_scans]



def summarize_occupancy_grid(result: OccupancyGridResult) -> Dict[str, Any]:
    """Restituisce un piccolo riepilogo stampabile."""
    return asdict(result.summary)


# =========================
# CLI
# =========================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Costruzione Occupancy Grid 2D da output preprocess_bag.py"
    )

    # Input bag: la CLI richiama il preprocess e poi il mapping.
    parser.add_argument("bag_path", type=str, help="Percorso della cartella del bag ROS2")
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Cartella generale dentro cui creare una sottocartella dedicata per ogni test",
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default="experiment",
        help="Etichetta libera da inserire nel nome della cartella del test",
    )

    # Parametri preprocess già esistenti.
    parser.add_argument("--scan-topic", type=str, default=DEFAULT_SCAN_TOPIC)
    parser.add_argument("--odom-topic", type=str, default=DEFAULT_ODOM_TOPIC)
    parser.add_argument("--zed-to-lidar-dx", type=float, default=DEFAULT_ZED_TO_LIDAR_DX)
    parser.add_argument("--zed-to-lidar-dy", type=float, default=DEFAULT_ZED_TO_LIDAR_DY)
    parser.add_argument("--zed-to-lidar-dyaw", type=float, default=DEFAULT_ZED_TO_LIDAR_DYAW)
    parser.add_argument("--zed-yaw-correction", type=float, default=DEFAULT_ZED_YAW_CORRECTION)
    parser.add_argument(
        "--keep-max-range-returns",
        action="store_true",
        default=DEFAULT_KEEP_MAX_RANGE_RETURNS,
        help="Mantiene i raggi con range massimo durante il preprocess",
    )
    parser.add_argument(
        "--time-tolerance-ns",
        type=int,
        default=DEFAULT_TIME_TOLERANCE_NS,
        help="Tolleranza opzionale di clamp temporale nel preprocess",
    )

    # Parametri mapping.
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--p-occ", type=float, default=0.70)
    parser.add_argument("--p-free", type=float, default=0.35)
    parser.add_argument("--log-odds-min", type=float, default=-4.0)
    parser.add_argument("--log-odds-max", type=float, default=4.0)
    parser.add_argument("--free-threshold", type=float, default=0.35)
    parser.add_argument("--occ-threshold", type=float, default=0.65)
    parser.add_argument("--scan-stride", type=int, default=1)
    parser.add_argument(
        "--exclude-origin-cell-from-free",
        action="store_true",
        help="Non aggiorna come free la cella iniziale del LiDAR",
    )

    return parser



def main() -> None:
    args = build_arg_parser().parse_args()

    data = process_bag_for_mapping(
        bag_path=args.bag_path,
        scan_topic=args.scan_topic,
        odom_topic=args.odom_topic,
        zed_to_lidar_dx=args.zed_to_lidar_dx,
        zed_to_lidar_dy=args.zed_to_lidar_dy,
        zed_to_lidar_dyaw=args.zed_to_lidar_dyaw,
        zed_yaw_correction=args.zed_yaw_correction,
        keep_max_range_returns=args.keep_max_range_returns,
        time_tolerance_ns=args.time_tolerance_ns,
    )

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

    result = build_occupancy_grid(data, config=config)
    lidar_trajectory = extract_lidar_trajectory(data)
    output_dir = create_experiment_output_dir(
        bag_path=args.bag_path,
        results_root=args.results_root,
        experiment_tag=args.experiment_tag,
    )
    save_all_outputs(result, output_dir, lidar_trajectory=lidar_trajectory)

    print("\n=== OCCUPANCY GRID SUMMARY ===")
    for key, value in summarize_occupancy_grid(result).items():
        print(f"{key}: {value}")

    print(f"\nOutput salvati in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
