from __future__ import annotations

"""
Plot di debug per verificare il funzionamento di preprocess_bag.py.

Plot prodotti:
1. traiettoria ZED e LiDAR nel piano globale
2. tutti i punti globali + traiettoria LiDAR
3. una singola scansione nel frame locale del LiDAR

Uso tipico:
    python plot_preprocessed_data.py D:/tesi/acquisizioni/rosbag2_2024_12_03-14_22_49

Prerequisiti:
- preprocess_bag.py nello stesso progetto / cartella importabile
- matplotlib
- numpy
- rosbags
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from preprocess_bag import (
    DEFAULT_KEEP_MAX_RANGE_RETURNS,
    DEFAULT_ODOM_TOPIC,
    DEFAULT_SCAN_TOPIC,
    DEFAULT_ZED_TO_LIDAR_DX,
    DEFAULT_ZED_TO_LIDAR_DY,
    DEFAULT_ZED_TO_LIDAR_DYAW,
    DEFAULT_ZED_YAW_CORRECTION,
    PreprocessedBagData,
    process_bag_for_mapping,
)


# =========================
# Utility semplici
# =========================


def ensure_output_dir(bag_path: str | Path) -> Path:
    """Crea una cartella di output accanto al bag."""
    bag = Path(bag_path)
    out_dir = bag.parent / f"{bag.name}_plots_occgrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir



def set_equal_axes() -> None:
    """Mantiene le stesse unità su asse x e y per non deformare la geometria."""
    plt.axis("equal")



def select_scan_index(data: PreprocessedBagData, scan_index: int | None) -> int:
    """
    Sceglie quale scansione plottare nel frame locale.

    Se l'utente non specifica un indice, usiamo quella centrale tra le scansioni
    allineate: spesso è più rappresentativa della prima.
    """
    if not data.aligned_scans:
        raise ValueError("Nessuna scansione allineata disponibile per i plot.")

    if scan_index is None:
        return len(data.aligned_scans) // 2

    if scan_index < 0 or scan_index >= len(data.aligned_scans):
        raise IndexError(
            f"scan_index fuori range: {scan_index}. "
            f"Valori ammessi: 0 .. {len(data.aligned_scans) - 1}"
        )
    return scan_index



def collect_all_global_points(data: PreprocessedBagData) -> Tuple[np.ndarray, np.ndarray]:
    """Appiattisce tutti i punti globali in due array x/y."""
    xs: List[float] = []
    ys: List[float] = []

    for scan in data.aligned_scans:
        for p in scan.global_points:
            xs.append(p.x)
            ys.append(p.y)

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)



def collect_zed_trajectory(data: PreprocessedBagData) -> Tuple[np.ndarray, np.ndarray]:
    """Estrae la traiettoria ZED dalle pose odometriche grezze."""
    xs = np.asarray([s.x for s in data.odom_samples], dtype=float)
    ys = np.asarray([s.y for s in data.odom_samples], dtype=float)
    return xs, ys



def collect_lidar_trajectory_from_aligned_scans(
    data: PreprocessedBagData,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estrae la traiettoria LiDAR dalle scansioni allineate."""
    xs = np.asarray([s.lidar_pose.x for s in data.aligned_scans], dtype=float)
    ys = np.asarray([s.lidar_pose.y for s in data.aligned_scans], dtype=float)
    return xs, ys


# =========================
# Plot 1
# =========================


def plot_trajectory_zed_lidar(data: PreprocessedBagData, output_dir: Path) -> Path:
    """Plot 2D della traiettoria ZED e della traiettoria LiDAR."""
    zed_x, zed_y = collect_zed_trajectory(data)
    lidar_x, lidar_y = collect_lidar_trajectory_from_aligned_scans(data)

    plt.figure(figsize=(9, 7))
    plt.plot(zed_x, zed_y, marker="o", linestyle="-", markersize=3, label="ZED odom")
    plt.plot(lidar_x, lidar_y, marker="o", linestyle="-", markersize=3, label="LiDAR aligned")

    if len(zed_x) > 0:
        plt.scatter([zed_x[0]], [zed_y[0]], marker="x", s=80, label="Start ZED")
    if len(lidar_x) > 0:
        plt.scatter([lidar_x[-1]], [lidar_y[-1]], marker="x", s=80, label="End LiDAR")

    plt.title("Plot 1 - Traiettoria ZED e LiDAR")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.grid(True)
    set_equal_axes()
    plt.tight_layout()

    out_path = output_dir / "plot_1_trajectory_zed_lidar.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# =========================
# Plot 2
# =========================


def plot_global_points_with_lidar_trajectory(data: PreprocessedBagData, output_dir: Path) -> Path:
    """Plot di tutti i punti globali sovrapposti alla traiettoria del LiDAR."""
    points_x, points_y = collect_all_global_points(data)
    lidar_x, lidar_y = collect_lidar_trajectory_from_aligned_scans(data)

    plt.figure(figsize=(10, 8))
    if len(points_x) > 0:
        plt.scatter(points_x, points_y, s=4, alpha=0.5, label="Global points")
    plt.plot(lidar_x, lidar_y, marker="o", linestyle="-", markersize=3, label="LiDAR trajectory")

    if len(lidar_x) > 0:
        plt.scatter([lidar_x[0]], [lidar_y[0]], marker="x", s=80, label="Start LiDAR")

    plt.title("Plot 2 - Tutti i punti globali + traiettoria LiDAR")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.grid(True)
    set_equal_axes()
    plt.tight_layout()

    out_path = output_dir / "plot_2_global_points_with_lidar_trajectory.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# =========================
# Plot 3
# =========================


def plot_single_local_scan(
    data: PreprocessedBagData,
    output_dir: Path,
    scan_index: int | None = None,
) -> Path:
    """Plot di una singola scansione nel frame locale del LiDAR."""
    idx = select_scan_index(data, scan_index)
    scan = data.aligned_scans[idx]

    local_x = np.asarray([p.x for p in scan.local_points], dtype=float)
    local_y = np.asarray([p.y for p in scan.local_points], dtype=float)

    plt.figure(figsize=(8, 8))
    if len(local_x) > 0:
        plt.scatter(local_x, local_y, s=6, alpha=0.7, label="Local points")

    # Origine del LiDAR nel suo frame locale.
    plt.scatter([0.0], [0.0], marker="x", s=100, label="LiDAR origin")

    plt.title(f"Plot 3 - Singola scansione locale (scan_index={idx})")
    plt.xlabel("x_local [m]")
    plt.ylabel("y_local [m]")
    plt.legend()
    plt.grid(True)
    set_equal_axes()
    plt.tight_layout()

    out_path = output_dir / f"plot_3_single_local_scan_idx_{idx}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# =========================
# Runner principale
# =========================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera plot di debug a partire dall'output di preprocess_bag.py"
    )
    parser.add_argument("bag_path", type=str, help="Percorso della cartella del bag ROS2")
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
        help="Mantiene i raggi a range massimo anche nei plot",
    )
    parser.add_argument(
        "--scan-index",
        type=int,
        default=None,
        help="Indice della scansione da usare per il plot locale. Se omesso usa la centrale.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostra i plot a schermo oltre a salvarli su file.",
    )
    return parser



def generate_debug_plots(
    bag_path: str | Path,
    scan_topic: str = DEFAULT_SCAN_TOPIC,
    odom_topic: str = DEFAULT_ODOM_TOPIC,
    zed_to_lidar_dx: float = DEFAULT_ZED_TO_LIDAR_DX,
    zed_to_lidar_dy: float = DEFAULT_ZED_TO_LIDAR_DY,
    zed_to_lidar_dyaw: float = DEFAULT_ZED_TO_LIDAR_DYAW,
    zed_yaw_correction: float = DEFAULT_ZED_YAW_CORRECTION,
    keep_max_range_returns: bool = DEFAULT_KEEP_MAX_RANGE_RETURNS,
    scan_index: int | None = None,
) -> Tuple[PreprocessedBagData, List[Path]]:
    """
    Esegue preprocess_bag, genera i 3 plot richiesti e restituisce:
    - i dati preprocessati
    - i percorsi dei file PNG salvati
    """
    data = process_bag_for_mapping(
        bag_path=bag_path,
        scan_topic=scan_topic,
        odom_topic=odom_topic,
        zed_to_lidar_dx=zed_to_lidar_dx,
        zed_to_lidar_dy=zed_to_lidar_dy,
        zed_to_lidar_dyaw=zed_to_lidar_dyaw,
        zed_yaw_correction=zed_yaw_correction,
        keep_max_range_returns=keep_max_range_returns,
    )

    output_dir = ensure_output_dir(bag_path)
    outputs = [
        plot_trajectory_zed_lidar(data, output_dir),
        plot_global_points_with_lidar_trajectory(data, output_dir),
        plot_single_local_scan(data, output_dir, scan_index=scan_index),
    ]
    return data, outputs



def main() -> None:
    args = build_arg_parser().parse_args()

    data, output_paths = generate_debug_plots(
        bag_path=args.bag_path,
        scan_topic=args.scan_topic,
        odom_topic=args.odom_topic,
        zed_to_lidar_dx=args.zed_to_lidar_dx,
        zed_to_lidar_dy=args.zed_to_lidar_dy,
        zed_to_lidar_dyaw=args.zed_to_lidar_dyaw,
        zed_yaw_correction=args.zed_yaw_correction,
        keep_max_range_returns=args.keep_max_range_returns,
        scan_index=args.scan_index,
    )

    print("\n=== DEBUG PLOTS GENERATI ===")
    print(f"Bag: {data.bag_path}")
    print(f"Scansioni allineate: {len(data.aligned_scans)}")
    for path in output_paths:
        print(path)

    if args.show:
        for path in output_paths:
            img = plt.imread(path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.title(path.name)
        plt.show()


if __name__ == "__main__":
    main()
