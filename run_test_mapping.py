from __future__ import annotations

"""
Script minimo di test per preprocess + occupancy grid.

Uso:
    python run_test_mapping.py

Prima di eseguirlo, aggiorna soprattutto:
- BAG_PATH
- RESULTS_ROOT se vuoi un'altra cartella generale risultati
- EXPERIMENT_TAG se vuoi distinguere i run
"""

from pathlib import Path

from preprocess_bag import process_bag_for_mapping
from build_occupancy_grid import (
    OccupancyGridConfig,
    build_occupancy_grid,
    create_experiment_output_dir,
    extract_lidar_trajectory,
    save_all_outputs,
    summarize_occupancy_grid,
)


# =========================
# Parametri da cambiare qui
# =========================

BAG_PATH = r"D:\tesi\acquisizioni\testInAula\rosbag2_2024_11_15-16_36_27"
RESULTS_ROOT = Path("results")
EXPERIMENT_TAG = "experiment"

CONFIG = OccupancyGridConfig(
    resolution=0.1,
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



def main() -> None:
    data = process_bag_for_mapping(BAG_PATH)
    result = build_occupancy_grid(data, CONFIG)

    output_dir = create_experiment_output_dir(
        bag_path=BAG_PATH,
        results_root=RESULTS_ROOT,
        experiment_tag=EXPERIMENT_TAG,
    )

    save_all_outputs(
        result,
        output_dir,
        lidar_trajectory=extract_lidar_trajectory(data),
    )

    print("\n=== OCCUPANCY GRID SUMMARY ===")
    for key, value in summarize_occupancy_grid(result).items():
        print(f"{key}: {value}")

    print(f"\nOutput salvati in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
