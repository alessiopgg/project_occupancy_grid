from __future__ import annotations

"""
Pipeline di preprocessing per Occupancy Grid 2D da bag ROS2.

Obiettivo:
- leggere in memoria i messaggi `/zed/zed_node/odom` e `/scan`
- usare `bag_time_ns` come riferimento temporale comune
- convertire il quaternione in yaw
- interpolare la posa ZED al tempo di ogni scan
- applicare una trasformazione rigida approssimata ZED -> LiDAR
- convertire i raggi LiDAR in punti 2D locali
- trasformare i punti nel frame globale `odom`

Dipendenze principali:
- rosbags
- numpy

Esempio d'uso:
    python preprocess_bag.py /percorso/al_bag_folder
"""

import argparse
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


# =========================
# Configurazione di default
# =========================

DEFAULT_SCAN_TOPIC = "/scan"
DEFAULT_ODOM_TOPIC = "/zed/zed_node/odom"

# Ipotesi iniziale ricavata dal contesto del progetto:
# il LiDAR è circa 15 cm dietro la ZED, senza offset laterale sul piano.
DEFAULT_ZED_TO_LIDAR_DX = -0.15
DEFAULT_ZED_TO_LIDAR_DY = 0.0
DEFAULT_ZED_TO_LIDAR_DYAW = 0.0
DEFAULT_ZED_YAW_CORRECTION = 0.0

# Se True, i raggi a distanza massima vengono tenuti come "free-only".
# In questo file li convertiamo comunque in punti; la distinzione servirà poi
# soprattutto nel mapping. Per partire in modo semplice li scartiamo.
DEFAULT_KEEP_MAX_RANGE_RETURNS = False

# Tolleranza per considerare uno scan fuori dal range temporale dell'odom.
DEFAULT_TIME_TOLERANCE_NS = 0


# =========================
# Dataclass principali
# =========================

@dataclass(frozen=True)
class OdomSample:
    bag_time_ns: int
    x: float
    y: float
    qx: float
    qy: float
    qz: float
    qw: float
    yaw: float


@dataclass(frozen=True)
class ScanSample:
    bag_time_ns: int
    ranges: Tuple[float, ...]


@dataclass(frozen=True)
class LaserConfig:
    frame_id: str
    angle_min: float
    angle_max: float
    angle_increment: float
    time_increment: float
    scan_time: float
    range_min: float
    range_max: float
    ray_count: int


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class LocalPoint2D:
    x: float
    y: float
    angle: float
    range_value: float
    ray_index: int


@dataclass(frozen=True)
class GlobalPoint2D:
    x: float
    y: float
    angle_local: float
    range_value: float
    ray_index: int


@dataclass(frozen=True)
class AlignedScan:
    bag_time_ns: int
    zed_pose: Pose2D
    lidar_pose: Pose2D
    local_points: Tuple[LocalPoint2D, ...]
    global_points: Tuple[GlobalPoint2D, ...]


@dataclass(frozen=True)
class PreprocessedBagData:
    bag_path: str
    scan_topic: str
    odom_topic: str
    laser_config: LaserConfig
    odom_samples: Tuple[OdomSample, ...]
    scan_samples: Tuple[ScanSample, ...]
    aligned_scans: Tuple[AlignedScan, ...]


# =========================
# Utility matematiche
# =========================


def normalize_angle(angle: float) -> float:
    """Riporta un angolo in radianti nell'intervallo [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))



def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """
    Converte un quaternione in yaw (rotazione attorno a z).

    In questo progetto usiamo la componente planare della posa, quindi ci basta
    lo yaw. La formula è quella standard per estrarre lo yaw dal quaternione.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return normalize_angle(math.atan2(siny_cosp, cosy_cosp))



def lerp(a: float, b: float, alpha: float) -> float:
    """Interpolazione lineare standard."""
    return a + alpha * (b - a)



def interpolate_yaw(yaw0: float, yaw1: float, alpha: float) -> float:
    """
    Interpolazione dello yaw evitando errori sul salto -pi / +pi.

    Invece di interpolare direttamente yaw1-yaw0, usiamo la differenza angolare
    normalizzata, così se per esempio passiamo da 179° a -179° otteniamo una
    piccola variazione e non una rotazione enorme nel verso sbagliato.
    """
    dyaw = normalize_angle(yaw1 - yaw0)
    return normalize_angle(yaw0 + alpha * dyaw)


# =========================
# Lettura bag ROS2
# =========================


def _ensure_ros2_bag_path(bag_path: str | Path) -> Path:
    """Valida il percorso del bag ROS2."""
    path = Path(bag_path)
    if not path.exists():
        raise FileNotFoundError(f"Bag path non trovato: {path}")
    return path



def _get_typestore():
    """Restituisce il typestore ROS2 generico usato da rosbags."""
    return get_typestore(Stores.ROS2_HUMBLE)



def extract_odom_data(
    bag_path: str | Path,
    odom_topic: str = DEFAULT_ODOM_TOPIC,
) -> List[OdomSample]:
    """
    Estrae in memoria i campioni odometrici utili alla pipeline.

    Per ogni messaggio odom teniamo:
    - bag_time_ns
    - x, y
    - quaternione completo
    - yaw già calcolato e normalizzato
    """
    bag_path = _ensure_ros2_bag_path(bag_path)
    typestore = _get_typestore()
    samples: List[OdomSample] = []

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == odom_topic]
        if not connections:
            raise ValueError(f"Topic odom non trovato nel bag: {odom_topic}")

        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
            qx = float(msg.pose.pose.orientation.x)
            qy = float(msg.pose.pose.orientation.y)
            qz = float(msg.pose.pose.orientation.z)
            qw = float(msg.pose.pose.orientation.w)
            yaw = quaternion_to_yaw(qx, qy, qz, qw)

            samples.append(
                OdomSample(
                    bag_time_ns=int(timestamp_ns),
                    x=x,
                    y=y,
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    qw=qw,
                    yaw=yaw,
                )
            )

    samples.sort(key=lambda s: s.bag_time_ns)
    if not samples:
        raise ValueError(f"Nessun campione odom estratto da {odom_topic}")
    return samples



def extract_scan_data(
    bag_path: str | Path,
    scan_topic: str = DEFAULT_SCAN_TOPIC,
) -> Tuple[LaserConfig, List[ScanSample]]:
    """
    Estrae in memoria i dati essenziali delle scansioni LiDAR.

    Scelta progettuale:
    - per ogni scan teniamo solo bag_time_ns + ranges
    - i parametri geometrici del laser vengono letti una volta e messi in
      LaserConfig, assumendo che siano costanti durante il bag
    """
    bag_path = _ensure_ros2_bag_path(bag_path)
    typestore = _get_typestore()

    laser_config: Optional[LaserConfig] = None
    samples: List[ScanSample] = []

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == scan_topic]
        if not connections:
            raise ValueError(f"Topic scan non trovato nel bag: {scan_topic}")

        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            if laser_config is None:
                ranges_tuple = tuple(float(r) for r in msg.ranges)
                laser_config = LaserConfig(
                    frame_id=str(msg.header.frame_id),
                    angle_min=float(msg.angle_min),
                    angle_max=float(msg.angle_max),
                    angle_increment=float(msg.angle_increment),
                    time_increment=float(msg.time_increment),
                    scan_time=float(msg.scan_time),
                    range_min=float(msg.range_min),
                    range_max=float(msg.range_max),
                    ray_count=len(ranges_tuple),
                )
                ranges = ranges_tuple
            else:
                ranges = tuple(float(r) for r in msg.ranges)

            samples.append(ScanSample(bag_time_ns=int(timestamp_ns), ranges=ranges))

    samples.sort(key=lambda s: s.bag_time_ns)
    if laser_config is None:
        raise ValueError(f"Nessuna scansione estratta da {scan_topic}")
    return laser_config, samples


# =========================
# Interpolazione temporale
# =========================


def _find_bracketing_odom_indices(
    odom_samples: Sequence[OdomSample],
    target_time_ns: int,
) -> Optional[Tuple[int, int]]:
    """
    Cerca i due indici odom che racchiudono temporalmente target_time_ns.

    Restituisce:
    - (i, i) se esiste un match esatto
    - (i0, i1) con i0 < i1 se serve interpolare
    - None se il tempo è fuori dall'intervallo dell'odom
    """
    if not odom_samples:
        return None

    first_t = odom_samples[0].bag_time_ns
    last_t = odom_samples[-1].bag_time_ns
    if target_time_ns < first_t or target_time_ns > last_t:
        return None

    # Ricerca binaria semplice su lista ordinata.
    lo = 0
    hi = len(odom_samples) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        t_mid = odom_samples[mid].bag_time_ns
        if t_mid == target_time_ns:
            return mid, mid
        if t_mid < target_time_ns:
            lo = mid + 1
        else:
            hi = mid - 1

    # A questo punto hi è l'indice immediatamente prima, lo subito dopo.
    i0 = hi
    i1 = lo
    if i0 < 0 or i1 >= len(odom_samples):
        return None
    return i0, i1



def interpolate_pose(
    odom_samples: Sequence[OdomSample],
    target_time_ns: int,
) -> Optional[Pose2D]:
    """
    Stima la posa ZED al tempo target_time_ns.

    Se troviamo un match esatto usiamo quella posa.
    Altrimenti interpoliamo x, y e yaw tra i due campioni odom adiacenti.
    """
    bracket = _find_bracketing_odom_indices(odom_samples, target_time_ns)
    if bracket is None:
        return None

    i0, i1 = bracket
    s0 = odom_samples[i0]
    s1 = odom_samples[i1]

    if i0 == i1 or s0.bag_time_ns == s1.bag_time_ns:
        return Pose2D(x=s0.x, y=s0.y, yaw=s0.yaw)

    alpha = (target_time_ns - s0.bag_time_ns) / (s1.bag_time_ns - s0.bag_time_ns)
    x = lerp(s0.x, s1.x, alpha)
    y = lerp(s0.y, s1.y, alpha)
    yaw = interpolate_yaw(s0.yaw, s1.yaw, alpha)
    return Pose2D(x=x, y=y, yaw=yaw)


# =========================
# Trasformazioni 2D
# =========================


def transform_local_point_to_global(
    point_xy: Tuple[float, float],
    pose: Pose2D,
) -> Tuple[float, float]:
    """
    Applica una trasformazione rigida 2D: punto locale -> punto globale.

    Formula standard:
    p_global = R(yaw) * p_local + t
    """
    px, py = point_xy
    cy = math.cos(pose.yaw)
    sy = math.sin(pose.yaw)
    gx = pose.x + cy * px - sy * py
    gy = pose.y + sy * px + cy * py
    return gx, gy



def compose_lidar_pose(
    zed_pose: Pose2D,
    zed_to_lidar_dx: float = DEFAULT_ZED_TO_LIDAR_DX,
    zed_to_lidar_dy: float = DEFAULT_ZED_TO_LIDAR_DY,
    zed_to_lidar_dyaw: float = DEFAULT_ZED_TO_LIDAR_DYAW,
    zed_yaw_correction: float = DEFAULT_ZED_YAW_CORRECTION,
) -> Pose2D:
    """
    Ricava la posa globale del LiDAR dalla posa globale ZED.

    Prima applichiamo un'eventuale correzione yaw alla ZED, poi componiamo la
    trasformazione rigida fissa ZED -> LiDAR sul piano.
    """
    corrected_zed_yaw = normalize_angle(zed_pose.yaw + zed_yaw_correction)
    corrected_zed_pose = Pose2D(x=zed_pose.x, y=zed_pose.y, yaw=corrected_zed_yaw)

    lidar_x, lidar_y = transform_local_point_to_global(
        (zed_to_lidar_dx, zed_to_lidar_dy), corrected_zed_pose
    )
    lidar_yaw = normalize_angle(corrected_zed_yaw + zed_to_lidar_dyaw)
    return Pose2D(x=lidar_x, y=lidar_y, yaw=lidar_yaw)


# =========================
# Scan -> punti 2D
# =========================


def is_valid_range(
    range_value: float,
    laser_config: LaserConfig,
    keep_max_range_returns: bool = DEFAULT_KEEP_MAX_RANGE_RETURNS,
) -> bool:
    """
    Filtra i range invalidi.

    Regole minime adottate nel progetto:
    - scarta 0.0
    - scarta non finiti
    - scarta <= range_min
    - gestione del range_max configurabile
    """
    if not math.isfinite(range_value):
        return False
    if range_value == 0.0:
        return False
    if range_value <= laser_config.range_min:
        return False
    if range_value > laser_config.range_max:
        return False
    if not keep_max_range_returns and math.isclose(range_value, laser_config.range_max):
        return False
    return True



def scan_to_local_points(
    scan_sample: ScanSample,
    laser_config: LaserConfig,
    keep_max_range_returns: bool = DEFAULT_KEEP_MAX_RANGE_RETURNS,
) -> List[LocalPoint2D]:
    """
    Converte una scansione in punti 2D locali del LiDAR.

    Ogni misura valida (r, theta) viene convertita in:
        x = r * cos(theta)
        y = r * sin(theta)
    """
    local_points: List[LocalPoint2D] = []

    for i, range_value in enumerate(scan_sample.ranges):
        if not is_valid_range(range_value, laser_config, keep_max_range_returns):
            continue

        angle = laser_config.angle_min + i * laser_config.angle_increment
        x_local = range_value * math.cos(angle)
        y_local = range_value * math.sin(angle)

        local_points.append(
            LocalPoint2D(
                x=x_local,
                y=y_local,
                angle=angle,
                range_value=range_value,
                ray_index=i,
            )
        )

    return local_points



def local_to_global_points(
    local_points: Sequence[LocalPoint2D],
    lidar_pose: Pose2D,
) -> List[GlobalPoint2D]:
    """Trasforma una lista di punti locali del LiDAR nel frame globale odom."""
    global_points: List[GlobalPoint2D] = []

    for p in local_points:
        gx, gy = transform_local_point_to_global((p.x, p.y), lidar_pose)
        global_points.append(
            GlobalPoint2D(
                x=gx,
                y=gy,
                angle_local=p.angle,
                range_value=p.range_value,
                ray_index=p.ray_index,
            )
        )

    return global_points


# =========================
# Pipeline principale
# =========================


def process_bag_for_mapping(
    bag_path: str | Path,
    scan_topic: str = DEFAULT_SCAN_TOPIC,
    odom_topic: str = DEFAULT_ODOM_TOPIC,
    zed_to_lidar_dx: float = DEFAULT_ZED_TO_LIDAR_DX,
    zed_to_lidar_dy: float = DEFAULT_ZED_TO_LIDAR_DY,
    zed_to_lidar_dyaw: float = DEFAULT_ZED_TO_LIDAR_DYAW,
    zed_yaw_correction: float = DEFAULT_ZED_YAW_CORRECTION,
    keep_max_range_returns: bool = DEFAULT_KEEP_MAX_RANGE_RETURNS,
    time_tolerance_ns: int = DEFAULT_TIME_TOLERANCE_NS,
) -> PreprocessedBagData:
    """
    Esegue tutta la pipeline di preprocessing in memoria.

    Per ogni scan:
    1. usa bag_time_ns come tempo di riferimento
    2. stima la posa ZED a quel tempo
    3. ricava la posa LiDAR con la trasformazione fissa ZED -> LiDAR
    4. converte i raggi validi in punti 2D locali
    5. trasforma i punti nel frame globale odom
    """
    odom_samples = extract_odom_data(bag_path, odom_topic=odom_topic)
    laser_config, scan_samples = extract_scan_data(bag_path, scan_topic=scan_topic)

    aligned_scans: List[AlignedScan] = []
    skipped_scans_outside_time = 0

    for scan in scan_samples:
        zed_pose = interpolate_pose(odom_samples, scan.bag_time_ns)
        if zed_pose is None:
            # Piccola estensione opzionale: se è fuori di poco dall'intervallo
            # consentiamo il clamp al campione più vicino, ma solo se richiesto.
            if time_tolerance_ns > 0:
                first_dt = abs(scan.bag_time_ns - odom_samples[0].bag_time_ns)
                last_dt = abs(scan.bag_time_ns - odom_samples[-1].bag_time_ns)
                if first_dt <= time_tolerance_ns:
                    zed_pose = Pose2D(odom_samples[0].x, odom_samples[0].y, odom_samples[0].yaw)
                elif last_dt <= time_tolerance_ns:
                    zed_pose = Pose2D(odom_samples[-1].x, odom_samples[-1].y, odom_samples[-1].yaw)

        if zed_pose is None:
            skipped_scans_outside_time += 1
            continue

        lidar_pose = compose_lidar_pose(
            zed_pose,
            zed_to_lidar_dx=zed_to_lidar_dx,
            zed_to_lidar_dy=zed_to_lidar_dy,
            zed_to_lidar_dyaw=zed_to_lidar_dyaw,
            zed_yaw_correction=zed_yaw_correction,
        )

        local_points = scan_to_local_points(
            scan,
            laser_config,
            keep_max_range_returns=keep_max_range_returns,
        )
        global_points = local_to_global_points(local_points, lidar_pose)

        aligned_scans.append(
            AlignedScan(
                bag_time_ns=scan.bag_time_ns,
                zed_pose=zed_pose,
                lidar_pose=lidar_pose,
                local_points=tuple(local_points),
                global_points=tuple(global_points),
            )
        )

    if skipped_scans_outside_time > 0:
        print(f"[INFO] Scan saltati perché fuori dal range temporale odom: {skipped_scans_outside_time}")

    return PreprocessedBagData(
        bag_path=str(Path(bag_path)),
        scan_topic=scan_topic,
        odom_topic=odom_topic,
        laser_config=laser_config,
        odom_samples=tuple(odom_samples),
        scan_samples=tuple(scan_samples),
        aligned_scans=tuple(aligned_scans),
    )


# =========================
# Funzioni opzionali di debug
# =========================


def summarize_preprocessed_data(data: PreprocessedBagData) -> Dict[str, Any]:
    """Restituisce un piccolo riepilogo utile per debug rapido."""
    first_odom_t = data.odom_samples[0].bag_time_ns if data.odom_samples else None
    last_odom_t = data.odom_samples[-1].bag_time_ns if data.odom_samples else None
    first_scan_t = data.scan_samples[0].bag_time_ns if data.scan_samples else None
    last_scan_t = data.scan_samples[-1].bag_time_ns if data.scan_samples else None

    total_local_points = sum(len(s.local_points) for s in data.aligned_scans)
    total_global_points = sum(len(s.global_points) for s in data.aligned_scans)
    mean_points_per_scan = (
        total_local_points / len(data.aligned_scans) if data.aligned_scans else 0.0
    )

    return {
        "bag_path": data.bag_path,
        "scan_topic": data.scan_topic,
        "odom_topic": data.odom_topic,
        "num_odom_samples": len(data.odom_samples),
        "num_scan_samples": len(data.scan_samples),
        "num_aligned_scans": len(data.aligned_scans),
        "first_odom_bag_time_ns": first_odom_t,
        "last_odom_bag_time_ns": last_odom_t,
        "first_scan_bag_time_ns": first_scan_t,
        "last_scan_bag_time_ns": last_scan_t,
        "laser_frame_id": data.laser_config.frame_id,
        "laser_ray_count": data.laser_config.ray_count,
        "laser_angle_min": data.laser_config.angle_min,
        "laser_angle_max": data.laser_config.angle_max,
        "laser_angle_increment": data.laser_config.angle_increment,
        "total_local_points": total_local_points,
        "total_global_points": total_global_points,
        "mean_points_per_scan": mean_points_per_scan,
    }



def debug_first_aligned_scan(data: PreprocessedBagData, max_points: int = 10) -> Dict[str, Any]:
    """Restituisce un piccolo dump della prima scansione allineata."""
    if not data.aligned_scans:
        return {"message": "Nessuna scansione allineata disponibile."}

    scan = data.aligned_scans[0]
    return {
        "bag_time_ns": scan.bag_time_ns,
        "zed_pose": asdict(scan.zed_pose),
        "lidar_pose": asdict(scan.lidar_pose),
        "num_local_points": len(scan.local_points),
        "num_global_points": len(scan.global_points),
        "local_points_preview": [asdict(p) for p in scan.local_points[:max_points]],
        "global_points_preview": [asdict(p) for p in scan.global_points[:max_points]],
    }


# =========================
# CLI minima per PyCharm / terminale
# =========================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocessing bag ROS2 per Occupancy Grid 2D da LiDAR + ZED"
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
        help="Mantiene i raggi con range massimo invece di scartarli",
    )
    parser.add_argument(
        "--time-tolerance-ns",
        type=int,
        default=DEFAULT_TIME_TOLERANCE_NS,
        help="Tolleranza opzionale per clamp temporale fuori dall'intervallo odom",
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

    summary = summarize_preprocessed_data(data)
    first_scan_debug = debug_first_aligned_scan(data, max_points=5)

    print("\n=== SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n=== FIRST ALIGNED SCAN DEBUG ===")
    for key, value in first_scan_debug.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
