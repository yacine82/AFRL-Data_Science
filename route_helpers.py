from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


M_TO_FT = 3.28084


@dataclass
class Waypoint:
    """
    Simple 2D waypoint plus altitude.

    x_nm, y_nm are top-down coordinates in NM.
    alt_ft is altitude in feet (internal model units).
    """
    name: str
    x_nm: float
    y_nm: float
    alt_ft: float


# ---------- parsing helpers ----------

def parse_waypoint_line(line: str, default_name: str) -> Waypoint:
    """
    Parse a waypoint line of the form:
        name x_nm y_nm alt_m
    or:
        x_nm y_nm alt_m   (name implied by default_name)

    Returns a Waypoint with altitude converted to feet.
    """
    cleaned = line.strip()
    if not cleaned:
        raise ValueError(f"Empty waypoint line where '{default_name}' was expected.")

    parts = [p for p in cleaned.replace(",", " ").split() if p]

    if len(parts) == 3:
        name = default_name
        xs, ys, zs = parts
    elif len(parts) >= 4:
        name = parts[0]
        xs, ys, zs = parts[1:4]
    else:
        raise ValueError(
            f"Invalid waypoint line '{line}'. Expected: name x y alt_m  or  x y alt_m"
        )

    try:
        x_nm = float(xs)
        y_nm = float(ys)
        alt_m = float(zs)
    except ValueError:
        raise ValueError(
            f"Invalid numeric value in waypoint line '{line}'. "
            "Format: name x y alt_m  or  x y alt_m"
        )

    alt_ft = alt_m * M_TO_FT
    return Waypoint(name=name, x_nm=x_nm, y_nm=y_nm, alt_ft=alt_ft)


def build_waypoints_from_inputs(
    start_line: str,
    holds_text: str,
    end_line: str,
) -> List[Waypoint]:
    """
    Build an ordered waypoint list from start / holds / end text.
    """
    waypoints: List[Waypoint] = []
    start_wp = parse_waypoint_line(start_line, "START")
    waypoints.append(start_wp)

    if holds_text:
        idx = 1
        for line in holds_text.splitlines():
            if not line.strip():
                continue
            default = f"H{idx}"
            wp = parse_waypoint_line(line, default)
            waypoints.append(wp)
            idx += 1

    end_wp = parse_waypoint_line(end_line, "TRGT")
    waypoints.append(end_wp)
    return waypoints


# ---------- legs and derived waypoints ----------

def legs_from_waypoint_segments(
    waypoints: List[Waypoint],
) -> Tuple[List[dict], float]:
    """
    Convert a waypoint chain into leg metadata suitable for Leg objects.
    """
    legs_meta: List[dict] = []
    total_dist = 0.0

    for a, b in zip(waypoints[:-1], waypoints[1:]):
        dx = b.x_nm - a.x_nm
        dy = b.y_nm - a.y_nm
        seg_dist_nm = math.hypot(dx, dy)
        dh_ft = b.alt_ft - a.alt_ft

        if seg_dist_nm <= 0.0:
            continue

        if dh_ft > 0:
            phase = "climb"
            tas = 250.0
        elif dh_ft < 0:
            phase = "descent"
            tas = 260.0
        else:
            phase = "cruise"
            tas = 420.0

        legs_meta.append(
            {
                "phase": phase,
                "dx_nm": seg_dist_nm,
                "dh_ft": dh_ft,
                "tas_kt": tas,
                "target_alt_ft": b.alt_ft,
            }
        )
        total_dist += seg_dist_nm

    return legs_meta, total_dist


def waypoints_from_profile(
    profile,
    start_wp: Waypoint,
    end_wp: Waypoint,
) -> List[Waypoint]:
    """
    Build waypoints along the straight line from start_wp to end_wp,
    at each leg boundary in the MissionProfile.
    """
    waypoints: List[Waypoint] = []
    total = max(profile.total_distance_nm, 1e-6)

    dx = end_wp.x_nm - start_wp.x_nm
    dy = end_wp.y_nm - start_wp.y_nm

    dist_cum = 0.0
    alt = profile.start_alt_ft
    waypoints.append(Waypoint(start_wp.name, start_wp.x_nm, start_wp.y_nm, alt))

    for idx, leg in enumerate(profile.legs, start=1):
        dist_cum += leg.dx_nm
        frac = dist_cum / total
        x = start_wp.x_nm + frac * dx
        y = start_wp.y_nm + frac * dy
        alt = leg.target_alt_ft
        name = f"WP{idx}"
        if idx == len(profile.legs):
            name = end_wp.name
        waypoints.append(Waypoint(name, x, y, alt))

    return waypoints


# ---------- leg summary formatting ----------

def _fmt_mmss(time_s: float) -> str:
    m = int(time_s // 60)
    s = int(round(time_s - m * 60))
    return f"{m:02d}:{s:02d}"


def format_leg_summary(
    waypoints: List[Waypoint],
    profile,
    per_leg: List[tuple],
    fuel_used: float,
    time_s: float,
    arrival_fuel: float,
    objective_desc: str,
    gross_weight_kg: float,
) -> str:
    """
    Produce a planner style leg summary text block.

    per_leg is a list of (fuel_lb, time_s) aligned with profile.legs.
    """
    lines: List[str] = []
    lines.append("Leg  From   To     Phase   Alt(m)  Dist(NM)  Time   Fuel(lb)")
    lines.append("-------------------------------------------------------------")

    assert len(waypoints) >= 2
    assert len(waypoints) == len(profile.legs) + 1

    for idx, (leg, (fuel_lb, t_s)) in enumerate(zip(profile.legs, per_leg), start=1):
        wp_from = waypoints[idx - 1]
        wp_to = waypoints[idx]
        alt_m = leg.target_alt_ft / M_TO_FT
        dist_nm = leg.dx_nm
        time_str = _fmt_mmss(t_s)

        lines.append(
            f"{idx:>2}  {wp_from.name:<5} {wp_to.name:<5} "
            f"{leg.phase[:5].upper():<6} "
            f"{int(round(alt_m)):>6}  {dist_nm:>7.1f}  {time_str:>5}  {fuel_lb:>7.0f}"
        )

    lines.append("")
    lines.append(
        f"Total: {profile.total_distance_nm:,.1f} NM, "
        f"{fuel_used:,.0f} lb used, arrival fuel {arrival_fuel:,.0f} lb"
    )
    lines.append(f"Block time: {_fmt_mmss(time_s)}")
    lines.append(f"Objective: {objective_desc}")
    lines.append(f"Gross weight: {gross_weight_kg:,.0f} kg")

    return "\n".join(lines)
