from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from models_mock import MOCK_MODELS, FuelModel
from route_helpers import (
    Waypoint,
    parse_waypoint_line,
    build_waypoints_from_inputs,
    legs_from_waypoint_segments,
    waypoints_from_profile,
    format_leg_summary,
)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


M_TO_FT = 3.28084


# -------------------------------
# Core data structures
# -------------------------------

@dataclass
class Leg:
    phase: str                 # "climb" | "cruise" | "descent"
    dx_nm: float
    dh_ft: float
    tas_kt: float
    target_alt_ft: float


@dataclass
class MissionProfile:
    start_alt_ft: float
    end_alt_ft: float
    total_distance_nm: float
    isa_dev_c: float
    initial_fuel_lb: float
    legs: List[Leg]


# -------------------------------
# 3-leg profile builder
# -------------------------------

def build_three_leg_profile(
    start_alt_ft: float,
    end_alt_ft: float,
    distance_nm: float,
    isa_dev_c: float,
    initial_fuel_lb: float,
    cruise_alt_ft: float,
) -> MissionProfile:
    """
    Simple 3-leg profile: climb, cruise, descent.
    Distance split 20 / 60 / 20 percent with minimum distance per leg.
    """
    total = max(distance_nm, 0.01)
    climb_d = 0.20 * total
    cruise_d = 0.60 * total
    descent_d = 0.20 * total

    min_leg = 0.01
    parts = [max(min_leg, climb_d), max(min_leg, cruise_d), max(min_leg, descent_d)]
    s = sum(parts)
    scale = total / s
    climb_d, cruise_d, descent_d = [p * scale for p in parts]

    dh_climb = float(cruise_alt_ft - start_alt_ft)
    dh_desc = float(end_alt_ft - cruise_alt_ft)

    legs = [
        Leg("climb", climb_d, dh_climb, 250.0, cruise_alt_ft),
        Leg("cruise", cruise_d, 0.0, 420.0, cruise_alt_ft),
        Leg("descent", descent_d, dh_desc, 260.0, end_alt_ft),
    ]
    return MissionProfile(start_alt_ft, end_alt_ft, total, isa_dev_c, initial_fuel_lb, legs)


# -------------------------------
# Optimizer
# -------------------------------

MIN_CRUISE_FT = 18000.0
MAX_CRUISE_FT = 41000.0


class Optimizer:
    def __init__(self, model: FuelModel) -> None:
        self.model = model

    def evaluate_leg(self, w0_lb: float, leg: Leg, isa_dev_c: float) -> Tuple[float, float]:
        if leg.phase == "climb":
            return self.model.climb(
                w0_lb, dh_ft=leg.dh_ft, dx_nm=leg.dx_nm, tas_kt=leg.tas_kt, isa_dev_c=isa_dev_c
            )
        if leg.phase == "cruise":
            return self.model.cruise(
                w0_lb,
                dx_nm=leg.dx_nm,
                tas_kt=leg.tas_kt,
                isa_dev_c=isa_dev_c,
                alt_ft=leg.target_alt_ft,
            )
        if leg.phase == "descent":
            return self.model.descent(
                w0_lb, dh_ft=leg.dh_ft, dx_nm=leg.dx_nm, tas_kt=leg.tas_kt, isa_dev_c=isa_dev_c
            )
        raise ValueError("Unknown phase")

    def simulate(self, prof: MissionProfile) -> Tuple[float, float, float]:
        fuel_used = 0.0
        time_s = 0.0
        fuel_remaining = prof.initial_fuel_lb
        for leg in prof.legs:
            f, t = self.evaluate_leg(fuel_remaining, leg, prof.isa_dev_c)
            fuel_used += f
            fuel_remaining = max(0.0, fuel_remaining - f)
            time_s += t
        return fuel_used, time_s, fuel_remaining

    def simulate_with_details(
        self, prof: MissionProfile
    ) -> Tuple[float, float, float, List[Tuple[float, float]]]:
        fuel_used = 0.0
        time_s = 0.0
        fuel_remaining = prof.initial_fuel_lb
        per_leg: List[Tuple[float, float]] = []
        for leg in prof.legs:
            f, t = self.evaluate_leg(fuel_remaining, leg, prof.isa_dev_c)
            fuel_used += f
            fuel_remaining = max(0.0, fuel_remaining - f)
            time_s += t
            per_leg.append((f, t))
        return fuel_used, time_s, fuel_remaining, per_leg

    def _generate_cruise_candidates(
        self, base: MissionProfile, selected_cruise_ft: float
    ) -> List[float]:
        """
        Generate candidate cruise altitudes around the center, with some fixed levels.
        """
        candidates = [24000.0, 28000.0, 30000.0, 32000.0, 34000.0]
        span = base.total_distance_nm
        step = 4000.0 if span >= 200.0 else 2000.0
        for delta in (0.0, -step, step):
            candidates.append(selected_cruise_ft + delta)

        out: List[float] = []
        for c in candidates:
            c = max(MIN_CRUISE_FT, min(MAX_CRUISE_FT, c))
            c = max(c, base.start_alt_ft)
            if c not in out:
                out.append(c)
        out.sort()
        return out

    def enumerate_cruise_solutions(
        self,
        base: MissionProfile,
        min_arrival_lb: Optional[float],
        center_cruise_ft: float,
    ) -> List[Dict]:
        """
        Evaluate all candidate cruise altitudes and return a list of feasible solutions.

        Each item is:
        {
            "prof": MissionProfile,
            "fuel": fuel_used,
            "time": time_s,
            "arrival": arrival_fuel,
            "per_leg": [(fuel_lb, time_s), ...],
        }

        Solutions violating min_arrival_lb (if given) are discarded.
        """
        candidates = self._generate_cruise_candidates(base, center_cruise_ft)
        items: List[Dict] = []

        for cruise_ft in candidates:
            prof = build_three_leg_profile(
                start_alt_ft=base.start_alt_ft,
                end_alt_ft=base.end_alt_ft,
                distance_nm=base.total_distance_nm,
                isa_dev_c=base.isa_dev_c,
                initial_fuel_lb=base.initial_fuel_lb,
                cruise_alt_ft=cruise_ft,
            )
            fuel_used, time_s, arrival_fuel, per_leg = self.simulate_with_details(prof)

            if (min_arrival_lb is not None) and (arrival_fuel < min_arrival_lb):
                continue

            items.append(
                {
                    "prof": prof,
                    "fuel": fuel_used,
                    "time": time_s,
                    "arrival": arrival_fuel,
                    "per_leg": per_leg,
                }
            )

        return items


# -------------------------------
# Tkinter app with two views
# -------------------------------

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        # changed: simpler window title
        self.title("CRONUS Flight Planner")
        self.geometry("1150x700")

        self.input_frame = ttk.Frame(self, padding=10)
        self.output_frame = ttk.Frame(self, padding=10)

        self.start_wp: Optional[Waypoint] = None
        self.end_wp: Optional[Waypoint] = None
        self.last_gw_kg: float = 0.0

        self._build_input_view()
        self._build_output_view()

        self.input_frame.pack(fill=tk.BOTH, expand=True)

    # ---------- view builders ----------

    def _build_input_view(self) -> None:
        f = self.input_frame

        title = ttk.Label(f, text="Mission and Model Inputs", font=("TkDefaultFont", 14, "bold"))
        title.pack(pady=(0, 10))

        grid = ttk.Frame(f)
        grid.pack()

        def add_row_entry(row: int, label: str, default: str, width: int = 18) -> tk.Entry:
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky="e", padx=(0, 8), pady=3)
            entry = ttk.Entry(grid, width=width)
            entry.insert(0, default)
            entry.grid(row=row, column=1, sticky="w", pady=3)
            return entry

        r = 0
        # Start / target waypoints as "name x y alt_m"
        self.start_line = add_row_entry(
            r, "Start waypoint (name x y alt_m)", "START 0 0 0", width=26
        ); r += 1
        self.end_line = add_row_entry(
            r, "Target waypoint (name x y alt_m)", "TRGT 300 0 900", width=26
        ); r += 1

        self.gw_kg_entry = add_row_entry(r, "Gross weight (kg)", "70000"); r += 1
        self.fuel_lb_entry = add_row_entry(r, "Initial fuel (lb)", "8000"); r += 1
        self.isa_dev_entry = add_row_entry(r, "ISA deviation (°C)", "0"); r += 1

        # Min arrival fuel constraint
        self.arrival_lb_entry = add_row_entry(r, "Min arrival fuel (lb)", "", width=18); r += 1

        # Profile type (only meaningful when no min arrival fuel)
        ttk.Label(grid, text="Profile type").grid(
            row=r, column=0, sticky="ne", padx=(0, 8), pady=(8, 3)
        )
        self.profile_mode = tk.StringVar(value="fuel")
        profile_frame = ttk.Frame(grid)
        profile_frame.grid(row=r, column=1, sticky="w", pady=(8, 3))
        self.profile_radios: List[ttk.Radiobutton] = []

        for text, val in [
            ("Fuel-optimal", "fuel"),
            ("Time-optimal", "time"),
            ("Balanced", "balanced"),
        ]:
            rb = ttk.Radiobutton(
                profile_frame, text=text, variable=self.profile_mode, value=val
            )
            rb.pack(anchor="w")
            self.profile_radios.append(rb)
        r += 1

        # Model choice
        ttk.Label(grid, text="Model").grid(row=r, column=0, sticky="e", padx=(0, 8), pady=3)
        self.model_name = tk.StringVar(value=list(MOCK_MODELS.keys())[0])
        ttk.Combobox(
            grid,
            textvariable=self.model_name,
            values=list(MOCK_MODELS.keys()),
            state="readonly",
            width=16,
        ).grid(row=r, column=1, sticky="w", pady=3)
        r += 1

        # Hold points
        ttk.Label(grid, text="Hold points\n(name x y alt_m)").grid(
            row=r, column=0, sticky="ne", padx=(0, 8), pady=(8, 0)
        )
        self.holds_text = tk.Text(grid, height=4, width=26)
        self.holds_text.insert("1.0", "HLD1 40 0 7500\nTANKR 55 0 7600")
        self.holds_text.grid(row=r, column=1, sticky="w", pady=(8, 0))
        r += 1

        btn_frame = ttk.Frame(f)
        btn_frame.pack(pady=12)
        ttk.Button(btn_frame, text="Run planner", command=self.run_planner).pack()

        # Update profile radios when min arrival fuel is edited
        self.arrival_lb_entry.bind("<KeyRelease>", lambda e: self._update_profile_mode_state())
        self._update_profile_mode_state()

    def _build_output_view(self) -> None:
        f = self.output_frame

        # changed: simpler top label
        title = ttk.Label(
            f, text="CRONUS Flight Planner",
            font=("TkDefaultFont", 14, "bold")
        )
        title.pack(pady=(0, 8))

        body = ttk.Frame(f)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        right = ttk.Frame(body)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        # Figure: top-down and profile views
        self.fig = Figure(figsize=(8.0, 5.5))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0], width_ratios=[3.0, 2.0])
        self.ax_top = self.fig.add_subplot(gs[0, :])    # top row, full width
        self.ax_prof = self.fig.add_subplot(gs[1, 0])   # bottom-left

        self.ax_top.set_title("ROUTE VIEW (Top Down)")
        self.ax_top.set_xlabel("X (NM)")
        self.ax_top.set_ylabel("Y (NM)")
        self.ax_top.grid(True)

        self.ax_prof.set_title("PROFILE VIEW (Altitude vs Distance)")
        self.ax_prof.set_xlabel("Distance (NM)")
        self.ax_prof.set_ylabel("Altitude (m)")
        self.ax_prof.grid(True)

        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right side: profile label + summary
        self.profile_mode_label = ttk.Label(right, text="Profile: (none)")
        self.profile_mode_label.pack(anchor="w", pady=(0, 4))

        ttk.Label(right, text="LEG SUMMARY (Planner Output)").pack(anchor="w", pady=(6, 0))
        self.summary = tk.Text(right, height=20, width=48, font=("Courier New", 9))
        self.summary.pack(fill=tk.BOTH, expand=True)

        ttk.Button(f, text="Back to inputs", command=self._back_to_inputs).pack(pady=6)

    # ---------- small helpers ----------

    def _update_profile_mode_state(self) -> None:
        """
        Disable profile-type radios if a min arrival fuel is specified.
        """
        txt = self.arrival_lb_entry.get().strip()
        has_min_arrival = bool(txt)
        for rb in self.profile_radios:
            if has_min_arrival:
                rb.state(["disabled"])
            else:
                rb.state(["!disabled"])

    def _back_to_inputs(self) -> None:
        self.output_frame.pack_forget()
        self.input_frame.pack(fill=tk.BOTH, expand=True)

    # ---------- main run ----------

    def run_planner(self) -> None:
        # Parse scalar inputs
        try:
            gw_kg = float(self.gw_kg_entry.get())
            fuel_lb = float(self.fuel_lb_entry.get())
            isa_dev_c = float(self.isa_dev_entry.get())
        except ValueError:
            messagebox.showerror("Input error", "Please enter valid numeric values.")
            return

        if fuel_lb < 0:
            messagebox.showerror("Input error", "Initial fuel must be nonnegative.")
            return

        # Min arrival fuel constraint
        min_arrival_lb: Optional[float] = None
        txt = self.arrival_lb_entry.get().strip()
        if txt:
            try:
                min_arrival_lb = float(txt)
            except ValueError:
                messagebox.showerror("Input error", "Min arrival fuel must be numeric.")
                return

        # Profile mode from inputs (may be overridden if min_arrival is set)
        chosen_profile_mode = self.profile_mode.get()   # "fuel" | "time" | "balanced"

        model = MOCK_MODELS[self.model_name.get()]
        opt = Optimizer(model)

        start_line = self.start_line.get().strip()
        end_line = self.end_line.get().strip()
        holds_raw = self.holds_text.get("1.0", tk.END).strip()

        # Parse start / target waypoints (altitudes in meters -> feet in helper)
        try:
            start_wp = parse_waypoint_line(start_line, "START")
            end_wp = parse_waypoint_line(end_line, "TRGT")
        except ValueError as e:
            messagebox.showerror("Waypoint error", str(e))
            return

        self.start_wp = start_wp
        self.end_wp = end_wp
        self.last_gw_kg = gw_kg

        start_alt_ft = start_wp.alt_ft
        end_alt_ft = end_wp.alt_ft

        # straight line horizontal distance between start and target
        dx = end_wp.x_nm - start_wp.x_nm
        dy = end_wp.y_nm - start_wp.y_nm
        distance_direct_nm = (dx ** 2 + dy ** 2) ** 0.5

        # heuristic initial guess for cruise altitude, used as center for search
        guess_cruise_ft = max(start_alt_ft, end_alt_ft) + 10000.0
        guess_cruise_ft = max(MIN_CRUISE_FT, min(MAX_CRUISE_FT, guess_cruise_ft))

        # ---------- case 1: no holds, multi-solution cruise optimization ----------
        if not holds_raw:
            base_prof = build_three_leg_profile(
                start_alt_ft=start_alt_ft,
                end_alt_ft=end_alt_ft,
                distance_nm=distance_direct_nm,
                isa_dev_c=isa_dev_c,
                initial_fuel_lb=fuel_lb,
                cruise_alt_ft=guess_cruise_ft,
            )

            items = opt.enumerate_cruise_solutions(
                base=base_prof,
                min_arrival_lb=min_arrival_lb,
                center_cruise_ft=guess_cruise_ft,
            )

            if not items:
                messagebox.showerror(
                    "Optimization error",
                    "No feasible cruise altitude meets the arrival fuel constraint."
                )
                return

            fuels = [it["fuel"] for it in items]
            times = [it["time"] for it in items]

            fuel_idx = min(range(len(items)), key=lambda i: fuels[i])
            time_idx = min(range(len(items)), key=lambda i: times[i])

            # Balanced: closest to "ideal" (min fuel, min time) in normalized space
            min_fuel, max_fuel = min(fuels), max(fuels)
            min_time, max_time = min(times), max(times)

            def norm(val: float, vmin: float, vmax: float) -> float:
                if vmax <= vmin:
                    return 0.0
                return (val - vmin) / (vmax - vmin)

            best_bal_idx = fuel_idx
            best_bal_score = None
            for i, it in enumerate(items):
                f_n = norm(it["fuel"], min_fuel, max_fuel)
                t_n = norm(it["time"], min_time, max_time)
                score = (f_n ** 2 + t_n ** 2) ** 0.5
                if (best_bal_score is None) or (score < best_bal_score):
                    best_bal_score = score
                    best_bal_idx = i

            # If min arrival fuel is specified, force fuel-optimal within constraint
            if min_arrival_lb is not None:
                profile_mode_effective = "fuel"
            else:
                profile_mode_effective = chosen_profile_mode

            if profile_mode_effective == "time":
                idx = time_idx
                mode_name = "time-optimal"
            elif profile_mode_effective == "balanced":
                idx = best_bal_idx
                mode_name = "balanced (fuel/time compromise)"
            else:
                idx = fuel_idx
                mode_name = "fuel-optimal"

            item = items[idx]
            prof = item["prof"]
            fuel_used = item["fuel"]
            time_s = item["time"]
            arrival_fuel = item["arrival"]
            per_leg = item["per_leg"]

            waypoints = waypoints_from_profile(prof, start_wp, end_wp)
            self._update_plots(waypoints, prof)

            self.profile_mode_label.configure(text=f"Profile: {mode_name}")

            self.summary.delete("1.0", tk.END)
            summary_text = format_leg_summary(
                waypoints=waypoints,
                profile=prof,
                per_leg=per_leg,
                fuel_used=fuel_used,
                time_s=time_s,
                arrival_fuel=arrival_fuel,
                objective_desc=mode_name,
                gross_weight_kg=self.last_gw_kg,
            )
            self.summary.insert(tk.END, summary_text)

        # ---------- case 2: holds present, multi-waypoint route (no optimization) ----------
        else:
            try:
                waypoints = build_waypoints_from_inputs(start_line, holds_raw, end_line)
            except ValueError as e:
                messagebox.showerror("Hold point error", str(e))
                return

            leg_meta, total_dist = legs_from_waypoint_segments(waypoints)
            legs = [Leg(**meta) for meta in leg_meta]

            prof = MissionProfile(
                start_alt_ft=start_alt_ft,
                end_alt_ft=end_alt_ft,
                total_distance_nm=total_dist,
                isa_dev_c=isa_dev_c,
                initial_fuel_lb=fuel_lb,
                legs=legs,
            )

            fuel_used, time_s, arr, per_leg = opt.simulate_with_details(prof)

            if (min_arrival_lb is not None) and (arr < min_arrival_lb):
                messagebox.showwarning(
                    "Arrival fuel warning",
                    "Route arrives with less fuel than the specified minimum."
                )

            self.profile_mode_label.configure(text="Profile: fixed route (holds)")

            self._update_plots(waypoints, prof)
            self.summary.delete("1.0", tk.END)
            summary_text = format_leg_summary(
                waypoints=waypoints,
                profile=prof,
                per_leg=per_leg,
                fuel_used=fuel_used,
                time_s=time_s,
                arrival_fuel=arr,
                objective_desc="fixed route (holds, no cruise optimization)",
                gross_weight_kg=self.last_gw_kg,
            )
            self.summary.insert(tk.END, summary_text)

        # Switch to output view
        self.input_frame.pack_forget()
        self.output_frame.pack(fill=tk.BOTH, expand=True)

    # ---------- plotting ----------

    def _update_plots(self, waypoints: List[Waypoint], prof: MissionProfile) -> None:
        # Top down
        self.ax_top.clear()
        xs = [wp.x_nm for wp in waypoints]
        ys = [wp.y_nm for wp in waypoints]
        self.ax_top.plot(xs, ys, marker="o")

        # Add labels
        for wp in waypoints:
            alt_m = wp.alt_ft / M_TO_FT
            label = f"{wp.name}\n{int(round(alt_m))} m"
            self.ax_top.text(wp.x_nm, wp.y_nm, label, ha="center", va="bottom", fontsize=8)

        # New: add some padding so labels do not go off the axes
        if xs and ys:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            dx = max(x_max - x_min, 1.0)
            dy = max(y_max - y_min, 1.0)
            pad_x = 0.05 * dx
            pad_y = 0.20 * dy  # extra vertical space for text above points
            self.ax_top.set_xlim(x_min - pad_x, x_max + pad_x)
            self.ax_top.set_ylim(y_min - pad_y, y_max + pad_y)

        self.ax_top.set_title("ROUTE VIEW (Top Down)")
        self.ax_top.set_xlabel("X (NM)")
        self.ax_top.set_ylabel("Y (NM)")
        self.ax_top.grid(True)

        # Profile (altitude in meters)
        self.ax_prof.clear()
        dist_cum = 0.0
        xs_prof = [0.0]
        ys_prof = [waypoints[0].alt_ft / M_TO_FT]
        for leg in prof.legs:
            dist_cum += leg.dx_nm
            xs_prof.append(dist_cum)
            ys_prof.append(leg.target_alt_ft / M_TO_FT)
        self.ax_prof.plot(xs_prof, ys_prof, marker="o")
        self.ax_prof.set_title("PROFILE VIEW (Altitude vs Distance)")
        self.ax_prof.set_xlabel("Distance (NM)")
        self.ax_prof.set_ylabel("Altitude (m)")
        self.ax_prof.grid(True)

        self.fig.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    App().mainloop()
