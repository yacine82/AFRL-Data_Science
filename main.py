from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Mock models interface and catalog
from models_mock import MOCK_MODELS, FuelModel

# Matplotlib embed
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Leg:
    phase: str               # "climb" | "cruise" | "descent"
    dx_nm: float
    dh_ft: float
    tas_kt: float
    target_alt_ft: Optional[float] = None


@dataclass
class MissionProfile:
    start_alt_ft: float
    end_alt_ft: float
    total_distance_nm: float
    isa_dev_c: float
    initial_fuel_lb: float
    legs: List[Leg]


# -----------------------------
# Helper formatting
# -----------------------------
def fmt_time_hms(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_time_minsec(sec: float) -> str:
    sec = max(0, int(round(sec)))
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"


# -----------------------------
# Profile construction
# -----------------------------
def build_three_leg_profile(
    start_alt_ft: float,
    end_alt_ft: float,
    distance_nm: float,
    isa_dev_c: float,
    initial_fuel_lb: float,
    cruise_alt_ft: float,
) -> MissionProfile:
    """
    Build a simple 3 leg profile with robust handling for short distances.
    Dist split is 20 percent climb, 60 percent cruise, 20 percent descent,
    with minimum positive distance per leg and exact total sum.
    """
    total = max(distance_nm, 0.01)
    # provisional split
    climb_d = 0.20 * total
    cruise_d = 0.60 * total
    descent_d = 0.20 * total

    # ensure no zero length legs; re-normalize if needed
    min_leg = 0.01
    parts = [max(min_leg, climb_d), max(min_leg, cruise_d), max(min_leg, descent_d)]
    s = sum(parts)
    scale = total / s
    climb_d, cruise_d, descent_d = [p * scale for p in parts]

    # vertical changes
    dh_climb = float(cruise_alt_ft - start_alt_ft)
    dh_desc = float(end_alt_ft - cruise_alt_ft)

    legs = [
        Leg("climb", dx_nm=climb_d, dh_ft=dh_climb, tas_kt=250.0, target_alt_ft=cruise_alt_ft),
        Leg("cruise", dx_nm=cruise_d, dh_ft=0.0, tas_kt=420.0, target_alt_ft=cruise_alt_ft),
        Leg("descent", dx_nm=descent_d, dh_ft=dh_desc, tas_kt=260.0, target_alt_ft=end_alt_ft),
    ]
    return MissionProfile(start_alt_ft, end_alt_ft, total, isa_dev_c, initial_fuel_lb, legs)


# -----------------------------
# Optimizer
# -----------------------------
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
                w0_lb, dx_nm=leg.dx_nm, tas_kt=leg.tas_kt, isa_dev_c=isa_dev_c, alt_ft=leg.target_alt_ft or 0.0
            )
        if leg.phase == "descent":
            return self.model.descent(
                w0_lb, dh_ft=leg.dh_ft, dx_nm=leg.dx_nm, tas_kt=leg.tas_kt, isa_dev_c=isa_dev_c
            )
        raise ValueError("Unknown phase")

    def simulate(self, prof: MissionProfile) -> Tuple[float, float, float]:
        """
        Returns fuel_used, time_s, fuel_remaining.
        """
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
        """
        Returns fuel_used, time_s, fuel_remaining, and a list per leg of (fuel, time).
        """
        fuel_used = 0.0
        time_s = 0.0
        fuel_remaining = prof.initial_fuel_lb
        per_leg: List[Tuple[float, float]] = []

        for leg in prof.legs:
            f, t = self.evaluate_leg(fuel_remaining, leg, prof.isa_dev_c)
            per_leg.append((f, t))
            fuel_used += f
            fuel_remaining = max(0.0, fuel_remaining - f)
            time_s += t

        return fuel_used, time_s, fuel_remaining, per_leg

    def _generate_cruise_candidates(self, selected_ft: float, start_alt_ft: float, distance_nm: float) -> List[int]:
        """
        Include the user selection and nearby sensible points.
        Adapt the spread lightly to total distance.
        """
        # Base set
        base = [24000, 28000, 30000, 32000, 34000]

        # Include selected and neighbors
        step = 4000 if distance_nm >= 200 else 2000
        bundle = [selected_ft, selected_ft - step, selected_ft + step]

        # Combine, clamp, dedupe, sort
        raw = base + bundle
        clamped = []
        for a in raw:
            a = max(MIN_CRUISE_FT, min(MAX_CRUISE_FT, float(a)))
            # never plan cruise below start altitude
            a = max(a, start_alt_ft, MIN_CRUISE_FT)
            clamped.append(int(round(a)))
        return sorted(set(clamped))

    def optimize_cruise_alt(
        self, base: MissionProfile, objective: str, min_arrival_lb: Optional[float]
    ) -> Tuple[MissionProfile, float, float, float]:
        """
        Sweep cruise altitude candidates and choose best by objective.
        If objective is arrival-fuel constrained, optimize fuel subject to feasibility.
        """
        selected = base.legs[1].target_alt_ft or 30000.0
        candidates = self._generate_cruise_candidates(selected, base.start_alt_ft, base.total_distance_nm)

        best = None
        best_score = math.inf

        for alt in candidates:
            prof = build_three_leg_profile(
                start_alt_ft=base.start_alt_ft,
                end_alt_ft=base.end_alt_ft,
                distance_nm=base.total_distance_nm,
                isa_dev_c=base.isa_dev_c,
                initial_fuel_lb=base.initial_fuel_lb,
                cruise_alt_ft=float(alt),
            )
            f, t, arr = self.simulate(prof)

            if min_arrival_lb is not None and arr < min_arrival_lb:
                continue

            score = f if objective == "min_fuel" else t
            if score < best_score:
                best = (prof, f, t, arr)
                best_score = score

        if best is None:
            raise ValueError("No feasible solution met the arrival fuel constraint")
        return best


# -----------------------------
# GUI application
# -----------------------------
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Flight Path Optimizer (Mock)")
        self.geometry("980x640")

        # Left: inputs
        frm = ttk.Frame(self, padding=8)
        frm.pack(side=tk.LEFT, fill=tk.Y)

        self.start_alt = self._add_labeled_entry(frm, "Start altitude ft", "3000")
        self.end_alt = self._add_labeled_entry(frm, "End altitude ft", "3000")
        self.distance = self._add_labeled_entry(frm, "Distance nm", "300")
        self.fuel_lb = self._add_labeled_entry(frm, "Initial fuel lb", "8000")
        self.isa_dev = self._add_labeled_entry(frm, "ISA deviation C", "0")

        ttk.Label(frm, text="Objective").pack(anchor="w", pady=(8, 0))
        self.objective = tk.StringVar(value="min_fuel")
        for k, v in [
            ("Minimum fuel", "min_fuel"),
            ("Minimum time", "min_time"),
            ("Required arrival fuel", "arrival_fuel"),
        ]:
            ttk.Radiobutton(frm, text=k, variable=self.objective, value=v, command=self._toggle_arrival).pack(anchor="w")

        self.arrival_lb = self._add_labeled_entry(frm, "Min arrival fuel lb", "1200")
        self.arrival_lb.configure(state="disabled")

        ttk.Label(frm, text="Cruise altitude (select)").pack(anchor="w", pady=(8, 0))
        self.cruise_alt = tk.StringVar(value="30000")
        alt_box = ttk.Combobox(
            frm,
            textvariable=self.cruise_alt,
            values=["24000", "28000", "30000", "32000", "34000"],
            state="readonly",
        )
        alt_box.pack(fill=tk.X)

        ttk.Label(frm, text="Model").pack(anchor="w", pady=(8, 0))
        self.model_key = tk.StringVar(value="mock_v1")
        model_box = ttk.Combobox(frm, textvariable=self.model_key, values=list(MOCK_MODELS.keys()), state="readonly")
        model_box.pack(fill=tk.X)

        run_btn = ttk.Button(frm, text="Optimize", command=self.run_optimize)
        run_btn.pack(fill=tk.X, pady=10)

        # Right: outputs
        out = ttk.Frame(self, padding=8)
        out.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Plot
        self.fig = Figure(figsize=(6.2, 4.2))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Distance, nm")
        self.ax.set_ylabel("Altitude, ft")
        self.ax.set_title("Altitude vs Distance")
        self.canvas = FigureCanvasTkAgg(self.fig, master=out)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Text summary
        ttk.Label(out, text="Leg summary").pack(anchor="w")
        self.summary = tk.Text(out, height=10, wrap="word")
        self.summary.pack(fill=tk.BOTH, expand=False)

    def _add_labeled_entry(self, parent: tk.Widget, label: str, default: str) -> ttk.Entry:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
        e = ttk.Entry(parent)
        e.insert(0, default)
        e.pack(fill=tk.X)
        return e

    def _toggle_arrival(self) -> None:
        if self.objective.get() == "arrival_fuel":
            self.arrival_lb.configure(state="normal")
        else:
            self.arrival_lb.configure(state="disabled")

    # -----------------------------
    # Run optimization
    # -----------------------------
    def run_optimize(self) -> None:
        # Parse inputs
        try:
            start_alt = float(self.start_alt.get())
            end_alt = float(self.end_alt.get())
            distance_nm = float(self.distance.get())
            fuel_lb = float(self.fuel_lb.get())
            isa_dev_c = float(self.isa_dev.get())
            cruise_alt_ft = float(self.cruise_alt.get())
            model = MOCK_MODELS[self.model_key.get()]
            objective = self.objective.get()
            min_arrival = float(self.arrival_lb.get()) if objective == "arrival_fuel" else None
        except Exception as e:
            messagebox.showerror("Input error", f"Please check your inputs. {e}")
            return

        # Basic validation with helpful messages
        if distance_nm <= 0:
            messagebox.showerror("Input error", "Distance must be positive.")
            return
        if fuel_lb < 0:
            messagebox.showerror("Input error", "Initial fuel cannot be negative.")
            return
        # Gentle clamp on ISA for sanity, keep value flowing to the model
        if isa_dev_c < -80 or isa_dev_c > 80:
            messagebox.showwarning("ISA deviation", "ISA deviation is unusual; using entered value anyway.")

        # Build a base profile with the user selected cruise altitude
        base_prof = build_three_leg_profile(
            start_alt_ft=start_alt,
            end_alt_ft=end_alt,
            distance_nm=distance_nm,
            isa_dev_c=isa_dev_c,
            initial_fuel_lb=fuel_lb,
            cruise_alt_ft=cruise_alt_ft,
        )

        opt = Optimizer(model)

        # Optimize
        try:
            if objective in ("min_fuel", "min_time"):
                prof, fuel_used, time_s, arr = opt.optimize_cruise_alt(base_prof, objective, None)
            else:
                prof, fuel_used, time_s, arr = opt.optimize_cruise_alt(base_prof, "min_fuel", min_arrival)
        except Exception as e:
            messagebox.showerror("Optimization error", str(e))
            return

        # For detailed summaries, simulate once more to get per-leg numbers
        fuel_used2, time_s2, arr2, per_leg = opt.simulate_with_details(prof)
        # Keep totals from the optimization run, but the detailed list is from the second pass
        # They should be identical for deterministic models

        # -----------------------------
        # Update plot
        # -----------------------------
        xs = [0.0]
        ys = [prof.start_alt_ft]
        d = 0.0
        for leg in prof.legs:
            d += leg.dx_nm
            ys.append(ys[-1] + leg.dh_ft)
            xs.append(d)

        self.ax.clear()
        self.ax.plot(xs, ys, marker="o")
        self.ax.set_xlabel("Distance, nm")
        self.ax.set_ylabel("Altitude, ft")
        self.ax.set_title("Altitude vs Distance")
        # axis limits for clarity
        self.ax.set_xlim(0.0, prof.total_distance_nm)
        ymin = min(ys)
        ymax = max(ys)
        pad = max(300.0, 0.05 * max(500.0, ymax - ymin))
        self.ax.set_ylim(ymin - pad, ymax + pad)
        self.ax.grid(True)
        self.canvas.draw_idle()

        # -----------------------------
        # Text summary
        # -----------------------------
        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, f"Fuel used lb: {fuel_used:.1f}\n")
        self.summary.insert(tk.END, f"Time s: {time_s:.1f} ({fmt_time_hms(time_s)})\n")
        self.summary.insert(tk.END, f"Arrival fuel lb: {arr:.1f}\n")

        # Per leg lines
        for idx, (leg, (lfuel, ltime)) in enumerate(zip(prof.legs, per_leg), start=1):
            time_str = fmt_time_minsec(ltime)
            if leg.phase == "climb":
                angle_deg = math.degrees(math.atan2(leg.dh_ft, max(1e-6, leg.dx_nm * 6076.12)))
                text = (
                    f"Leg {idx}: climb to {int(leg.target_alt_ft)} ft over {leg.dx_nm:.1f} nm at {leg.tas_kt:.0f} kt\n"
                    f"    Angle: {angle_deg:.2f}°, Time: {time_str}, Fuel: {lfuel:.1f} lb\n"
                )
            elif leg.phase == "cruise":
                text = (
                    f"Leg {idx}: cruise at {int(leg.target_alt_ft)} ft for {leg.dx_nm:.1f} nm at {leg.tas_kt:.0f} kt\n"
                    f"    Time: {time_str}, Fuel: {lfuel:.1f} lb\n"
                )
            else:
                angle_deg = math.degrees(math.atan2(leg.dh_ft, max(1e-6, leg.dx_nm * 6076.12)))
                text = (
                    f"Leg {idx}: descend to {int(leg.target_alt_ft)} ft over {leg.dx_nm:.1f} nm at {leg.tas_kt:.0f} kt\n"
                    f"    Angle: {angle_deg:.2f}°, Time: {time_str}, Fuel: {lfuel:.1f} lb\n"
                )
            self.summary.insert(tk.END, text)


if __name__ == "__main__":
    App().mainloop()
