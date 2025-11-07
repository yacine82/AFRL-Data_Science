from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

from models_mock import MOCK_MODELS, FuelModel

# Matplotlib embed
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Simple mission structures kept in this file to honor the 2-file requirement
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

class Optimizer:
    def __init__(self, model: FuelModel) -> None:
        self.model = model

    def evaluate_leg(self, w0_lb: float, leg: Leg, isa_dev_c: float) -> Tuple[float, float]:
        if leg.phase == "climb":
            return self.model.climb(w0_lb, dh_ft=leg.dh_ft, dx_nm=leg.dx_nm, tas_kt=leg.tas_kt, isa_dev_c=isa_dev_c)
        if leg.phase == "cruise":
            return self.model.cruise(w0_lb, dx_nm=leg.dx_nm, tas_kt=leg.tas_kt, isa_dev_c=isa_dev_c, alt_ft=leg.target_alt_ft or 0.0)
        if leg.phase == "descent":
            return self.model.descent(w0_lb, dh_ft=leg.dh_ft, dx_nm=leg.dx_nm, tas_kt=leg.tas_kt, isa_dev_c=isa_dev_c)
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

    def optimize_cruise_alt(self, base: MissionProfile, objective: str, min_arrival_lb: Optional[float]) -> Tuple[MissionProfile, float, float, float]:
        # Try a small set of cruise alts and choose best per objective
        candidates = [24000, 28000, 30000, 32000, 34000]
        best = None
        best_score = math.inf

        for alt in candidates:
            prof = build_three_leg_profile(
                start_alt_ft=base.start_alt_ft,
                end_alt_ft=base.end_alt_ft,
                distance_nm=base.total_distance_nm,
                isa_dev_c=base.isa_dev_c,
                initial_fuel_lb=base.initial_fuel_lb,
                cruise_alt_ft=alt
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

def build_three_leg_profile(start_alt_ft: float, end_alt_ft: float, distance_nm: float, isa_dev_c: float, initial_fuel_lb: float, cruise_alt_ft: float) -> MissionProfile:
    climb_d = distance_nm * 0.2
    cruise_d = distance_nm * 0.6
    descent_d = distance_nm * 0.2
    legs = [
        Leg("climb", dx_nm=climb_d, dh_ft=cruise_alt_ft - start_alt_ft, tas_kt=250.0, target_alt_ft=cruise_alt_ft),
        Leg("cruise", dx_nm=cruise_d, dh_ft=0.0, tas_kt=420.0, target_alt_ft=cruise_alt_ft),
        Leg("descent", dx_nm=descent_d, dh_ft=end_alt_ft - cruise_alt_ft, tas_kt=260.0, target_alt_ft=end_alt_ft),
    ]
    return MissionProfile(start_alt_ft, end_alt_ft, distance_nm, isa_dev_c, initial_fuel_lb, legs)

def fmt_time_sec_to_minsec(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Flight Path Optimizer (Mock)")
        self.geometry("980x640")

        # Inputs frame
        frm = ttk.Frame(self, padding=8)
        frm.pack(side=tk.LEFT, fill=tk.Y)

        self.start_alt = self._add_labeled_entry(frm, "Start altitude ft", "3000")
        self.end_alt = self._add_labeled_entry(frm, "End altitude ft", "3000")
        self.distance = self._add_labeled_entry(frm, "Distance nm", "300")
        self.fuel_lb = self._add_labeled_entry(frm, "Initial fuel lb", "8000")
        self.isa_dev = self._add_labeled_entry(frm, "ISA deviation C", "0")

        ttk.Label(frm, text="Objective").pack(anchor="w", pady=(8, 0))
        self.objective = tk.StringVar(value="min_fuel")
        for k, v in [("Minimum fuel", "min_fuel"), ("Minimum time", "min_time"), ("Required arrival fuel", "arrival_fuel")]:
            ttk.Radiobutton(frm, text=k, variable=self.objective, value=v, command=self._toggle_arrival).pack(anchor="w")

        self.arrival_lb = self._add_labeled_entry(frm, "Min arrival fuel lb", "1200")
        self.arrival_lb.configure(state="disabled")

        ttk.Label(frm, text="Cruise altitude (select)").pack(anchor="w", pady=(8, 0))
        self.cruise_alt = tk.StringVar(value="30000")
        alt_box = ttk.Combobox(frm, textvariable=self.cruise_alt, values=["24000", "28000", "30000", "32000", "34000"], state="readonly")
        alt_box.pack(fill=tk.X)

        ttk.Label(frm, text="Model").pack(anchor="w", pady=(8, 0))
        self.model_key = tk.StringVar(value="mock_v1")
        model_box = ttk.Combobox(frm, textvariable=self.model_key, values=list(MOCK_MODELS.keys()), state="readonly")
        model_box.pack(fill=tk.X)

        run_btn = ttk.Button(frm, text="Optimize", command=self.run_optimize)
        run_btn.pack(fill=tk.X, pady=10)

        # Output frame
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

    def run_optimize(self) -> None:
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

        # Build a base profile to pass parameters, then search cruise alt if needed
        base_prof = build_three_leg_profile(
            start_alt_ft=start_alt,
            end_alt_ft=end_alt,
            distance_nm=distance_nm,
            isa_dev_c=isa_dev_c,
            initial_fuel_lb=fuel_lb,
            cruise_alt_ft=cruise_alt_ft
        )

        opt = Optimizer(model)

        try:
            if objective in ("min_fuel", "min_time"):
                prof, fuel_used, time_s, arr = opt.optimize_cruise_alt(base_prof, objective, None)
            else:
                prof, fuel_used, time_s, arr = opt.optimize_cruise_alt(base_prof, "min_fuel", min_arrival)
        except Exception as e:
            messagebox.showerror("Optimization error", str(e))
            return

        # Update plot
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
        self.ax.grid(True)
        self.canvas.draw_idle()

        # Text summary
        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, f"Fuel used lb: {fuel_used:.1f}\n")
        self.summary.insert(tk.END, f"Time s: {time_s:.1f}\n")
        self.summary.insert(tk.END, f"Arrival fuel lb: {arr:.1f}\n")
        for idx, leg in enumerate(prof.legs, 1):
            time_leg_s = (leg.dx_nm / leg.tas_kt) * 3600.0
        time_str = fmt_time_sec_to_minsec(time_leg_s)
        angle_deg = math.degrees(math.atan2(leg.dh_ft, leg.dx_nm * 6076.12))

        if leg.phase == "climb":
            text = (
                f"Leg {idx}: climb to {int(leg.target_alt_ft)} ft over {leg.dx_nm:.1f} nm at {leg.tas_kt:.0f} kt\n"
                f"    Angle: {angle_deg:.2f}°, Time: {time_str}\n"
            )
        elif leg.phase == "cruise":
            text = (
                f"Leg {idx}: cruise at {int(leg.target_alt_ft)} ft for {leg.dx_nm:.1f} nm at {leg.tas_kt:.0f} kt\n"
                f"    Time: {time_str}\n"
            )
        else:
            text = (
                f"Leg {idx}: descend to {int(leg.target_alt_ft)} ft over {leg.dx_nm:.1f} nm at {leg.tas_kt:.0f} kt\n"
                f"    Angle: {angle_deg:.2f}°, Time: {time_str}\n"
            )

        self.summary.insert(tk.END, text)



if __name__ == "__main__":
    App().mainloop()
