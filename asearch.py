#!/usr/bin/env python3
"""
A* waypoint-based flight path planner (grid world)

Grid:
 - X: 0..100 nm at 1 nm resolution (inclusive) -> 101 points
 - Y: 0..100 nm at 1 nm resolution -> 101 points
 - Z: 0..60000 ft in 500 ft steps -> 121 levels

Neighbors:
 - 26-connected (dx,dy,dz in {-1,0,1}, not all zero)

Objective:
 - 'time' (minimize travel time)
 - 'fuel' (minimize fuel burn)

Inputs (GUI):
 - start x,y (nm), start alt (ft)
 - end x,y (nm), end alt (ft)
 - objective selector
 - obstacle list (JSON-like lines) OR predefined obstacles

Outputs:
 - Top-down X-Y path plot
 - Altitude vs along-track distance plot (z in ft)
 - Text summary with totals and per-step legs
"""

import math
import heapq
import json
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Tuple, List, Dict, Optional

# Matplotlib embed
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------- Grid params -------
X_MAX_NM = 100
Y_MAX_NM = 100
DX_NM = 1.0
DZ_FT = 500.0
Z_MAX_FT = 60000.0

NX = int(X_MAX_NM / DX_NM) + 1   # 101
NY = int(Y_MAX_NM / DX_NM) + 1
NZ = int(Z_MAX_FT / DZ_FT) + 1   # 121

# Safety: check sizes
assert NX * NY * NZ <= 2_000_000, "Grid is very large; modify parameters."

# ------- Base performance constants (phase nominals) -------
TAS_CRUISE_KT  = 420.0
TAS_CLIMB_KT   = 250.0
TAS_DESCENT_KT = 260.0

FUEL_CRUISE_LBPH  = 2600.0
FUEL_CLIMB_LBPH   = 3500.0
FUEL_DESCENT_LBPH = 900.0

# Vertical rates for time realism
CLIMB_RATE_FPM   = 2000.0
DESCENT_RATE_FPM = 2500.0

# Utility conversions
NM_TO_FT = 6076.12

# For heuristic we use optimistic rates
# Cap TAS improvements to keep heuristic conservative
CRUISE_TAS_CAP_FACTOR = 1.12  # up to +12 percent at high altitude
MAX_TAS_KT = TAS_CRUISE_KT * CRUISE_TAS_CAP_FACTOR
MIN_FUEL_RATE_LBPH = min(FUEL_CRUISE_LBPH, FUEL_CLIMB_LBPH, FUEL_DESCENT_LBPH)

# ------- Helper functions -------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def alt_to_level(alt_ft: float) -> int:
    """Convert alt in ft to nearest grid level index."""
    lvl = int(round(alt_ft / DZ_FT))
    return clamp(lvl, 0, NZ - 1)

def level_to_alt(lvl: int) -> float:
    return lvl * DZ_FT

def fmt_time_sec_to_minsec(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    m = int(sec // 60)
    s = int(round(sec % 60))
    return f"{m:02d}:{s:02d}"

def fmt_time_hms(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def euclidean_3d_nm(x1,y1,z1,x2,y2,z2) -> float:
    """Distance in NM: convert vertical ft -> nm via NM_TO_FT."""
    dx = (x1 - x2)
    dy = (y1 - y2)
    dz_ft = (z1 - z2) * DZ_FT
    dz_nm = dz_ft / NM_TO_FT
    return math.sqrt(dx*dx + dy*dy + dz_nm*dz_nm)

# ------- Altitude-aware performance shaping -------
def tas_at_alt_phase(alt_ft: float, phase: str) -> float:
    """
    Simple TAS model that increases with altitude up to a capped factor.
    """
    if phase == "climb":
        base = TAS_CLIMB_KT
        gain_per_kft = 0.002   # +0.2 percent per 1 kft
        cap = 1.08
    elif phase == "descent":
        base = TAS_DESCENT_KT
        gain_per_kft = 0.002
        cap = 1.08
    else:
        base = TAS_CRUISE_KT
        gain_per_kft = 0.003   # +0.3 percent per 1 kft
        cap = CRUISE_TAS_CAP_FACTOR
    kft = max(alt_ft, 0.0) / 1000.0
    factor = min(1.0 + gain_per_kft * kft, cap)
    return base * factor

def fuel_rate_at_alt_phase(alt_ft: float, phase: str) -> float:
    """
    Fuel rate improves with altitude, has a floor, and adds a low-alt penalty below 10 kft.
    """
    if phase == "climb":
        base = FUEL_CLIMB_LBPH
        reduce_per_kft = 0.006
        floor = 0.70
    elif phase == "descent":
        base = FUEL_DESCENT_LBPH
        reduce_per_kft = 0.004
        floor = 0.75
    else:
        base = FUEL_CRUISE_LBPH
        reduce_per_kft = 0.008
        floor = 0.65
    kft = max(alt_ft, 0.0) / 1000.0
    factor = max(floor, 1.0 - reduce_per_kft * kft)

    # penalty for low altitude to discourage "sea level cruise"
    low_alt_pen = 0.0
    if alt_ft < 10000.0:
        low_alt_pen = (10000.0 - alt_ft) / 10000.0 * 0.20   # up to +20 percent at ground

    return base * (factor + low_alt_pen)

# ------- Obstacles -------
# Obstacle structure: dict with xmin,xmax,ymin,ymax,zmin_ft,zmax_ft
DEFAULT_OBSTACLES = [
    {"xmin":20,"xmax":40,"ymin":30,"ymax":40,"zmin_ft":0,"zmax_ft":15000},
    {"xmin":60,"xmax":80,"ymin":60,"ymax":80,"zmin_ft":20000,"zmax_ft":40000},
]

def parse_obstacles(text: str) -> List[Dict]:
    """
    Accepts a JSON list or newline-separated simple dicts.
    If parsing fails, returns empty list.
    """
    if not text.strip():
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # Try line-by-line small JSON dicts
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
            out.append(o)
        except Exception:
            # attempt simple "xmin=..,xmax=..,..." pairs
            try:
                pairs = [p.strip() for p in line.split(",")]
                d = {}
                for p in pairs:
                    k,v = p.split("=")
                    d[k.strip()] = float(v.strip())
                out.append(d)
            except Exception:
                continue
    return out

def obstacle_blocks_node(obs: Dict, x:int,y:int,z:int) -> bool:
    z_ft = level_to_alt(z)
    return (obs.get("xmin", -1e9) <= x <= obs.get("xmax", 1e9) and
            obs.get("ymin", -1e9) <= y <= obs.get("ymax", 1e9) and
            obs.get("zmin_ft", -1e9) <= z_ft <= obs.get("zmax_ft", 1e9))

def edge_crosses_obstacle(a: Tuple[int,int,int], b: Tuple[int,int,int], obs: Dict) -> bool:
    """
    Conservative mid-sampling to prevent edges from sneaking through prisms.
    """
    ax, ay, az = a; bx, by, bz = b
    mx = int(round(0.5 * (ax + bx)))
    my = int(round(0.5 * (ay + by)))
    mz = int(round(0.5 * (az + bz)))
    return (obstacle_blocks_node(obs, ax, ay, az) or
            obstacle_blocks_node(obs, mx, my, mz) or
            obstacle_blocks_node(obs, bx, by, bz))

# ------- A* search -------
def neighbors_of(node: Tuple[int,int,int]) -> List[Tuple[int,int,int]]:
    x,y,z = node
    res = []
    for dx in (-1,0,1):
        nx = x + dx
        if nx < 0 or nx >= NX: continue
        for dy in (-1,0,1):
            ny = y + dy
            if ny < 0 or ny >= NY: continue
            for dz in (-1,0,1):
                nz = z + dz
                if nz < 0 or nz >= NZ: continue
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                res.append((nx,ny,nz))
    return res

def cost_between(a: Tuple[int,int,int], b: Tuple[int,int,int]) -> Tuple[float,float]:
    """
    Returns (fuel_lb, time_s) between adjacent nodes a->b.
    Altitude-aware TAS and fuel rate, plus vertical time via climb/descent rates.
    """
    (ax,ay,az) = a
    (bx,by,bz) = b
    dx = (bx - ax)
    dy = (by - ay)
    horiz_nm = math.hypot(dx, dy) * DX_NM
    dz_ft = (bz - az) * DZ_FT

    # use altitude at the "to" node for evaluation
    alt_ft = level_to_alt(bz)

    # phase and vertical time
    if dz_ft > 1e-6:
        phase = "climb"
        tas = tas_at_alt_phase(alt_ft, phase)
        fuel_rate = fuel_rate_at_alt_phase(alt_ft, phase)
        vert_time_hr = (dz_ft / max(CLIMB_RATE_FPM, 1e-3)) / 60.0
    elif dz_ft < -1e-6:
        phase = "descent"
        tas = tas_at_alt_phase(alt_ft, phase)
        fuel_rate = fuel_rate_at_alt_phase(alt_ft, phase)
        vert_time_hr = (abs(dz_ft) / max(DESCENT_RATE_FPM, 1e-3)) / 60.0
    else:
        phase = "cruise"
        tas = tas_at_alt_phase(alt_ft, phase)
        fuel_rate = fuel_rate_at_alt_phase(alt_ft, phase)
        vert_time_hr = 0.0

    horiz_time_hr = (horiz_nm / tas) if horiz_nm > 0 else 0.0
    time_hr = max(horiz_time_hr, vert_time_hr)  # crude coupling so climbs take time
    time_s = time_hr * 3600.0

    # base fuel proportional to time in this phase
    fuel = fuel_rate * time_hr

    # small vertical energy term
    vertical_work_factor = 0.01
    if dz_ft > 0:
        fuel += dz_ft * vertical_work_factor
    elif dz_ft < 0:
        fuel += abs(dz_ft) * (vertical_work_factor * 0.25)

    # ensure nonzero time
    if time_s == 0.0:
        time_s = (0.001 / MAX_TAS_KT) * 3600.0
        fuel += MIN_FUEL_RATE_LBPH * (time_s / 3600.0)

    return fuel, time_s

def heuristic(a: Tuple[int,int,int], goal: Tuple[int,int,int], objective: str) -> float:
    """
    Admissible heuristic:
      - For time: horizontal straight-line distance / max TAS, to seconds.
      - For fuel: time with min fuel rate, plus a tiny vertical fuel component that is ≤ real climb cost.
    """
    ax, ay, az = a
    gx, gy, gz = goal
    # horizontal distance only to keep it optimistic under our time model
    dx = gx - ax
    dy = gy - ay
    horiz_nm = math.hypot(dx, dy) * DX_NM
    if objective == "time":
        time_hr = horiz_nm / MAX_TAS_KT
        return time_hr * 3600.0
    else:
        time_hr = horiz_nm / MAX_TAS_KT
        fuel = MIN_FUEL_RATE_LBPH * time_hr
        dz_ft = abs((gz - az) * DZ_FT)
        fuel += dz_ft * 0.0025   # guaranteed ≤ actual climb penalty (0.01) and descent credit
        return fuel

def a_star(start: Tuple[int,int,int], goal: Tuple[int,int,int], obstacles: List[Dict], objective: str, max_expansions: int=1_000_000):
    """
    A* search. returns (path list of nodes from start to goal, total_fuel, total_time_s)
    or (None, None, None) if no path found.
    objective must be 'time' or 'fuel'
    """
    # Quick check if start/goal in obstacle
    for obs in obstacles:
        if obstacle_blocks_node(obs, *start):
            raise ValueError("Start location blocked by obstacle")
        if obstacle_blocks_node(obs, *goal):
            raise ValueError("Goal location blocked by obstacle")

    open_heap: List[Tuple[float, float, Tuple[int,int,int]]] = []
    gscore: Dict[Tuple[int,int,int], float] = {}
    came_from: Dict[Tuple[int,int,int], Tuple[int,int,int]] = {}
    closed: set = set()

    start_cost = 0.0
    gscore[start] = start_cost
    fscore = start_cost + heuristic(start, goal, objective)
    heapq.heappush(open_heap, (fscore, start_cost, start))

    expansions = 0

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        expansions += 1
        if expansions % 200000 == 0:
            print(f"expanded {expansions} nodes...")

        if current == goal:
            # reconstruct path
            path = [current]
            total_fuel = 0.0
            total_time = 0.0
            while current in came_from:
                prev = came_from[current]
                fcost, tcost = cost_between(prev, current)
                total_fuel += fcost
                total_time += tcost
                current = prev
                path.append(current)
            path.reverse()
            return path, total_fuel, total_time

        if expansions > max_expansions:
            break

        # expand neighbors
        for nb in neighbors_of(current):
            # check obstacle at node and along edge
            blocked = False
            for obs in obstacles:
                if obstacle_blocks_node(obs, *nb) or edge_crosses_obstacle(current, nb, obs):
                    blocked = True
                    break
            if blocked:
                continue

            fuel_cost, time_cost = cost_between(current, nb)
            step_cost = time_cost if objective == "time" else fuel_cost
            tentative_g = current_g + step_cost

            if nb not in gscore or tentative_g < gscore[nb]:
                came_from[nb] = current
                gscore[nb] = tentative_g
                f = tentative_g + heuristic(nb, goal, objective)
                heapq.heappush(open_heap, (f, tentative_g, nb))

    return None, None, None

# ------- GUI application -------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("A* Flight Path Planner (3D grid)")
        self.geometry("1200x760")

        left = ttk.Frame(self, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Start X (nm 0-100)").pack(anchor="w")
        self.start_x = ttk.Entry(left); self.start_x.insert(0,"5"); self.start_x.pack(fill=tk.X)
        ttk.Label(left, text="Start Y (nm 0-100)").pack(anchor="w")
        self.start_y = ttk.Entry(left); self.start_y.insert(0,"5"); self.start_y.pack(fill=tk.X)
        ttk.Label(left, text="Start Alt (ft)").pack(anchor="w")
        self.start_alt = ttk.Entry(left); self.start_alt.insert(0,"3000"); self.start_alt.pack(fill=tk.X)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        ttk.Label(left, text="Goal X (nm 0-100)").pack(anchor="w")
        self.goal_x = ttk.Entry(left); self.goal_x.insert(0,"95"); self.goal_x.pack(fill=tk.X)
        ttk.Label(left, text="Goal Y (nm 0-100)").pack(anchor="w")
        self.goal_y = ttk.Entry(left); self.goal_y.insert(0,"95"); self.goal_y.pack(fill=tk.X)
        ttk.Label(left, text="Goal Alt (ft)").pack(anchor="w")
        self.goal_alt = ttk.Entry(left); self.goal_alt.insert(0,"3000"); self.goal_alt.pack(fill=tk.X)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        ttk.Label(left, text="Objective").pack(anchor="w")
        self.objective = tk.StringVar(value="time")
        ttk.Radiobutton(left, text="Minimize time", variable=self.objective, value="time").pack(anchor="w")
        ttk.Radiobutton(left, text="Minimize fuel", variable=self.objective, value="fuel").pack(anchor="w")

        ttk.Label(left, text="Obstacles (JSON list or one JSON dict per line)").pack(anchor="w", pady=(8,0))
        self.obs_text = tk.Text(left, height=8, width=36)
        self.obs_text.pack(fill=tk.X)
        # defaults
        self.obs_text.insert("1.0", json.dumps(DEFAULT_OBSTACLES, indent=2))

        ttk.Button(left, text="Run A*", command=self.run_astar).pack(fill=tk.X, pady=8)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status_var, foreground="blue").pack(anchor="w", pady=(6,0))

        # Right side: 2 plots and summary
        right = ttk.Frame(self, padding=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # XY plot
        self.fig_xy = Figure(figsize=(6,4))
        self.ax_xy = self.fig_xy.add_subplot(111)
        self.ax_xy.set_title("Top-down X-Y (nm)")
        self.ax_xy.set_xlim(0, X_MAX_NM)
        self.ax_xy.set_ylim(0, Y_MAX_NM)
        self.canvas_xy = FigureCanvasTkAgg(self.fig_xy, master=right)
        self.canvas_xy.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Altitude plot
        self.fig_z = Figure(figsize=(6,2.4))
        self.ax_z = self.fig_z.add_subplot(111)
        self.ax_z.set_title("Altitude vs Along-track Distance")
        self.ax_z.set_xlabel("Distance (nm)")
        self.ax_z.set_ylabel("Altitude (ft)")
        self.canvas_z = FigureCanvasTkAgg(self.fig_z, master=right)
        self.canvas_z.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Summary text
        ttk.Label(right, text="Summary").pack(anchor="w")
        self.summary = tk.Text(right, height=8)
        self.summary.pack(fill=tk.BOTH, expand=False)

    def run_astar(self):
        try:
            sx = clamp(float(self.start_x.get()), 0.0, X_MAX_NM)
            sy = clamp(float(self.start_y.get()), 0.0, Y_MAX_NM)
            gx = clamp(float(self.goal_x.get()), 0.0, X_MAX_NM)
            gy = clamp(float(self.goal_y.get()), 0.0, Y_MAX_NM)
            salt = clamp(float(self.start_alt.get()), 0.0, Z_MAX_FT)
            galt = clamp(float(self.goal_alt.get()), 0.0, Z_MAX_FT)
            obj = self.objective.get()
        except Exception as e:
            messagebox.showerror("Input error", f"Invalid inputs: {e}")
            return

        # Convert continuous XY to grid indices
        sx_i = int(round(sx / DX_NM))
        sy_i = int(round(sy / DX_NM))
        gx_i = int(round(gx / DX_NM))
        gy_i = int(round(gy / DX_NM))
        sz = alt_to_level(salt)
        gz = alt_to_level(galt)

        # parse obstacles
        obs_raw = self.obs_text.get("1.0", tk.END)
        try:
            obstacles = parse_obstacles(obs_raw)
        except Exception as e:
            messagebox.showerror("Obstacle parse error", str(e))
            return

        # normalize obstacles: clamp box coords to grid indices/ft
        norm_obs = []
        for ob in obstacles:
            try:
                xmin = int(round(clamp(ob.get("xmin", 0), 0, X_MAX_NM)))
                xmax = int(round(clamp(ob.get("xmax", X_MAX_NM), 0, X_MAX_NM)))
                ymin = int(round(clamp(ob.get("ymin", 0), 0, Y_MAX_NM)))
                ymax = int(round(clamp(ob.get("ymax", Y_MAX_NM), 0, Y_MAX_NM)))
                zmin = clamp(ob.get("zmin_ft", 0), 0, Z_MAX_FT)
                zmax = clamp(ob.get("zmax_ft", Z_MAX_FT), 0, Z_MAX_FT)
                norm_obs.append({"xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"zmin_ft":zmin,"zmax_ft":zmax})
            except Exception:
                continue

        start_node = (sx_i, sy_i, sz)
        goal_node  = (gx_i, gy_i, gz)
        self.status_var.set("Searching...")
        self.update_idletasks()

        try:
            path, total_fuel, total_time = a_star(start_node, goal_node, norm_obs, obj, max_expansions=1_000_000)
        except ValueError as e:
            messagebox.showerror("Blocked", str(e))
            self.status_var.set("Ready")
            return

        if path is None:
            messagebox.showinfo("No path", "No path found (expanded limit reached or blocked).")
            self.status_var.set("Ready")
            return

        # Convert path into display coords (nm, ft)
        xs = [n[0]*DX_NM for n in path]
        ys = [n[1]*DX_NM for n in path]
        zs = [level_to_alt(n[2]) for n in path]

        # Compute per-leg distances and times
        leg_dists = []
        leg_times_s = []
        leg_fuels = []
        along = [0.0]
        cum = 0.0
        for i in range(1, len(path)):
            a = path[i-1]; b = path[i]
            horiz = math.hypot(b[0]-a[0], b[1]-a[1]) * DX_NM
            dz_ft = (b[2]-a[2]) * DZ_FT
            dist = math.hypot(horiz, dz_ft / NM_TO_FT)
            f,t = cost_between(a,b)
            leg_dists.append(dist)
            leg_times_s.append(t)
            leg_fuels.append(f)
            cum += dist
            along.append(cum)

        total_dist = sum(leg_dists)

        # Plot top-down
        self.ax_xy.clear()
        self.ax_xy.set_title("Top-down X-Y (nm)")
        self.ax_xy.set_xlim(0, X_MAX_NM)
        self.ax_xy.set_ylim(0, Y_MAX_NM)
        # draw obstacles top-down footprint
        for ob in norm_obs:
            rx = [ob["xmin"], ob["xmax"], ob["xmax"], ob["xmin"], ob["xmin"]]
            ry = [ob["ymin"], ob["ymin"], ob["ymax"], ob["ymax"], ob["ymin"]]
            self.ax_xy.plot(rx, ry, color="red")
        self.ax_xy.plot(xs, ys, marker="o", linestyle="-")
        self.ax_xy.scatter([xs[0], xs[-1]],[ys[0], ys[-1]], color=["green","blue"])
        self.ax_xy.set_xlabel("X (nm)")
        self.ax_xy.set_ylabel("Y (nm)")
        self.canvas_xy.draw_idle()

        # Plot altitude profile (z vs along-track)
        self.ax_z.clear()
        self.ax_z.set_title("Altitude vs Along-track Distance")
        self.ax_z.set_xlabel("Distance (nm)")
        self.ax_z.set_ylabel("Altitude (ft)")
        self.ax_z.plot(along, zs, marker="o")
        # nice vertical padding
        ymin = min(zs); ymax = max(zs)
        pad = max(300.0, 0.05 * max(500.0, ymax - ymin))
        self.ax_z.set_ylim(ymin - pad, ymax + pad)
        self.canvas_z.draw_idle()

        # Summary text
        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, f"Path nodes: {len(path)}\n")
        self.summary.insert(tk.END, f"Total distance (3D approx, nm): {total_dist:.2f}\n")
        self.summary.insert(tk.END, f"Total time: {fmt_time_sec_to_minsec(total_time)} ({fmt_time_hms(total_time)})\n")
        self.summary.insert(tk.END, f"Total fuel (lb, approx): {total_fuel:.1f}\n\n")

        self.summary.insert(tk.END, "Legs (adjacent steps):\n")
        for i, (d,t,f) in enumerate(zip(leg_dists, leg_times_s, leg_fuels), start=1):
            self.summary.insert(tk.END, f"{i:04d}: dist {d:.3f} nm | time {fmt_time_sec_to_minsec(t)} | fuel {f:.2f} lb\n")

        self.status_var.set("Done")
        return

if __name__ == "__main__":
    app = App()
    app.mainloop()
