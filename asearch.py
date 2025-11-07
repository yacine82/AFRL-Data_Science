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
 - Text summary with total time (mm:ss), fuel
"""

import math
import heapq
import json
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Tuple, List, Dict, Optional
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

# ------- Simple performance model (tunable) -------
# Horizontal TAS depending on phase:
TAS_CRUISE_KT = 420.0
TAS_CLIMB_KT = 250.0
TAS_DESCENT_KT = 260.0

# Fuel burn rates (lb/hr) approximate:
FUEL_CRUISE_LBPH = 2600.0
FUEL_CLIMB_LBPH = 3500.0
FUEL_DESCENT_LBPH = 900.0

# For heuristic we use optimistic rates (best case)
MAX_TAS_KT = max(TAS_CRUISE_KT, TAS_CLIMB_KT, TAS_DESCENT_KT)
MIN_FUEL_RATE_LBPH = min(FUEL_CRUISE_LBPH, FUEL_CLIMB_LBPH, FUEL_DESCENT_LBPH)

# Utility conversions
NM_TO_FT = 6076.12
KT_TO_NM_PER_HR = 1.0  # 1 kt = 1 nm/hr, used as scale

# ------- Helper functions -------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def alt_to_level(alt_ft: float) -> int:
    """Convert alt in ft to nearest grid level index"""
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

def node_to_key(x:int,y:int,z:int) -> Tuple[int,int,int]:
    return (x,y,z)

def euclidean_3d_nm(x1,y1,z1,x2,y2,z2) -> float:
    """Distance in NM: convert vertical ft -> nm via NM_TO_FT"""
    dx = (x1 - x2)
    dy = (y1 - y2)
    dz_ft = (z1 - z2) * DZ_FT
    dz_nm = dz_ft / NM_TO_FT
    return math.sqrt(dx*dx + dy*dy + dz_nm*dz_nm)

# ------- Obstacles -------
# Obstacle structure: dict with xmin,xmax,ymin,ymax,zmin_ft,zmax_ft
# Example default obstacles
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

# ------- A* search -------
from collections import defaultdict

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
    Returns (fuel_lb, time_s) to travel between adjacent grid nodes a->b.
    a and b are neighbors (max 1 step in each axis).
    Model is simplistic but tunable:
      - Horizontal distance (nm) based on dx,dy
      - Vertical step based on dz * DZ_FT
      - Phase: climb if dz>0, descent if dz<0 else cruise
      - Time = horiz_nm / TAS_phase (hours) -> seconds
      - Fuel = fuel_rate_phase * time_hours
    """
    (ax,ay,az) = a
    (bx,by,bz) = b
    dx = (bx - ax)
    dy = (by - ay)
    horiz_nm = math.hypot(dx, dy)  # in nm, since grid spacing is 1 nm
    dz_ft = (bz - az) * DZ_FT

    # phase
    if dz_ft > 1e-6:
        tas = TAS_CLIMB_KT
        fuel_rate = FUEL_CLIMB_LBPH
    elif dz_ft < -1e-6:
        tas = TAS_DESCENT_KT
        fuel_rate = FUEL_DESCENT_LBPH
    else:
        tas = TAS_CRUISE_KT
        fuel_rate = FUEL_CRUISE_LBPH

    # Horizontal time (hr) -- if horiz_nm == 0 but vertical change exists, we still model a small horizontal progress
    if horiz_nm <= 0:
        # vertical-only move: model as small horizontal equivalent of 0.001 nm to avoid zero division
        horiz_nm = 0.001

    time_hr = horiz_nm / tas
    time_s = time_hr * 3600.0

    # base fuel from horizontal travel
    fuel = fuel_rate * time_hr

    # additional fuel penalty for vertical change (very small)
    # climb energy cost approximated proportional to vertical ft
    # This coefficient is arbitrary/tunable (lb per ft)
    vertical_work_factor = 0.01  # ~0.01 lb per ft of climb (tunable)
    if dz_ft > 0:
        fuel += dz_ft * vertical_work_factor
    elif dz_ft < 0:
        # descending actually uses a little fuel but less (we'll subtract a small fraction)
        fuel += abs(dz_ft) * (vertical_work_factor * 0.25)

    return fuel, time_s

def heuristic(a: Tuple[int,int,int], goal: Tuple[int,int,int], objective: str) -> float:
    """
    Admissible heuristic:
      - For time: straight-line 3D distance / MAX_TAS_KT converted to seconds (optimistic).
      - For fuel: straight-line horizontal distance / 1 hr * MIN_FUEL_RATE (optimistic), plus tiny vertical fuel.
    """
    ax,ay,az = a
    gx,gy,gz = goal
    dist_nm = euclidean_3d_nm(ax,ay,az,gx,gy,gz)
    if objective == "time":
        # time in seconds (optimistic)
        time_hr = dist_nm / MAX_TAS_KT
        return time_hr * 3600.0
    else:
        # fuel in lb (optimistic)
        time_hr = dist_nm / MAX_TAS_KT
        fuel = MIN_FUEL_RATE_LBPH * time_hr
        # small vertical fuel estimate
        dz_ft = abs((gz - az) * DZ_FT)
        fuel += dz_ft * 0.005
        return fuel

def a_star(start: Tuple[int,int,int], goal: Tuple[int,int,int], obstacles: List[Dict], objective: str, max_expansions: int=5_000_000):
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

    open_heap = []
    # g-scores: cost from start to node. We keep two g scores depending objective: if objective == 'time' g is time_s, else fuel_lb
    gscore = {}
    came_from = {}

    start_cost = 0.0
    gscore[start] = start_cost
    fscore = start_cost + heuristic(start, goal, objective)
    heapq.heappush(open_heap, (fscore, start_cost, start))

    expansions = 0

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        expansions += 1
        if expansions % 100000 == 0:
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
            # check obstacle
            blocked = False
            for obs in obstacles:
                if obstacle_blocks_node(obs, *nb):
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
        # fill with defaults
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

        # Convert continuous XY to grid indices (nearest integer)
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
            # cost_between returns fuel and time for this adjacent step
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
        self.canvas_z.draw_idle()

        # Summary text
        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, f"Path nodes: {len(path)}\n")
        self.summary.insert(tk.END, f"Total horizontal/3D distance (nm, approx): {total_dist:.2f}\n")
        self.summary.insert(tk.END, f"Total time: {fmt_time_sec_to_minsec(total_time)} (hh:mm:ss approx {total_time/3600:.3f} hr)\n")
        self.summary.insert(tk.END, f"Total fuel (lb, approx): {total_fuel:.1f}\n\n")

        self.summary.insert(tk.END, "Legs (adjacent steps):\n")
        for i, (d,t,f) in enumerate(zip(leg_dists, leg_times_s, leg_fuels), start=1):
            self.summary.insert(tk.END, f"{i:04d}: dist {d:.3f} nm | time {fmt_time_sec_to_minsec(t)} | fuel {f:.2f} lb\n")

        self.status_var.set("Done")
        return

if __name__ == "__main__":
    app = App()
    app.mainloop()
