from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
z = sp.symbols('z')

from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

ComplexFunc = Callable[[np.ndarray], np.ndarray]

# --------------------------------- Params ---------------------------------- #

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = False
plt.rcParams['text.color'] = 'white'

# ----------------------------- Utility helpers ----------------------------- #

def _sanitize(Z: np.ndarray, clip: float = 1e6) -> np.ndarray:
    """Replace non-finite values and clip extremes to keep plots stable."""
    Z = Z.copy()
    mask = ~np.isfinite(Z)
    if np.any(mask):
        Z[mask] = np.nan
    if clip is not None:
        Z = np.clip(Z.real, -clip, clip) + 1j * np.clip(Z.imag, -clip, clip)
    return Z


def _ease(t: float, kind: str = "ease_in_out_quad") -> float:
    """Easing functions"""
    t = float(np.clip(t, 0.0, 1.0))
    if kind == "linear":
        return t
    if kind == "ease_in_out_quad":
        return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t
    if kind == "ease_in_out_cubic":
        return 4 * t**3 if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2
    if kind == "ease_out_back":
        c1, c3 = 1.70158, 1.70158 + 1
        return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2
    return t


# ------------------------------ Complex class ------------------------------ #

@dataclass
class ConformalAnimator:
    f: ComplexFunc
    sympy_f_expr: sp.Expr
    latex_title: str
    domain: Tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0)
    grid_steps: int = 21
    line_resolution: int = 200
    grid_width: float = 1.5
    grid_alpha: float = 0.9
    grid_color: Optional[str] = None
    
    n_frames: int = 180
    easing: str = "ease_in_out_quad"
    dpi: int = 160
    figsize: Tuple[float, float] = (6.5, 6.5)
    facecolor: str = "white"
    
    # --- UPDATED: More robust adaptive refinement ---
    adaptive: bool = True
    refinement_method: str = 'derivative'  # NEW: using derivative
    refinement_threshold: float = 1.5      # For 'derivative' method
    max_refinements: int = 5
    
    clip: float = 1e4
    _grid_h: List[np.ndarray] = field(default_factory=list, init=False)
    _grid_v: List[np.ndarray] = field(default_factory=list, init=False)
    f_prime: ComplexFunc = field(init=False)

    def __post_init__(self):
        if self.grid_steps < 3:
            raise ValueError("grid_steps should be ≥ 3 for a meaningful grid")
        
        self._build_derivative_func()
        self._build_grid()

    def _build_derivative_func(self) -> None:
        """Builds a callable function for the derivative using SymPy."""
        try:
            f_prime_expr = sp.diff(self.sympy_f_expr, z)
            self.f_prime: ComplexFunc = sp.lambdify(z, f_prime_expr, 'numpy')
        except (TypeError, ValueError) as e:
            warnings.warn(f"Failed to create derivative function: {e}. Falling back to default refinement.")
            self.f_prime: ComplexFunc = lambda z: np.full_like(z, 1.0)
            
    def _build_grid(self) -> None:
        x0, x1, y0, y1 = self.domain
        xs = np.linspace(x0, x1, self.line_resolution)
        ys = np.linspace(y0, y1, self.line_resolution)
        xs_grid_lines = np.linspace(x0, x1, self.grid_steps)
        ys_grid_lines = np.linspace(y0, y1, self.grid_steps)
        grid_h_initial = [xs + 1j * y for y in ys_grid_lines]
        grid_v_initial = [x + 1j * ys for x in xs_grid_lines]

        if self.adaptive:
            print("Applying adaptive refinement...")
            self._grid_h = [self._refine_line(line) for line in grid_h_initial]
            self._grid_v = [self._refine_line(line) for line in grid_v_initial]
            print("Refinement complete.")
        else:
            self._grid_h = grid_h_initial
            self._grid_v = grid_v_initial

    def _refine_line(self, Z: np.ndarray) -> np.ndarray:
        """Iteratively adds points to a line for smoother final curves using the derivative."""
        for _ in range(self.max_refinements):
            refined_Z = [Z[0]]
            points_added = 0
            
            try:
                derivatives = self.f_prime((Z[:-1] + Z[1:]) / 2)
            except Exception:
                derivatives = np.full(len(Z) - 1, 1.0)

            for i in range(len(Z) - 1):
                z1, z2 = Z[i], Z[i+1]
                stretching_factor = np.abs(derivatives[i])

                if stretching_factor > self.refinement_threshold:
                    refined_Z.append((z1 + z2) / 2.0)
                    points_added += 1
                
                refined_Z.append(z2)

            if points_added == 0:
                break
            Z = np.array(refined_Z)
        return Z

    def _apply(self, Z: np.ndarray) -> np.ndarray:
        try:
            W = self.f(Z)
        except Exception as e:
            warnings.warn(f"Function application failed on grid: {e}")
            W = np.full_like(Z, np.nan + 1j * np.nan)
        return _sanitize(W, clip=self.clip)

    def _setup_axis(self):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        fig.patch.set_facecolor('black')
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor(self.facecolor)
        x0, x1, y0, y1 = self.domain
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel('Re', color='white')
        ax.set_ylabel('Im', color='white')
        ax.set_facecolor('black')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        return fig, ax

    def _morph(self, Z: np.ndarray, t: float) -> np.ndarray:
        a = _ease(t, self.easing)
        W = self._apply(Z)
        return (1 - a) * Z + a * W

    def animate_grid(self, save_path: Optional[str] = None, fps: int = 30,
                     show: bool = False, blit: bool = False) -> FuncAnimation:
        fig, ax = self._setup_axis()
        lines_h = [ax.plot([], [], lw=self.grid_width, alpha=self.grid_alpha, color=self.grid_color)[0] for _ in self._grid_h]
        lines_v = [ax.plot([], [], lw=self.grid_width, alpha=self.grid_alpha, color=self.grid_color)[0] for _ in self._grid_v]
        ax.set_title(self.latex_title.replace('\\\\', '\\'))

        def init():
            for line in lines_h + lines_v:
                line.set_data([], [])
            return [*lines_h, *lines_v]

        def update(frame: int):
            t = frame / max(1, self.n_frames - 1)
            for lh, h_refined in zip(lines_h, self._grid_h):
                H = self._morph(h_refined, t)
                lh.set_data(H.real, H.imag)
            for lv, v_refined in zip(lines_v, self._grid_v):
                V = self._morph(v_refined, t)
                lv.set_data(V.real, V.imag)
            return [*lines_h, *lines_v]

        anim = FuncAnimation(fig, update, init_func=init,
                             frames=self.n_frames, interval=1000 / fps,
                             blit=blit)
        self._maybe_save(anim, save_path, fps)
        if show:
            plt.show()
        plt.close(fig)
        return anim

    def _maybe_save(self, anim: FuncAnimation, save_path: Optional[str], fps: int) -> None:
        if not save_path:
            return
        ext = save_path.lower().rsplit('.', 1)[-1]
        if ext in {"gif"}:
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer, dpi=self.dpi)
        elif ext in {"mp4", "m4v", "mov"}:
            try:
                writer = FFMpegWriter(fps=fps)
                anim.save(save_path, writer=writer, dpi=self.dpi)
            except Exception as e:
                warnings.warn(
                    f"FFmpeg save failed ({e}). Install ffmpeg or save as GIF instead.")
        else:
            warnings.warn("Unknown extension. Use .gif or .mp4")
    
# ------------------------------ Real class ------------------------------ #
    
@dataclass
class LineAnimator:
    f: Callable[[np.ndarray], np.ndarray]
    domain: Tuple[float, float] = (-2.5, 2.5)
    num_ticks: int = 41
    line_resolution: int = 200
    tick_height: float = 0.28
    moving_tick_height: float = 0.36
    tick_width: float = 1.2
    moving_tick_width: float = 2.2
    tick_alpha: float = 0.9
    moving_tick_alpha: float = 0.95
    point_size: int = 18
    point_alpha: float = 0.9
    
    n_frames: int = 120
    easing: str = "ease_in_out_cubic"
    dpi: int = 150
    figsize: Tuple[float, float] = (8, 2.8)
    facecolor: str = "black"
    
    _x0: np.ndarray = field(init=False)
    _x_line: np.ndarray = field(init=False)

    def __post_init__(self):
        self._build_line()

    def _build_line(self) -> None:
        """Build the initial line and tick positions"""
        xmin, xmax = self.domain
        self._x0 = np.linspace(xmin, xmax, self.num_ticks)
        self._x_line = np.linspace(xmin, xmax, self.line_resolution)

    def _setup_axis(self):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        fig.patch.set_facecolor(self.facecolor)
        ax.set_facecolor(self.facecolor)
        
        xmin, xmax = self.domain
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-1.25, 1.25)
        ax.set_yticks([])
        ax.set_xlabel("Real line (positions)", color='white')
        
        # Style the axis
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', colors='white')
        
        return fig, ax

    def _morph(self, x: np.ndarray, t: float) -> np.ndarray:
        """Interpolate between original and transformed positions"""
        a = _ease(t, self.easing)
        fx = self.f(x)
        return (1 - a) * x + a * fx

    def animate_line(self, save_path: Optional[str] = None, fps: int = 30,
                    show: bool = False, blit: bool = True) -> FuncAnimation:
        fig, ax = self._setup_axis()
        
        # Draw baseline
        ax.plot([self.domain[0], self.domain[1]], [0, 0], lw=2, color='white')
        
        # Draw static original ticks
        for xi in self._x0:
            ax.plot([xi, xi], [-self.tick_height/2, self.tick_height/2], 
                   lw=self.tick_width, alpha=self.tick_alpha, color='lightgray')
        
        # Create moving ticks
        moving_ticks = []
        for _ in self._x0:
            ln, = ax.plot([0, 0], [-self.moving_tick_height/2, self.moving_tick_height/2], 
                         lw=self.moving_tick_width, alpha=self.moving_tick_alpha, 
                         color='dodgerblue')
            moving_ticks.append(ln)
        
        # Create moving points
        moving_pts = ax.scatter([], [], s=self.point_size, alpha=self.point_alpha, 
                               color='crimson', edgecolors='white', linewidth=0.5)
        
        # Create title and subtitle
        title = ax.text(0.02, 0.88, "", transform=ax.transAxes, ha="left", va="center", 
                       fontsize=12, color='white', weight='bold')
        subtitle = ax.text(0.02, 0.72, "", transform=ax.transAxes, ha="left", va="center", 
                          fontsize=10, color='white', weight='bold')

        def init():
            t = 0.0
            xt = self._morph(self._x0, t)
            for line, xi in zip(moving_ticks, xt):
                line.set_xdata([xi, xi])
                line.set_ydata([-self.moving_tick_height/2, self.moving_tick_height/2])
            moving_pts.set_offsets(np.column_stack([xt, np.zeros_like(xt)]))
            title.set_text(r"Stretching of the line under $f(x)$")
            subtitle.set_text(r"Interpolation: $x_t = (1-t)\,x + t\, f(x)$,   $t=0 \rightarrow 1$")
            return moving_ticks + [moving_pts, title, subtitle]

        def update(t):
            xt = self._morph(self._x0, t)
            for line, xi in zip(moving_ticks, xt):
                line.set_xdata([xi, xi])
                line.set_ydata([-self.moving_tick_height/2, self.moving_tick_height/2])
            moving_pts.set_offsets(np.column_stack([xt, np.zeros_like(xt)]))
            subtitle.set_text(fr"Interpolation: $x_t = (1-t)\,x + t\, f(x)$     $t={t:.2f}$")
            return moving_ticks + [moving_pts, subtitle]

        # Create eased time values
        t_values = np.linspace(0, 1, self.n_frames)
        t_ease = 0.5 - 0.5 * np.cos(np.pi * t_values)  # Smooth cosine easing

        anim = FuncAnimation(
            fig, update, frames=list(t_ease),
            init_func=init, blit=blit, interval=1000/fps
        )
        
        self._maybe_save(anim, save_path, fps)
        if show:
            plt.show()
        plt.close(fig)
        return anim
    
def _maybe_save(self, anim: FuncAnimation, save_path: Optional[str], fps: int) -> None:
        if not save_path:
            return
        ext = save_path.lower().rsplit('.', 1)[-1]
        if ext in {"gif"}:
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer, dpi=self.dpi)
        elif ext in {"mp4", "m4v", "mov"}:
            try:
                writer = FFMpegWriter(fps=fps)
                anim.save(save_path, writer=writer, dpi=self.dpi)
            except Exception as e:
                warnings.warn(
                    f"FFmpeg save failed ({e}). Install ffmpeg or save as GIF instead.")
        else:
            warnings.warn("Unknown extension. Use .gif or .mp4")


# ------------------------------- Convenience ------------------------------- #

def demo():
    import numpy as np

    examples: List[Tuple[str, ComplexFunc, sp.Expr, str]] = [
       ("z^2", lambda z: z**2, z**2, "$f(z) = z^2$"),
       ("sin", lambda z: np.sin(z), sp.sin(z), "$f(z) = \\sin(z)$"),
       ("mobius", lambda z: (z - 1) / (z + 1), (z - 1) / (z + 1), "$f(z) = \\frac{z - 1}{z + 1}$"),
    ]

    for name, f, sympy_f, latex_title in examples:
        anim = ConformalAnimator(
            f=f,
            sympy_f_expr=sympy_f,
            latex_title=latex_title,
            domain=(-2, 2, -2, 2),
            grid_steps=21,
            n_frames=150,
        )
        anim.animate_grid(save_path=f"demo_{name}.gif", fps=30)


__all__ = [
    "ConformalAnimator",
    "demo",
]
