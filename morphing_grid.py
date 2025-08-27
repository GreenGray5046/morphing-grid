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

# ----------------------------------- Font ---------------------------------- #

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = False

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


# ------------------------------- Main class -------------------------------- #

@dataclass
class ConformalAnimator:
    f: ComplexFunc
    latex_title: str
    domain: Tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0)
    grid_steps: int = 21
    line_resolution: int = 200
    grid_width: float = 1.5
    grid_alpha: float = 0.9
    grid_color: Optional[str] = None  # None → use matplotlib cycle
    
    n_frames: int = 180
    easing: str = "ease_in_out_quad"
    dpi: int = 160
    figsize: Tuple[float, float] = (6.5, 6.5)
    facecolor: str = "white"
    
    # For numerical stability / outliers
    clip: float = 1e4

    # Internal fields
    _grid_h: List[np.ndarray] = field(default_factory=list, init=False)
    _grid_v: List[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.grid_steps < 3:
            raise ValueError("grid_steps should be ≥ 3 for a meaningful grid")
        self._build_grid()

    # --------------------------- Grid Construction ------------------------ #
    def _build_grid(self) -> None:
        x0, x1, y0, y1 = self.domain

        xs = np.linspace(x0, x1, self.line_resolution)
        ys = np.linspace(y0, y1, self.line_resolution)

        xs_grid_lines = np.linspace(x0, x1, self.grid_steps)
        ys_grid_lines = np.linspace(y0, y1, self.grid_steps)

        # Horizontal lines: y fixed, x varies
        self._grid_h = [xs + 1j * y for y in ys_grid_lines]
        # Vertical lines: x fixed, y varies
        self._grid_v = [x + 1j * ys for x in xs_grid_lines]

    def _apply(self, Z: np.ndarray) -> np.ndarray:
        try:
            W = self.f(Z)
        except Exception as e:
            warnings.warn(f"Function application failed on grid: {e}")
            W = np.full_like(Z, np.nan + 1j * np.nan)
        return _sanitize(W, clip=self.clip)

    # -------------------------- Figure management ------------------------- #
    def _setup_axis(self):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor(self.facecolor)
        x0, x1, y0, y1 = self.domain
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel('Re')
        ax.set_ylabel('Im')
        return fig, ax

    # ------------------------------ Animations ---------------------------- #
    def _morph(self, Z: np.ndarray, t: float) -> np.ndarray:
        """Interpolate each point from identity Z to f(Z) using easing."""
        a = _ease(t, self.easing)
        return (1 - a) * Z + a * self._apply(Z)

    def animate_grid(self, save_path: Optional[str] = None, fps: int = 30,
                     show: bool = False, blit: bool = False) -> FuncAnimation:
        """Animate the conformal image of a rectilinear grid under f."""
        fig, ax = self._setup_axis()
        # Initialize line artists
        (lines_h, lines_v) = ([], [])
        for h in self._grid_h:
            lh, = ax.plot(h.real, h.imag, lw=self.grid_width, alpha=self.grid_alpha,
                          color=self.grid_color)
            lines_h.append(lh)
        for v in self._grid_v:
            lv, = ax.plot(v.real, v.imag, lw=self.grid_width, alpha=self.grid_alpha,
                          color=self.grid_color)
            lines_v.append(lv)
        
        # The title can now use LaTeX for rendering.
        ax.set_title(self.latex_title.replace('\\\\', '\\'))

        def init():
            return [*lines_h, *lines_v]

        def update(frame: int):
            t = frame / max(1, self.n_frames - 1)
            for lh, h in zip(lines_h, self._grid_h):
                H = self._morph(h, t)
                lh.set_data(H.real, H.imag)
            for lv, v in zip(lines_v, self._grid_v):
                V = self._morph(v, t)
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

    # ------------------------------ Saving -------------------------------- #
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

    examples: List[Tuple[str, ComplexFunc, str]] = [
       ("z^2", lambda z: z**2, "$f(z) = z^2$"),
       ("sin", lambda z: np.sin(z), "$f(z) = \\sin(z)$"),
       ("mobius", lambda z: (z - 1) / (z + 1), "$f(z) = \\frac{z - 1}{z + 1}$"),
    ]

    for name, f, latex_title in examples:
        # Pass the function and LaTeX title directly to the constructor
        anim = ConformalAnimator(
            f=f,
            latex_title=latex_title,
            domain=(-2, 2, -2, 2),
            grid_steps=21,
            n_frames=150,
        )
        anim.animate_grid(save_path=f"demo_{name}.gif", fps=30)
      #  anim.animate_grid(show=True, fps=30)


__all__ = [
    "ConformalAnimator",
    "demo",
]
