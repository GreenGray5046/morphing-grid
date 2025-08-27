from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# Set global font to Times New Roman
rcParams["font.family"] = "Times New Roman"

ComplexFunc = Callable[[np.ndarray], np.ndarray]


# ----------------------------- Utility helpers ----------------------------- #

def _sanitize(Z: np.ndarray, clip: float = 1e6) -> np.ndarray:
    Z = Z.copy()
    mask = ~np.isfinite(Z)
    if np.any(mask):
        Z[mask] = np.nan
    if clip is not None:
        Z = np.clip(Z.real, -clip, clip) + 1j * np.clip(Z.imag, -clip, clip)
    return Z


def _ease(t: float, kind: str = "ease_in_out_quad") -> float:
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


# --------------------------- Domain coloring utils ------------------------- #

def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = (h % 1.0) * 6.0
    i = np.floor(h).astype(int)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    conds = [
        (i == 0, np.stack([v, t, p], axis=-1)),
        (i == 1, np.stack([q, v, p], axis=-1)),
        (i == 2, np.stack([p, v, t], axis=-1)),
        (i == 3, np.stack([p, q, v], axis=-1)),
        (i == 4, np.stack([t, p, v], axis=-1)),
        (i == 5, np.stack([v, p, q], axis=-1)),
    ]
    rgb = np.zeros(h.shape + (3,), dtype=float)
    for cond, val in conds:
        rgb[cond] = val[cond]
    return np.clip(rgb, 0.0, 1.0)


def _domain_coloring_image(W: np.ndarray,
                           lightness_period: float = math.e,
                           saturation: float = 0.85) -> np.ndarray:
    arg = np.angle(W)
    hue = (arg / (2 * np.pi)) % 1.0
    mag = np.abs(W)
    v = (np.log(mag + 1e-9) / np.log(lightness_period)) % 1.0
    v = 0.15 + 0.85 * v
    s = np.full_like(v, float(saturation))
    return _hsv_to_rgb(hue, s, v)


# ------------------------------- Main class -------------------------------- #

@dataclass
class ConformalAnimator:
    f: ComplexFunc
    domain: Tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0)
    grid_steps: int = 21
    grid_width: float = 1.5
    grid_alpha: float = 0.9
    grid_color: Optional[str] = None
    
    n_frames: int = 180
    easing: str = "ease_in_out_quad"
    dpi: int = 160
    figsize: Tuple[float, float] = (6.5, 6.5)
    facecolor: str = "white"
    
    clip: float = 1e4
    color_res: int = 600

    _grid_h: List[np.ndarray] = field(default_factory=list, init=False)
    _grid_v: List[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.grid_steps < 3:
            raise ValueError("grid_steps should be ≥ 3")
        self._build_grid()

    def _build_grid(self) -> None:
        x0, x1, y0, y1 = self.domain
        xs = np.linspace(x0, x1, self.grid_steps)
        ys = np.linspace(y0, y1, self.grid_steps)
        self._grid_h = [xs + 1j * y for y in ys]
        self._grid_v = [x + 1j * ys for x in xs]

    def _apply(self, Z: np.ndarray) -> np.ndarray:
        try:
            W = self.f(Z)
        except Exception as e:
            warnings.warn(f"Function failed on grid: {e}")
            W = np.full_like(Z, np.nan + 1j * np.nan)
        return _sanitize(W, clip=self.clip)

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

    def _morph(self, Z: np.ndarray, t: float) -> np.ndarray:
        a = _ease(t, self.easing)
        return (1 - a) * Z + a * self._apply(Z)

    # ----------------------- Animations with original grid ---------------- #

    def animate_grid(self, save_path: Optional[str] = None, fps: int = 30,
                     show: bool = False, blit: bool = False) -> FuncAnimation:
        fig, ax = self._setup_axis()

        # Draw original static grid (dark background grid)
        for h in self._grid_h:
            ax.plot(h.real, h.imag, lw=self.grid_width, alpha=0.3, color="black")
        for v in self._grid_v:
            ax.plot(v.real, v.imag, lw=self.grid_width, alpha=0.3, color="black")

        # Animated grid
        lines_h, lines_v = [], []
        for h in self._grid_h:
            lh, = ax.plot(h.real, h.imag, lw=self.grid_width,
                          alpha=self.grid_alpha, color=self.grid_color)
            lines_h.append(lh)
        for v in self._grid_v:
            lv, = ax.plot(v.real, v.imag, lw=self.grid_width,
                          alpha=self.grid_alpha, color=self.grid_color)
            lines_v.append(lv)
        ax.set_title("Conformal Grid Morph")

        def update(frame: int):
            t = frame / max(1, self.n_frames - 1)
            for lh, h in zip(lines_h, self._grid_h):
                H = self._morph(h, t)
                lh.set_data(H.real, H.imag)
            for lv, v in zip(lines_v, self._grid_v):
                V = self._morph(v, t)
                lv.set_data(V.real, V.imag)
            return [*lines_h, *lines_v]

        anim = FuncAnimation(fig, update, frames=self.n_frames,
                             interval=1000 / fps, blit=blit)
        self._maybe_save(anim, save_path, fps)
        if show:
            plt.show()
        plt.close(fig)
        return anim

    # Other animation methods (domain coloring, combo) remain unchanged
    # but will also inherit the Times New Roman font via rcParams.

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
