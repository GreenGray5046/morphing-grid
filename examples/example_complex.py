import math
import numpy as np
import morphing_grid as mg
from scipy.special import gamma

def zeta(z_val):
    # This is your function. It will be passed to f.
    return (z_val**2)-1

# Define the title for your custom function (supports LaTeX)
custom_latex_title = "$ f(z) = z^2 - 1 $"

# Create an instance of the ConformalAnimator class with your function
animator = mg.ConformalAnimator(
    f=zeta,
    latex_title=custom_latex_title,
    domain=(-2, 2, -2, 2),
    grid_steps=25,
    n_frames=200,
    line_resolution=200
)
# Show preview
animator.animate_grid(show=True, fps=30)

# Save file
# animator.animate_grid(save_path="custom_animation.gif", fps=40)
