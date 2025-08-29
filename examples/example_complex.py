import math
import numpy as np
import morphing_grid as mg
import sympy as sp

z = sp.symbols('z')

def zeta(z_val):
    # This is your vectorized function. It will be passed to f.
    # Note: `z_val` is a NumPy array of complex numbers.
    return np.sin(z_val)

# Define the symbolic SymPy expression for your custom function.
# This is required for the new derivative-based refinement.
custom_sympy_expr = sp.sin(z)

# Define the LaTeX title for your custom function
custom_latex_title = "$f(z) = sin(z) $"

# Create an instance of the ConformalAnimator class with your function
animator = mg.ConformalAnimator(
    f=zeta,
    sympy_f_expr=custom_sympy_expr, # Pass the new symbolic expression here
    latex_title=custom_latex_title,
    domain=(-4, 4, -4, 4),
    grid_steps=25,
    n_frames=200,
    line_resolution=200
)

# Run the animation and save it to a file
animator.animate_grid(show=True, fps=30)
# animator.animate_grid(save_path="custom_animation.gif", fps=40)
