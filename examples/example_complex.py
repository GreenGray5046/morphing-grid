#For generating GIFs of complex functions

import numpy as np
import morphing_grid as mg
import mpmath as mp

def zeta(z):
    return np.array([complex(mp.zeta(complex(val))) for val in np.ravel(z)]).reshape(z.shape)

# Define the title for your function (supports LaTeX)
custom_latex_title = "Riemann Zeta"

# Define a SymPy expression for adaptive resolution and flexibility with updates in the future
custom_sympy_expr = z ** (sp.exp(sp.I))

# Create an instance of the ConformalAnimator class with your function
animator = mg.ConformalAnimator(
    f=zeta,
    latex_title=custom_latex_title,
    sympy_f_expr=custom_sympy_expr,
    domain=(-2, 2, -2, 2),
    grid_steps=25,
    line_resolution=200,
    n_frames=200
)

# Run the animation and save it to a file
# animator.animate_grid(save_path="custom_animation.gif", fps=40)

# Show a preview
animator.animate_grid(show=True, fps=30)
