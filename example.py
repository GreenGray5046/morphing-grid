import numpy as np
import morphing_grid as mg
import mpmath as mp
import scipy as scp
import scipy.special as scps

def zeta(z):
    return np.array([complex(mp.zeta(complex(val))) for val in np.ravel(z)]).reshape(z.shape)

# Define the LaTeX title for your  function
custom_latex_title = "riemann zeta"

# Create an instance of the ConformalAnimator class with your function
animator = mg.ConformalAnimator(
    f=zeta,
    latex_title=custom_latex_title,
    domain=(-2, 2, -2, 2),
    grid_steps=25,
    line_resolution=200,
    n_frames=200
)

# Run the animation and save it to a file
# animator.animate_grid(save_path="custom_animation.gif", fps=40)

# Show a preview
animator.animate_grid(show=True, fps=30)

# You can also call the demo function
# demo()
