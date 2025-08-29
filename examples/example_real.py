from __future__ import annotations
import numpy as np
import sympy as sp
from morphing_grid import LineAnimator
    
    # Create an instance of the LineAnimator class
animator = LineAnimator(
    #Define your real function here
        f=lambda x: x**2,
  
        domain=(-2.5, 2.5),
        num_ticks=41,
        n_frames=120,
    )
    
animator.animate_line(
  #Saves it as a GIF
    #save_path="x_squared.gif",
  
    fps=30,
  
  #Show preview
    show=True,
)
