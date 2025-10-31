import argparse
import numpy as np

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Plot the direction field (2D) for the ODE:
# C'(u) = [exp(-u*C) * C] / [-exp(-u*C) + exp(-C) + alpha * C * exp(-C)]
# Domain: u in [0, 1], C in [0, 2]

import matplotlib.pyplot as plt

def dC_du(u, C, alpha):
    """
    Compute C'(u) on a grid for given alpha.
    u, C can be numpy arrays of the same shape.
    """
    E_uC = np.exp(-u * C)
    E_C = np.exp(-C)
    num = E_uC * C
    den = u*(-E_uC + E_C) + alpha * C * E_C

    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        slope = num / den

    return slope, den

def plot_direction_field(alpha=0.5, u_min=0.0, u_max=1.0, C_min=0.0, C_max=2.0, nx=200, ny=200, use_stream=True):
    # Build grid
    u = np.linspace(u_min, u_max, nx)
    C = np.linspace(C_min, C_max, ny)
    U, CG = np.meshgrid(u, C)

    # Compute slope field
    F, den = dC_du(U, CG, alpha)

    # Mask problematic regions: near singularities, non-finite, or extremely large slopes
    mask = (~np.isfinite(F)) | (~np.isfinite(den)) | (np.abs(den) < 1e-8) | (np.abs(F) > 1e3)
    F = np.ma.array(F, mask=mask)

    plt.figure(figsize=(7, 5.5))
    if use_stream:
        # Streamplot using the autonomous system: du/dt = 1, dC/dt = F(u, C)
        Ucomp = np.ma.ones_like(F)
        Vcomp = F
        # Color by slope magnitude (log-scaled for better dynamic range)
        color = np.log1p(np.abs(F))
        sp = plt.streamplot(
            U, CG, Ucomp, Vcomp,
            density=1.2,
            color=color,
            cmap='viridis',
            arrowsize=1.0,
            minlength=0.1
        )
        cbar = plt.colorbar(sp.lines, pad=0.02)
        cbar.set_label('log(1 + |C\'(u)|)')
    else:
        # Quiver (downsampled) as an alternative
        step_x = max(1, nx // 30)
        step_y = max(1, ny // 30)
        U_ds = U[::step_y, ::step_x]
        C_ds = CG[::step_y, ::step_x]
        F_ds = F[::step_y, ::step_x]

        # Normalize arrows so they have comparable lengths
        du = np.ma.ones_like(F_ds)
        dv = F_ds
        norm = np.ma.sqrt(du ** 2 + dv ** 2)
        du /= norm
        dv /= norm

        plt.quiver(U_ds, C_ds, du, dv, angles='xy', scale_units='xy', scale=25, width=0.002, pivot='mid')
        # Add a background showing slope magnitude
        plt.imshow(
            np.log1p(np.abs(F))[::-1, :],
            extent=[u_min, u_max, C_min, C_max],
            cmap='viridis',
            aspect='auto',
            alpha=0.6
        )
        cbar = plt.colorbar(pad=0.02)
        cbar.set_label('log(1 + |C\'(u)|)')

    plt.xlim(u_min, u_max)
    plt.ylim(C_min, C_max)
    plt.xlabel('u')
    plt.ylabel('C')
    plt.title(f"Direction field for C'(u) with alpha = {alpha}")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot the direction field for C'(u) over u∈[0,1], C∈[0,2].")
    parser.add_argument('--alpha', type=float, default=0.8, help='Parameter alpha in the ODE (default: 0.5)')
    parser.add_argument('--stream', action='store_true', help='Use streamplot (default)')
    parser.add_argument('--quiver', action='store_true', help='Use quiver instead of streamplot')
    parser.add_argument('--nx', type=int, default=200, help='Grid resolution in u (default: 200)')
    parser.add_argument('--ny', type=int, default=200, help='Grid resolution in C (default: 200)')
    args = parser.parse_args()

    use_stream = True
    if args.quiver:
        use_stream = False
    elif args.stream:
        use_stream = True

    plot_direction_field(alpha=args.alpha, nx=args.nx, ny=args.ny, use_stream=use_stream)

if __name__ == '__main__':
    main()