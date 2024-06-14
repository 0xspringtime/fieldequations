import numpy as np
from scipy.integrate import solve_ivp
from mayavi import mlab

def christoffel_symbols_schwarzschild(r, M):
    Rs = 2 * M  # Schwarzschild radius
    Γ_r_tt = Rs / (2 * r**2 * (1 - Rs / r))
    Γ_t_rt = Rs / (2 * r * (r - Rs))
    Γ_r_rr = Rs / (2 * r * (r - Rs))
    return Γ_r_tt, Γ_t_rt, Γ_r_rr

def geodesic_equations(t, y, M):
    τ, t, r, θ, φ, dτ_dτ, dt_dτ, dr_dτ, dθ_dτ, dφ_dτ = y
    Rs = 2 * M  # Schwarzschild radius

    Γ_r_tt, Γ_t_rt, Γ_r_rr = christoffel_symbols_schwarzschild(r, M)

    d2τ_dτ2 = 0  # Proper time parameterization, second derivative of τ w.r.t τ is zero
    d2t_dτ2 = -2 * Γ_t_rt * dt_dτ * dr_dτ
    d2r_dτ2 = -Γ_r_tt * dt_dτ**2 - Γ_r_rr * dr_dτ**2
    d2θ_dτ2 = -2 / r * dr_dτ * dθ_dτ
    d2φ_dτ2 = -2 / r * dr_dτ * dφ_dτ

    return [dτ_dτ, dt_dτ, dr_dτ, dθ_dτ, dφ_dτ, d2τ_dτ2, d2t_dτ2, d2r_dτ2, d2θ_dτ2, d2φ_dτ2]

# Parameters
M = 1  # Mass of the central object (e.g., black hole)
r0 = 10  # Initial radial distance
θ0 = np.pi / 2  # Initial polar angle (equatorial plane)
φ0 = 0  # Initial azimuthal angle
dr_dτ0 = 0  # Initial radial velocity
dθ_dτ0 = 0  # Initial polar velocity
dφ_dτ0 = 0.3  # Initial azimuthal velocity for stable orbit

# Initial conditions: [τ, t, r, θ, φ, dτ/dτ, dt/dτ, dr/dτ, dθ/dτ, dφ/dτ]
y0 = [0, 0, r0, θ0, φ0, 1, 1, dr_dτ0, dθ_dτ0, dφ_dτ0]

# Time span (dummy variable for parameterization)
t_span = [0, 2000]  # Extended time span for multiple orbits

# Solve the geodesic equations
sol = solve_ivp(geodesic_equations, t_span, y0, args=(M,), t_eval=np.linspace(0, 2000, 10000))  # Increased number of points for smoother animation

# Extract the solution
τ_sol, t_sol, r_sol, θ_sol, φ_sol = sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4]

# Convert to Cartesian coordinates for visualization
x_sol = r_sol * np.sin(θ_sol) * np.cos(φ_sol)
y_sol = r_sol * np.sin(θ_sol) * np.sin(φ_sol)
z_sol = r_sol * np.cos(θ_sol)

# Define a function to represent the curvature of spacetime more accurately
def curvature_z(x, y, M):
    Rs = 2 * M
    r = np.sqrt(x**2 + y**2)
    return -Rs / np.sqrt(r + 1e-10)  # Improved curvature function

# Create a mesh grid for the spacetime curvature
grid_size = 50
x_grid, y_grid = np.meshgrid(np.linspace(-20, 20, grid_size), np.linspace(-20, 20, grid_size))
z_grid = curvature_z(x_grid, y_grid, M)

# Set up the figure and axis using Mayavi
mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))

# Plot the curvature grid using Mayavi
mlab.mesh(x_grid, y_grid, z_grid, color=(0.5, 0.5, 0.5), opacity=0.5)

# Plot the massive object (black hole) as a sphere
mlab.points3d(0, 0, 0, scale_factor=2, color=(0, 0, 0))

# Initialize the plot for the geodesic path
geodesic_plot = mlab.plot3d([x_sol[0]], [y_sol[0]], [z_sol[0]], color=(0, 0, 1), tube_radius=0.1)

# Animation function
@mlab.animate(delay=100)  # Adjust the delay as needed
def anim():
    for i in range(1, len(x_sol)):
        geodesic_plot.mlab_source.reset(x=x_sol[:i+1], y=y_sol[:i+1], z=z_sol[:i+1])
        yield

# Run the animation
anim()
mlab.show()

