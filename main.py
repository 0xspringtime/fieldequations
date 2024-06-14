import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
def christoffel_symbols_schwarzschild(r, M):
    """
    Calculate the Christoffel symbols for Schwarzschild metric.
    
    Parameters:
    r : float
        Radial coordinate.
    M : float
        Mass of the central object.
    
    Returns:
    Γ^r_tt, Γ^t_rt, Γ^r_rr : float
        Non-zero Christoffel symbols for Schwarzschild metric.
    """
    Rs = 2 * M  # Schwarzschild radius
    Γ_r_tt = Rs / (2 * r**2 * (1 - Rs / r))
    Γ_t_rt = Rs / (2 * r * (r - Rs))
    Γ_r_rr = Rs / (2 * r * (r - Rs))
    return Γ_r_tt, Γ_t_rt, Γ_r_rr
def geodesic_equations(t, y, M):
    """
    Geodesic equations for Schwarzschild metric.
    
    Parameters:
    t : float
        Time variable (dummy in this case, needed for solve_ivp).
    y : array-like
        State vector [t, r, θ, φ, dt/dτ, dr/dτ, dθ/dτ, dφ/dτ].
    M : float
        Mass of the central object.
    
    Returns:
    dydt : array-like
        Derivatives of the state vector.
    """
    t, r, θ, φ, dt_dτ, dr_dτ, dθ_dτ, dφ_dτ = y
    Rs = 2 * M  # Schwarzschild radius

    Γ_r_tt, Γ_t_rt, Γ_r_rr = christoffel_symbols_schwarzschild(r, M)

    d2t_dτ2 = -2 * Γ_t_rt * dt_dτ * dr_dτ
    d2r_dτ2 = -Γ_r_tt * dt_dτ**2 - Γ_r_rr * dr_dτ**2
    d2θ_dτ2 = -2 / r * dr_dτ * dθ_dτ
    d2φ_dτ2 = -2 / r * dr_dτ * dφ_dτ

    return [dt_dτ, dr_dτ, dθ_dτ, dφ_dτ, d2t_dτ2, d2r_dτ2, d2θ_dτ2, d2φ_dτ2]
# Parameters
M = 1  # Mass of the central object (e.g., black hole)
r0 = 10  # Initial radial distance
θ0 = np.pi / 2  # Initial polar angle (equatorial plane)
φ0 = 0  # Initial azimuthal angle
dr_dτ0 = 0  # Initial radial velocity
dθ_dτ0 = 0  # Initial polar velocity
dφ_dτ0 = 0.1  # Initial azimuthal velocity

# Initial conditions
y0 = [0, r0, θ0, φ0, 1, dr_dτ0, dθ_dτ0, dφ_dτ0]

# Time span (dummy variable for parameterization)
t_span = [0, 100]

# Solve the geodesic equations
sol = solve_ivp(geodesic_equations, t_span, y0, args=(M,), t_eval=np.linspace(0, 100, 1000))

# Extract the solution
t_sol, r_sol, θ_sol, φ_sol = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
# Convert to Cartesian coordinates for visualization
x_sol = r_sol * np.sin(θ_sol) * np.cos(φ_sol)
y_sol = r_sol * np.sin(θ_sol) * np.sin(φ_sol)
z_sol = r_sol * np.cos(θ_sol)

# Plot the geodesic path
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_sol, y_sol, z_sol, label='Geodesic Path')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Geodesic Path in Schwarzschild Spacetime')
plt.legend()
plt.show()

