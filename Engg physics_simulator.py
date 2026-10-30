import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Projectile Motion Simulator", layout="wide")

st.title("🎯 Interactive Projectile Motion Simulator")
st.markdown("**Engineering Physics Project** — Live Interactive Demonstration")

# ----------------------------------------------------------------------------
# PHYSICS FUNCTIONS (Preserved from your code)
# ----------------------------------------------------------------------------
PLANETS = {"Earth": 9.81, "Moon": 1.62, "Mars": 3.71, "Jupiter": 24.79}

def ideal_trajectory(v0, angle_deg, h0, g, n=300):
    th = np.radians(angle_deg)
    vx, vy = v0 * np.cos(th), v0 * np.sin(th)
    t_flight = (vy + np.sqrt(vy ** 2 + 2 * g * h0)) / g
    t = np.linspace(0, t_flight, n)
    x = vx * t
    y = h0 + vy * t - 0.5 * g * t ** 2
    return t, x, y

def drag_trajectory(v0, angle_deg, h0, g, c, steps=2000):
    th = np.radians(angle_deg)
    vx0, vy0 = v0 * np.cos(th), v0 * np.sin(th)
    t_ideal = (vy0 + np.sqrt(vy0 ** 2 + 2 * g * h0)) / g
    dt = max(t_ideal, 1e-3) / steps * 1.05

    def deriv(vx, vy):
        v = (vx * vx + vy * vy) ** 0.5
        return vx, vy, -c * v * vx, -g - c * v * vy

    x, y, vx, vy = 0.0, float(h0), float(vx0), float(vy0)
    ts, xs, ys = [0.0], [x], [y]
    t = 0.0
    for _ in range(steps * 20):
        k1 = deriv(vx, vy)
        k2 = deriv(vx + 0.5 * dt * k1[2], vy + 0.5 * dt * k1[3])
        k3 = deriv(vx + 0.5 * dt * k2[2], vy + 0.5 * dt * k2[3])
        k4 = deriv(vx + dt * k3[2], vy + dt * k3[3])
        nx = x + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        ny = y + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        nvx = vx + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        nvy = vy + dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        t += dt
        if ny < 0:
            f = y / (y - ny) if y != ny else 0.0
            ts.append(t - dt + f * dt)
            xs.append(x + f * (nx - x))
            ys.append(0.0)
            break
        x, y, vx, vy = nx, ny, nvx, nvy
        ts.append(t)
        xs.append(x)
        ys.append(y)
    return np.array(ts), np.array(xs), np.array(ys)

# ----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Simulation Parameters")
planet_name = st.sidebar.radio("Select Planet", list(PLANETS.keys()))
g = PLANETS[planet_name]

v0 = st.sidebar.slider("Launch Speed (m/s)", 1.0, 100.0, 30.0, 1.0)
angle = st.sidebar.slider("Launch Angle (deg)", 0.0, 90.0, 45.0, 1.0)
h0 = st.sidebar.slider("Initial Height (m)", 0.0, 50.0, 0.0, 1.0)
drag_c = st.sidebar.slider("Drag Coefficient k/m (1/m)", 0.0, 0.2, 0.01, 0.001)

# Compute trajectories on parameter change
t1, x1, y1 = ideal_trajectory(v0, angle, h0, g)
t2, x2, y2 = drag_trajectory(v0, angle, h0, g, drag_c)

# ----------------------------------------------------------------------------
# MAIN LAYOUT (Graph + Metrics)
# ----------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x1, y1, "b-", lw=2, label="Ideal (no air resistance)")
    ax.plot(x2, y2, "r--", lw=2, label="With air resistance")
    ax.set_xlabel("Horizontal distance (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title(f"Projectile Motion on {planet_name} ($g = {g} \\text{{ m/s}}^2$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    
    ax.set_xlim(0, max(x1.max(), x2.max()) * 1.1 + 1)
    ax.set_ylim(0, max(y1.max(), y2.max()) * 1.2 + 1)
    
    st.pyplot(fig)

with col2:
    st.markdown("### 📊 Live Results")
    
    st.markdown("#### 🔵 Ideal Trajectory")
    st.metric("Time of Flight", f"{t1[-1]:.2f} s")
    st.metric("Max Height", f"{y1.max():.2f} m")
    st.metric("Range", f"{x1[-1]:.2f} m")
    
    st.markdown("#### 🔴 With Air Resistance")
    st.metric("Time of Flight", f"{t2[-1]:.2f} s")
    st.metric("Max Height", f"{y2.max():.2f} m")
    st.metric("Range", f"{x2[-1]:.2f} m")

# ----------------------------------------------------------------------------
# EXTRA FEATURE: RANGE VS ANGLE ANALYSIS
# ----------------------------------------------------------------------------
if st.button("📈 Run Range vs. Angle Analysis (No Drag)"):
    angles = np.linspace(1, 89, 177)
    ranges = [ideal_trajectory(v0, a, h0, g)[1][-1] for a in angles]
    best_angle = angles[int(np.argmax(ranges))]
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(angles, ranges, "b-", lw=2)
    ax2.axvline(best_angle, color="r", ls="--", label=f"Max range at {best_angle:.0f}°")
    ax2.set_xlabel("Launch angle (deg)")
    ax2.set_ylabel("Range (m)")
    ax2.set_title(f"Range vs. Angle ($v_0 = {v0:.0f}$ m/s, $h_0 = {h0:.0f}$ m, {planet_name})")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)
    st.success(f"💡 Optimal angle for maximum range without drag is **{best_angle:.0f}°**!")
