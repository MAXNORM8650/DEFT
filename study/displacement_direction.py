# We'll visualize per-point displacements (arrows) from z = W0 x to the updated h
# for both cases: plain (PP^T) and ReLU(P)ReLU(P)^T. Each chart is separate.
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Dimensions and components (reuse structure from earlier)
d = 2
P = np.array([[1.0], [-1.0]])
P_relu = np.maximum(P, 0)

W0 = np.array([[1.2, 0.3],
               [0.1, 0.8]])
R = np.array([[0.5],
              [0.2]])

G = P @ P.T
G_relu = P_relu @ P_relu.T

# Sample inputs
X = np.random.randn(200, d)

# Base transform
Z = (W0 @ X.T).T  # shape (N,2)

# Updates
PRx = (P @ (R.T @ X.T)).T            # (N,2)
H_plain = Z - (G @ Z.T).T + PRx
H_relu  = Z - (G_relu @ Z.T).T + PRx

# Displacements
U_plain = H_plain - Z
U_relu  = H_relu  - Z

# --- Plot 1: Plain PP^T ---
plt.figure(figsize=(7,6))
plt.scatter(Z[:,0], Z[:,1], alpha=0.5, s=15)
plt.quiver(Z[:,0], Z[:,1], U_plain[:,0], U_plain[:,1], angles="xy", scale_units="xy", scale=1, alpha=0.6)
plt.title("Displacement:  $h = W_0x - PP^T W_0x + PRx$")
plt.axhline(0, lw=0.5)
plt.axvline(0, lw=0.5)
plt.xlim(-3,3); plt.ylim(-3,3)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()

# --- Plot 2: ReLU(P) ---
plt.figure(figsize=(7,6))
plt.scatter(Z[:,0], Z[:,1], alpha=0.5, s=15)
plt.quiver(Z[:,0], Z[:,1], U_relu[:,0], U_relu[:,1], angles="xy", scale_units="xy", scale=1, alpha=0.6)
plt.title("Displacement:  $h = W_0x - \\mathrm{ReLU}(P)\\mathrm{ReLU}(P)^T W_0x + PRx$")
plt.axhline(0, lw=0.5)
plt.axvline(0, lw=0.5)
plt.xlim(-3,3); plt.ylim(-3,3)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()