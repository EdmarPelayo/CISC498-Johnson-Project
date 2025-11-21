import sys, os
import numpy as np
import matplotlib.pyplot as plt
from src.traj.genkspace import genkspace
from src.traj.genspivd_kim import genspivd_kim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


print("=== Constant-density spiral ===")
kx, ky, gx, gy = genkspace(
    FOV=24.0, N=120, ld=0, nint=1,
    gamp=4.0, gslew=400.0, tsamp=5e-6,
    rotamount=0, rev_flag=0, gts=5e-6,
    flag_vd=0, int_rotation=1, alpha_vd=4
)

print("kx shape:", kx.shape)
print("ky shape:", ky.shape)
print("Gx max:", np.max(gx))
print("Gy max:", np.max(gy))

plt.figure()
plt.plot(kx, ky)
plt.title("Constant-density Spiral")
plt.axis("equal")


print("\n=== Variable-density spiral (Kim) ===")
Gx, Gy, kx2, ky2, sx, sy = genspivd_kim(
    D=24.0, N=120, nl=1,
    gamp=4.0, gslew=400.0, gts=10e-6, alphavd=4
)

print("kx2 shape:", kx2.shape)
print("ky2 shape:", ky2.shape)
print("Gx2 max:", np.max(Gx))
print("Gy2 max:", np.max(Gy))

plt.figure()
plt.plot(kx2, ky2)
plt.title("VD Spiral (Kim)")
plt.axis("equal")

plt.show()
