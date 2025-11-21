import numpy as np
import matplotlib.pyplot as plt
from src.traj.genkspace import genkspace

FOV = 24.0
N = 120
nint = 8

kx, ky, gx, gy = genkspace(
    FOV, N, ld=0, nint=nint,
    gamp=4.0, gslew=400.0, tsamp=5e-6,
    rotamount=0, rev_flag=0, gts=5e-6,
    flag_vd=0, int_rotation=1, alpha_vd=4
)

print("total samples:", kx.size)
shot_len = kx.size // nint
print("interleaves:", nint, "samples per interleaf:", shot_len)

# Plot each interleaf in a different segment color
plt.figure(figsize=(5,5))
for i in range(nint):
    seg = slice(i*shot_len, (i+1)*shot_len)
    plt.plot(kx[seg], ky[seg], linewidth=1)
plt.gca().set_aspect("equal", adjustable="box")
plt.title(f"Spiral k-space ({nint} interleaves, {shot_len} pts each)")
plt.xlabel("kx"); plt.ylabel("ky")
plt.show()

# Build timing vector for one shot (needed later for time-seg field correction)
tsamp = 5e-6  # seconds, must match what you passed to genkspace
timingVec = np.arange(shot_len, dtype=np.float32) * tsamp
print("timingVec shape:", timingVec.shape, "first/last (ms):",
      timingVec[0]*1e3, timingVec[-1]*1e3)
