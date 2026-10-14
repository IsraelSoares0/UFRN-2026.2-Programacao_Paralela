#!/usr/bin/env python3
"""
Visualiza a saída do solver (snapshots.bin) de difusão viscosa 2D.

Gera:
  - difusao_figura.png : mapas de calor em 4 instantes, perfil central
                         (numérico x analítico) e decaimento do pico
  - difusao.gif        : animação do campo u(x,y,t) + perfil central

Uso:  python3 visualizar.py [snapshots.bin]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

arq = sys.argv[1] if len(sys.argv) > 1 else "snapshots.bin"

# ---------- leitura ----------
with open(arq, "rb") as f:
    N, nq = np.fromfile(f, dtype=np.int32, count=2)
    nu, A, s0, L = np.fromfile(f, dtype=np.float64, count=4)
    N, nq = int(N), int(nq)
    t = np.empty(nq)
    U = np.empty((nq, N, N))
    V = np.empty((nq, N, N))
    for k in range(nq):
        t[k] = np.fromfile(f, dtype=np.float64, count=1)[0]
        U[k] = np.fromfile(f, dtype=np.float64, count=N * N).reshape(N, N)
        V[k] = np.fromfile(f, dtype=np.float64, count=N * N).reshape(N, N)

dx = L / N
x = (np.arange(N) + 0.5) * dx
jc = N // 2                                   # linha/coluna central
tem_pert = (U[0].max() - U[0].min()) > 1e-9   # há perturbação? (modo 2)


def analitica(tt, xx):
    """Gaussiana 2D difundindo, avaliada no eixo y = L/2."""
    s2 = s0**2 + 2 * nu * tt
    return A * (s0**2 / s2) * np.exp(-((xx - 0.5 * L) ** 2) / (2 * s2))


vmin, vmax = float(U.min()), float(U.max())
if vmax - vmin < 1e-12:                       # campo uniforme: evita escala degenerada
    vmin, vmax = vmin - 1, vmax + 1
extent = [0, L, 0, L]
cmap = "magma"

# ---------- figura estática ----------
idx = np.linspace(0, nq - 1, 4).astype(int)
fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])

axs = []
for c, k in enumerate(idx):
    ax = fig.add_subplot(gs[0, c])
    # transposta: eixo horizontal = x (índice i), vertical = y (índice j)
    im = ax.imshow(U[k].T, origin="lower", extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(f"t = {t[k]:.3f}  |  max u = {U[k].max():.3f}")
    ax.set_xlabel("x")
    if c == 0:
        ax.set_ylabel("y")
    axs.append(ax)
fig.colorbar(im, ax=axs, shrink=0.85, pad=0.01, label="velocidade u")

ax_p = fig.add_subplot(gs[1, 0:2])
cores = plt.cm.viridis(np.linspace(0.05, 0.9, len(idx)))
for k, cor in zip(idx, cores):
    ax_p.plot(x, U[k][:, jc], "o", ms=2.5, color=cor, label=f"numérico t={t[k]:.2f}")
    if tem_pert:
        ax_p.plot(x, analitica(t[k], x), "-", lw=1, color="k", alpha=0.6)
if tem_pert:
    ax_p.plot([], [], "-", color="k", alpha=0.6, label="analítico")
ax_p.set_title("Perfil central u(x, y=L/2)")
ax_p.set_xlabel("x")
ax_p.set_ylabel("u")
ax_p.legend(fontsize=8)
ax_p.grid(alpha=0.3)

ax_d = fig.add_subplot(gs[1, 2:4])
ax_d.plot(t, U.max(axis=(1, 2)), "o", ms=3, label="numérico: max u")
if tem_pert:
    tt = np.linspace(0, t[-1], 200)
    ax_d.plot(tt, A * s0**2 / (s0**2 + 2 * nu * tt), "k-", lw=1, label=r"analítico: $A\,\sigma_0^2/(\sigma_0^2+2\nu t)$")
ax_d.set_title("Decaimento do pico")
ax_d.set_xlabel("t")
ax_d.set_ylabel("max u")
ax_d.legend(fontsize=9)
ax_d.grid(alpha=0.3)

fig.suptitle(f"Difusão viscosa 2D (Navier-Stokes sem pressão/advecção)  —  "
             f"N={N}, " + r"$\nu$" + f"={nu:g}", fontsize=13)
fig.savefig("difusao_figura.png", dpi=130)
plt.close(fig)
print("Gerado: difusao_figura.png")

# ---------- animação ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
im = a1.imshow(U[0].T, origin="lower", extent=extent, cmap=cmap,
               vmin=vmin, vmax=vmax, interpolation="nearest")
fig.colorbar(im, ax=a1, label="velocidade u")
a1.set_xlabel("x")
a1.set_ylabel("y")
tit = a1.set_title("")

(lin_n,) = a2.plot([], [], "o", ms=3, label="numérico")
(lin_a,) = a2.plot([], [], "k-", lw=1, label="analítico")
a2.set_xlim(0, L)
a2.set_ylim(vmin - 0.05 * (vmax - vmin), vmax + 0.05 * (vmax - vmin))
a2.set_xlabel("x")
a2.set_ylabel("u(x, y=L/2)")
a2.set_title("Perfil central")
a2.grid(alpha=0.3)
a2.legend(loc="upper right")


def atualiza(k):
    im.set_data(U[k].T)
    tit.set_text(f"t = {t[k]:.3f}")
    lin_n.set_data(x, U[k][:, jc])
    if tem_pert:
        lin_a.set_data(x, analitica(t[k], x))
    return im, lin_n, lin_a, tit


ani = FuncAnimation(fig, atualiza, frames=nq, blit=False)
ani.save("difusao.gif", writer=PillowWriter(fps=12))
plt.close(fig)
print("Gerado: difusao.gif")