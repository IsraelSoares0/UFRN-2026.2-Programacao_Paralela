#!/usr/bin/env python3
"""Gera os graficos da Tarefa 14 e ajusta o modelo de Hockney T(n) = alfa + n/beta.

Uso: python3 plot_pingpong.py resultados_intra.csv [resultados_inter.csv ...]
Saidas: grafico_tempo.pdf, grafico_banda.pdf, ajuste_hockney.csv
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def carregar(caminho):
    d = np.genfromtxt(caminho, delimiter=",", names=True)
    return d["tamanho_bytes"], d["tempo_unidirecional_s"], d["banda_MBps"]


def hockney(n, t):
    """Minimos quadrados com pesos 1/t (erro relativo), para nao deixar
    os tamanhos grandes dominarem o ajuste da latencia."""
    A = np.vstack([np.ones_like(n), n]).T / t[:, None]
    b = np.ones_like(t)
    (alfa, inv_beta), *_ = np.linalg.lstsq(A, b, rcond=None)
    return alfa, 1.0 / inv_beta  # s, bytes/s


def rotulo(caminho):
    nome = os.path.splitext(os.path.basename(caminho))[0]
    return nome.replace("resultados_", "").replace("_", "-")


arquivos = sys.argv[1:]
if not arquivos:
    sys.exit(__doc__)

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
linhas = ["caso,latencia_us,banda_MBps,n_meio_bytes"]

for i, arq in enumerate(arquivos):
    n, t, bw = carregar(arq)
    nome = rotulo(arq)
    cor = f"C{i}"
    alfa, beta = hockney(n, t)
    n_meio = alfa * beta
    linhas.append(f"{nome},{alfa*1e6:.3f},{beta/1e6:.1f},{n_meio:.0f}")

    ax1.loglog(n, t * 1e6, "o-", color=cor, ms=4, label=f"{nome} (medido)")
    nn = np.logspace(np.log10(n.min()), np.log10(n.max()), 200)
    ax1.loglog(nn, (alfa + nn / beta) * 1e6, "--", color=cor, lw=1,
               label=f"{nome}: $\\alpha$={alfa*1e6:.2f} µs, "
                     f"$\\beta$={beta/1e6:.0f} MB/s")
    ax1.axvline(n_meio, color=cor, ls=":", lw=1)

    ax2.semilogx(n, bw, "o-", color=cor, ms=4, label=nome)
    ax2.axhline(beta / 1e6, color=cor, ls="--", lw=1)
    ax2.axvline(n_meio, color=cor, ls=":", lw=1)

ax1.set_xlabel("Tamanho da mensagem (bytes)")
ax1.set_ylabel("Tempo unidirecional (µs)")
ax1.set_title("Tempo vs. tamanho (ping-pong MPI)\n"
              "linhas pontilhadas: $n_{1/2}=\\alpha\\beta$")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(fontsize=8)
fig1.tight_layout()
fig1.savefig("graficos/grafico_tempo.pdf")

ax2.set_xlabel("Tamanho da mensagem (bytes)")
ax2.set_ylabel("Largura de banda efetiva (MB/s)")
ax2.set_title("Largura de banda vs. tamanho")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend()
fig2.tight_layout()
fig2.savefig("graficos/grafico_banda.pdf")

with open("graficos/ajuste_hockney.csv", "w") as f:
    f.write("\n".join(linhas) + "\n")
print("\n".join(linhas))