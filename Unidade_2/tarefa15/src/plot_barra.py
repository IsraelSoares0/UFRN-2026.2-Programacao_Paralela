"""Plota a evolução da temperatura de uma barra 1D (difusão de calor).

Formato esperado do CSV:
  - cabeçalho: "tempo" seguido das posições x de cada célula;
  - cada linha: o instante de tempo seguido da temperatura em cada posição.

Uso:
    python plot_barra.py                      # usa temperaturas.csv
    python plot_barra.py outro.csv            # outro arquivo
    python plot_barra.py --tempo 0.5          # apenas o instante mais próximo de 0.5
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def carregar(caminho):
    df = pd.read_csv(caminho)
    tempos = df["tempo"].to_numpy(dtype=float)
    posicoes = df.columns[1:].to_numpy(dtype=float)
    temperaturas = df.iloc[:, 1:].to_numpy(dtype=float)
    return tempos, posicoes, temperaturas


def plotar_todos(tempos, posicoes, temps, saida):
    """Uma faixa de calor por instante de tempo, empilhadas verticalmente."""
    n = len(tempos)
    vmin, vmax = temps.min(), temps.max()
    extent = [posicoes[0], posicoes[-1], 0, 1]

    fig, axes = plt.subplots(n, 1, figsize=(10, 0.55 * n + 1.5), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, t, linha in zip(axes, tempos, temps):
        im = ax.imshow(
            linha[np.newaxis, :],
            aspect="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        ax.set_yticks([])
        ax.set_ylabel(f"t={t:.3g}", rotation=0, ha="right", va="center", fontsize=9)

    axes[-1].set_xlabel("Posição na barra (x)")
    axes[0].set_title("Difusão de calor em uma barra 1D")
    fig.colorbar(im, ax=axes.tolist(), label="Temperatura", pad=0.02)

    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.show()


def plotar_um(tempos, posicoes, temps, t_alvo, saida):
    """Uma única barra de calor, no instante mais próximo de t_alvo."""
    i = int(np.argmin(np.abs(tempos - t_alvo)))
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(
        temps[i][np.newaxis, :],
        aspect="auto",
        cmap="inferno",
        extent=[posicoes[0], posicoes[-1], 0, 1],
    )
    ax.set_yticks([])
    ax.set_xlabel("Posição na barra (x)")
    ax.set_title(f"Barra 1D em t = {tempos[i]:.3g}")
    fig.colorbar(im, ax=ax, label="Temperatura")

    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="temperaturas.csv")
    p.add_argument("--tempo", type=float, default=None,
                   help="plota só o instante mais próximo deste valor")
    p.add_argument("--saida", default="barra_calor.png")
    args = p.parse_args()

    tempos, posicoes, temps = carregar(args.csv)

    if args.tempo is None:
        plotar_todos(tempos, posicoes, temps, args.saida)
    else:
        plotar_um(tempos, posicoes, temps, args.tempo, args.saida)


if __name__ == "__main__":
    main()