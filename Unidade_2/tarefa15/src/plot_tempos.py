"""Análise dos tempos de execução das versões V1, V2 e V3.

Entrada: resultados_todos.csv (colunas: caso, versao, processos, pontos, tempo_s)
Saídas:  tempos_barras.png, speedup_vs_v1.png, escalabilidade.png

Uso: python plot_tempos.py [resultados_todos.csv]
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "resultados_todos.csv"
CORES = {"V1": "tab:red", "V2": "tab:blue", "V3": "tab:green"}
ROTULOS = {
    "V1": "V1 (Send/Recv)",
    "V2": "V2 (Isend/Irecv/Wait)",
    "V3": "V3 (sobreposição)",
}

df = pd.read_csv(CSV)
df["tipo"] = df["caso"].str.split("_").str[0]  # intra ou inter
df = df.sort_values(["pontos", "tipo", "processos"])
tamanhos = sorted(df["pontos"].unique())


def ordem_casos(sub):
    return (sub[["caso", "tipo", "processos"]].drop_duplicates()
            .sort_values(["tipo", "processos"])["caso"].tolist())


# ---- 1. Tempo de cada versão por caso (um painel por tamanho) ----
fig, axes = plt.subplots(1, len(tamanhos), figsize=(6 * len(tamanhos), 4.5),
                         squeeze=False)
for ax, n in zip(axes[0], tamanhos):
    sub = df[df["pontos"] == n]
    piv = sub.pivot(index="caso", columns="versao", values="tempo_s")
    piv = piv.loc[ordem_casos(sub)]
    piv.plot.bar(ax=ax, color=[CORES[v] for v in piv.columns], width=0.8)
    ax.set_title(f"{n} pontos")
    ax.set_xlabel("")
    ax.set_ylabel("Tempo (s)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend([ROTULOS[v] for v in piv.columns], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Tempo de execução por versão e configuração")
fig.tight_layout()
fig.savefig("tempos_barras.png", dpi=150)

# ---- 2. Speedup de V2 e V3 em relação à V1 ----
fig, axes = plt.subplots(1, len(tamanhos), figsize=(6 * len(tamanhos), 4.5),
                         squeeze=False)
for ax, n in zip(axes[0], tamanhos):
    sub = df[df["pontos"] == n]
    piv = sub.pivot(index="caso", columns="versao", values="tempo_s")
    piv = piv.loc[ordem_casos(sub)]
    speed = pd.DataFrame({v: piv["V1"] / piv[v] for v in ("V2", "V3")})
    speed.plot.bar(ax=ax, color=[CORES["V2"], CORES["V3"]], width=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{n} pontos")
    ax.set_xlabel("")
    ax.set_ylabel("Speedup sobre V1 (T_V1 / T_Vx)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend([ROTULOS["V2"], ROTULOS["V3"]], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Ganho das versões não bloqueantes em relação à V1")
fig.tight_layout()
fig.savefig("speedup_vs_v1.png", dpi=150)

# ---- 3. Tempo x nº de processos (intra-nó vs inter-nó) ----
fig, axes = plt.subplots(1, len(tamanhos), figsize=(6 * len(tamanhos), 4.5),
                         squeeze=False)
estilos = {"intra": "-", "inter": "--"}
for ax, n in zip(axes[0], tamanhos):
    sub = df[df["pontos"] == n]
    for (versao, tipo), g in sub.groupby(["versao", "tipo"]):
        g = g.sort_values("processos")
        ax.plot(g["processos"], g["tempo_s"], estilos.get(tipo, "-"),
                marker="o", color=CORES[versao], label=f"{versao} {tipo}-nó")
    ax.set_title(f"{n} pontos")
    ax.set_xlabel("Nº de processos")
    ax.set_ylabel("Tempo (s)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sub["processos"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.suptitle("Tempo de execução vs. número de processos")
fig.tight_layout()
fig.savefig("escalabilidade.png", dpi=150)

plt.show()