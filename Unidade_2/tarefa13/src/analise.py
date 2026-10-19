"""Tarefa 13: recalcula medias/speedup a partir do CSV bruto, confere com o resumo
gerado no NPAD, produz as figuras (PDF) e a tabela LaTeX de speedup."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import pandas as pd

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

# Cor fixa por caso (slot da paleta categorica validada) + marcador por familia
CORES = {
    "sem_afinidade":  "#2a78d6",  # azul
    "true_cores":     "#eb6834",  # laranja
    "close_threads":  "#1baf7a",  # aqua
    "close_cores":    "#eda100",  # amarelo
    "close_sockets":  "#e87ba4",  # magenta
    "spread_threads": "#008300",  # verde
    "spread_cores":   "#6250d6",  # violeta
    "spread_sockets": "#e34948",  # vermelho
}
MARC = {"sem_afinidade": "o", "true_cores": "D", "close_threads": "s", "close_cores": "s",
        "close_sockets": "s", "spread_threads": "^", "spread_cores": "^", "spread_sockets": "^"}
ESTILO = {"cores": "-", "threads": "--", "sockets": ":"}
ROTULO = {
    "sem_afinidade": "false (sem afinidade)",
    "true_cores": "true | cores", "close_cores": "close | cores",
    "close_threads": "close | threads", "close_sockets": "close | sockets",
    "spread_cores": "spread | cores", "spread_threads": "spread | threads",
    "spread_sockets": "spread | sockets", "master_cores": "master | cores",
}
ORDEM = ["sem_afinidade", "true_cores", "close_threads", "close_cores", "close_sockets",
         "spread_threads", "spread_cores", "spread_sockets", "master_cores"]

df = pd.read_csv("resultados_t13.csv")
assert len(df) == 9 * 9 * 5, len(df)

m = df.groupby(["caso", "threads"], as_index=False).agg(
    t_calc=("tempo_calculo_s", "mean"), t_tot=("tempo_total_s", "mean"),
    desvio=("tempo_calculo_s", "std"), migr=("migracoes", "mean"))
base = m[(m.caso == "sem_afinidade") & (m.threads == 1)].t_calc.iloc[0]
m["speedup"] = base / m.t_calc
m["efic"] = m.speedup / m.threads
m["cv"] = m.desvio / m.t_calc

# Conferencia com o resumo gerado no NPAD
r = pd.read_csv("resumo_t13.csv")
j = m.merge(r, on=["caso", "threads"], suffixes=("", "_npad"))
assert len(j) == 81
print("base (s):", round(base, 4))
print("max |dif speedup| vs resumo NPAD:", float((j.speedup - j.speedup_npad).abs().max()))
print("max |dif tempo|   vs resumo NPAD:", float((j.t_calc - j.tempo_calculo_medio_s).abs().max()))

pv = m.pivot(index="caso", columns="threads", values="speedup").loc[ORDEM]
ev = m.pivot(index="caso", columns="threads", values="efic").loc[ORDEM]
print("\nSPEEDUP\n", pv.round(2).to_string())
print("\nEFICIENCIA\n", ev.round(2).to_string())
print("\nCV (desvio/media do tempo) maximo por caso\n", m.groupby("caso").cv.max().round(2).to_string())
print("\nMigracoes medias maximas por caso\n", m.groupby("caso").migr.max().round(1).to_string())
print("\nMelhor por numero de threads")
for t in sorted(m.threads.unique()):
    s = m[m.threads == t].sort_values("speedup", ascending=False)
    print(f"  T={t:>3}: " + ", ".join(f"{c}={v:.1f}" for c, v in zip(s.caso.head(3), s.speedup.head(3)))
          + " | pior: " + f"{s.caso.iloc[-1]}={s.speedup.iloc[-1]:.2f}")

# ---------- Figuras ----------
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})
TH = sorted(m.threads.unique())

def estilo_eixos(ax, ymin, ymax):
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(TH))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in TH]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(ymin, ymax)
    ax.grid(True, which="major", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("Threads")

# Curvas que coincidem sao desenhadas com espessuras diferentes (a grossa por baixo)
LW = {"true_cores": 4.2, "spread_cores": 4.2}

def linha(ax, caso, **kw):
    d = m[m.caso == caso].sort_values("threads")
    ax.plot(d.threads, d.speedup, color=CORES[caso], marker=MARC[caso],
            linestyle=ESTILO[caso.split("_")[-1] if caso != "sem_afinidade" else "cores"],
            linewidth=LW.get(caso, 1.5), markersize=5, markeredgecolor="white", markeredgewidth=0.8,
            label=ROTULO[caso], **kw)

fig, axs = plt.subplots(2, 1, figsize=(7.4, 8.0), sharex=True, sharey=True)
for ax, casos, tit in (
    (axs[0], ["sem_afinidade", "true_cores", "close_cores", "close_threads", "spread_cores", "spread_threads"],
     "Lugares = cores ou threads"),
    (axs[1], ["sem_afinidade", "close_sockets", "spread_sockets"], "Lugares = sockets"),
):
    ax.plot(TH, TH, color=MUTED, linestyle=(0, (1, 2)), linewidth=1, label="ideal")
    for c in casos:
        linha(ax, c)
    estilo_eixos(ax, 0.8, 300)
    ax.axvline(128, color=MUTED, linewidth=0.8, alpha=0.6)
    ax.text(118, 1.0, "128 cores\nfísicos", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_title(tit, fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
for ax in axs:
    ax.set_ylabel("Speedup (base: false, 1 thread)")
axs[0].set_xlabel("")
fig.tight_layout()
fig.savefig("fig/fig1_speedup.pdf"); fig.savefig("fig/fig1_speedup.png", dpi=160)
plt.close(fig)

# Figura 2: master (oversubscription) x true|cores
fig, ax = plt.subplots(figsize=(4.8, 3.1))
dm = m[m.caso == "master_cores"].sort_values("threads")
dt = m[m.caso == "true_cores"].sort_values("threads")
ax.axhline(base, color=MUTED, linestyle=(0, (1, 2)), linewidth=1)
ax.text(5.5, base * 0.8, "tempo serial (1 thread)", fontsize=7.5, color=MUTED, va="top")
ax.plot(dt.threads, dt.t_calc, color=CORES["true_cores"], marker="D", linewidth=1.6, markersize=5,
        markeredgecolor="white", label="true | cores")
ax.plot(dm.threads, dm.t_calc, color="#52514e", marker="o", linewidth=1.6, markersize=5,
        markeredgecolor="white", label="master | cores")
estilo_eixos(ax, 0.02, 300)
ax.set_ylabel("Tempo de cálculo (s)")
ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))
fig.tight_layout()
fig.savefig("fig/fig2_master.pdf"); fig.savefig("fig/fig2_master.png", dpi=160)
plt.close(fig)

# ---------- Tabela LaTeX (speedup medio) ----------
def esc(s): return s.replace("_", r"\_")
linhas = []
for c in ORDEM:
    nome = r"\texttt{" + esc(ROTULO[c].replace(" (sem afinidade)", "")).replace("|", r"\textbar") + "}"
    vals = " & ".join((f"{pv.loc[c, t]:.1f}" if pv.loc[c, t] >= 1 else f"{pv.loc[c, t]:.2f}").replace(".", ",")
                      for t in TH)
    linhas.append(f"{nome} & {vals} \\\\")
cab = " & ".join(str(t) for t in TH)
with open("tabela_speedup.tex", "w") as f:
    f.write("\\begin{tabular}{l" + "r" * len(TH) + "}\n\\toprule\n")
    f.write(f"Política \\textbar\\ lugares & {cab} \\\\\n\\midrule\n")
    f.write("\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n")
print("\nOK: figuras e tabela_speedup.tex gerados")
