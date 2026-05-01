"""
visualizations.py — 12 gráficos para análisis de concesionaria
Autora: Micaela Feriale
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

P = ["#2563EB","#7C3AED","#059669","#D97706","#DC2626","#0891B2","#9333EA","#EA580C"]
BG = "#F8FAFC"; DK = "#1E293B"
DIR = "outputs/plots"
os.makedirs(DIR, exist_ok=True)

def _s():
    plt.rcParams.update({
        "figure.facecolor":BG,"axes.facecolor":"#FFF","axes.edgecolor":"#E2E8F0",
        "axes.labelcolor":DK,"axes.titlesize":13,"axes.titleweight":"bold",
        "axes.titlepad":10,"xtick.color":"#64748B","ytick.color":"#64748B",
        "text.color":DK,"grid.color":"#F1F5F9","grid.linewidth":0.8,
    })

def _w(fig, name):
    fig.savefig(os.path.join(DIR, name), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ {name}")


# 01 — Revenue mensual + ganancia
def plot_monthly(m):
    _s(); fig, ax = plt.subplots(2,1,figsize=(14,8),gridspec_kw={"height_ratios":[3,1]})
    fig.suptitle("Revenue y Ganancia Mensual 2022–2024",fontsize=14,fontweight="bold")
    x = range(len(m))
    ax[0].bar(x, m["revenue"]/1e9, color=P[0], alpha=0.75, label="Revenue")
    ax2 = ax[0].twinx()
    ax2.plot(x, m["ganancia"]/1e9, color=P[2], lw=2.5, marker="o", ms=5, label="Ganancia")
    ax[0].set_ylabel("Revenue (B$)"); ax2.set_ylabel("Ganancia (B$)", color=P[2])
    ax[0].set_xticks(x); ax[0].set_xticklabels(m["year_month"], rotation=45, ha="right", fontsize=7)
    ax[0].grid(axis="y", alpha=0.4)
    h1,l1 = ax[0].get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax[0].legend(h1+h2, l1+l2, loc="upper left")
    colors_m = [P[2] if v>=0 else P[4] for v in m["revenue_mom"].fillna(0)]
    ax[1].bar(x, m["revenue_mom"].fillna(0), color=colors_m, alpha=0.8)
    ax[1].axhline(0, color=DK, lw=0.8); ax[1].set_ylabel("Var MoM (%)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(m["year_month"], rotation=45, ha="right", fontsize=7)
    plt.tight_layout(); _w(fig,"01_monthly_revenue.png")


# 02 — Top modelos
def plot_top_models(models_df):
    _s(); top = models_df.head(10)
    fig, axes = plt.subplots(1,2,figsize=(15,6))
    fig.suptitle("Top 10 Modelos por Revenue y Ganancia",fontsize=14,fontweight="bold")
    seg_colors = {"Económico":P[3],"Intermedio":P[0],"Premium":P[1],"Luxury":P[4]}
    colors = [seg_colors.get(s,P[5]) for s in top["segmento"]]
    axes[0].barh(top["modelo"][::-1], top["revenue"][::-1]/1e9, color=colors[::-1], edgecolor="white")
    axes[0].set_xlabel("Revenue (B$)"); axes[0].set_title("Revenue por Modelo")
    axes[1].barh(top["modelo"][::-1], top["margen"][::-1], color=colors[::-1], alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Margen promedio (%)"); axes[1].set_title("Margen por Modelo")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor=v,label=k) for k,v in seg_colors.items()], fontsize=9)
    plt.tight_layout(); _w(fig,"02_top_models.png")


# 03 — Segmentos
def plot_segments(seg_df):
    _s(); fig, axes = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle("Análisis por Segmento de Vehículo",fontsize=14,fontweight="bold")
    axes[0].bar(seg_df["segmento"], seg_df["revenue"]/1e9, color=P[:4], edgecolor="white")
    axes[0].set_ylabel("Revenue (B$)"); axes[0].set_title("Revenue por Segmento")
    axes[1].bar(seg_df["segmento"], seg_df["margen"], color=P[:4], alpha=0.8, edgecolor="white")
    axes[1].axhline(seg_df["margen"].mean(), color=P[4], linestyle="--", lw=1.5, label="Promedio")
    axes[1].set_ylabel("Margen (%)"); axes[1].set_title("Margen por Segmento"); axes[1].legend()
    axes[2].bar(seg_df["segmento"], seg_df["dias_cierre"], color=P[:4], alpha=0.8, edgecolor="white")
    axes[2].set_ylabel("Días promedio"); axes[2].set_title("Días de Cierre por Segmento")
    plt.tight_layout(); _w(fig,"03_segments.png")


# 04 — Vendedores
def plot_sellers(seller_df):
    _s(); fig, axes = plt.subplots(2,2,figsize=(14,10))
    fig.suptitle("Performance de Vendedores",fontsize=14,fontweight="bold")
    axes[0,0].bar(seller_df["vendedor"], seller_df["ventas"], color=P[0], edgecolor="white", alpha=0.85)
    axes[0,0].set_title("Ventas por Vendedor"); axes[0,0].tick_params(axis="x",rotation=30)
    axes[0,1].bar(seller_df["vendedor"], seller_df["ganancia"]/1e6, color=P[2], edgecolor="white", alpha=0.85)
    axes[0,1].set_title("Ganancia por Vendedor (M$)"); axes[0,1].tick_params(axis="x",rotation=30)
    axes[1,0].bar(seller_df["vendedor"], seller_df["margen"], color=P[1], edgecolor="white", alpha=0.85)
    axes[1,0].axhline(seller_df["margen"].mean(), color=P[4], linestyle="--", lw=1.3)
    axes[1,0].set_title("Margen % Promedio"); axes[1,0].tick_params(axis="x",rotation=30)
    axes[1,1].bar(seller_df["vendedor"], seller_df["dias_cierre"], color=P[3], edgecolor="white", alpha=0.85)
    axes[1,1].set_title("Días Promedio de Cierre"); axes[1,1].tick_params(axis="x",rotation=30)
    plt.tight_layout(); _w(fig,"04_sellers.png")


# 05 — Financiación y origen
def plot_financing_origin(fin_df, orig_df):
    _s(); fig, axes = plt.subplots(2,2,figsize=(14,10))
    fig.suptitle("Financiación y Origen de Leads",fontsize=14,fontweight="bold")
    axes[0,0].bar(fin_df["financiacion"], fin_df["ventas"], color=P[:len(fin_df)], edgecolor="white")
    axes[0,0].set_title("Ventas por Tipo de Financiación"); axes[0,0].tick_params(axis="x",rotation=20)
    axes[0,1].bar(fin_df["financiacion"], fin_df["ticket"]/1e6, color=P[:len(fin_df)], alpha=0.8, edgecolor="white")
    axes[0,1].set_title("Ticket Promedio por Financiación (M$)"); axes[0,1].tick_params(axis="x",rotation=20)
    axes[1,0].bar(orig_df["origen_lead"], orig_df["ventas"], color=P[:len(orig_df)], edgecolor="white")
    axes[1,0].set_title("Ventas por Origen de Lead"); axes[1,0].tick_params(axis="x",rotation=20)
    axes[1,1].bar(orig_df["origen_lead"], orig_df["dias_cierre"], color=P[:len(orig_df)], alpha=0.8, edgecolor="white")
    axes[1,1].set_title("Días Cierre por Origen"); axes[1,1].tick_params(axis="x",rotation=20)
    plt.tight_layout(); _w(fig,"05_financing_origin.png")


# 06 — Descuentos
def plot_discounts(df, disc_df):
    _s(); fig, axes = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle("Impacto de los Descuentos",fontsize=14,fontweight="bold")
    pcts = df["descuento_pct"].value_counts().sort_index()
    axes[0].bar(pcts.index.astype(str)+"%", pcts.values, color=P[0], edgecolor="white", alpha=0.85)
    axes[0].set_title("Distribución de Descuentos"); axes[0].set_ylabel("Cantidad de ventas")
    axes[1].plot(disc_df["descuento_pct"], disc_df["margen"], color=P[4], lw=2.5, marker="o", ms=7)
    axes[1].fill_between(disc_df["descuento_pct"], disc_df["margen"], alpha=0.12, color=P[4])
    axes[1].set_xlabel("Descuento (%)"); axes[1].set_ylabel("Margen promedio (%)")
    axes[1].set_title("Descuento vs. Margen"); axes[1].grid(True, alpha=0.4)
    axes[2].plot(disc_df["descuento_pct"], disc_df["dias_cierre"], color=P[0], lw=2.5, marker="o", ms=7)
    axes[2].set_xlabel("Descuento (%)"); axes[2].set_ylabel("Días cierre promedio")
    axes[2].set_title("Descuento vs. Días de Cierre"); axes[2].grid(True, alpha=0.4)
    plt.tight_layout(); _w(fig,"06_discounts.png")


# 07 — Accesorios y permuta
def plot_extras(df):
    _s(); fig, axes = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle("Accesorios, Permuta y Test Drive",fontsize=14,fontweight="bold")
    acc_counts = df["num_accesorios"].value_counts().sort_index()
    axes[0].bar(acc_counts.index.astype(str), acc_counts.values, color=P[0], edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Cantidad de accesorios"); axes[0].set_ylabel("Ventas")
    axes[0].set_title("Ventas por Nro. de Accesorios")
    extras = {
        "Sin permuta": df[~df["permuta"]]["margen_pct"].mean(),
        "Con permuta": df[df["permuta"]]["margen_pct"].mean(),
        "Sin test drive": df[~df["test_drive"]]["margen_pct"].mean(),
        "Con test drive": df[df["test_drive"]]["margen_pct"].mean(),
    }
    axes[1].barh(list(extras.keys()), list(extras.values()),
                 color=[P[4],P[2],P[4],P[2]], edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Margen promedio (%)"); axes[1].set_title("Impacto Permuta / Test Drive en Margen")
    seg_acc = df.groupby("segmento")["accesorios_monto"].mean() / 1e3
    axes[2].bar(seg_acc.index, seg_acc.values, color=P[:4], edgecolor="white", alpha=0.85)
    axes[2].set_ylabel("Accesorios promedio (K$)"); axes[2].set_title("Accesorios Promedio por Segmento")
    plt.tight_layout(); _w(fig,"07_extras.png")


# 08 — Heatmap vendedor x segmento
def plot_heatmap(df):
    _s()
    pivot = df.pivot_table(index="vendedor", columns="segmento",
                           values="ganancia_neta", aggfunc="sum") / 1e6
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(pivot, ax=ax, cmap="Blues", annot=True, fmt=".0f",
                linewidths=0.5, linecolor="#E2E8F0",
                cbar_kws={"label":"Ganancia (M$)"})
    ax.set_title("Ganancia por Vendedor y Segmento (M$)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Segmento"); ax.set_ylabel("Vendedor")
    plt.tight_layout(); _w(fig,"08_heatmap_seller_segment.png")


# 09 — Feature importance ML
def plot_fi(fi_margin, fi_days):
    _s(); fig, axes = plt.subplots(1,2,figsize=(14,6))
    fig.suptitle("Feature Importance — Gradient Boosting",fontsize=14,fontweight="bold")
    top_m = fi_margin.head(12)
    axes[0].barh(top_m["feature"][::-1], top_m["importance"][::-1], color=P[0], edgecolor="white", alpha=0.85)
    axes[0].set_title("Predicción de Margen %"); axes[0].set_xlabel("Importancia")
    top_d = fi_days.head(12)
    axes[1].barh(top_d["feature"][::-1], top_d["importance"][::-1], color=P[1], edgecolor="white", alpha=0.85)
    axes[1].set_title("Predicción de Días de Cierre"); axes[1].set_xlabel("Importancia")
    plt.tight_layout(); _w(fig,"09_feature_importance.png")


# 10 — Modelo comparison
def plot_model_comparison(results_dict):
    _s(); fig, axes = plt.subplots(1,2,figsize=(14,5))
    fig.suptitle("Comparación de Modelos ML",fontsize=14,fontweight="bold")
    for ax, (target, results_df) in zip(axes, results_dict.items()):
        r = results_df.sort_values("R2", ascending=False)
        colors_c = [P[2] if i==0 else P[0] for i in range(len(r))]
        ax.bar(r["model"], r["R2"], color=colors_c, edgecolor="white")
        ax.set_ylim(0,1); ax.set_title(f"R² — Target: {target}")
        ax.tick_params(axis="x", rotation=15)
        for bar, v in zip(ax.patches, r["R2"]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout(); _w(fig,"10_model_comparison.png")


# 11 — Pred vs real
def plot_pred_real(arts):
    _s(); fig, axes = plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("Predicciones vs. Real — Gradient Boosting",fontsize=14,fontweight="bold")
    for ax, key, label in zip(axes,
        ["gb_margen_pct","gb_dias_cierre"],
        ["Margen %","Días de Cierre"]):
        m = arts["models"][key]
        yt = np.array(m["yte"]); yp = np.array(m["ypred"])
        idx = np.random.choice(len(yt), min(400,len(yt)), replace=False)
        ax.scatter(yt[idx], yp[idx], alpha=0.4, color=P[0], s=15, edgecolors="none")
        mn,mx = min(yt.min(),yp.min()), max(yt.max(),yp.max())
        ax.plot([mn,mx],[mn,mx],"r--",lw=1.5,label="Predicción perfecta")
        ax.set_xlabel(f"Real ({label})"); ax.set_ylabel(f"Predicho ({label})")
        ax.set_title(f"Real vs. Predicho — {label}"); ax.legend()
    plt.tight_layout(); _w(fig,"11_pred_vs_real.png")


# 12 — Zona y cierre
def plot_zone_timing(df):
    _s(); fig, axes = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle("Zona, Timing y Rating",fontsize=14,fontweight="bold")
    zona_r = df.groupby("zona")["precio_total"].sum()
    axes[0].bar(zona_r.index, zona_r.values/1e9, color=P[:len(zona_r)], edgecolor="white")
    axes[0].set_ylabel("Revenue (B$)"); axes[0].set_title("Revenue por Zona")
    axes[1].hist(df["dias_cierre"].clip(0,60), bins=30, color=P[3], edgecolor="white", alpha=0.85)
    axes[1].axvline(df["dias_cierre"].median(), color=P[4], linestyle="--", lw=1.8,
                    label=f"Mediana: {df['dias_cierre'].median():.0f} días")
    axes[1].set_xlabel("Días de cierre"); axes[1].set_title("Distribución Días de Cierre"); axes[1].legend()
    rating_c = df["rating_cliente"].value_counts().sort_index()
    axes[2].bar(rating_c.index, rating_c.values,
                color=[P[4],P[3],P[5],P[3],P[2]], edgecolor="white")
    axes[2].set_xlabel("Rating"); axes[2].set_title("Distribución de Ratings")
    plt.tight_layout(); _w(fig,"12_zone_timing_rating.png")
