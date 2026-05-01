"""
main.py — Pipeline completo: ETL → Análisis → ML → Visualizaciones
Autora: Micaela Feriale | github.com/micaelaferiale
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from src.etl import load, clean, save, summary
from src.analysis import (kpis, monthly_sales, by_model, by_segment,
                           by_seller, by_financing, by_origin, yoy,
                           discount_impact, seller_ranking)
from src.model import train
from src.visualizations import (
    plot_monthly, plot_top_models, plot_segments, plot_sellers,
    plot_financing_origin, plot_discounts, plot_extras, plot_heatmap,
    plot_fi, plot_model_comparison, plot_pred_real, plot_zone_timing,
)

def main():
    print("\n" + "="*55)
    print("  ANÁLISIS DE VENTAS — CONCESIONARIA MULTIMARCA")
    print("  Autora: Micaela Feriale")
    print("="*55)

    print("\n[1/4] ETL...")
    df = clean(load())
    summary(df)
    save(df)

    print("[2/4] Análisis...")
    m   = monthly_sales(df)
    mod = by_model(df)
    seg = by_segment(df)
    sel = by_seller(df)
    fin = by_financing(df)
    ori = by_origin(df)
    disc = discount_impact(df)
    rank = seller_ranking(df)

    m.to_csv("data/processed/monthly.csv", index=False)
    mod.to_csv("data/processed/by_model.csv", index=False)
    seg.to_csv("data/processed/by_segment.csv", index=False)
    sel.to_csv("data/processed/by_seller.csv", index=False)
    rank.to_csv("data/processed/seller_ranking.csv", index=False)
    print("  ✓ Tablas guardadas")

    print("\n[3/4] ML (2 targets: margen % + días cierre)...")
    arts = train(df)

    print("\n[4/4] Visualizaciones...")
    plot_monthly(m)
    plot_top_models(mod)
    plot_segments(seg)
    plot_sellers(sel)
    plot_financing_origin(fin, ori)
    plot_discounts(df, disc)
    plot_extras(df)
    plot_heatmap(df)
    plot_fi(arts["feature_importance"]["gb_margen_pct"],
            arts["feature_importance"]["gb_dias_cierre"])
    plot_model_comparison(arts["results"])
    plot_pred_real(arts)
    plot_zone_timing(df)

    best_m = arts["results"]["margen"].iloc[0]
    best_d = arts["results"]["dias"].iloc[0]
    print("\n" + "="*55)
    print(f"  ✅ Pipeline completo.")
    print(f"  Margen   → {best_m['model']} R²={best_m['R2']:.3f} MAPE={best_m['MAPE']:.1f}%")
    print(f"  Días     → {best_d['model']} R²={best_d['R2']:.3f} MAPE={best_d['MAPE']:.1f}%")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()
