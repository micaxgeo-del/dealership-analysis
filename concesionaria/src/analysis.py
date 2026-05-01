"""
analysis.py — KPIs y análisis de ventas de concesionaria
Autora: Micaela Feriale
"""
import pandas as pd
import numpy as np


def kpis(df):
    return pd.Series({
        "ventas_totales": len(df),
        "revenue_total": df["precio_total"].sum(),
        "ganancia_neta": df["ganancia_neta"].sum(),
        "margen_promedio": df["margen_pct"].mean(),
        "ticket_promedio": df["precio_final"].mean(),
        "comisiones_total": df["comision_vendedor"].sum(),
        "accesorios_total": df["accesorios_monto"].sum(),
        "pct_con_descuento": df["tiene_descuento"].mean() * 100,
        "pct_con_permuta": df["permuta"].mean() * 100,
        "pct_test_drive": df["test_drive"].mean() * 100,
        "dias_cierre_prom": df["dias_cierre"].mean(),
        "rating_promedio": df["rating_cliente"].mean(),
    })


def monthly_sales(df):
    m = (df.groupby("year_month")
         .agg(ventas=("venta_id","count"),
              revenue=("precio_total","sum"),
              ganancia=("ganancia_neta","sum"),
              margen=("margen_pct","mean"),
              ticket=("precio_final","mean"))
         .reset_index())
    m["revenue_mom"] = m["revenue"].pct_change() * 100
    return m


def by_model(df):
    return (df.groupby(["modelo","marca","segmento","tipo"])
              .agg(ventas=("venta_id","count"),
                   revenue=("precio_total","sum"),
                   ganancia=("ganancia_neta","sum"),
                   margen=("margen_pct","mean"),
                   ticket=("precio_final","mean"),
                   dias_cierre=("dias_cierre","mean"),
                   rating=("rating_cliente","mean"))
              .reset_index()
              .sort_values("revenue", ascending=False))


def by_segment(df):
    return (df.groupby("segmento")
              .agg(ventas=("venta_id","count"),
                   revenue=("precio_total","sum"),
                   ganancia=("ganancia_neta","sum"),
                   margen=("margen_pct","mean"),
                   ticket=("precio_final","mean"),
                   dias_cierre=("dias_cierre","mean"))
              .reset_index()
              .sort_values("revenue", ascending=False))


def by_seller(df):
    s = (df.groupby(["vendedor","zona","senioridad"])
           .agg(ventas=("venta_id","count"),
                revenue=("precio_total","sum"),
                ganancia=("ganancia_neta","sum"),
                comisiones=("comision_vendedor","sum"),
                margen=("margen_pct","mean"),
                dias_cierre=("dias_cierre","mean"),
                rating=("rating_cliente","mean"),
                pct_premium=("es_premium_luxury","mean"))
           .reset_index()
           .sort_values("revenue", ascending=False))
    s["revenue_share"] = s["revenue"] / s["revenue"].sum() * 100
    s["ganancia_por_venta"] = s["ganancia"] / s["ventas"]
    return s


def by_financing(df):
    return (df.groupby("financiacion")
              .agg(ventas=("venta_id","count"),
                   revenue=("precio_total","sum"),
                   ticket=("precio_final","mean"),
                   margen=("margen_pct","mean"),
                   dias_cierre=("dias_cierre","mean"))
              .reset_index()
              .sort_values("ventas", ascending=False))


def by_origin(df):
    o = (df.groupby("origen_lead")
           .agg(ventas=("venta_id","count"),
                revenue=("precio_total","sum"),
                ticket=("precio_final","mean"),
                dias_cierre=("dias_cierre","mean"),
                pct_premium=("es_premium_luxury","mean"))
           .reset_index()
           .sort_values("ventas", ascending=False))
    o["conversion_value"] = o["revenue"] / o["ventas"]
    return o


def yoy(df):
    return (df.pivot_table(index="mes", columns="año",
                           values="precio_total", aggfunc="sum")
              .assign(crecimiento=lambda x:
                  ((x[2024]-x[2023])/x[2023]*100) if (2024 in x and 2023 in x) else None)
              .reset_index())


def discount_impact(df):
    return (df.groupby("descuento_pct")
              .agg(ventas=("venta_id","count"),
                   margen=("margen_pct","mean"),
                   ticket=("precio_final","mean"),
                   dias_cierre=("dias_cierre","mean"))
              .reset_index())


def seller_ranking(df):
    """Ranking de vendedores con múltiples métricas."""
    r = by_seller(df).copy()
    for col, asc in [("ventas",False),("ganancia",False),("margen",False),("dias_cierre",True)]:
        r[f"rank_{col}"] = r[col].rank(ascending=asc).astype(int)
    r["score_total"] = (r["rank_ventas"] + r["rank_ganancia"] + r["rank_margen"] + r["rank_dias_cierre"])
    r["ranking_final"] = r["score_total"].rank().astype(int)
    return r.sort_values("ranking_final")
