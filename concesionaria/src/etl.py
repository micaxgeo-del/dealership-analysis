"""
etl.py — Limpieza y feature engineering para ventas de concesionaria
Autora: Micaela Feriale
"""
import pandas as pd
import numpy as np
import logging, os

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

RAW = "data/raw/ventas_concesionaria.csv"
CLEAN = "data/processed/ventas_clean.csv"


def load(path=RAW):
    log.info(f"Cargando: {path}")
    df = pd.read_csv(path)
    log.info(f"  → {len(df):,} filas | {df.shape[1]} columnas")
    return df


def clean(df):
    log.info("Procesando...")
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["year_month"] = df["fecha"].dt.to_period("M").astype(str)
    df["semana"] = df["fecha"].dt.isocalendar().week.astype(int)
    df["dia_semana"] = df["fecha"].dt.day_name()
    df["es_fin_semana"] = df["fecha"].dt.dayofweek >= 5

    # Derived KPIs
    df["precio_log"] = np.log1p(df["precio_final"])
    df["margen_log"] = np.log1p(df["margen_bruto"])
    df["tiene_descuento"] = (df["descuento_pct"] > 0).astype(int)
    df["tiene_accesorios"] = (df["num_accesorios"] > 0).astype(int)
    df["cierre_rapido"] = (df["dias_cierre"] <= 7).astype(int)
    df["es_premium_luxury"] = df["segmento"].isin(["Premium","Luxury"]).astype(int)
    df["revenue_total_log"] = np.log1p(df["precio_total"])

    # Segment price order
    seg_ord = {"Económico":1,"Intermedio":2,"Premium":3,"Luxury":4}
    df["segmento_num"] = df["segmento"].map(seg_ord)

    df["ganancia_por_dia"] = (df["ganancia_neta"] / df["dias_cierre"].replace(0,1)).round(0)
    df["pct_accesorios"] = (df["accesorios_monto"] / df["precio_total"].replace(0,1) * 100).round(2)
    df["permuta"] = df["permuta"].astype(bool)
    df["test_drive"] = df["test_drive"].astype(bool)
    df["tiene_descuento"] = df["tiene_descuento"].astype(bool)

    log.info(f"  ✓ {len(df):,} filas limpias | {df.shape[1]} columnas")
    return df.reset_index(drop=True)


def save(df, path=CLEAN):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Guardado: {path}")


def summary(df):
    print("\n" + "="*52)
    print("  CONCESIONARIA — RESUMEN EJECUTIVO")
    print("="*52)
    print(f"  Ventas totales     : {len(df):,}")
    print(f"  Período            : {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"  Revenue total      : ${df['precio_total'].sum()/1e9:.2f}B")
    print(f"  Ganancia neta total: ${df['ganancia_neta'].sum()/1e9:.2f}B")
    print(f"  Margen promedio    : {df['margen_pct'].mean():.1f}%")
    print(f"  Ticket promedio    : ${df['precio_final'].mean():,.0f}")
    print(f"  Modelos únicos     : {df['modelo'].nunique()}")
    print(f"  Vendedores         : {df['vendedor'].nunique()}")
    print(f"  Comisiones totales : ${df['comision_vendedor'].sum()/1e6:.0f}M")
    print(f"  Con permuta        : {df['permuta'].sum()} ({df['permuta'].mean()*100:.1f}%)")
    print(f"  Días cierre prom.  : {df['dias_cierre'].mean():.1f}")
    print(f"  Rating promedio    : {df['rating_cliente'].mean():.2f}/5")
    print("="*52 + "\n")


if __name__ == "__main__":
    df = load()
    df = clean(df)
    summary(df)
    save(df)
