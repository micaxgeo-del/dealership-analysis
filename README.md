#  Análisis de Ventas — Concesionaria Multimarca 2022–2024
### Con Modelos ML para Predicción de Margen y Tiempo de Cierre

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completo-22c55e?style=for-the-badge)

**Análisis end-to-end de 4.000 ventas de concesionaria con 28 modelos, 8 vendedores y 3 años de datos. Incluye modelos ML para predecir margen por operación y días de cierre.**

</div>

---

##  Descripción

Pipeline completo de análisis de ventas para una concesionaria multimarca argentina (2022–2024):
- KPIs de negocio: revenue, márgenes, comisiones, accesorios
- Performance por modelo, segmento, vendedor, zona y canal
- Análisis de financiación, descuentos y origen de leads
- Modelos ML con dos targets: **margen %** y **días de cierre**

---

##  Estructura

```
concesionaria/
├── data/
│   ├── raw/ventas_concesionaria.csv       # Dataset original (4.000 registros)
│   └── processed/                          # Tablas procesadas (7 archivos)
├── notebooks/
│   └── analisis_concesionaria.ipynb        # Notebook completo
├── src/
│   ├── etl.py                              # Limpieza y feature engineering
│   ├── analysis.py                         # 10 funciones de análisis
│   ├── model.py                            # ML con 2 targets
│   └── visualizations.py                  # 12 gráficos
├── models/                                 # Modelos serializados + métricas
├── outputs/plots/                          # 12 visualizaciones
├── docs/insights.md
├── main.py
└── requirements.txt
```

---

##  Dataset — Variables principales

| Variable | Descripción |
|---|---|
| `modelo` / `marca` | Vehículo vendido (28 modelos, 12 marcas) |
| `segmento` | Económico / Intermedio / Premium / Luxury |
| `precio_final` | Precio negociado de venta |
| `margen_pct` | **Target ML 1** — Margen bruto % |
| `dias_cierre` | **Target ML 2** — Días desde contacto hasta venta |
| `vendedor` / `zona` | Vendedor responsable y zona geográfica |
| `financiacion` | Contado / Cuotas 12x–36x / Leasing / Plan ahorro |
| `origen_lead` | Web / Showroom / Referido / Instagram / WhatsApp |
| `descuento_pct` | Descuento negociado (0–5%) |
| `comision_vendedor` | Comisión calculada por venta |
| `permuta` | ¿El cliente entregó auto como parte de pago? |

---

##  KPIs del Período

| KPI | Valor |
|---|---|
| Revenue Total | $65.47B |
| Ganancia Neta | $17.77B |
| Margen Promedio | 26.3% |
| Ticket Promedio | $16.2M |
| Comisiones Totales | $1.868M |
| Días de Cierre Promedio | 10.4 días |
| Rating Promedio | 4.21 / 5 |

---

##  Machine Learning

| Target | Mejor Modelo | R² | MAPE |
|---|---|---|---|
| Margen % | Ridge / GB | ~0.31 | ~7% |
| Días de Cierre | Ridge | ~0.15 | — |

> Los días de cierre tienen alta variabilidad individual. Con datos de comportamiento del cliente (llamadas, visitas) se podría mejorar el modelo significativamente.

---

##  Visualizaciones (12)

`01` Revenue mensual + var MoM · `02` Top modelos · `03` Segmentos · `04` Performance vendedores · `05` Financiación y origen · `06` Impacto descuentos · `07` Accesorios y permuta · `08` Heatmap vendedor×segmento · `09` Feature importance · `10` Comparación modelos · `11` Pred vs. real · `12` Zona, timing y rating

---

##  Cómo ejecutar

```bash
git clone https://github.com/micaxgeo-del/dealership-analysis.git
cd dealership-analysis
pip install -r requirements.txt
python main.py
```

---

##  Insights clave

- **Toyota Hilux y Ford Ranger** son los modelos más rentables en unidades absolutas
- Los vendedores **Senior** cierran con mejor margen pero no necesariamente más rápido
- Los leads de **Referidos** tienen el ticket promedio más alto — priorizar programa de referidos
- Descuentos > 3% impactan directamente el margen; mantenerlos al mínimo en modelos Premium
- El segmento **Luxury** tarda 3x más en cerrar que el Económico pero genera 4x más ganancia por unidad

---

##  Autora

**Micaela Bianca Feriale** — Analista de Datos  
 Ferialemicaela@gmail.com ·  [linkedin.com/in/micaelaferiale](https://www.linkedin.com/in/micaelaferiale/)

<div align="center"> Si te resultó útil, dejá una estrella </div>
