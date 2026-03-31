# Estratificación de Riesgo de Diabetes — CDC BRFSS 2015

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Data-CDC%20BRFSS%202015-lightgrey)](https://www.cdc.gov/brfss/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Pipeline de ML clínico multiclase para estratificación de riesgo de diabetes usando datos conductuales y socioeconómicos a nivel poblacional.**  
> 253.680 adultos · CDC BRFSS 2015 · Sin diabetes / Prediabetes / Diabetes

---

## Contexto Clínico

El Sistema de Vigilancia de Factores de Riesgo del Comportamiento (BRFSS) de los CDC es la mayor encuesta de salud de seguimiento continuo del mundo, y recoge anualmente datos sobre conductas de salud, condiciones crónicas y prácticas de salud preventiva en adultos estadounidenses.

Este proyecto construye un pipeline de clasificación multiclase sobre los datos BRFSS 2015 para estratificar a los adultos en tres categorías: sin diabetes, prediabetes y diabetes — usando únicamente variables disponibles en el contexto de una encuesta de salud rutinaria: IMC, presión arterial, actividad física, tabaquismo, autoevaluación del estado de salud general e indicadores socioeconómicos.

**Encuadre clínico:** ¿Pueden los datos conductuales y socioeconómicos a nivel poblacional, sin ningún valor de laboratorio, estratificar de forma significativa el riesgo de diabetes en tres categorías clínicas? Este proyecto responde a esa pregunta — incluyendo su respuesta incómoda para la clase de prediabetes.

---

## Hallazgo Principal: El Problema de la Detección de Prediabetes

Un resultado central de este proyecto es que **la prediabetes es prácticamente indetectable usando únicamente variables de encuesta**, independientemente del modelo o la estrategia de balanceo de clases empleada.

Esto no es un fallo del modelo. Es una limitación de los datos con implicaciones clínicas directas:

- La prediabetes se define bioquímicamente (HbA1c 5,7–6,4% o GPB 100–125 mg/dL), no por marcadores conductuales o sintomáticos
- El BRFSS no contiene valores de laboratorio — solo conductas y condiciones autorreferidas
- Los individuos con prediabetes son en gran medida asintomáticos e indistinguibles conductualmente de adultos normoglucémicos en datos de encuesta
- Este hallazgo refuerza por qué las guías clínicas recomiendan el cribado bioquímico, y no la estratificación por puntuación de riesgo, para la detección de prediabetes

Esta limitación está documentada explícitamente en el análisis y visible en la interfaz de la aplicación.

---

## Qué Demuestra Este Proyecto

| Capa | Contenido |
|---|---|
| Razonamiento clínico | Encuadre multiclase con justificación explícita; tratamiento honesto de las limitaciones del modelo como hallazgos clínicos |
| Pipeline de ML | SMOTE para desbalance severo de clases (46:1), validación cruzada estratificada, GridSearchCV, métricas multiclase |
| Explicabilidad | SHAP multiclase — contribuciones de variables por clase, importancia global |
| Análisis de equidad | Rendimiento estratificado por sexo, grupo de edad, nivel de ingresos y nivel educativo |
| Despliegue | Aplicación Streamlit con entrada tipo encuesta, salida de riesgo en tres clases y explicación SHAP |

---

## Dataset

**Fuente:** [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)  
**Origen:** CDC BRFSS 2015 — 253.680 respuestas de encuesta, población adulta general de EE.UU.

**Variable objetivo — `Diabetes_012`:**

| Clase | Etiqueta | N | % |
|---|---|---|---|
| 0 | Sin diabetes | 213.703 | 84,2% |
| 1 | Prediabetes | 4.631 | 1,8% |
| 2 | Diabetes | 35.346 | 13,9% |

**Variables (21 en total):**

| Variable | Tipo | Significado clínico |
|---|---|---|
| HighBP | Binaria | Diagnóstico de hipertensión autorreferido |
| HighChol | Binaria | Diagnóstico de colesterol alto autorreferido |
| CholCheck | Binaria | Control de colesterol en los últimos 5 años |
| BMI | Continua | Índice de Masa Corporal |
| Smoker | Binaria | ≥100 cigarrillos a lo largo de la vida |
| Stroke | Binaria | Historia de ictus |
| HeartDiseaseorAttack | Binaria | Historia de cardiopatía coronaria o infarto |
| PhysActivity | Binaria | Actividad física en los últimos 30 días |
| Fruits | Binaria | Consumo de fruta ≥1 vez/día |
| Veggies | Binaria | Consumo de verdura ≥1 vez/día |
| HvyAlcoholConsump | Binaria | Consumo elevado de alcohol |
| AnyHealthcare | Binaria | Cualquier cobertura sanitaria |
| NoDocbcCost | Binaria | No pudo ver al médico por motivos económicos |
| GenHlth | Ordinal (1–5) | Autoevaluación del estado de salud general |
| MentHlth | 0–30 días | Días de mala salud mental |
| PhysHlth | 0–30 días | Días de mala salud física |
| DiffWalk | Binaria | Dificultad para caminar o subir escaleras |
| Sex | Binaria | 0=Mujer, 1=Hombre |
| Age | Ordinal (1–13) | Categorías de edad (18–24 hasta 80+) |
| Education | Ordinal (1–6) | Nivel educativo |
| Income | Ordinal (1–8) | Nivel de ingresos del hogar |

### Limitaciones Conocidas

- **Sin valores de laboratorio.** HbA1c, glucemia en ayunas e insulina están ausentes. Esto limita fundamentalmente la detección de prediabetes.
- **Datos autorreferidos.** El diagnóstico de diabetes y prediabetes es autorreferido, no verificado clínicamente.
- **Población estadounidense, 2015.** Puede no generalizarse a otros sistemas sanitarios, poblaciones o periodos temporales.
- **Diseño transversal.** No es posible realizar inferencia causal; no se distingue entre casos incidentes y prevalentes.
- **Desbalance severo de clases.** La clase de prediabetes (1,8%) requiere manejo explícito e interpretación cuidadosa de las métricas.

---

## Rendimiento del Modelo

| Modelo | Exactitud Balanceada | F1 Macro | Recall Diabetes | Recall Prediabetes |
|---|---|---|---|---|
| Regresión Logística | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |

> Resultados completos en `notebooks/03_evaluation.ipynb`

**Métrica principal: Exactitud Balanceada** (recall promediado macro entre clases). La exactitud convencional es engañosa con una distribución de clases del 84/2/14%.

---

## Estructura del Proyecto
```
diabetes-brfss-cdss/
├── data/
│   └── diabetes_012_health_indicators_BRFSS2015.csv
├── notebooks/
│   ├── 01_eda.ipynb              # EDA, desbalance de clases, análisis de variables
│   ├── 02_preprocessing.ipynb   # SMOTE, escalado, partición train/test
│   └── 03_evaluation.ipynb      # Comparación de modelos, SHAP, análisis de equidad
├── src/
│   ├── model_xgb.pkl
│   ├── scaler.pkl
│   └── shap_explainer.pkl
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## Ejecución Local
```bash
git clone https://github.com/Aram9574/diabetes-brfss-cdss.git
cd diabetes-brfss-cdss
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Demo en Vivo

[Abrir aplicación](https://your-streamlit-url.streamlit.app) ← *enlace actualizado tras el despliegue*

---

## Aviso Legal Clínico

Esta herramienta tiene fines exclusivamente educativos y de investigación. No constituye consejo médico y no está validada para uso clínico. El diagnóstico de prediabetes y diabetes requiere confirmación bioquímica según las guías clínicas vigentes (ADA 2024, OMS).

---

## Autor

**Alejandro Zakzuk** — Médico · IA Aplicada a la Sanidad (CEMP) · Salud Digital (Universidad Europea de Madrid)

[LinkedIn](https://linkedin.com/in/alejandrozakzuk-ia-salud-digital) · [Sitio web](https://alejandrozakzuk.com) · [GitHub](https://github.com/Aram9574)
