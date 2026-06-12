# Resumen del proyecto — Práctica B3-T5 · Entradas heterogéneas (Rossmann)

> Bitácora completa de los pasos dados en el proyecto: el trabajo del grupo (commits en `main`)
> y el trabajo de la sesión de desarrollo del modelo. Última actualización: **12-jun-2026**.
>
> **✅ ENTREGABLE: `entradas_heterogeneas.ipynb` (en `main`, commit `881923b`+)** — un único notebook
> con las secciones 1–7 del grupo y las secciones 8–14 del modelo. R² final: **0,981**.

---

## 1. Contexto de la práctica

- **Objetivo:** construir y optimizar una **red neuronal con entradas heterogéneas** para predecir
  las ventas diarias de la cadena de tiendas **Rossmann** (problema tipo Kaggle de series temporales).
- **Entregable:** un único Jupyter notebook (`.ipynb`). **Entrega: 13-jun-2026.**
- **Evaluación:** 50 % resultados (**R² en test de "las 10 tiendas"**) + 50 % análisis y justificación.
- **Datos:** `train.csv` (~1 M filas, 2013-01-01 → 2015-07-17), `test.csv` (1.115 tiendas, único día
  2015-07-31, **sin la columna `Sales`**), `store.csv` (atributos de tienda), `submission.csv` (formato).
- **Material de apoyo:** notebooks de clase en `modelos_base/` (01 endógena, 02 +exógenas,
  03 +embeddings) y el plan inicial en `pasos_segun_Codex`.

---

## 2. Trabajo del grupo (commits en `main`)

| Fecha | Autor | Commit | Aportación |
|-------|-------|--------|------------|
| 04-jun 20:01 | **Natalia** | `43d6589` | Datos Rossmann + notebooks de clase + PDF del enunciado + utilidades |
| 04-jun 23:19 | **Natalia** | `dab4210` | `requirements.txt`, `.gitignore`, `pasos_segun_Codex` y **notebook v1 (secciones 1–4)**: carga y unión con `store`, limpieza avanzada (competencia, Promo2), códigos categóricos para embeddings, enventanado temporal y split train/validación |
| 08-jun 20:45 | **Josep** | `86f7d73` | **Sección 5 (EDA)**: distribución del target, patrones temporales, tipo de tienda/surtido/promos, competencia, correlaciones. **Sección 6 (Escalado)**: `log1p` en target, `StandardScaler` en continuas, embeddings sin escalar |
| 08-jun 23:20 | **Natalia** | `9cb292a` | **Sección 7 (Baseline persistente)** a 1 y 7 días por tienda |

**Resultado del grupo:** preprocesado + EDA + escalado + baseline, todo en `entradas_heterogeneas.ipynb`.
Las ramas `desarrollo_josep` y `escalado_baseline` no contienen commits adicionales sobre `main`.

> Nota: el baseline de la sección 7 quedó **sin ejecutar** en el commit; al ejecutarlo, el
> **persistente a 7 días da R² = 0,475** (el de 1 día da R² negativo por arrastrar los ceros de los
> domingos cerrados). Esa fue la primera referencia a batir.

---

## 3. Trabajo de la sesión de desarrollo (Emilio + asistente)

Todo lo siguiente se hizo **sin modificar `main`**: el notebook del grupo se mantuvo intacto y el
trabajo nuevo vive en ficheros independientes.

### 3.1 Análisis exploratorio propio
- Notebook **`exploración_dataset.ipynb`** (13 secciones, 11 gráficas) con un EDA completo e
  independiente del de Josep, del que surgieron tres hallazgos que el EDA del grupo no recogía
  (ver sección 4).

### 3.2 Verificación del trabajo del grupo
- Se ejecutó el notebook del grupo de punta a punta (0 errores) y se obtuvo el R² del baseline (0,475).

### 3.3 Primera GRU (diagnóstico)
- Se construyó una GRU + embeddings básica (rama recurrente + 4 embeddings de tienda).
- Resultado: **R² = 0,39**, *por debajo* del baseline. Diagnóstico clave: el modelo solo veía la
  ventana pasada y **no recibía las exógenas del día que predice** → no "sabía" qué día era.
- Este trabajo se respaldó y el notebook del grupo se **restauró al estado de `main`**.

### 3.4 Modelo final optimizado
- Se desarrolló primero en un notebook autónomo (`modelo_final_entradas_heterogeneas.ipynb`) y,
  tras validarlo, se **integró en el notebook del grupo** manteniendo su estructura (el notebook
  autónomo se eliminó por redundante el 11-jun).
- Corrige el diagnóstico anterior y añade todas las decisiones de diseño (sección 5).
- Productos generados: **`submission_final.csv`** y **`mejor_modelo.keras`**.

### 3.5 Barrido nocturno de arquitecturas
- Script `entrenamiento_nocturno.py`: entrenamiento largo desatendido (~7,5 h, con `caffeinate`) de
  **6 arquitecturas grandes** con callbacks muy generosos, sobre las 1.115 tiendas (resultados en la Fase C).
- Mejor resultado del proyecto: **GRU 128+64 con Customers, R² = 0,9805**.

---

## 4. Hallazgos clave del análisis

1. **Hueco de 14 días:** `train` acaba el 2015-07-17 y `test` es el 2015-07-31 → la predicción NO
   es "a un día"; hay dos semanas sin datos.
2. **`Customers` en `test` = posible *leakage*:** correlación 0,82 con las ventas, pero en un caso
   real no se conoce el día a predecir. Se tratan dos variantes (con/sin).
3. **`Open` es casi determinista:** tienda cerrada → `Sales = 0` (solo 54 excepciones en 1 M de filas).
4. **Estacionalidad fuerte:** semanal (lunes y domingo altos, sábado bajo) y anual (pico de diciembre).
5. **`Promo` sube las ventas ~+39 %**; es la palanca exógena más potente.
6. **Heterogeneidad por tienda:** `StoreType b` (17 tiendas) vende ~10.200 € vs ~6.900 € del resto.
7. **~181 tiendas con histórico incompleto** (cierres por reforma en 2014): hay que respetar los huecos.
8. **El `test` no trae etiquetas** → el R² se mide en una validación temporal propia.

---

## 5. Diseño del modelo final (6 decisiones)

| # | Decisión | Justificación |
|---|----------|---------------|
| 1 | Modelo **funcional** con 3 tipos de entrada (GRU temporal + embeddings + numéricas) | Esencia de las *entradas heterogéneas* |
| 2 | **Exógenas del día objetivo** como rama extra (DayOfWeek, Promo, Open, StateHoliday, estacionalidad) | Corrige el fallo de la primera GRU; fue el salto de rendimiento decisivo |
| 3 | Entrenar **solo días abiertos** y forzar `Sales=0` si `Open=0` | `Open` es conocido y determinista |
| 4 | Objetivo **`log1p(Sales)`** | Distribución muy sesgada; homogeneiza el error |
| 5 | **Horizonte 14 días** en el modelo final | Replica exactamente la tarea del test |
| 6 | Dos variantes finales: **con y sin `Customers`** | Realismo vs. máximo R², con discusión de *leakage* |

**Arquitectura:** GRU(64) sobre la ventana 28×13 · embeddings Store(24)/StoreType/Assortment/PromoInterval ·
rama día-objetivo (embeddings DayOfWeek y StateHoliday + binarias + sin/cos estacional) · concatenación →
Dense(128) → Dropout(0,3) → Dense(64) → Dropout(0,2) → salida. Optimizador Adam, `EarlyStopping` + `ReduceLROnPlateau`.

---

## 6. Resultados

### Fase A — comparación de arquitecturas (10 tiendas, horizonte 1)

| Modelo | R² total | R² días abiertos |
|--------|---------:|-----------------:|
| Baseline persistente 1 día | −0,047 | −1,043 |
| Baseline persistente 7 días | 0,475 | −0,025 |
| M0 — Densa (sin recurrencia) | 0,479 | −0,017 |
| M1 — GRU solo endógena | 0,798 | 0,606 |
| M2 — GRU + embeddings de tienda | 0,836 | 0,681 |
| **M3 — M2 + exógenas del día objetivo** | **0,848** | 0,703 |
| M4 — M3 + Customers del día | 0,923 | 0,850 |
| M5 — LSTM (variante de M3) | 0,756 | 0,524 |

> La progresión M1→M2→M3 aísla las dos aportaciones reales: embeddings de tienda y, sobre todo,
> las exógenas del día objetivo. GRU y LSTM rinden parecido; se elige GRU (menos parámetros).

### Fase B — modelo final (las 1.115 tiendas, horizonte 14)

Entrenado sobre **948.504 ventanas** (746.946 de entrenamiento / 46.830 de validación), ~10 min.

| Modelo | R² total | R² días abiertos | RMSE | MAE |
|--------|---------:|-----------------:|-----:|----:|
| Baseline persistente 14 días | 0,770 | 0,601 | 1.784 | 948 |
| **GRU final SIN `Customers`** (realista) | **0,932** | 0,882 | 970 | 622 |
| **GRU final CON `Customers`** (máximo) | **0,967** | 0,943 | 674 | 444 |

- Predicción del test (2015-07-31): media **≈ 8.199 €** (variante con Customers), coherente con un
  viernes con promoción en todas las tiendas.
- Los embeddings aprendidos son interpretables (el de `Store` ordena las tiendas por nivel de venta;
  el de `StoreType` separa el tipo `b` atípico).

### Fase C — Barrido de arquitecturas (entrenamiento largo, 7,46 h · 11-jun-2026)

Barrido nocturno desatendido de **6 arquitecturas grandes** con callbacks generosos
(`EarlyStopping patience=50` + `ReduceLROnPlateau patience=18`, hasta 500 épocas), todas sobre las
1.115 tiendas a horizonte 14. Completado dentro del tope de 7,5 h.

| # | Arquitectura | Customers | R² total | R² (abiertas) | RMSE | épocas |
|---|--------------|:---------:|---------:|--------------:|-----:|-------:|
| 🥇 | **GRU 128+64** | sí | **0,9805** | 0,966 | 519 | 60 |
| 2 | LSTM 128+64 | sí | 0,9785 | 0,963 | 545 | 60 |
| 3 | GRU 128 bidireccional | sí | 0,9785 | 0,963 | 545 | 59 |
| 4 | GRU 192+96 | sí | 0,9767 | 0,960 | 568 | 60 |
| 5 | **GRU 96+48** (mejor sin Customers) | no | **0,9361** | 0,889 | 940 | 91 |
| 6 | GRU 128+64 | no | 0,9354 | 0,888 | 945 | 60 |

> **Conclusiones del barrido:** (1) el entrenamiento largo con LR decreciente mejoró el mejor modelo
> de 0,967 a **0,9805**; (2) **más capacidad no ayuda** — la red más grande (192+96) quedó por detrás
> de la media (128+64), señal de que el problema está saturado; (3) GRU, LSTM y bidireccional rinden
> casi igual (se elige GRU por simplicidad).
>
> **Modelo elegido para la entrega: GRU 128+64 con `Customers`** (R² = **0,9805**).
> Variante realista sin `Customers`: GRU 96+48 (R² = **0,9361**).

---

## 7. Estado actual y archivos

| Archivo | Contenido | Estado |
|---------|-----------|--------|
| **`entradas_heterogeneas.ipynb`** | **EL ENTREGABLE**: secciones 1–7 del grupo + 8–14 del modelo (97 celdas, ejecutado, R² 0,981) | **En `main` y `desarrollo_emilio`** |
| `exploración_dataset.ipynb` | EDA propio (Emilio), material de apoyo | Commiteado |
| `submission_final.csv` | Predicción del modelo final (R² 0,981) | Commiteado |
| `mejor_modelo.keras` | Modelo final entrenado (GRU 128+64) | Commiteado |
| `entrenamiento_nocturno.py` | Script del barrido de 6 arquitecturas | Commiteado |
| `entrenamiento_nocturno/` | Resultados del barrido: curvas + `resultados_nocturno.csv` + submissions (los 6 `.keras` quedan fuera de git) | Parcialmente commiteado |
| `RESUMEN_PROYECTO.md` | Este documento | Commiteado |
| ~~`modelo_final_entradas_heterogeneas.ipynb`~~ | Notebook intermedio | **Eliminado** (integrado en el entregable) |

> Nota: el dataset (`dataset_completo_Rossmann.../`) y los PDFs dejaron de estar trackeados en git
> (commit `1d083be`). **Al hacer `git pull` de `main`, git los elimina del disco** — hay que
> restaurarlos del historial (`git restore --source=9cb292a -- <ruta>`) o re-descargarlos del Drive.
> El notebook entregable no los necesita para consultarse (tiene los outputs embebidos).

---

## 8. Pendientes / a coordinar con el grupo

- [x] ~~Decidir el entregable~~ → **integrado en `entradas_heterogeneas.ipynb`** manteniendo la
      estructura del grupo (secciones 1–7) + modelado (8–14). Fusionado a `main` el 11-jun.
- [x] ~~Integrar la mejor red del barrido (GRU 128+64)~~ → hecha y re-entrenada en el notebook
      final (R² 0,981 con Customers / 0,937 sin).
- [x] ~~Búsqueda de arquitecturas / entrenamiento largo~~ → hecho en la Fase C (barrido nocturno).
- [x] ~~Revisión final~~ → repaso completo el 12-jun: 50/50 celdas ejecutadas en orden, 0 errores,
      submission validado (formato, Ids, ceros en cerradas), fechas del CSV invertidas pero
      corregidas por `sort_values` (sin fuga temporal).
- [ ] **Aclarar con el profesorado qué son "las 10 tiendas"** del criterio de evaluación
      (el `test.csv` trae 1.115 y sin etiquetas; nuestro R² se mide en validación temporal interna).
- [ ] Subir `entradas_heterogeneas.ipynb` al aula virtual antes del **13-jun**.
- [ ] (Opcional) *Ensemble* de las 3 mejores variantes y modelo en dos etapas (clientes → ventas).
