#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Barrido de arquitecturas para entrenamiento largo desatendido (~8 h).

Variables de entorno:
  DRY=1          -> prueba rápida (15 tiendas, 3 épocas, 2 arquitecturas) para validar el flujo
  LIMITE_HORAS=7.5  -> tope de tiempo global; al alcanzarlo termina ordenadamente

Salidas (carpeta entrenamiento_nocturno/):
  resultados_nocturno.csv     tabla de métricas (se reescribe tras CADA arquitectura)
  <nombre>.keras              mejor modelo de cada arquitectura
  <nombre>_history.csv        curva de entrenamiento de cada arquitectura
  entrenamiento_nocturno.log  log con marcas de tiempo
  submission_nocturno_max.csv      submission de la mejor variante global (máximo R²)
  submission_nocturno_realista.csv submission de la mejor variante SIN Customers
"""
import os, gc, time, traceback
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import keras
from keras.layers import (Input, GRU, LSTM, Bidirectional, Embedding, Flatten,
                          Dense, Dropout, concatenate)
from keras import Model, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.utils import set_random_seed

PROJECT = Path("/Users/emiliosanchez/Taller-B3--T5-Entradas-Heterogeneas")
OUT_DIR = PROJECT / "entrenamiento_nocturno"
OUT_DIR.mkdir(exist_ok=True)
LOGFILE = OUT_DIR / "entrenamiento_nocturno.log"

DRY = os.environ.get("DRY", "0") == "1"
LIMITE_SEG = float(os.environ.get("LIMITE_HORAS", "7.5")) * 3600
T0 = time.time()
set_random_seed(7)


def log(msg):
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S} | +{(time.time()-T0)/60:6.1f} min] {msg}"
    print(linea, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(linea + "\n")


log(f"=== INICIO {'(DRY RUN)' if DRY else '(ENTRENAMIENTO COMPLETO)'} | "
    f"límite {LIMITE_SEG/3600:.1f} h ===")

# ============================================================ PREPROCESADO
DATASET_DIR = PROJECT / "dataset_completo_Rossmann-20260604T174330Z-3-001" / "dataset_completo_Rossmann"
train = pd.read_csv(DATASET_DIR / "train.csv", parse_dates=["Date"], low_memory=False)
test  = pd.read_csv(DATASET_DIR / "test.csv",  parse_dates=["Date"], low_memory=False)
store = pd.read_csv(DATASET_DIR / "store.csv", low_memory=False)
for df_ in (train, test):
    df_["StateHoliday"] = df_["StateHoliday"].astype(str).replace({"0.0": "0"})

MESES_PROMO = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}


def preparar_tabla(df, mediana_distancia):
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(mediana_distancia)
    df["PromoInterval"] = df["PromoInterval"].fillna("None")
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                "Promo2SinceWeek", "Promo2SinceYear"]:
        df[col] = df[col].fillna(0)
    hay = (df["CompetitionOpenSinceMonth"] > 0) & (df["CompetitionOpenSinceYear"] > 0)
    meses = (df["Year"] - df["CompetitionOpenSinceYear"]) * 12 + (df["Month"] - df["CompetitionOpenSinceMonth"])
    df["CompetitionOpen"] = hay.astype(int)
    df["CompetitionOpenMonths"] = meses.where(hay, 0).clip(lower=0)
    activo = (df["Promo2"] == 1) & (df["Promo2SinceWeek"] > 0) & (df["Promo2SinceYear"] > 0)
    inicio = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if activo.any():
        codigo_iso = (df.loc[activo, "Promo2SinceYear"].astype(int).astype(str)
                      + df.loc[activo, "Promo2SinceWeek"].astype(int).astype(str).str.zfill(2) + "1")
        inicio.loc[activo] = pd.to_datetime(codigo_iso, format="%G%V%u", errors="coerce")
    df["Promo2Weeks"] = ((df["Date"] - inicio).dt.days / 7.0).fillna(0).clip(lower=0)
    mes_label = df["Month"].map(MESES_PROMO)
    en_int = np.array([m in s for m, s in zip(mes_label, df["PromoInterval"].astype(str))])
    df["Promo2ActiveMonth"] = (en_int & activo.to_numpy() & (df["Promo2Weeks"].to_numpy() > 0)).astype(int)
    return df


def crear_codigos(tr, te, columnas):
    cardinalidades, categorias = {}, {}
    for col in columnas:
        cats = pd.Index(pd.concat([tr[col], te[col]], ignore_index=True).astype(str).unique()).sort_values()
        mapa = {c: i for i, c in enumerate(cats)}
        tr[f"{col}Code"] = tr[col].astype(str).map(mapa).astype(int)
        te[f"{col}Code"] = te[col].astype(str).map(mapa).astype(int)
        cardinalidades[f"{col}Code"] = len(cats)
        categorias[col] = list(cats)
    return cardinalidades, categorias


mediana_distancia = store["CompetitionDistance"].median()
train_full = preparar_tabla(train.merge(store, on="Store", how="left", validate="many_to_one"), mediana_distancia)
test_full  = preparar_tabla(test.merge(store, on="Store", how="left", validate="many_to_one"), mediana_distancia)
cardinalidades, categorias = crear_codigos(
    train_full, test_full, ["Store", "DayOfWeek", "StateHoliday", "StoreType", "Assortment", "PromoInterval"])
train_full = train_full.sort_values(["Store", "Date"]).reset_index(drop=True)
test_full  = test_full.sort_values("Store").reset_index(drop=True)
log(f"datos preparados | train {train_full.shape} | cardinalidades {cardinalidades}")

# ============================================================ ENVENTANADO
COLS_TEMPORALES = ["Sales", "Customers", "Open", "Promo", "SchoolHoliday",
                   "CompetitionDistance", "CompetitionOpenMonths", "Promo2Weeks", "Promo2ActiveMonth",
                   "DayOfWeek", "Month", "WeekOfYear", "DayOfYear"]
COLS_ESTATICAS = ["StoreCode", "StoreTypeCode", "AssortmentCode", "PromoIntervalCode",
                  "CompetitionDistance", "CompetitionOpen", "Promo2"]
COLS_OBJETIVO_BASE = ["DayOfWeekCode", "StateHolidayCode", "Promo", "Open", "SchoolHoliday", "DayOfYear", "Customers"]
IDX_SALES_T = COLS_TEMPORALES.index("Sales")
LOOKBACK, HORIZONTE = 28, 14


def construir_ventanas(df, lookback, horizonte, tiendas=None):
    if tiendas is not None:
        df = df[df["Store"].isin(tiendas)]
    df = df.sort_values(["Store", "Date"])
    span = np.timedelta64(lookback - 1 + horizonte, "D")
    Xt_l, Xe_l, Xo_l, y_l, open_l, store_l, fecha_l = [], [], [], [], [], [], []
    descartadas = 0
    for store_id, g in df.groupby("Store", sort=True):
        n = len(g); k = n - lookback - horizonte + 1
        if k <= 0:
            continue
        arr_t = g[COLS_TEMPORALES].to_numpy(np.float32)
        arr_e = g[COLS_ESTATICAS].to_numpy(np.float32)
        arr_o = g[COLS_OBJETIVO_BASE].to_numpy(np.float32)
        fechas = g["Date"].to_numpy()
        ventanas = sliding_window_view(arr_t, lookback, axis=0).transpose(0, 2, 1)[:k]
        idx_obj = np.arange(lookback + horizonte - 1, n)
        ok = (fechas[idx_obj] - fechas[:k]) == span
        descartadas += int((~ok).sum())
        if not ok.any():
            continue
        doy = arr_o[idx_obj, 5]
        xo = np.column_stack([arr_o[idx_obj, 0], arr_o[idx_obj, 1], arr_o[idx_obj, 2],
                              arr_o[idx_obj, 3], arr_o[idx_obj, 4],
                              np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25),
                              arr_o[idx_obj, 6]]).astype(np.float32)
        Xt_l.append(ventanas[ok]); Xe_l.append(arr_e[idx_obj][ok]); Xo_l.append(xo[ok])
        y_l.append(arr_t[idx_obj, IDX_SALES_T][ok]); open_l.append(arr_o[idx_obj, 3][ok])
        store_l.append(np.full(int(ok.sum()), store_id, np.int32)); fecha_l.append(fechas[idx_obj][ok])
    return {"Xt": np.concatenate(Xt_l), "Xe": np.concatenate(Xe_l), "Xo": np.concatenate(Xo_l),
            "y": np.concatenate(y_l), "open": np.concatenate(open_l),
            "meta": pd.DataFrame({"Store": np.concatenate(store_l),
                                  "DateObjetivo": pd.to_datetime(np.concatenate(fecha_l))}),
            "descartadas": descartadas}

# ============================================================ ESCALADO
IDX_LOG_T  = [COLS_TEMPORALES.index(c) for c in ["Sales", "Customers", "CompetitionDistance"]]
IDX_BIN_T  = {COLS_TEMPORALES.index(c) for c in ["Open", "Promo", "SchoolHoliday", "Promo2ActiveMonth"]}
IDX_CONT_T = [i for i in range(len(COLS_TEMPORALES)) if i not in IDX_BIN_T]
IDX_DIST_E = COLS_ESTATICAS.index("CompetitionDistance")
IDX_CUST_O = 7


def ajustar_y_escalar(V, mask):
    Xt, Xe, Xo = V["Xt"], V["Xe"], V["Xo"]
    params = {"t": {}}
    for i in IDX_LOG_T:
        np.log1p(np.maximum(Xt[:, :, i], 0), out=Xt[:, :, i])
    for i in IDX_CONT_T:
        v = Xt[mask, :, i]; mu, sd = float(v.mean()), float(v.std() + 1e-8)
        Xt[:, :, i] = (Xt[:, :, i] - mu) / sd; params["t"][i] = (mu, sd)
    np.log1p(np.maximum(Xe[:, IDX_DIST_E], 0), out=Xe[:, IDX_DIST_E])
    mu, sd = float(Xe[mask, IDX_DIST_E].mean()), float(Xe[mask, IDX_DIST_E].std() + 1e-8)
    Xe[:, IDX_DIST_E] = (Xe[:, IDX_DIST_E] - mu) / sd; params["e"] = (mu, sd)
    np.log1p(np.maximum(Xo[:, IDX_CUST_O], 0), out=Xo[:, IDX_CUST_O])
    mu, sd = float(Xo[mask, IDX_CUST_O].mean()), float(Xo[mask, IDX_CUST_O].std() + 1e-8)
    Xo[:, IDX_CUST_O] = (Xo[:, IDX_CUST_O] - mu) / sd; params["o"] = (mu, sd)
    return params


def aplicar_escalado(params, Xt, Xe, Xo):
    for i in IDX_LOG_T:
        np.log1p(np.maximum(Xt[:, :, i], 0), out=Xt[:, :, i])
    for i, (mu, sd) in params["t"].items():
        Xt[:, :, i] = (Xt[:, :, i] - mu) / sd
    np.log1p(np.maximum(Xe[:, IDX_DIST_E], 0), out=Xe[:, IDX_DIST_E])
    mu, sd = params["e"]; Xe[:, IDX_DIST_E] = (Xe[:, IDX_DIST_E] - mu) / sd
    np.log1p(np.maximum(Xo[:, IDX_CUST_O], 0), out=Xo[:, IDX_CUST_O])
    mu, sd = params["o"]; Xo[:, IDX_CUST_O] = (Xo[:, IDX_CUST_O] - mu) / sd


def preparar_datos(V, dias_validacion=42):
    corte = V["meta"]["DateObjetivo"].max() - pd.Timedelta(days=dias_validacion)
    mask_tr = (V["meta"]["DateObjetivo"] <= corte).to_numpy()
    abiertos = V["open"] > 0
    mask_fit, mask_va = mask_tr & abiertos, ~mask_tr
    params = ajustar_y_escalar(V, mask_fit)
    D = {"params": params,
         "Xt_fit": V["Xt"][mask_fit], "Xe_fit": V["Xe"][mask_fit], "Xo_fit": V["Xo"][mask_fit],
         "y_fit_log": np.log1p(V["y"][mask_fit]),
         "Xt_vo": V["Xt"][mask_va & abiertos], "Xe_vo": V["Xe"][mask_va & abiertos],
         "Xo_vo": V["Xo"][mask_va & abiertos], "y_vo_log": np.log1p(V["y"][mask_va & abiertos]),
         "Xt_val": V["Xt"][mask_va], "Xe_val": V["Xe"][mask_va], "Xo_val": V["Xo"][mask_va],
         "y_val": V["y"][mask_va], "open_val": V["open"][mask_va]}
    log(f"corte {corte.date()} | fit {len(D['y_fit_log']):,} | val {len(D['y_val']):,}")
    return D


TIENDAS = sorted(train_full["Store"].unique())[:15] if DRY else None
t_env = time.time()
V = construir_ventanas(train_full, LOOKBACK, HORIZONTE, TIENDAS)
log(f"ventanas: {len(V['y']):,} (descartadas {V['descartadas']:,}) | "
    f"Xt {V['Xt'].nbytes/1e9:.2f} GB | {(time.time()-t_env):.0f}s")
D = preparar_datos(V)
del V; gc.collect()

# Entradas de test (una vez)
ultimas = train_full.sort_values(["Store", "Date"]).groupby("Store").tail(LOOKBACK)
stores_orden = np.array(sorted(train_full["Store"].unique()))
test_eval = test_full[test_full["Store"].isin(stores_orden)].sort_values("Store")
if DRY:
    ultimas = ultimas[ultimas["Store"].isin(TIENDAS)]
    test_eval = test_eval[test_eval["Store"].isin(TIENDAS)]
n_test = test_eval["Store"].nunique()
Xt_test = ultimas[COLS_TEMPORALES].to_numpy(np.float32).reshape(n_test, LOOKBACK, len(COLS_TEMPORALES))
Xe_test = test_eval[COLS_ESTATICAS].to_numpy(np.float32)
doy_t = test_eval["DayOfYear"].to_numpy(np.float32)
Xo_test = np.column_stack([test_eval["DayOfWeekCode"].to_numpy(np.float32),
                           test_eval["StateHolidayCode"].to_numpy(np.float32),
                           test_eval["Promo"].to_numpy(np.float32), test_eval["Open"].to_numpy(np.float32),
                           test_eval["SchoolHoliday"].to_numpy(np.float32),
                           np.sin(2*np.pi*doy_t/365.25), np.cos(2*np.pi*doy_t/365.25),
                           test_eval["Customers"].to_numpy(np.float32)]).astype(np.float32)
aplicar_escalado(D["params"], Xt_test, Xe_test, Xo_test)
open_test = test_eval["Open"].to_numpy(np.float32)
ids_test = test_eval["Id"].to_numpy()

# ============================================================ MODELO
def crear_modelo(rama="gru", capas_rnn=(128, 64), bidireccional=False,
                 densas=(256, 128, 64), dropouts=(0.35, 0.25, 0.1),
                 dim_emb_store=32, usar_dia_objetivo=True, usar_customers=True, lr=1e-3):
    set_random_seed(7)
    RNN = GRU if rama == "gru" else LSTM
    in_t = Input(shape=(LOOKBACK, len(COLS_TEMPORALES)), name="temporal")
    x = in_t
    for i, u in enumerate(capas_rnn):
        capa = RNN(u, return_sequences=(i < len(capas_rnn) - 1))
        x = (Bidirectional(capa) if bidireccional else capa)(x)
    entradas, ramas = [in_t], [x]
    for nombre, col, dim in [("store", "StoreCode", dim_emb_store), ("storetype", "StoreTypeCode", 3),
                             ("assortment", "AssortmentCode", 2), ("promointerval", "PromoIntervalCode", 2)]:
        inp = Input(shape=(1,), name=nombre)
        entradas.append(inp)
        ramas.append(Flatten()(Embedding(cardinalidades[col], dim, name=f"emb_{nombre}")(inp)))
    in_num = Input(shape=(3,), name="numerico"); entradas.append(in_num); ramas.append(in_num)
    if usar_dia_objetivo:
        in_dow = Input(shape=(1,), name="dow_obj"); entradas.append(in_dow)
        ramas.append(Flatten()(Embedding(cardinalidades["DayOfWeekCode"], 3, name="emb_dow_obj")(in_dow)))
        in_fes = Input(shape=(1,), name="festivo_obj"); entradas.append(in_fes)
        ramas.append(Flatten()(Embedding(cardinalidades["StateHolidayCode"], 2, name="emb_festivo_obj")(in_fes)))
        in_ex = Input(shape=(5,), name="exog_obj"); entradas.append(in_ex); ramas.append(in_ex)
    if usar_customers:
        in_c = Input(shape=(1,), name="customers_obj"); entradas.append(in_c); ramas.append(in_c)
    x = concatenate(ramas)
    for u, dr in zip(densas, dropouts):
        x = Dense(u, activation="relu")(x)
        if dr > 0:
            x = Dropout(dr)(x)
    out = Dense(1, name="sales_log")(x)
    m = Model(entradas, out)
    m.compile(loss="mse", optimizer=optimizers.Adam(learning_rate=lr), metrics=["mae"])
    return m


def construir_xs(Xt, Xe, Xo, usar_dia_objetivo=True, usar_customers=True, **_):
    d = {"temporal": Xt}
    for j, nombre in enumerate(["store", "storetype", "assortment", "promointerval"]):
        d[nombre] = Xe[:, j:j + 1].astype("int32")
    d["numerico"] = Xe[:, 4:7].astype("float32")
    if usar_dia_objetivo:
        d["dow_obj"] = Xo[:, 0:1].astype("int32")
        d["festivo_obj"] = Xo[:, 1:2].astype("int32")
        d["exog_obj"] = Xo[:, 2:7].astype("float32")
    if usar_customers:
        d["customers_obj"] = Xo[:, 7:8].astype("float32")
    return d


def predecir_real(modelo, Xs, open_obj):
    p = np.expm1(np.clip(modelo.predict(Xs, batch_size=4096, verbose=0).ravel(), 0.0, 12.5))
    p[open_obj <= 0] = 0.0
    return p


class LimiteYLog(keras.callbacks.Callback):
    def __init__(self, nombre, cada=5):
        self.nombre, self.cada = nombre, cada
    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.cada == 0) or self.model.stop_training:
            lr = float(self.model.optimizer.learning_rate)
            log(f"  [{self.nombre}] época {epoch+1}: loss={logs['loss']:.4f} "
                f"val_loss={logs['val_loss']:.4f} lr={lr:.1e}")
        if time.time() - T0 > LIMITE_SEG:
            self.model.stop_training = True
            log(f"  [{self.nombre}] ⏱ tope de tiempo global alcanzado")

# ============================================================ BARRIDO
EPOCHS  = 3 if DRY else 500
PAT_ES  = 2 if DRY else 50
PAT_RLR = 1 if DRY else 18
BATCH   = 256 if DRY else 512

ARQUITECTURAS = [
    ("GRU128x64_CON",     dict(rama="gru", capas_rnn=(128, 64), dim_emb_store=32, usar_customers=True)),
    ("GRU128x64_SIN",     dict(rama="gru", capas_rnn=(128, 64), dim_emb_store=32, usar_customers=False)),
    ("GRU192x96_CON",     dict(rama="gru", capas_rnn=(192, 96), densas=(384, 192, 96),
                               dropouts=(0.4, 0.3, 0.15), dim_emb_store=48, usar_customers=True)),
    ("GRU128_bidir_CON",  dict(rama="gru", capas_rnn=(128,), bidireccional=True, dim_emb_store=32, usar_customers=True)),
    ("LSTM128x64_CON",    dict(rama="lstm", capas_rnn=(128, 64), dim_emb_store=32, usar_customers=True)),
    ("GRU96x48_SIN",      dict(rama="gru", capas_rnn=(96, 48), densas=(192, 96),
                               dropouts=(0.3, 0.2), dim_emb_store=24, usar_customers=False)),
]
if DRY:
    ARQUITECTURAS = ARQUITECTURAS[:2]

Xs_fit = None  # se construye una vez (mismo para todas las arquitecturas salvo claves no usadas)
resultados, preds_test = [], {}

for nombre, cfg in ARQUITECTURAS:
    if time.time() - T0 > LIMITE_SEG:
        log(f"tope de tiempo alcanzado antes de {nombre}; se detiene el barrido")
        break
    log(f"--- entrenando {nombre} | cfg={cfg} ---")
    try:
        modelo = crear_modelo(**cfg)
        Xs_tr = construir_xs(D["Xt_fit"], D["Xe_fit"], D["Xo_fit"], **cfg)
        Xs_vo = construir_xs(D["Xt_vo"], D["Xe_vo"], D["Xo_vo"], **cfg)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=PAT_ES, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PAT_RLR, min_lr=1e-6),
            ModelCheckpoint(str(OUT_DIR / f"{nombre}.keras"), monitor="val_loss", save_best_only=True),
            CSVLogger(str(OUT_DIR / f"{nombre}_history.csv")),
            LimiteYLog(nombre),
        ]
        t_ini = time.time()
        h = modelo.fit(Xs_tr, D["y_fit_log"], validation_data=(Xs_vo, D["y_vo_log"]),
                       epochs=EPOCHS, batch_size=BATCH, callbacks=callbacks, verbose=0)
        pred_val = predecir_real(modelo, construir_xs(D["Xt_val"], D["Xe_val"], D["Xo_val"], **cfg), D["open_val"])
        ab = D["open_val"] > 0
        fila = {"arquitectura": nombre, "usar_customers": cfg.get("usar_customers", True),
                "R2_total": r2_score(D["y_val"], pred_val),
                "R2_abiertas": r2_score(D["y_val"][ab], pred_val[ab]),
                "RMSE": float(np.sqrt(mean_squared_error(D["y_val"], pred_val))),
                "MAE": float(mean_absolute_error(D["y_val"], pred_val)),
                "epocas": len(h.history["loss"]), "minutos": round((time.time() - t_ini) / 60, 1)}
        resultados.append(fila)
        preds_test[nombre] = predecir_real(modelo, construir_xs(Xt_test, Xe_test, Xo_test, **cfg), open_test)
        pd.DataFrame(resultados).sort_values("R2_total", ascending=False).to_csv(
            OUT_DIR / "resultados_nocturno.csv", index=False)
        log(f"  ✔ {nombre}: R2_total={fila['R2_total']:.4f} R2_abiertas={fila['R2_abiertas']:.4f} "
            f"| {fila['epocas']} épocas | {fila['minutos']} min")
        # Submissions parciales (se actualizan tras cada modelo, por seguridad)
        df = pd.DataFrame(resultados)
        mejor = df.sort_values("R2_total", ascending=False).iloc[0]["arquitectura"]
        pd.DataFrame({"Id": ids_test, "Sales": preds_test[mejor]}).to_csv(
            OUT_DIR / "submission_nocturno_max.csv", index=False)
        sin = df[~df["usar_customers"]].sort_values("R2_total", ascending=False)
        if len(sin):
            pd.DataFrame({"Id": ids_test, "Sales": preds_test[sin.iloc[0]["arquitectura"]]}).to_csv(
                OUT_DIR / "submission_nocturno_realista.csv", index=False)
        del modelo, Xs_tr, Xs_vo
        keras.backend.clear_session(); gc.collect()
    except Exception:
        log(f"  [ERROR] en {nombre}:\n{traceback.format_exc()}")
        continue

# ============================================================ CIERRE
if resultados:
    final = pd.DataFrame(resultados).sort_values("R2_total", ascending=False).reset_index(drop=True)
    log("=== RESULTADOS FINALES ===\n" + final.to_string(index=False))
    log(f"mejor global: {final.iloc[0]['arquitectura']} (R2={final.iloc[0]['R2_total']:.4f})")
else:
    log("No se completó ninguna arquitectura.")
log(f"=== FIN en {(time.time()-T0)/3600:.2f} h ===")
