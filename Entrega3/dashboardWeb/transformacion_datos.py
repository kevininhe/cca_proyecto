import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import shapely
import streamlit
from holidays import country_holidays


MAP_PATH = "https://raw.githubusercontent.com/kevininhe/cca_proyecto/main/Entrega3/data/bogota_cadastral.json"


def asignar_cuadrante(df, cols=13, rows=21):
    """
    Asigna a cada punto un cuadrante dado el tamaño de maya definida por cols y rows
    """

    mapa_bogota_no_sumapaz = gpd.read_file(MAP_PATH).cx[:, 4.45422:]

    bottomLeft = (4.45422, -74.22446)
    bottomRight = (4.45422, -73.99208494428275)
    topLeft = (4.833779672812246, -74.22446)

    cols = np.linspace(bottomLeft[1], bottomRight[1], num=cols + 1)
    rows = np.linspace(bottomLeft[0], topLeft[0], num=rows + 1)
    col = np.searchsorted(cols, df["X"]) - 1
    row = np.searchsorted(rows, df["Y"]) - 1
    df["cuadrante"] = row * (len(cols) - 1) + col
    df["columna"] = row + 1
    df["fila"] = col + 1

    cuadrantes_cord = []
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            poly = shapely.geometry.box(cols[j], rows[i], cols[j + 1], rows[i + 1])
            cuadrantes_cord.append(
                {"geometry": poly, "cuadrante": i * (len(cols) - 1) + j}
            )

    grid = (
        gpd.GeoDataFrame(
            cuadrantes_cord,
            crs=mapa_bogota_no_sumapaz.crs,
        )
        .sjoin(mapa_bogota_no_sumapaz)[["geometry", "cuadrante"]]
        .drop_duplicates()
    )

    return df, grid


def enriquecer_fechas(df) -> pd.DataFrame:
    """
    Agrega información adicional al dataset como festivos, días de la semana, mes, entre otroe
    :param df:
    :return:
    """
    df["hora"] = df.fecha_y_hora.dt.hour
    df["semana_del_año"] = df.fecha_y_hora.dt.isocalendar().week
    df["dia_de_la_semana"] = df.fecha_y_hora.dt.weekday
    df["mes"] = df.fecha_y_hora.dt.month

    colombian_holidays = country_holidays("CO")

    holidays = pd.DataFrame(
        index=pd.date_range(
            df.fecha_y_hora.min(),
            df.fecha_y_hora.max(),
            freq="d",
            normalize=True,
        )
    )

    holidays["festivo"] = np.where(
        holidays.index.map(lambda x: x in colombian_holidays), 1, 0
    )

    df["fecha_truncada"] = df["fecha_y_hora"].dt.floor("d")

    df = df.join(holidays, on="fecha_truncada").drop(columns="fecha_truncada")

    return df


@streamlit.cache_data
def crear_dataset(cols=13, rows=21, lapso="4h"):
    """
    Crea un dataset de una grilla espacio temporal y marca cada cuadrante según el número de accidentes ocurridos
    """

    df = pd.read_csv("data/dataset_preparado.csv.gz", low_memory=False)

    # Asignamos los cuadrantes
    df, grid = asignar_cuadrante(df, cols, rows)

    fecha_y_hora = pd.to_datetime(
        df["FECHA_OCURRENCIA_ACC"].str[:11] + df["HORA_OCURRENCIA_ACC"].str[:2]
    )

    fecha_corte = fecha_y_hora.max()

    # Creamos el índice de fechas agregando 7 días más para la predicción
    fechas = pd.DataFrame(
        pd.date_range(
            fecha_y_hora.min(), fecha_corte + pd.DateOffset(days=7), freq="1h"
        ),
        columns=["fecha_y_hora"],
    )

    cuadrantes = df[["cuadrante", "fila", "columna"]].drop_duplicates()

    # Creamos una tabla del producto cartesiano entre los cuadrantes y las fechas
    df_base = fechas.merge(cuadrantes, how="cross")

    # Agregamos el número de accidentes y marcamos en cero cuando no hay accidentes.
    df_base = df_base.join(
        df.groupby([fecha_y_hora, "cuadrante"]).size().rename("n_accidentes"),
        on=["fecha_y_hora", "cuadrante"],
    ).fillna(0)

    # Re-muestreamos para tener los intervalos de tiempo del tamaño deseado
    result = df_base.groupby(
        ["cuadrante", "columna", "fila", pd.Grouper(key="fecha_y_hora", freq=lapso)],
        as_index=False,
    ).sum()

    # creamos la variable accidentes indicando si hubo o no accidentes
    result["accidentes"] = np.where(result["n_accidentes"] > 1, 1, 0)

    result = enriquecer_fechas(result)

    # Trabajamos con datos rezagados 7 para poder hacer predicciones 7 días adelante
    df_rezagados = (
        result.set_index(["fecha_y_hora"]).groupby("cuadrante").shift(7, freq="d")
    )

    # Hubo accidentes hace n días a la misma hora:
    df_accidente_dias_rezagados = pd.concat(
        [
            df_rezagados.groupby("cuadrante")
            .shift(n, freq="d")
            .set_index("cuadrante", append=True)["accidentes"]
            .rename(f"accidente_{7 + n}_dias_antes_misma_hora")
            for n in range(30)
        ],
        axis=1,
    ).dropna()

    # Proporción de horas con accidentes en los últimos n días a la misma hora:
    df_accidente_dias_media_movil = pd.concat(
        [
            df_accidente_dias_rezagados[
                [f"accidente_{7 + n}_dias_antes_misma_hora" for n in range(n)]
            ]
            .mean(axis=1)
            .fillna(0)
            .rename(f"accidente_{7 + n}_dias_media_móvil")
            for n in [1, 7, 14, 21]
        ],
        axis=1,
    )

    # Proporción de horas con accidentes en los últimos n días:
    df_horas_con_accidentes_rezagado = pd.concat(
        [
            df_rezagados.groupby("cuadrante")["accidentes"]
            .rolling(f"{n}d")
            .mean()
            .rename(f"accidentes_{7 + n}_media_móvil_día")
            for n in [1, 7, 14, 21]
        ],
        axis=1,
    ).reorder_levels(["fecha_y_hora", "cuadrante"])

    df_características = pd.concat(
        [
            df_accidente_dias_rezagados,
            df_accidente_dias_media_movil,
            df_horas_con_accidentes_rezagado,
        ],
        axis=1,
    ).dropna()

    result = result.join(df_características, on=["fecha_y_hora", "cuadrante"]).dropna()

    # Mantenemos solo los datos futuros para hacer la predicción
    result = result[result["fecha_y_hora"] >= fecha_corte + pd.DateOffset(days=1)]

    return result, fecha_corte, grid


@streamlit.cache_resource
def load_model(ncols, nrows):
    return joblib.load(f"models/model_{ncols}x{nrows}.joblib")


@streamlit.cache_data
def predict(df, ncols, nrows):
    """
    Realiza la predicción para todas las fechas del df
    :param df:
    :param ncols:
    :param nrows:
    :return:
    """

    model = load_model(ncols, nrows)
    probas = model.predict_proba(df)
    return probas[:, 1]
