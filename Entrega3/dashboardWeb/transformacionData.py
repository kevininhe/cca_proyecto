import pandas as pd
import numpy as np
import geopandas
import shapely
from holidays import country_holidays
from joblib import load
import os

colombian_holidays = country_holidays("CO")
bottomLeft = (4.45422, -74.22446)
bottomRight = (4.45422, -73.99208494428275)
topLeft = (4.833779672812246, -74.22446)
topRight = (4.833779672812246, -73.99208494428275)
mapa_bogota = geopandas.read_file("https://raw.githubusercontent.com/kevininhe/cca_proyecto/main/Entrega3/data/bogota_cadastral.json")
features =[
        # "cuadrante",
        "X",
        "Y",
        "hora",
        "semana_del_año",
        "dia_de_la_semana",
        "festivo",
        "accidentes_15_dias_misma_hora",
        "mes"]

model = load("./models/xgb_model_v2.joblib")
accidentes_df = pd.read_csv("data/dataset_preparado.csv.gz")

def asignarCuadrantes(accidentes_df,ncols,nrows):
    cols = np.linspace(bottomLeft[1], bottomRight[1], num=ncols)
    rows = np.linspace(bottomLeft[0], topLeft[0], num=nrows)
    accidentes_df["col"] = np.searchsorted(cols, accidentes_df["X"])
    accidentes_df["row"] = np.searchsorted(rows, accidentes_df["Y"])

    accidentes_df = accidentes_df.loc[accidentes_df["col"] > 0]
    accidentes_df = accidentes_df.loc[accidentes_df["row"] > 0]
    accidentes_df['cuadrante'] = ((accidentes_df["row"] - 1) * (len(cols) - 1)) + (accidentes_df["col"])

    cuadrantes_cord = []
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            poly = shapely.geometry.box(cols[j], rows[i], cols[j + 1], rows[i + 1])
            cuadrantes_cord.append(poly)

    cuadrantes_gdf = geopandas.GeoDataFrame(
        cuadrantes_cord, columns=["geometry"], crs=mapa_bogota.crs
    )
    cuadrantes_gdf["id_cuadrante"] = cuadrantes_gdf.index + 1

    return accidentes_df,cuadrantes_gdf

def transformarEntrada(accidentes_df):
    # Eliminar variables no necesarias
    accidentes_df = accidentes_df.dropna(subset=['Y'])
    accidentes_df = accidentes_df.drop("GEOMETRICA_A_via", axis=1)
    accidentes_df = accidentes_df.drop("GEOMETRICA_B_via", axis=1)
    accidentes_df = accidentes_df.drop("GEOMETRICA_C_via", axis=1)
    accidentes_df = accidentes_df.drop("UTILIZACION_via", axis=1)
    accidentes_df = accidentes_df.drop("CALZADAS_via", axis=1)
    accidentes_df = accidentes_df.drop("CARRILES_via", axis=1)
    accidentes_df = accidentes_df.drop("MATERIAL_via", axis=1)
    accidentes_df = accidentes_df.drop_duplicates()

    fecha_hora = pd.to_datetime(
        accidentes_df["FECHA_OCURRENCIA_ACC"].str[:11]
        + accidentes_df["HORA_OCURRENCIA_ACC"].str[:2]
    )

    fechas = pd.DataFrame(
        pd.date_range(
            fecha_hora.min(),
            fecha_hora.max(),
            freq="1h",
        ),
        columns=["fecha"],
    )
    cuadrantes = pd.DataFrame(
        range(1, accidentes_df["cuadrante"].max()),
        columns=["cuadrante"],
    )

    df_base = fechas.merge(cuadrantes, how="cross")
    # Agregamos la información de los accidentes
    df_base = df_base.join(
        accidentes_df.groupby([fecha_hora, "cuadrante"]).size().rename("numero_de_accidentes"),
        on=["fecha", "cuadrante"]
    )
    df_base["numero_de_accidentes"] = df_base["numero_de_accidentes"].fillna(0)
    # Remuestramos para obtener grupos de 4 horas
    df_base = df_base.groupby(["cuadrante", pd.Grouper(key="fecha", freq="4h")], as_index=True).sum().reset_index()
    # Coordenadas promedio
    df_base = df_base.join(
        accidentes_df.groupby("cuadrante").agg(X=("X", "mean"), Y=("Y", "mean")),
        on="cuadrante",
    )
    # Eliminamos los cuadrantes donde nunca han habido accidentes:
    df_base = df_base.dropna(subset=["X", "Y"]).copy()
    df_base["hora"] = df_base.fecha.dt.hour
    df_base["semana_del_año"] = df_base.fecha.dt.isocalendar().week
    df_base["dia_de_la_semana"] = df_base.fecha.dt.weekday
    df_base["mes"] = df_base.fecha.dt.month

    # Marcamos los festivos
    holidays = pd.DataFrame(
        index=pd.date_range(fecha_hora.min(), fecha_hora.max(), freq="d", normalize=True)
    )
    holidays["festivo"] = np.where(
        holidays.index.map(lambda x: x in colombian_holidays), 1, 0
    )
    df_base["fecha_truncada"] = df_base["fecha"].dt.floor("d")
    df_base = df_base.join(holidays, on="fecha_truncada").drop(columns="fecha_truncada")

    # Rezago de accidentes 15 dias antes
    dias = 15
    accidentes_15_dias_misma_hora = sum(
        df_base.groupby("cuadrante").apply(lambda x: x.set_index("fecha").shift(d, freq="d"))["numero_de_accidentes"]
        for d in range(dias)
    ).rename("accidentes_15_dias_misma_hora")
    df_base = df_base.join(accidentes_15_dias_misma_hora, on=["cuadrante", "fecha"])

    # Eliminamos las primeras filas que no tienen resago
    df_base = df_base.dropna(subset=["accidentes_15_dias_misma_hora"])

    # Probabilidad de accidente
    df_base["accidente"] = np.where(df_base["numero_de_accidentes"] > 0, 1, 0)

    return df_base

def generarBase(accidentes_df,ncols,nrows):
    accidentes_df, cuadrantes_gdf = asignarCuadrantes(accidentes_df,ncols,nrows)
    df_base = transformarEntrada(accidentes_df)
    # Filtrar cuadrantes
    cuadrantes_gdf = cuadrantes_gdf[cuadrantes_gdf['id_cuadrante'].isin(df_base['cuadrante'].unique())]

    df_base = df_base.sort_values(["fecha", "cuadrante"])

    # Se agrega información de coordenadas promedio a los cuadrantes
    cuadrantes_gdf = pd.merge(cuadrantes_gdf,
        df_base[["cuadrante","X","Y"]].groupby("cuadrante").agg(X=("X", "max"), Y=("Y", "max")),
        left_on="id_cuadrante",
        right_on="cuadrante"
    )
    return df_base,cuadrantes_gdf

def validarFechaFestiva(x):
    if pd.Timestamp(x).floor("d") in colombian_holidays: 
        return 1
    else:
        return 0

# Solo deberia tener la fecha como entrada, pero como debe generar la base, recibe los parametros correspondientes a la base
def calcularProbabilidades(accidentes_df,model,ncols,nrows,fecha):
    df_base,cuadrantes_gdf = generarBase(accidentes_df,ncols,nrows)
    fechas = pd.DataFrame(
        [[fecha,validarFechaFestiva(fecha)]],
        columns=["fecha","festivo"]
    )
    
    # Cruce de una fecha con todos los cuadrantes  
    df_final = fechas.merge(cuadrantes_gdf, how="cross")

    # Se agrega información del dia como tal
    df_final["hora"] = df_final.fecha.dt.hour
    df_final["semana_del_año"] = df_final.fecha.dt.isocalendar().week
    df_final["dia_de_la_semana"] = df_final.fecha.dt.weekday
    df_final["mes"] = df_final.fecha.dt.month

    # Se agrega información de accidentes de ultimos 15 dias por cuadrante y hora. Por ahora, la tomamos de los datos históricos
    df_final = df_final.merge(df_base[["fecha","cuadrante","hora","accidentes_15_dias_misma_hora"]].groupby(["fecha","cuadrante","hora"]).agg(accidentes_15_dias_misma_hora=("accidentes_15_dias_misma_hora", "max")),
        left_on=["fecha","id_cuadrante","hora"],
        right_on=["fecha","cuadrante","hora"]
    )
    df_final = df_final.drop(['geometry'], axis=1)

    X = df_final[features]
    test_predictions = X.assign(proba=model.predict_proba(X)[:, 1])[["proba"]].join(df_final)
    return test_predictions, cuadrantes_gdf

def obtenerCuadrantesDef(fecha,ncols,nrows):
    test_predictions, cuadrantes_gdf = calcularProbabilidades(accidentes_df,model,ncols,nrows,fecha)
    cuadrantes_def = cuadrantes_gdf[["geometry","id_cuadrante"]].join(
        test_predictions[test_predictions['fecha'] == fecha].set_index("id_cuadrante"),
        on="id_cuadrante",
        how="inner"
    )
    return cuadrantes_def