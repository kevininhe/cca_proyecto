import datetime

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from transformacion_datos import crear_dataset, predict


def run():
    st.set_page_config(
        page_title="Dashboard",
        page_icon="🗺️",
    )

    st.sidebar.subheader("GSD+ & SDM Bogotá")
    st.sidebar.markdown(
        """
        Mediante este dashboard usted podrá determinar los lugares y momentos prioritarios para llevar a cabo
        intervenciones enfocadas en reducir la siniestralidad y orientar el tipo de acciones a desplegar allí."""
    )

    mallas_disponibles = [f"{cols}x{2*cols}" for cols in range(1, 16)]
    tamaño = st.sidebar.selectbox("Tamaño de la malla", mallas_disponibles, index=5)

    ncols, nrows = map(int, tamaño.split("x"))

    df, fecha_corte, grilla = crear_dataset(cols=ncols, rows=nrows)
    df["proba"] = predict(df, nrows=nrows, ncols=ncols)

    fecha_s = st.sidebar.date_input(
        "Fecha de predicción",
        df["fecha_y_hora"].min() + pd.DateOffset(days=1),
        min_value=df["fecha_y_hora"].min(),
        max_value=df["fecha_y_hora"].max(),
    )

    hora_s = st.sidebar.time_input(
        "Hora predicción", datetime.time(4, 0), step=datetime.timedelta(hours=4)
    )
    fecha = datetime.datetime.combine(fecha_s, hora_s)
    df = df[df["fecha_y_hora"] == fecha]

    st.header("Reducción de accidentes viales en Bogotá D.C.")

    mapa = folium.Map(
        location=[4.6163, -74.103], zoom_start=11, tiles="CartoDB positron"
    )

    cp = folium.Choropleth(
        geo_data=grilla.to_json(),
        name="choropleth",
        data=df[["cuadrante", "proba"]],
        columns=["cuadrante", "proba"],
        key_on="feature.properties.cuadrante",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.0,
        legend_name="Probabilidades de accidentes",
        nan_fill_color="none",
    ).add_to(mapa)

    probas = df.set_index("cuadrante")["proba"].to_dict()
    for s in cp.geojson.data["features"]:
        s["properties"]["proba"] = format(
            probas.get(s["properties"]["cuadrante"], 0), "0.2f"
        )

    folium.GeoJsonTooltip(
        ["cuadrante", "proba"], ["Cuadrante ⁣ ⁣", "Probabilidad ⁣ ⁣"]
    ).add_to(cp.geojson)

    folium.LayerControl().add_to(mapa)

    st_folium(mapa, width=600)


if __name__ == "__main__":
    run()
