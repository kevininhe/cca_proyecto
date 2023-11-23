import streamlit as st
from streamlit.logger import get_logger
import folium
from streamlit_folium import st_folium
import transformacionData
import datetime

LOGGER = get_logger(__name__)
ncols = 6
nrows = 10

@st.cache_data
def getCuadrantes(fecha):
    return transformacionData.obtenerCuadrantesDef(fecha,ncols,nrows)

def run():
    st.set_page_config(
        page_title="Dashboard",
        page_icon="🗺️",
    )

    st.write("# **Política pública para la reducción de accidentes viales en Bogotá**")

    st.markdown(
        """
        ### GSD+ & SDM Bogotá

        Mediante este dashboard, se podrán determinar los lugares y momentos prioritarios para llevar a cabo intervenciones enfocadas en reducir la siniestralidad y orientar el tipo de acciones a desplegar allí.
        """
    )

    m = folium.Map(location=[4.6163, -74.103], zoom_start=11, tiles="CartoDB positron")
    fecha_s = st.date_input("Fecha de predicción", datetime.date(2022, 1, 1),key='date_picker')
    hora_s = st.time_input('Hora prediccion', datetime.time(4, 0),step=14400)
    fecha = datetime.datetime.combine(fecha_s, hora_s)

    cuadrantes_def = getCuadrantes(fecha)
    geo_j = cuadrantes_def[["geometry","id_cuadrante"]].to_json()
    cp = folium.Choropleth(
    geo_data=geo_j,
    name="choropleth",
    data=cuadrantes_def[["id_cuadrante","proba"]],
    columns=["id_cuadrante", "proba"],
    key_on="feature.properties.id_cuadrante",
    fill_color="YlGn",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Probabilidades de accidentes",
    ).add_to(m)

    for s in cp.geojson.data['features']:
        proba = cuadrantes_def.loc[cuadrantes_def["id_cuadrante"] == s['properties']["id_cuadrante"]]["proba"].values[0]
        s['properties']['proba'] = format((proba*100),'.4f')

    folium.GeoJsonTooltip(["id_cuadrante","proba"]).add_to(cp.geojson)
    folium.LayerControl().add_to(m)

    st_data = st_folium(m, width=500)

if __name__ == "__main__":
    run()