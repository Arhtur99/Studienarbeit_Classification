import plotly.figure_factory as ff
import geopandas
import numpy as np
import pandas as pd
import _plotly_geo

df_sample = pd.read_csv('training.csv')

c = df_sample.groupby(["VNZIP1"]).mean()["IsBadBuy"]

vnzip = c.index
isbadbuy = c.values
"""
df_sample['State FIPS Code'] = df_sample['State FIPS Code'].apply(lambda x: str(x).zfill(2))
df_sample['County FIPS Code'] = df_sample['County FIPS Code'].apply(lambda x: str(x).zfill(3))

df_sample['FIPS'] = df_sample['State FIPS Code'] + df_sample['County FIPS Code']
"""

colorscale = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",
              "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",
              "#08519c","#0b4083","#08306b"]
endpts = list(np.linspace(1, 12, len(colorscale) - 1))
fips = vnzip.tolist()
values = isbadbuy .tolist()

fig = ff.create_choropleth(
    fips=fips, values=values,
    binning_endpoints=endpts,
    colorscale=colorscale,
    show_state_data=False,
    show_hover=True, centroid_marker={'opacity': 0},
    asp=2.9, title='USA by Unemployment %',
    legend_title='% unemployed'
)

fig.layout.template = None
fig.show()

