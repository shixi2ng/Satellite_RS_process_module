import ee
import folium
import geopandas as gp
import json
import geetools

ee.Initialize()

def
# Define the ROI
shpfile = 'E:\\A_Veg_phase2\\Entire_YTR\\shpfile\\MID_YZR.shp'
shpfile_gp = gp.read_file(shpfile)
shpfile_gp.to_crs('EPSG:4326')
roi = ee.Geometry.Polygon(json.loads(shpfile_gp.geometry.to_json())["features"][0]["geometry"]["coordinates"][0])
S2_sr = ee.ImageCollection('COPERNICUS/S2_SR').filterDate('2020-01-01', '2021-12-31').filterBounds(roi)


# indexList = S2_sr.reduceColumns(ee.Reducer.toList(), ["system:index"]).get("list")
geetools.batch.Export.imagecollection.toDrive(
    S2_sr,
    'S2',
    namePattern='{id}',
    scale=10,
    dataType="float",
    region=roi,
)
