import icechunk
import pystac
import geopandas as gpd
import xarray as xr

from datetime import date
from pathlib import Path

def get_gefs_forecast_xarray(variables, date, lat_range, lon_range,
        lead_times_limit=None):
    """
    Downloads and returns the GEFS forecast ensemble for a list of variables
    on a single day within provided latitude and longitude bounds as an
    xarray Dataset
    """
    cat = pystac.Catalog.from_file("https://stac.dynamical.org/catalog.json")
    col = cat.get_child("noaa-gefs-forecast-35-day")
    asset = col.assets["icechunk-https"]
    repo = icechunk.Repository.open(icechunk.http_storage(asset.href))
    ses = repo.readonly_session("main")
    ds = xr.open_zarr(ses.store, chunks=None)
    sub = ds[variables].sel(
        init_time=date.strftime(f"%Y-%m-%dT00"),
        latitude=slice(*lat_range[::-1]),
        longitude=slice(*lon_range),
        ).isel(lead_time=slice(0, lead_times_limit))
    return sub

if __name__=="__main__":
    domain_poly_path = Path("data/vector/usfs_r8_domain_small.geojson")
    dom = gpd.read_file(domain_poly_path)
    ref_date = date(2024, 7, 22)
    degree_buffer = .25
    ztime = 0
    lon_min, lat_min, lon_max, lat_max = map(float, dom.total_bounds)
    #'''
    x = get_gefs_forecast_xarray(
        variables=[
            "temperature_2m",
            "relative_humidity_2m",
            "wind_u_10m",
            "wind_v_10m",
            "wind_gust_surface",
            ],
        date=ref_date,
        init_time=ztime,
        lat_range=[lat_min-degree_buffer, lat_max+degree_buffer],
        lon_range=[lon_min-degree_buffer, lon_max+degree_buffer],
        )
    print(x)
    #'''
