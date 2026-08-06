import zarr
import icechunk
import pystac
import xarray as xr
import geopandas as gpd
import rasterio as rio
import shapely
import numpy as np
import pyproj
from affine import Affine
from rasterio.warp import calculate_default_transform
from rasterio.transform import array_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from pathlib import Path
from datetime import date

from utils import get_gefs_forecast_xarray
from config import cfg_gefs,cfg_gefs_backend

def get_gefs_region_mapping(domain_polygon, ref_date, ref_feat, degree_buffer,
        mask_oversample_factor=16, mask_coverage_cutoff=.3):
    """
    Given a domain polygon, use metadata from dynamical's gefs forecast
    repository to determine the output grid dimensions and geometric profile
    before and after reprojection to Web Mercator, and return that information
    alongside a boolean mask, index mapping, and request parameters.

    :@param domain_polygon: shapely MultiPolygon bounding the regional domain
    :@param ref_date: date to use to grab metadata and coordinates
    :@param ref_feat: variable to use to grab metadata and coordinates
    :@param degree_buffer: coordinate degreess to pad the retrieved array with
        to ensure everything fits in the domain.
    :@param mask_oversample_factor: resolution multiple to use in determining
        fractional coverage of polygons within pixels
    :@param mask_coverage_cutoff: Threshold number of pixels below which a
        partially-covered pixel is excluded from the valid mask.
    """
    ## get bounding box around the domain polygon
    lon_min, lat_min, lon_max, lat_max = domain_polygon.bounds
    lat_range = [lat_min-degree_buffer, lat_max+degree_buffer]
    lon_range = [lon_min-degree_buffer, lon_max+degree_buffer]

    ## load metadata around the subdomain
    sub = get_gefs_forecast_xarray(
        variables=[ref_feat],
        date=ref_date,
        lat_range=lat_range,
        lon_range=lon_range,
        )

    ## build the source profile
    gefs_dims = list(sub.dims)
    gefs_shape = sub[ref_feat].shape
    geo_ref_src = {
        "crs":rio.crs.CRS.from_wkt(sub["spatial_ref"].crs_wkt).to_string(),
        "width":gefs_shape[gefs_dims.index("longitude")],
        "height":gefs_shape[gefs_dims.index("latitude")],
        }
    geo_ref_src["transform"] = rio.transform.from_bounds(
        lon_min,
        lat_min,
        lon_max,
        lat_max,
        geo_ref_src["width"],
        geo_ref_src["height"],
        )

    ## get the destination geometry and build the destination profile
    t_out,w_out,h_out = calculate_default_transform(
        geo_ref_src["crs"],
        cfg_gefs_backend["crs_out"],
        geo_ref_src["width"],
        geo_ref_src["height"],
        *array_bounds(
            geo_ref_src["height"],
            geo_ref_src["width"],
            geo_ref_src["transform"],
            )
        )
    geo_ref_out = {
        "crs":cfg_gefs_backend["crs_out"],
        "width":w_out,
        "height":h_out,
        "transform":t_out,
        }

    ## determine an index mapping from the source on the destination grid
    j_out,i_out = np.meshgrid(
        np.arange(h_out),
        np.arange(w_out),
        indexing="ij"
        )
    ## convert from indices to spatial coordinates
    x_out,y_out = map(np.asarray, rio.transform.xy(t_out, j_out, i_out))
    ## transform the spatial coordinates to the source domain
    x_src,y_src = rio.warp.transform(
        geo_ref_out["crs"],
        geo_ref_src["crs"],
        x_out,
        y_out,
        )
    ## convert coordinates to spatial indices on the source domain
    j_src,i_src = rio.transform.rowcol(
        geo_ref_src["transform"],
        x_src,
        y_src,
        )
    ## reshape back to the destination grid dimensions
    j_src = np.asarray(j_src).reshape(h_out, w_out)
    i_src = np.asarray(i_src).reshape(h_out, w_out)

    ## develop a latlon array and boolean mask for the destination grid
    lon_out,lat_out = rio.warp.transform(
        geo_ref_out["crs"],
        "EPSG:4326", ## lat/lon coordinates
        x_out.ravel(),
        y_out.ravel(),
        )
    lon_out = np.asarray(lon_out).reshape(h_out, w_out)
    lat_out = np.asarray(lat_out).reshape(h_out, w_out)

    ## calculate a boolean mask on a super-resolution array given the threshold
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS("EPSG:4326"),
        pyproj.CRS(geo_ref_out["crs"]),
        always_xy=True,
        ).transform
    ppoly = shapely.ops.transform(transformer, domain_polygon)
    fine_mask = rasterize(
        [(ppoly, 1)],
        out_shape=(h_out*mask_oversample_factor, w_out*mask_oversample_factor),
        transform=geo_ref_out["transform"] * Affine.scale(
            1/mask_oversample_factor, 1/mask_oversample_factor),
        fill=0,
        default_value=1,
        dtype=np.uint8
        )
    frac = fine_mask.reshape(
            h_out,
            mask_oversample_factor,
            w_out,
            mask_oversample_factor,
            ).mean(axis=(1,3))
    m_valid = frac >= mask_coverage_cutoff

    geo_ref_src["transform"] = geo_ref_src["transform"].to_gdal()
    geo_ref_out["transform"] = geo_ref_out["transform"].to_gdal()

    return (lat_range,lon_range), \
        (geo_ref_src,geo_ref_out), \
        np.stack((j_src,i_src), axis=0), \
        (lat_out,lon_out), \
        m_valid

if __name__=="__main__":
    out_zarr_path = Path("data/store/rxburn.zarr")
    domains_poly_path = Path("data/vector/usfs_domains.geojson")

    ref_date = date(2024, 1, 1)
    ref_feat = "temperature_2m"
    degree_buffer = 0.5

    """ ------------( end normal configuration )------------ """

    if out_zarr_path.exists():
        raise ValueError("exists: ", out_zarr_path.as_posix())

    doms = gpd.read_file(domains_poly_path)

    '''
    coord_bounds,geo_refs,ix_map,latlon_out,m_valid = get_gefs_region_mapping(
        domain_polygon=doms.geometry[3],
        ref_date=ref_date,
        ref_feat=ref_feat,
        degree_buffer=degree_buffer,
        )
    print(coord_bounds)
    print(latlon_out)
    print(np.count_nonzero(m_valid), m_valid.shape)
    '''

    #'''
    zgrp = zarr.open(out_zarr_path, mode="w")
    zgrp.create_group("regions")
    for r,g in zip(doms.region, doms.geometry):
        if r not in cfg_gefs_backend["get_regions"]:
            continue
        cb, (grs,gro), ixm, (lat,lon), m = get_gefs_region_mapping(
            domain_polygon=g,
            ref_date=ref_date,
            ref_feat=ref_feat,
            degree_buffer=degree_buffer,
            )
        zgrp["regions"].create_group(f"r{r}")
        zgrp[f"/regions/r{r}"].attrs.update({
            "geo_ref_src":grs,
            "geo_ref_out":gro,
            "lat_bounds":cb[0],
            "lon_bounds":cb[1],
            })
        zgrp[f"/regions/r{r}"].create_array(
            "m_valid",
            shape=m.shape,
            dtype=bool,
            )
        zgrp[f"/regions/r{r}/m_valid"][...] = m
        zgrp[f"/regions/r{r}"].create_array(
            "index_map",
            shape=ixm.shape,
            dtype=np.uint32,
            )
        zgrp[f"/regions/r{r}/index_map"][...] = ixm
        zgrp[f"/regions/r{r}"].create_array(
            "latitude",
            shape=lat.shape,
            dtype=np.float32,
            )
        zgrp[f"/regions/r{r}/latitude"][...] = lat
        zgrp[f"/regions/r{r}"].create_array(
            "longitude",
            shape=lon.shape,
            dtype=np.float32,
            )
        zgrp[f"/regions/r{r}/longitude"][...] = lon

    #'''
