import numpy as np
import netCDF4 as nc
import zarr
from datetime import datetime,timedelta
from pathlib import Path
import rasterio as rio
from rasterio.warp import calculate_default_transform
from rasterio.transform import array_bounds
from rasterio.warp import reproject, Resampling
from affine import Affine

from config import cfg_gefs,cfg_gefs_backend

def rescale(x, feat_key, metric_key):
    """
    apply mask and re-normalize to uint16 according to the configuration
    """
    m_invalid = (~np.isfinite(x)) | (x >=cfg_gefs["invalid_thresh"])
    tmp_min,tmp_max = cfg_gefs["norm_bounds"][feat_key][metric_key]
    x = (np.clip(x, tmp_min, tmp_max) - tmp_min) / (tmp_max - tmp_min)
    ## each value should correspond to the lower bound of its bin range
    x = np.floor(np.clip(x*(cfg_gefs["norm_res"]+1), 0, cfg_gefs["norm_res"]))
    x[m_invalid] = cfg_gefs["mask_val"]
    return x.astype(np.uint16)

if __name__=="__main__":
    data_dir = Path("data")
    src_dir = data_dir.joinpath("source/gefs")
    zarr_out_path = data_dir.joinpath("store/rxburn.zarr")

    ## If True, completely overwrite any existing ensemble runs by init time.
    ## Coordinate and attribute data is always overwritten, so if appending
    ## with overwrite_existing=False, make sure they still apply.
    overwrite_existing = True

    ## If True, delete existing init times from the zarr store if they fall
    ## outside the ingest_init_date_range
    eliminate_out_of_range = False

    ## inclusive range of initialization times of ensemble files to acquire
    ingest_init_date_range = [datetime(2026,4,8), datetime(2026,4,9)]

    """ -----( IFS ingest pipeline )----- """

    metrics_spatial = [
        "min", "max", "mean", "stddev",
        "10pct", "25pct", "50pct", "75pct", "90pct",
        "max-min", "90-10pct", "75-25pct",
        ]

    ## Identify fefs files with init times in the requested range.
    ingest_ncs = list(sorted([
        (p,t.strftime("%Y%m%d%H")) for p,t in map(
            lambda p:(p,datetime.strptime(p.stem.split("_")[0], "%Y%m%d%H")),
            src_dir.iterdir()
            )
        if t>=ingest_init_date_range[0] and t<=ingest_init_date_range[1]
        ], key=lambda pt:pt[1]))

    ## Extract and ensure consistency of variable shapes from gefs files,
    ## and store geographic reference data.
    gefs_dims,gefs_shape,gefs_dim_dtypes = None,None,None
    geo_ref = None
    for p,t in ingest_ncs:
        ds = nc.Dataset(p.as_posix(), mode="r")
        for fk in cfg_gefs_backend["get_raw_gefs_feats"]:
            assert fk in ds.variables.keys(), f"{fk} not in {p.as_posix()}"
            if gefs_dims is None:
                gefs_dims = ds.variables[fk].dimensions
                gefs_shape = ds.variables[fk].shape
                gefs_dim_dtypes = [ds.variables[dk].dtype for dk in gefs_dims]
                geo_ref = {
                    "crs":rio.crs.CRS.from_wkt(ds["spatial_ref"].crs_wkt),
                    "transform":list(map(
                        float, str(ds["spatial_ref"].GeoTransform).split(" ")
                        )),
                    "width":gefs_shape[gefs_dims.index("longitude")],
                    "height":gefs_shape[gefs_dims.index("latitude")],
                    }
            else:
                assert ds.variables[fk].dimensions == gefs_dims, \
                    (gefs_dims, ds.variables[fk].dimensions)
                assert ds.variables[fk].shape == gefs_shape, \
                    (gefs_shape, ds.variables[fk].shape)
        ds.close()

    t_out,w_out,h_out = calculate_default_transform(
        geo_ref["crs"],
        cfg_gefs_backend["crs_out"],
        geo_ref["width"],
        geo_ref["height"],
        *array_bounds(
            geo_ref["height"],
            geo_ref["width"],
            Affine.from_gdal(*geo_ref["transform"]),
            ),
        )

    geo_ref_out = {
        "crs":cfg_gefs_backend["crs_out"],
        "width":w_out,
        "height":h_out,
        "transform":t_out,
        }

    print(geo_ref_out)

    exit(0)


    ## open the zarr store
    zstor = zarr.storage.LocalStore(zarr_out_path)

    ## Write the coordinate arrays to the zarr store, always overwriting.
    zgrp_gefs_coords = zarr.open(zstor, path="/ens/gefs/coords", mode="w")
    ## get all the relevant coords directly from the first source file.
    ## the user is responsible for ensuring they are consistent across netCDFs.
    tmp_ds = nc.Dataset(ingest_ncs[0][0].as_posix(), mode="r")
    for dk,dshape,ddtype in zip(gefs_dims,gefs_shape,gefs_dim_dtypes):
        zgrp_gefs_coords.create_array(dk, shape=dshape, dtype=ddtype)
        zgrp_gefs_coords[dk][...] = tmp_ds.variables[dk][...]
    tmp_ds.close()

    ## Declare separate groups for the spatial and temporal arrays
    ## 'a' mode doesn't destroy the store if it exists, but enables writing
    gefs_dim_sizes = dict(zip(gefs_dims, gefs_shape))
    zgrp_gefs_spatial = zarr.open(zstor, path="/ens/gefs/spatial", mode="a")
    zgrp_gefs_temporal = zarr.open(zstor, path="/ens/gefs/temporal", mode="a")

    ## labels and shapes for the dimensions of both
    dims_spatial = (
        "feats", "metrics", "horizon_time", "latitude", "longitude"
        )
    dims_temporal = (
        "feats", "horizon_time", "latitude", "longitude", "ensemble_member"
        )
    shape_spatial = (
            len(cfg_gefs_backend["get_raw_gefs_feats"]),
            len(metrics_spatial),
            gefs_dim_sizes["lead_time"],
            geo_ref_out["height"],
            geo_ref_out["width"],
            )
    shape_temporal = (
            len(cfg_gefs_backend["get_raw_gefs_feats"]),
            gefs_dim_sizes["lead_time"],
            geo_ref_out["height"],
            geo_ref_out["width"],
            gefs_dim_sizes["ensemble_member"],
            )

    init_times_str = []
    horizon_times_str = {}
    for src_path,tstr in sorted(ingest_ncs):
        ds = nc.Dataset(src_path, mode="r")
        if tstr in zgrp_gefs_spatial.keys():
            if overwrite_existing:
                del zgrp_gefs_spatial[tstr]
                del zgrp_gefs_temporal[tstr]
            else:
                continue

        itstr,_ = src_path.stem.split("_")
        init_time = datetime.strptime(itstr, "%Y%m%d%H")
        init_times_str.append(itstr)
        horizon_times_str[itstr] = [
            (init_time + timedelta(seconds=int(lts))).strftime("%Y%m%d%H%M")
            for lts in list(ds["lead_time"][...].data)
            ]

        zgrp_gefs_spatial.create_array(
            tstr,
            shape=shape_spatial,
            chunks=tuple(
                cfg_gefs["spatial_chunks"].get(k, shape_spatial[i])
                for i,k in enumerate(dims_spatial)
                ),
            shards=shape_spatial,
            dtype=np.uint16,
            )
        zgrp_gefs_temporal.create_array(
            tstr,
            shape=shape_temporal,
            chunks=tuple(
                cfg_gefs["temporal_chunks"].get(k, shape_temporal[i])
                for i,k in enumerate(dims_temporal)
                ),
            shards=shape_temporal,
            dtype=np.uint16,
            )
        for fix,fk in enumerate(cfg_gefs["get_raw_gefs_feats"]):
            ## originally (horizon, ensemble, lat, lon)
            farr = ds.variables[fk][...].data
            ## temporal array maintains individual ensemble members, and lets
            ## the client side calculate statistics
            ## store shape: (feat, horizon, lat, lon, ensemble)
            zgrp_gefs_temporal[tstr][fix] = rescale(
                    farr.transpose(0,2,3,1), fk, "default")
            pim = cfg_gefs["pctl_interp_method"]
            ## reduce along the ensemble axis to (horizon, lat, lon)
            ## stack to (metric, horizon, lat, lon)
            ## store shape: (feat, metric, horizon, lat, lon)
            xmin = np.amin(farr,axis=1)
            xmax = np.amax(farr,axis=1)
            p10 = np.percentile(farr,10,method=pim,axis=1)
            p25 = np.percentile(farr,25,method=pim,axis=1)
            p75 = np.percentile(farr,75,method=pim,axis=1)
            p90 = np.percentile(farr,90,method=pim,axis=1)

            zgrp_gefs_spatial[tstr][fix] = np.stack([
                rescale(xmin,fk,"min"),
                rescale(xmax,fk,"max"),
                rescale(np.average(farr,axis=1),fk,"mean"),
                rescale(np.std(farr,axis=1),fk,"stddev"),
                rescale(p10,fk,"10pct"),
                rescale(p25,fk,"25pct"),
                rescale(np.percentile(farr,50,method=pim,axis=1),fk,"50pct"),
                rescale(p75,fk,"75pct"),
                rescale(p90,fk,"90pct"),
                rescale(xmax-xmin,fk,"max-min"),
                rescale(p90-p10,fk,"90-10pct"),
                rescale(p75-p25,fk,"75-25pct"),
                ], axis=0)
        ds.close()

    ## Remove dates the fall out of the ingest range, if requested
    if eliminate_out_of_range:
        for tstr in zgrp_gefs_temporal.keys():
            tmpt = datetime.strptime(tstr, "%Y%m%d%H")
            t0,tf = ingest_init_date_range
            if tmpt < t0 or tmpt > tf:
                del zgrp_gefs_temporal[tmpt]

    ## consolidate attribute data for the gefs ensemble group
    zgrp_gefs = zarr.open(zstor, path="/ens/gefs", mode="a")
    zgrp_gefs.attrs.update({
        ## structure
        "feats":cfg_gefs["get_raw_gefs_feats"],
        "metrics":metrics_spatial,
        "dims_spatial":dims_spatial,
        "dims_temporal":dims_temporal,

        "init_times":init_times_str,
        "horizon_times":horizon_times_str,

        ## geometry
        "latitude_bounds":[
            np.amin(zgrp_gefs_coords["latitude"]),
            np.amax(zgrp_gefs_coords["latitude"]),
            ],
        "longitude_bounds":[
            np.amin(zgrp_gefs_coords["longitude"]),
            np.amax(zgrp_gefs_coords["longitude"]),
            ],
        "shape_spatial":shape_spatial,
        "shape_temporal":shape_temporal,
        "geo_ref":geo_ref,

        ## data normalization
        "norm_bounds":cfg_gefs["norm_bounds"],
        "norm_res":cfg_gefs["norm_res"],
        "mask_val":cfg_gefs["mask_val"],
        "cmap_default_bounds":cfg_gefs["cmap_default_bounds"],

        "spread_metrics":cfg_gefs["spread_metrics"],

        ## labels
        "long_labels_feats":cfg_gefs["long_labels_feats"],
        "long_labels_metrics":cfg_gefs["long_labels_metrics"],
        "long_labels_units":cfg_gefs["long_labels_units"],
        "short_labels_units":cfg_gefs["short_labels_units"],
        })
    print(f"finished")
