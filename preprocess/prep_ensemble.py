import numpy as np
import netCDF4 as nc
import zarr
from datetime import datetime,timedelta
from pathlib import Path

if __name__=="__main__":
    data_dir = Path("data")
    src_ifs_dir = data_dir.joinpath("source/ifs_ens")
    src_gefs_dir = data_dir.joinpath("source/gefs")
    zarr_out_path = data_dir.joinpath("store/rxburn.zarr")

    ## If True, completely overwrite any existing ensemble runs by init time.
    ## Coordinate and attribute data is always overwritten, so if appending
    ## with overwrite_existing=False, make sure they still apply.
    overwrite_existing = False

    ## If True, delete existing init times from the zarr store if they fall
    ## outside the ingest_init_date_range
    eliminate_out_of_range = False

    ## inclusive range of initialization times of ensemble files to acquire
    ingest_init_date_range = [datetime(2026,4,8), datetime(2026,4,9)]

    ## (lead_time, ensemble_member, latitude, longitude)
    get_raw_ifs_feats = [
            "temperature_2m", "relative_humidity_2m", "wspd",
            ]

    ## numpy percentile interpolation method for percentiles falling between
    ## data points (linear, lower, higher, midpoint, or nearest)
    pctl_interp_method = "linear"

    ## None -> chunk size is data size
    ifs_temporal_chunks = {
        #"ensemble_member":None,
        "latitude":1,
        "longitude":1,
        #"lead_time":None,
        #"feat":None,
        }
    ifs_spatial_chunks = {
        #"metric":None,
        #"latitude":None,
        #"longitude":None,
        "lead_time":1,
        "feat":1,
        }

    """ -----( IFS ingest pipeline )----- """

    metrics_spatial = [
        "min", "max", "mean", "stddev",
        "10pct", "25pct", "50pct", "75pct", "90pct"
        ]

    ## Identify ifs files with init times in the requested range.
    ingest_ifs_ncs = list(sorted([
        (p,t.strftime("%Y%m%d%H")) for p,t in map(
            lambda p:(p,datetime.strptime(p.stem.split("_")[0], "%Y%m%d%H")),
            src_ifs_dir.iterdir()
            )
        if t>=ingest_init_date_range[0] and t<=ingest_init_date_range[1]
        ], key=lambda pt:pt[1]))

    ## Extract and ensure consistency of variable shapes from ifs files,
    ## and store geographic reference data.
    ifs_dims,ifs_shape,ifs_dim_dtypes = None,None,None
    geo_ref = None
    for p,t in ingest_ifs_ncs:
        ds = nc.Dataset(p.as_posix(), mode="r")
        for fk in get_raw_ifs_feats:
            assert fk in ds.variables.keys(), f"{fk} not in {p.as_posix()}"
            if ifs_dims is None:
                ifs_dims = ds.variables[fk].dimensions
                ifs_shape = ds.variables[fk].shape
                ifs_dim_dtypes = [ds.variables[dk].dtype for dk in ifs_dims]
                geo_ref = {
                    "crs_wkt":str(ds["spatial_ref"].crs_wkt),
                    "geo_transform":list(map(
                        float, str(ds["spatial_ref"].GeoTransform).split(" ")
                        )),
                    "semi_major_axis":float(ds["spatial_ref"].semi_major_axis),
                    "semi_minor_axis":float(ds["spatial_ref"].semi_minor_axis),
                    }
            else:
                assert ds.variables[fk].dimensions == ifs_dims, \
                    (ifs_dims, ds.variables[fk].dimensions)
                assert ds.variables[fk].shape == ifs_shape, \
                    (ifs_shape, ds.variables[fk].shape)
        ds.close()

    ## consolidate attribute data for the ifs ensemble group
    ifs_attrs = {
        "dims":ifs_dims,
        "feats":get_raw_ifs_feats,
        "shape":(len(get_raw_ifs_feats), *ifs_shape),
        "geo_ref":geo_ref
        }

    ## open the zarr store
    zstor = zarr.storage.LocalStore(zarr_out_path)

    ## Write the coordinate arrays to the zarr store, always overwriting.
    zgrp_ifs_coords = zarr.open(zstor, path="/ens/ifs/coords", mode="w")
    ## get all the relevant coords directly from the first source file.
    ## the user is responsible for ensuring they are consistent across netCDFs.
    tmp_ds = nc.Dataset(ingest_ifs_ncs[0][0].as_posix(), mode="r")
    for dk,dshape,ddtype in zip(ifs_dims,ifs_shape,ifs_dim_dtypes):
        zgrp_ifs_coords.create_array(dk, shape=dshape, dtype=ddtype)
        zgrp_ifs_coords[dk][...] = tmp_ds.variables[dk][...]
    tmp_ds.close()

    ## spatial arrays are shaped (metric, time, lat, lon, feat)
    ## temporal arrays are shaped (member, time, lat, lon, feat)
    ## 'a' mode doesn't destroy the store if it exists, but enables writing
    ifs_dim_sizes = dict(zip(ifs_dims, ifs_shape))
    zgrp_ifs_spatial = zarr.open(zstor, path="/ens/ifs/spatial", mode="a")
    zgrp_ifs_temporal = zarr.open(zstor, path="/ens/ifs/temporal", mode="a")
    dims_spatial = (
        "lead_time", "latitude", "longitude", "feat", "metric")
    dims_temporal = (
        "lead_time", "latitude", "longitude", "feat", "ensemble_member")
    shape_spatial = (
            ifs_dim_sizes["lead_time"],
            ifs_dim_sizes["latitude"],
            ifs_dim_sizes["longitude"],
            len(get_raw_ifs_feats),
            len(metrics_spatial),
            )
    shape_temporal = (
            ifs_dim_sizes["lead_time"],
            ifs_dim_sizes["latitude"],
            ifs_dim_sizes["longitude"],
            len(get_raw_ifs_feats),
            ifs_dim_sizes["ensemble_member"],
            )
    for src_path,tstr in ingest_ifs_ncs:
        ds = nc.Dataset(src_path, mode="r")
        if tstr in zgrp_ifs_spatial.keys():
            if overwrite_existing:
                del zgrp_ifs_spatial[tstr]
                del zgrp_ifs_temporal[tstr]
            else:
                continue
        zgrp_ifs_spatial.create_array(
            tstr,
            shape=shape_spatial,
            chunks=tuple(
                ifs_spatial_chunks.get(k, shape_spatial[i])
                for i,k in enumerate(dims_spatial)
                ),
            dtype=np.float32,
            )
        zgrp_ifs_temporal.create_array(
            tstr,
            shape=shape_temporal,
            chunks=tuple(
                ifs_temporal_chunks.get(k, shape_temporal[i])
                for i,k in enumerate(dims_temporal)
                ),
            dtype=np.float32,
            )
        for fix,fk in enumerate(get_raw_ifs_feats):
            farr = ds.variables[fk][...].transpose(0,2,3,1)
            ## temporal array maintains individual ensemble members, and lets
            ## the client side calculate statistics
            zgrp_ifs_temporal[tstr][:,:,:,fix,:] = farr
            ## "min", "max", "mean", "stddev",
            ## "10pct", "25pct", "50pct", "75pct", "90pct"
            zgrp_ifs_spatial[tstr][:,:,:,fix,:] = np.stack([
                np.amin(farr, axis=-1),
                np.amax(farr, axis=-1),
                np.average(farr, axis=-1),
                np.std(farr, axis=-1),
                np.percentile(farr, 10, method=pctl_interp_method, axis=-1),
                np.percentile(farr, 25, method=pctl_interp_method, axis=-1),
                np.percentile(farr, 50, method=pctl_interp_method, axis=-1),
                np.percentile(farr, 75, method=pctl_interp_method, axis=-1),
                np.percentile(farr, 90, method=pctl_interp_method, axis=-1),
                ], axis=-1)
        ds.close()
    exit(0)

    zgrp_ifs_data = zarr.open(zstor, path="/ens/ifs/data", mode="a")

    if "ifs" not in zgrp.keys():
        zgrp.create_array(
            "ifs_spatial",
            shape=ifs_attrs["shape"],
            chunks=tuple(ifs_chunks[k] for k in ifs_attrs["dims"])
            )
        zgrp.create_array(
            "ifs_temporal",
            shape=ifs_attrs["shape"],
            chunks=tuple(ifs_chunks[k] for k in ifs_attrs["dims"])
            )
    print(list(zgrp.keys()))

    """ -----( GEFS ingest pipeline )----- """

    exit(0)

    ingest_gefs_ncs = list(sorted([
        (p,t) for p,t in map(
            lambda p:(p,datetime.strptime(p.stem.split("_")[0], "%Y%m%d%H")),
            src_gefs_dir.iterdir()
            )
        if t>=ingest_init_date_range[0] and t<=ingest_init_date_range[1]
        ], key=lambda pt:pt[1]))
