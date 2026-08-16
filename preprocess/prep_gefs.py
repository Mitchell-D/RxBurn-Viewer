import numpy as np
import zarr
import multiprocessing  as mp
## necessary so that each process has its own network memory space
## during data retrieval from dynamical icechunk repo
mp.set_start_method("spawn", force=True)
import numcodecs
numcodecs.blosc.set_nthreads(1)
from datetime import date,datetime,timedelta
from pathlib import Path

from utils import get_gefs_forecast_xarray
from config import cfg_gefs,cfg_gefs_backend
from derived_feats import derived_feats

'''
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
'''

def rescale(x, feat_key, metric_key):
    m_invalid = (~np.isfinite(x)) | (x >=cfg_gefs["invalid_thresh"])
    tmp_min,tmp_max = cfg_gefs["norm_bounds"][feat_key][metric_key]
    x = (np.clip(x, tmp_min, tmp_max) - tmp_min) / (tmp_max - tmp_min)
    x = np.round(np.clip(x * cfg_gefs["norm_res"], 0, cfg_gefs["norm_res"]))
    x[m_invalid] = cfg_gefs["mask_val"]
    return x.astype(np.uint16)

def mp_acquire_gefs_forecast(args):
    return args,acquire_gefs_forecast(**args)

def acquire_gefs_forecast(zarr_path, region_key, date):
    ## open the region group and make sure this date doesn't already exist
    print(f"starting {region_key} {date}")
    dstr = date.strftime("%Y%m%d")
    zgrp_root = zarr.open(zarr_path, mode="r")
    attrs = dict(zgrp_root[f"/regions/{region_key}"].attrs)
    zgrp = zarr.open(
        zarr_path,
        path=f"/regions/{region_key}/runs/{dstr}",
        mode="a",
        )
    assert len(cfg_gefs["labels"]["metrics"]) == 12, \
        "configuration must match the number of metrics in this method"

    ## grab raw and derived features to acquire
    raw_feats = cfg_gefs_backend["get_raw_gefs_feats"]
    all_feats = cfg_gefs["labels"]["feats"]
    fmap = cfg_gefs_backend["file_feat_mapping"]
    fmap_r = {v:k for k,v in fmap.items()}

    ## download metadata for the requested forecast subset.
    sub = get_gefs_forecast_xarray(
        variables=raw_feats,
        date=date,
        lat_range=attrs["lat_bounds_src"],
        lon_range=attrs["lon_bounds_src"],
        lead_times_limit=cfg_gefs_backend["get_lead_times"],
        )

    ## make sure the array has the anticipated dimensions
    src_spatial_shape = (
        attrs["geo_ref_src"]["height"],
        attrs["geo_ref_src"]["width"],
        )
    out_grid_shape = (
        attrs["geo_ref_out"]["height"],
        attrs["geo_ref_out"]["width"],
        )
    assert sub[raw_feats[0]].shape[-2:] == src_spatial_shape, \
        f"shape mismatch {sub[raw_feats[0]].shape=} {src_spatial_shape=}"
    src_shape = sub[raw_feats[0]].shape

    ## download data for all needed raw feats
    sub = sub.load()

    ## create the run day group and load the valid times
    vtimes = sub["valid_time"][...]
    zgrp.create_array(
        "valid_time",
        shape=vtimes.shape,
        dtype="M8[ns]",
        )
    zgrp["valid_time"][...] = vtimes

    ## load the region mask and index map
    m_valid = zgrp_root[f"/regions/{region_key}/m_valid"][...]
    ixmap = zgrp_root[f"/regions/{region_key}/index_map"][...][:,m_valid]

    ## iterate over all features, calculate derived ones, and use the index
    ## map to resample to the output domain
    out_shape_spatial = (
        len(all_feats), ## variables
        len(cfg_gefs["labels"]["metrics"]), ## metrics
        src_shape[1], ## lead times
        out_grid_shape[0], ## latitude
        out_grid_shape[1], ## longitude
        )
    out_shape_temporal = (
        len(all_feats), ## variables
        src_shape[0], ## ensemble members
        src_shape[1], ## lead times
        out_grid_shape[0], ## latitude
        out_grid_shape[1], ## longitude
        )
    zgrp.create_array(
        "spatial",
        shape=out_shape_spatial,
        chunks=cfg_gefs_backend["spatial_chunks"],
        shards=cfg_gefs_backend["spatial_shards"],
        dtype=np.uint16,
        )
    zgrp.create_array(
        "temporal",
        shape=out_shape_temporal,
        chunks=cfg_gefs_backend["temporal_chunks"],
        shards=cfg_gefs_backend["temporal_shards"],
        dtype=np.uint16,
        )
    for fix,fk in enumerate(all_feats):
        print("acquiring", dstr, fk)
        tmp = np.full(out_shape_temporal[1:], np.nan)
        if not fk in [fmap.get(k, k) for k in raw_feats]:
            if not fk in derived_feats.keys():
                raise ValueError(f"{fk} not a raw or derived feat")
            args = [sub[ak].to_numpy() for ak in derived_feats[fk][0]]
            v = derived_feats[fk][1](*args)
        else:
            v = sub[fmap_r.get(fk)].to_numpy()
        tmp[...,m_valid] = v[...,ixmap[0],ixmap[1]]
        tmp_min = np.amin(tmp, axis=0)
        tmp_max = np.amax(tmp, axis=0)
        tmp_mean = np.average(tmp, axis=0)
        tmp_stddev = np.std(tmp, axis=0)
        tmp_p10,tmp_p25,tmp_p50,tmp_p75,tmp_p90 = np.split(
            np.percentile(tmp, [10,25,50,75,90], method="linear", axis=0),
            5,
            axis=0,
            )
        tmp_max_min = tmp_max - tmp_min
        tmp_p90_p10 = tmp_p90 - tmp_p10
        tmp_p75_p25 = tmp_p75 - tmp_p25
        print(dstr, fk, np.nanmin(tmp_min), np.nanmax(tmp_max))
        zgrp["spatial"][fix,...] = np.stack([
            rescale(tmp_min, fk, "min"),
            rescale(tmp_max, fk, "max"),
            rescale(tmp_mean, fk, "mean"),
            rescale(tmp_stddev, fk, "stddev"),
            rescale(tmp_p10[0], fk, "p10"),
            rescale(tmp_p25[0], fk, "p25"),
            rescale(tmp_p50[0], fk, "p50"),
            rescale(tmp_p75[0], fk, "p75"),
            rescale(tmp_p90[0], fk, "p90"),
            rescale(tmp_max_min, fk, "max-min"),
            rescale(tmp_p90_p10[0], fk, "p90-10"),
            rescale(tmp_p75_p25[0], fk, "p75-25"),
            ], axis=0)
        zgrp["temporal"][fix,...] = rescale(tmp, fk, "default")

if __name__=="__main__":
    data_dir = Path("data")
    src_dir = data_dir.joinpath("source/gefs")
    zarr_out_path = data_dir.joinpath("store/rxburn.zarr")

    ## If True, completely overwrite any existing ensemble runs by init time.
    ## Coordinate and attribute data is always overwritten, so if appending
    ## with overwrite_existing=False, make sure they still apply.
    overwrite_existing = False

    ## If True, delete existing init times from the zarr store if they fall
    ## outside the ingest_init_date_range
    eliminate_out_of_range = True

    ## inclusive range of initialization times of ensemble files to acquire
    ingest_init_date_range = [
        date.today() - timedelta(days=3),
        date.today()
        ]

    nworkers = 8

    """ -----( GEFS ingest pipeline )----- """

    dr = ingest_init_date_range
    get_dates = {
        rn:[(dr[0] + timedelta(days=i))
            for i in range((dr[1]-dr[0]).days+1)]
        for rn in cfg_gefs_backend["get_regions"]
        }
    zgrp = zarr.open(zarr_out_path, mode="a")
    for rn in cfg_gefs_backend["get_regions"]:
        ## make sure all requested regions have been initialized
        if f"r{rn}" not in zgrp["regions"].keys():
            raise ValueError(f"Region {rn} needs to be initialized first.")
        ## make sure the runs group exists
        if "runs" not in zgrp[f"/regions/r{rn}"].keys():
            zgrp[f"/regions/r{rn}"].create_group("runs")
        ## check for already-acqured dates and remove stale ones
        for dk in zgrp[f"/regions/r{rn}/runs"].keys():
            stored_date = datetime.strptime(dk, "%Y%m%d").date()
            if stored_date in get_dates[rn]:
                if overwrite_existing:
                    del zgrp[f"/regions/r{rn}/runs/{dk}"]
                else:
                    get_dates[rn].remove(stored_date)
            elif eliminate_out_of_range:
                del zgrp[f"/regions/r{rn}/runs/{dk}"]
        for d in get_dates[rn]:
            zgrp[f"/regions/r{rn}/runs"].create_group(d.strftime("%Y%m%d"))

    args = [
        {"zarr_path":zarr_out_path, "region_key":f"r{rn}", "date":d}
        for rn in get_dates.keys()
        for d in get_dates[rn]
        ]

    with mp.Pool(nworkers) as pool:
        for a,_ in pool.imap_unordered(mp_acquire_gefs_forecast, args):
            print(f"finished", a["date"], a["region_key"])
