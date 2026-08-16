"""
Update metadata from the config to the zarr database, including labels,
color maps, and static (non-selectable) polygons.
"""
import zarr
import numpy as np
from pathlib import Path
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as pltc

import config

def get_cmaps(cmap_list, cmap_resolution, use_cmasher=False):
    """
    Given a list of string color map labels, generate a list table of
    flattened uint8 color map lookup tables and a list of slices that
    capture the slice partitioning each one if they are concatenated together.

    :@param cmap_list: matplotlib/cmasher string labels of color maps
    :@param cmap_resolution: integer resolution for all color maps
    :@param use_cmasher: If True, enables using cmasher color map strings too.
    """
    ## import dynamically so that cmasher dependency is optional, and heavy
    ## matplotlib load isn't default for the config script
    if use_cmasher:
        import cmasher as cmr
    cmap_arrays = []
    cmap_slices = []
    prv_ix = 0
    for cml in cmap_list:
        ## retrieve the color map and append a nan value last for transparent
        if cml in config.custom_cmaps.keys():
            ccmc = config.custom_cmaps[cml]
            if ccmc["type"] == "listed":
                cm = pltc.ListedColormap(
                    colors=ccmc["colors"],
                    name=cml,
                    )
            else:
                raise ValueError("unrecognized custom cmap type:",ccmc["type"])
        else:
            cm = plt.get_cmap(cml)
        tmp_cmap = cm(np.append(
            np.linspace(0, 1, int(cmap_resolution)),
            np.array(np.nan),
            ))

        ## convert to uint8 and flatten.
        cmap_arrays.append((tmp_cmap*255).astype(np.uint8).reshape(-1))
        new_ix = prv_ix + tmp_cmap.size
        cmap_slices.append((prv_ix, new_ix))
        prv_ix = new_ix
    return cmap_arrays,cmap_slices

if __name__=="__main__":
    out_zarr_path = Path("data/store/rxburn.zarr")
    zgrp = zarr.open(out_zarr_path, mode="a")
    vec_dir = Path("data/vector")
    domains_geojson = vec_dir.joinpath("usfs_domains.geojson")

    load_meta = True
    load_cmaps = True
    load_vectors = True

    if load_meta:
        rconf = {}
        for rk in zgrp["regions"].keys():
            ra = dict(zgrp[f"/regions/{rk}"].attrs)
            rn = int(rk[1:])
            rconf[rn] = {
                "width":ra["geo_ref_out"]["width"],
                "height":ra["geo_ref_out"]["height"],
                "lat_bounds":ra["lat_bounds_out"],
                "lon_bounds":ra["lon_bounds_out"],
                }
        zgrp.attrs.update({
            "gefs":{
                **config.cfg_gefs,
                "regions":rconf,
                "nvtimes":config.cfg_gefs_backend["get_lead_times"],
                }
            })

    if load_cmaps:
        cmarr,cms = get_cmaps(
            cmap_list=config.cfg_cmap["options"],
            cmap_resolution=config.cfg_cmap["resolution"],
            use_cmasher=True,
            )
        if "cmaps" in zgrp.keys():
            del zgrp["cmaps"]
        cmarr = np.concatenate(cmarr, axis=0)
        zgrp.create_array("cmaps", shape=cmarr.shape, dtype=np.uint8)
        zgrp["cmaps"][...] = cmarr
        zgrp.attrs.update({"cmaps":{**config.cfg_cmap, "slices":cms}})
        print("got color maps")

    if load_vectors:
        vecs = {}
        gjp = config.cfg_gefs_backend["geojson_precision"]
        '''
        doms = gpd.read_file(domains_geojson)
        doms["geometry"] = doms["geometry"].set_precision(
            grid_size=1*10**(-1*gjp)
            )
        print("got domains")
        vecs["domains"] = doms.to_geo_dict()
        '''
        for vl in config.cfg_gefs["labels"]["vgroups"]:
            vecs[vl] = {}
            for rn in config.cfg_gefs_backend["get_regions"]:
                print(f"getting r{rn} {vl}")
                keep_cols = config.cfg_gefs_backend["keep_vec_properties"][vl]
                keep_cols.append("geometry")
                gj_path = vec_dir.joinpath(f"usfs_{vl}_r{rn}.geojson")
                tmpgj = gpd.read_file(gj_path)
                tmpgj["geometry"] = tmpgj["geometry"].set_precision(
                    grid_size=1*10**(-1*gjp)
                    )
                drop_cols = [
                    c for c in tmpgj.columns
                    if c not in keep_cols
                    ]
                tmpgj = tmpgj.drop(columns=drop_cols)
                vecs[vl][rn] = tmpgj.to_geo_dict()
        zgrp.attrs.update({"vectors":vecs})
