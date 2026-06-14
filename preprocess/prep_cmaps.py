""" Load color maps and attribute data into the zarr """
import zarr
import numpy as np
from pathlib import Path

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
    from matplotlib.pyplot import get_cmap
    if use_cmasher:
        import cmasher as cmr
    cmap_arrays = []
    cmap_slices = []
    prv_ix = 0
    for cml in cmap_list:
        ## retrieve the color map and append a nan value last for transparent
        cm = get_cmap(cml)
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
    out_zarr_dir = Path("data/store/")
    out_zarr_path = out_zarr_dir.joinpath("rxburn.zarr")

    update_cmaps = True
    update_labels = True

    zstor = zarr.storage.LocalStore(out_zarr_path.as_posix())

    """ --( load color maps based on config.cmap_config )-- """

    if update_cmaps:
        zgrp = zarr.open(zstor, path="/", mode="a")
        #zgrp_cmap = zarr.open(zstor, path="/cmap", mode="a")
        ## destroy any existing cmap array and overwrite it
        if "cmap" in zgrp.keys():
            del zgrp["cmap"]

        ## generate new color map lookup tables and add them
        cm_arrs,cm_slices = get_cmaps(
            cmap_list=config.cfg_cmap["options"],
            cmap_resolution=config.cfg_cmap["resolution"],
            use_cmasher=True,
            )
        cm_arrs = np.concatenate(cm_arrs, axis=0)
        zgrp.create_array("cmap", shape=cm_arrs.shape, dtype=np.uint8)
        zgrp["cmap"][...] = cm_arrs

        ## load the color map configuration and LUT slices
        prv_attrs = dict(zgrp["cmap"].attrs)
        prv_attrs.update({**config.cfg_cmap, "slices":cm_slices})
        zgrp["cmap"].attrs.put(prv_attrs)

print("finished")
