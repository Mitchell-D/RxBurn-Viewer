"""
Defines fastapi endpoints for RxBurn Dashboard

This script is run by the ASGI server (uvicorn) once on startup, then the
decorated functions implementing the endpionts are invoked asynchronously
whenever a request is issued.
"""
import json
import zarr
import os
import numpy as np
from time import perf_counter
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pathlib import Path
import redis.asyncio as redis
import asyncio

## amount of time to keep a cache key after a hit
CACHE_TTL = 600 ## 10 minutes
## amount of time a lock can be held
LOCK_TTL = 10 ## 10 seconds
## how long each waiting process waits between cache pings
LOCK_WAIT = .01 ## seconds
DEBUG = True

""" ---( data sourcing )--- """

## zarr store reference
zarr.config.set({"async.concurrency": 64})
zgrp = zarr.open("rxburn.zarr", mode="r")
zgrp_ifs = zgrp["/ens/ifs"]

## explicitly collect metadata relevant to IFS ensemble data.
meta_ifs = {
    ## yyyymmddhh string init times and yyyymmddhh:yyyymmddhhmm horizon times
    #"init_times":list(sorted(zgrp_ifs["temporal"].keys()))
    "init_times":zgrp_ifs.attrs["init_times"],
    "horizon_times":zgrp_ifs.attrs["horizon_times"],

    ## data variables (ie temp, humidity, wind)
    "feats":zgrp_ifs.attrs["feats"],
    ## ensemble aggregation metrics
    "metrics":zgrp_ifs.attrs["metrics"],

    ## geometry (hard coding in javascript for now)
    #"bounds_lat":zgrp_ifs.attrs["latitude_bounds"],
    #"bounds_lon":zgrp_ifs.attrs["longitude_bounds"],
    "grid_shape":zgrp_ifs.attrs["shape_spatial"][1:3],
    #"geo_ref":{
    #    "crs_wkt":zgrp_ifs.attrs["geo_ref"]["crs_wkt"],
    #    "geo_transform":zgrp_ifs.attrs["geo_ref"]["geo_transform"],
    #    ## ignoring semi major and minor axes
    #    },

    ## data normalization
    "norm_bounds":zgrp_ifs.attrs["norm_bounds"],
    "norm_res":zgrp_ifs.attrs["norm_res"],
    "mask_val":zgrp_ifs.attrs["mask_val"],

    ## labels
    "long_labels_metrics":zgrp_ifs.attrs["long_labels_metrics"],
    "long_labels_units":zgrp_ifs.attrs["long_labels_units"],
    "short_labels_units":zgrp_ifs.attrs["short_labels_units"],
    }

## color map metadata and concatenated color map array
cmap_info = {
    **zgrp["cmap"].attrs,
    "cmaps":zgrp["cmap"][...].tolist(),
    }

""" ---( cache methods )--- """

async def raster_cache_get(request:Request, background:BackgroundTasks,
        rcache:redis.Redis, ckey:tuple, cache_other:list):
    """
    retrieve raster data from the redis cache. includes logic for distributed
    mutex so that multiple workers don't try to read the same data from disc
    at the same time (which will be very common in frame-by-frame context)

    the lock works by manipulating a 1-byte value in the cache which has the
    "NX" flag (set if doesn't exist)

    :@param request: redis Request object for this worker
    :@param background: redis BackgroundTasks manager for bulk caching
    :@param rcache: redis cache connection object
    :@param ckey: 5-tuple (dataset, itime, feat, metric, frame)
        for the current request
    :@param cache_other: list of 3-tuples (dataset, itime, feat, metric)
        for other combinations to load in a background task.
    """
    dataset,itime,feat,metric,frame = ckey
    ## key for the group name
    gkey = f"{dataset}_{itime}_{feat}"
    ## key for the current frame
    fkey = f"{metric}_{frame}"
    ## if possible, immediately get from the cache
    cached = await rcache.hget(gkey, fkey)
    if not cached is None:
        return cached

    ## if the cache missed, determine the lock that must be acquired.
    lkey = f"{dataset}_{itime}_{feat}_{metric}"
    cache_being_filled = False
    while True:
        ## returns True only for the one worker that wins the race
        acquired = await rcache.hsetex(
                name="lock",
                key=lkey,
                value="1",
                data_persist_option=HDPO.FNX,
                ex=LOCK_TTL
                )
        if acquired:
            try:
                ## make sure cache hasn't been populated since requesting lock
                cached = await rcache.hget(gkey, fkey)
                if cached is not None:
                    ## detect if the request was aborted while waiting on
                    ## the lock to resolve
                    if await request.is_disconnected():
                        return None
                    return cached

                if DEBUG:
                    print(f"{os.getpid()} setting cache")

                ## lock acquired... see which other hashes we can populate.
                ## currently, assume if the first frame is present for a
                ## (dataset, itime, feat, metric) combo, all the frames are
                ## already in the cache.
                other_locks = await asyncio.gather(*[
                    (
                        co,
                        rcache.hsetex(
                            name="lock",
                            key="_".join(co),
                            value="1",
                            data_persist_option=HDPO.FNX,
                            ex=LOCK_TTL
                            ),
                        )
                    for co in cache_other
                    if not rcache.hexists(
                        "_".join(co[:-1]),
                        f"{co[-1]}_0"
                        )
                    ])
                ## only keep hashes for other features if semaphore acquired
                other_locks = [co for co,acq in other_locks if acq]

                ## go ahead and set the current cache value first
                fix = meta_ifs["feats"].index(feat)
                mix = meta_ifs["metrics"].index(metric)
                nht = len(meta_ifs["horizon_times"][itime])
                ## stored as: (feat, metric, horizon, lat, lon)
                X = zgrp_ifs["spatial"][itime][fix,mix]
                frames = {
                    f"{metric}_{hix}":X[hix].tobytes()
                    for hix in range(nht)
                    }
                cur_frame = frames.pop(fkey)
                await rcache.hsetex(
                    gkey, fkey, cur_frame,
                    data_persist_option=HDPO.FNX,
                    ex=LOCK_TTL,
                    )

                ## dispatch a background task for loading the requested data
                background.add_task(
                    populate_locked_range,
                    rcache=rcache,
                    lkey=lkey,
                    group=gkey,
                    mapping=frames,
                    )

                ## also dispatch background tasks for populating other data
                for co in other_locks:
                    olk = "_".join(co)
                    ogk = "_".join(co[:-1]) ## group keys don't have metric
                    fix = meta_ifs["feats"].index(c0[2])
                    mix = meta_ifs["metrics"].index(c0[3])
                    X = zgrp_ifs["spatial"][co[1]][fix,mix]
                    background.add_task(
                        populate_locked_range,
                        rcache=rcache,
                        lkey=olk,
                        group=ogk,
                        mapping={
                            f"{co[-1]}_{hix}":X[hix].tobytes()
                            for hix in range(nht)
                            },
                        )

                ## indicate that the cache is being filled, so the background
                ## task will handle releasing the semaphores
                cache_being_filled = True

                ## check whether the request disconnected while handling above
                if await request.is_disconnected():
                    return None
                ## otherwise return the currently-requested frame
                return cur_frame

            finally:
                ## If the lock has been acquired and the try block exits,
                ## either the value has been populated by a different worker
                ## and loaded from the cache, the request disconnected, or
                ## this worker kicked off a background process to load other
                ## frames. In all but the latter case, the lock needs to
                ## be immediately released.
                ## cache_other background locks don't need to be released here
                ## since they are only acquired in the latter case.
                if not cache_being_filled:
                    await rcache.hdel("lock", lkey)
        else:
            ## another worker has the lock...
            ## see if the request has been cancelled
            if await request.is_disconnected():
                return None
            ## otherwise see if the worker with the lock has added the
            ## requested value to the cache
            cached = await rcache.hget(gkey, fkey)
            if cached is not None:
                return cached
            ## if neither of the above, wait before re-checking
            await asyncio.sleep(LOCK_WAIT)

""" ---( app initialization )--- """

## declare app and add middleware for logging requests
app = FastAPI(title="RxBurn Database")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"], ## all HTTP methods (GET, POST, PUT, etc.)
    allow_headers=["*"], ## all headers
    )



""" ---( app endpoints )--- """

@app.get("/raster/ens/ifs/{feat}/{metric}")
async def raster_ens_ifs(request:Request, background:BackgroundTasks,
        feat:str, metric:str, itime:str|None=None, frame:str|None=None,
        ):
    """
    return a byte stream for the requested raster frame in the provided times

    If init time and frame aren't provided, default to the latest init time
    and the first horizon time.

    The same defaults assumption needs to be mirrored on the javascript side
    when it receives the metadata
    """
    if DEBUG:
        dbt0 = perf_counter()
    ## validate the inputs
    if not feat in meta_ifs["feats"]:
        raise HTTPException(status_code=400, detail=f"Invalid feat:{feat}")
    if not metric in meta_ifs["metrics"]:
        raise HTTPException(status_code=400, detail=f"Invalid metric:{metric}")
    if not itime is None or itime in meta_ifs["init_times"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid init time:{itime}"
            )
    if not frame is None or frame.isdigit():
        raise HTTPException(status_code=400, detail=f"Frame must be an int")
    if not frame is None and not 0 <= frame < len(horizon_times[itime]):
        raise HTTPException(status_code=400, detail=f"Invalid frame {frame}")

    if itime is None:
        itime = meta_ifs["init_times"][-1]
    if frame is None:
        frame = "0"

    fix = int(frame)

    ## retrieve the cache reference from the app state namespace
    rc = app.state.redis

    ## get the data from the cache
    ckey = ("ifs", itime, feat, metric, fix)
    if DEBUG:
        print(f"retrieving {ckey}")
    cached = await raster_cache_get(
            request=request,
            background=background,
            rcache=rc,
            ckey=ckey,
            )
    cshape = meta_ifs["grid_shape"]
    carr = np.frombuffer(cached, dtype=np.float32).reshape(cshape)
    nbytes = str(carr.nbytes)

    ## return as a byte stream
    r = Response(
        content=carr.tobytes(),
        media_type="application/octet-stream",
        headers={
            "Content-Type":"application/octet-stream",
            "X-Array-Shape":",".join(map(str, cshape)),
            "Content-Length":nbytes,
            }
        )
    if DEBUG:
        print(f"{os.getpid()} processed {nbytes} in {perf_counter()-dbt0:.3f}")

    return r

@app.get("/poly/{pgroup}")
def poly(pgroup:str):
    """ endpoint for map polygon geojsons """
    if not pgroup in zgrp.attrs["polygons"].keys():
        raise HTTPException(status_code=400, detail=f"Invalid pgroup:{pgroup}")
    return zgrp.attrs["polygons"][pgroup]

@app.get("/menu/ens/ifs")
def menu_ens_ifs():
    """ endpoint for menu information (labels, time range, etc) """
    return meta_ifs

@app.get("/cmaps")
def cmaps():
    """ endpoint for concatenated color maps array and its metadata """
    return cmap_info
