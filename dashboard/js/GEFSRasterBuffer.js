import {dates_between} from "./utils.js";

export class GEFSRasterBuffer {
    constructor({
        url_formatter, // function mapping a request object to a url
        max_num_arrays, // maximum number of arrays to retain at a time
        region_dimensions, // array of objects with width, height, ntimes
        norm_bounds, // object mapping features to metrics to [min, max] bounds
        data_resolution, // integer data resolution of data arrays
        mask_value, // integer mask value for data arrays
    }) {
        this.url_formatter = url_formatter;
        this.max_num_arrays = max_num_arrays;
        this.region_dimensions = region_dimensions;
        this.norm = norm_bounds;
        this.dres = data_resolution;
        this.mval = mask_value;
        this.active_array = null;
        this.arrays = new Map();
        //this.buffered = [];
        this.mask = []; // objects with properties for {feat, metric, min, max}
        //this.promises = new Map();

        this.worker = new Worker("./js/array_worker.js", {type:"module"});
        this.worker.addEventListener("error", e => { console.error(e); });

        // worker message id
        this.next_id = 1;
        // maps message ids to associated promise callback functions
        this.pending = new Map();
        this.worker.onmessage = ({data}) => {
            const { id, ok, result, error } = data;
            const req = this.pending.get(id);
            if (!req) return;
            this.pending.delete(id);
            // if the operation was an array load, set the state accordingly
            if (req.type === "load-array") {
                const entry = this.arrays.get(this._akey(req.info.config));
                entry.state = "finished";
            }
            if (ok) {
                req.resolve(result);
            } else {
                req.reject(new Error(error));
            }
        }
    }

    async get_rgb({
        itime,
        region,
        feat,
        metric,
        time_index,
        cmap,
        threshold_bounds,
        cmap_bounds,
    }) {
        const id = this.next_id++;
        const p_rgb = new Promise((resolve, reject) => {
            this.pending.set(id, {
                resolve:resolve,
                reject:reject,
                type:"get-rgb",
                info:null,
            });

            try {
                this.worker.postMessage({
                    type:"get-rgb",
                    id:id,
                    args:{
                        key:this._akey({itime, region, feat, metric}),
                        time_index:time_index,
                        cmap:cmap,
                        resolution:this.dres,
                        mask_val:this.mval,
                        norm:{
                            min:this.norm[feat][metric][0],
                            max:this.norm[feat][metric][1],
                        },
                        cmap_bounds:cmap_bounds,
                        threshold_bounds:threshold_bounds,
                    },
                });
            } catch (error) {
                this.pending.delete(id);
                reject(error);
            }

        });
    }

    // get a mask for the active itime and region. CRITICALLY, this method
    // assumes that all arrays for the requested masks are present in the
    // buffer, but does not check for the sake of efficiency. It's important
    // to await the mask promise after update_array before calling this.
    async get_mask({
        masks, // objects with feat, metric, min, and max properties
        time_index, // time index within current buffer
    }) {
        const id = this.next_id++;
        const p_mask = new Promise((resolve, reject) => {
            this.pending.set(id, {
                resolve:resolve,
                reject:reject,
                type:"get-mask",
                info:null,
            });

            const mask_args = [];
            const norm = [];
            for (const m of masks) {
                norm.push({
                    min:this.norm[m.feat][m.metric][0],
                    max:this.norm[m.feat][m.metric][1],
                });
                mask_args.push({
                    key:this._akey({
                        region:this.active_array.region,
                        itime:this.active_array.itime,
                        feat:m.feat,
                        metric:m.metric,
                    }),
                    min:m.min,
                    max:m.max,
                });
            }

            try {
                this.worker.postMessage(
                    {
                        type: "get-mask",
                        id:id,
                        args: {
                            mask:mask_args,
                            time_index:time_index,
                            resolution:this.dres,
                            norm:norm,
                            mask_val:this.mval,
                        },
                    },
                );
            } catch (error) {
                this.pending.delete(id);
                reject(error);
            }
        });
    }

    // request and buffer a new array configuration and return promises.
    // If the requested configuration is already buffered, return the promises.
    update_array({
        array_request, // object with region, feat, metric, and itime
    }) {
        const ar = array_request;

        // determine arrays that are needed for global mask given new config
        let new_itime = false;
        let new_region = false;
        if (this.active_array !== null) {
            new_itime = this.active_array.itime !== ar.itime;
            new_region = this.active_array.region !== ar.region;
        }
        let cur_req_in_mask = false;
        const get_mask_arrays = [];
        if (new_itime || new_region) {
            for (const mo of this.mask) {
                // the requested array will be retrieved no matter what, so
                // don't get it twice, but instruct the mask promise to wait
                // until it is returned.
                if (mo.feat===ar.feat && mo.metric===ar.metric) {
                    cur_req_in_mask = true;
                    continue;
                }

                get_mask_arrays.push({
                    feat:mo.feat,
                    metric:mo.metric,
                    region:ar.region,
                    itime:ar.itime,
                });
            }
        }

        // set the new active array to this configuration. This should run
        // before _add_array so that buffer cleanup has an up-to-date active
        // array for reference.
        this.active_array = ar;
        const active_promise = this._add_array(ar);

        const mask_promises = [];
        for (const nm of get_mask_arrays) {
            mask_promises.push(this._add_array(nm));
        }

        return {
            array:active_promise,
            masks:Promise.all(mask_promises),
        };
    }

    // arg: list of objects with feat, metric, max, and min for current mask
    update_mask(mask_config) {
        if (this.active_array === null){
            console.error("masks can't update until there is an active array");
        }
        const new_masks = [];
        for (const m of mask_config) {
            const mac = {
                feat:m.feat,
                metric:m.metric,
                itime:this.active_array.itime,
                region:this.active_array.region,
            };
            //const mix = this._get_buffer_index(mac);
            //if (mix == -1) {
            //    console.error("mask array not loaded:", mac);
            //}
            if (!this.arrays.has(this._akey(mac))){
                console.error("mask array not loaded:", mac);
            }
            new_masks.push({feat:m.feat, metric:m.metric});
        }
        this.mask = new_masks;
    }

    /*
    // returns index if array in buffer, or -1 if it isn't
    _get_buffer_index(array_config) {
        // if requested array is buffered, return its promises
        const ent = Object.entries(array_config);
        for (const ix in this.buffered) {
            if (ent.every(([k,v]) => { return this.buffered[ix][k] === v })) {
                return ix;
            }
        }
        return -1;
    }
    */

    // get the key in the array map given an array configuration
    _akey(array_config){
        const ac = array_config;
        return `${ac.region}:${ac.itime}:${ac.feat}:${ac.metric}`;
    }

    // move the array config to the end of the buffer
    _promote_in_buffer(array_config) {
        const k = this._akey(array_config);
        const v = this.arrays.get(k);
        this.arrays.delete(k);
        this.arrays.set(k, v);
        //const [aconf] = this.buffered.splice(buffer_index, 1);
        //this.buffered.push(aconf);
    }

    /*
    async _add_array(array_config) {
        const {region, feat, metric, itime} = array_config;
        const cur_ix = this._get_buffer_index(array_config);
        if (cur_ix !== -1) {
            this._promote_in_buffer(cur_ix);
            return this.promises.get(this._akey(array_config));
        }
        // clean up the buffer if this request will overfill it.
        if (this.buffered.length + 1 > this.max_num_arrays) {
            this._clean_buffer(1);
        }

        // make sure there is space in the promise tree for this array
        this._create_promise_entry(array_config);

        // fetch the data as an arrayBuffer object
        const p_request = await fetch(this.url_formatter(array_config))
            .then(async v => {
                if (!v.ok) {
                    throw new Error(`server-side error: ${v.status}`);
                }
                return await v.arrayBuffer();
            });

        // transfer ownership to the worker and copy to WASM memory
        const id = this.next_id++;
        const p_transfer = new Promise((resolve, reject) => {
            this.pending.set(id, {resolve, reject});
        });
        const adims = this.region_dimensions[this.active_array.region];
        const message = {
            type:"load-array",
            id:id,
            args:{
                key:array_config,
                buffer:p_request,
                width:adims.width,
                height:adims.height,
                ntimes:adims.ntimes,
            },
        };
        this.worker.postMessage(message, [p_request]);

        // store return the promise that resolves when the array is in WASM
        this.promises.set(this._akey(array_config), p_transfer);
        this.buffered.push(array_config);
        return p_transfer;
    }
    */

    // given an array configuration, ensure its promises are buffered and
    // promote it to the most recent selection, making requests if necessary.
    _add_array(array_config) {
        const { region, feat, metric, itime } = array_config;

        // promote and return the promises if they already exist
        //const cur_ix = this._get_buffer_index(array_config);
        //if (cur_ix !== -1) {
        //    this._promote_in_buffer(cur_ix);
        //    return this.promises.get(this._akey(array_config));
        //}
        const ak = this._akey(array_config);
        const cur_array = this.arrays.get(ak);
        if (cur_array) {
            this._promote_in_buffer(array_config);
            return cur_array.promise;
        }

        // clean up the buffer if this request will overfill it
        if (this.arrays.size + 1 > this.max_num_arrays) {
            this._clean_buffer(1);
        }

        // make sure there's a place in the promise tree for the new array
        //this._create_promise_entry(array_config);

        const id = this.next_id++;
        const adims = this.region_dimensions[this.active_array.region];

        const p_transfer = this._fetch_and_transfer_array(
            array_config,
            id,
            adims
        );

        // synchronously buffer and store the new array's config and promises
        //this.promises.set(this._akey(array_config), p_transfer);
        //this.buffered.push(array_config);
        this.arrays.set(ak, {
            config:array_config,
            promise:p_transfer,
            state:"loading",
        });

        // if the request fails, remove the array so it can be re-requested.
        p_transfer.catch(() => { this._remove_array(array_config) });

        return p_transfer;
    }

    // return a promise that resolves when the array is downloaded and moved
    async _fetch_and_transfer_array(array_config, id, adims) {
        //const ac = new AbortController();
        const response = await fetch(
            this.url_formatter(array_config),
            //signal:ac.signal,
        );

        if (!response.ok) {
            throw new Error(`server-side error: ${response.status}`);
        }

        const buffer = await response.arrayBuffer();

        const p_transfer = new Promise((resolve, reject) => {
            this.pending.set(id, {
                resolve:resolve,
                reject:reject,
                type:"load-array",
                info:{config:array_config},
            });

            try {
                this.worker.postMessage(
                    {
                        type: "load-array",
                        id,
                        args: {
                            key: this._akey(array_config),
                            buffer,
                            width: adims.width,
                            height: adims.height,
                            ntimes: adims.ntimes,
                        },
                    },
                    [buffer]
                );
            } catch (error) {
                this.pending.delete(id);
                reject(error);
            }
        });

        await p_transfer;
    }

    // reduce the size of the array buffer to fit new requests
    _clean_buffer(reduction_size) {
        const priority = {
            // not in current itime, region, or mask
            low:[],
            // in one of itime or region but not in mask
            medium:[],
            // in itime and region but not in mask
            // OR in mask but not in itime or region
            high:[],
        }
        let same_feat = false;
        let same_metric = false;
        let same_itime = false;
        let same_region = false;
        let in_mask = false;
        let is_active = false;
        let arrays_removed = 0;
        // iterate over a copy of the buffer so it can be mutated
        for (const b of [...this.arrays.values()]) {
            // skip any array requests that are still in progress.
            if (b.state === "loading") {
                continue;
            }
            const c = b.config;
            same_itime = c.itime == this.active_array.itime;
            same_region = c.region == this.active_array.region;
            same_feat = c.feat == this.active_array.feat;
            same_metric = c.metric == this.active_array.metric;
            // always skip the active array
            if (same_itime && same_region && same_feat && same_metric){
                continue;
            }
            const mkeys = new Set(this.mask.map(m => `${m.feat}:${m.metric}`));
            in_mask = mkeys.has(`${c.feat}:${c.metric}`);

            if (!in_mask && !same_itime && !same_region) {
                // immediately remove very low-priority buffer items
                if (!same_feat && !same_metric) {
                    this._remove_array(c);
                    arrays_removed++;
                    if (arrays_removed === reduction_size) return;
                    continue;
                }
                priority.low.push(c);
                continue;
            }
            if (!in_mask && (same_itime || same_region)) {
                priority.medium.push(c);
                continue;
            }
            if (
                ((same_itime && same_region) && !in_mask)
                || (!(same_itime && same_region) && in_mask)
            ) {
                priority.high.push(c);
                continue;
            }
        }
        console.log(priority);
        for (const c of priority.low) {
            this._remove_array(c);
            arrays_removed++;
            if (arrays_removed === reduction_size) return;
        }
        for (const c of priority.medium) {
            this._remove_array(c);
            arrays_removed++;
            if (arrays_removed === reduction_size) return;
        }
        for (const c of priority.high) {
            this._remove_array(c);
            arrays_removed++;
            if (arrays_removed === reduction_size) return;
        }
    }

    _remove_array(array_config) {
        console.log("removing ", array_config);
        this.arrays.delete(this._akey(array_config));
        const p_delete = new Promise((resolve, reject) => {
            this.pending.set(id, {
                resolve:resolve,
                reject:reject,
                type:"delete-array",
                info:{},
            });

            try {
                this.worker.postMessage(
                    {
                        type: "delete-array",
                        id,
                        args: { key: this._akey(array_config) },
                    },
                );
            } catch (error) {
                this.pending.delete(id);
                reject(error);
            }
        });
        return p_delete;
    }
}
