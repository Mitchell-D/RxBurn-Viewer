import {dates_between} from "./utils.js";

export class GEFSRasterBuffer {
    constructor({
        url_formatter, // function mapping a request object to a url
        max_num_arrays, // maximum number of arrays to retain at a time
        region_dimensions, // array of objects with width, height, ntimes
    }) {
        this.url_formatter = url_formatter;
        this.max_num_arrays = max_num_arrays;
        this.region_dimensions = region_dimensions;
        this.active_array = null;
        this.buffered = [];
        this.mask = []; // objects with properties for {feat, metric, min, max}
        this.promises = {};

        this.worker = new Worker("./js/array_worker.js", {type:"module"});
        // worker message id
        this.next_id = 1;
        // maps message ids to associated promise callback functions
        this.pending = new Map();
        this.worker.onmessage = ({data}) => {
            const { id, ok, result, error } = data;
            const req = this.pending.get(id);
            if (!req) return;
            this.pending.delete(id);
            if (ok) {
                req.resolve(result);
            } else {
                req.reject(new Error(error));
            }
        }
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
        if (!this.active_array === null) {
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

        // set the new active array to this configuration
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

    update_mask({
        // list of objects with feat, metric, max, and min for current mask
        mask_settings,
    }) {
    }

    _is_array_buffered(array_config) {
        // if requested array is buffered, return its promises
        const ent = Object.entries(array_config);
        return this.buffered.some(item => {
            ent.every(([k,v]) => item[k] === v);
        });
    }

    // ensure there is a place in the nested promises object for a new array
    _create_promise_entry({region, feat, metric, itime}) {
        // make an entry in the promises object for this request
        if (!Object.hasOwn(this.promises, region))
            this.promises[region] = {}
        if (!Object.hasOwn(this.promises[region], feat))
            this.promises[region][feat] = {}
        if (!Object.hasOwn(this.promises[region][feat], metric))
            this.promises[region][feat][metric] = {}
        if (!Object.hasOwn(this.promises[region][feat][metric], itime))
            this.promises[region][feat][metric][itime] = null;
    }

    async _add_array(array_config) {
        const {region, feat, metric, itime} = array_config;
        if (this._is_array_buffered(array_config)) {
            return this.promises[region][feat][metric][itime];
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
        this.promises[region][feat][metric][itime] = p_transfer;
        return p_transfer;
    }

    // reduce the size of the array buffer to fit new requests
    _clean_buffer(reduction_size) {
        console.log("cleaning buffer");
    }

    // add a callback to a feed. Any time the feed is updated, subscriber
    // callbacks will be called with the new selection as an argument.
    subscribe(callback)  {
        if (!typeof callback == "function") {
            throw new Error("Must provide a callback function not "+callback);
        }
        this.subscriptions.push(callback);
    }

    // send the entire new selection to subscribers and let them decide which
    // properties are relevant to what they are doing.
    notify_subscribers(date_strings, promises) {
        this.subscriptions.forEach((f) => f({
            date_strings:date_strings,
            promises:promises,
        }));
    }
}
