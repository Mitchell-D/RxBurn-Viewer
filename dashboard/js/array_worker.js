import init, { RasterStore } from "/wasm/wasm_raster.js";

let wasm_ready = false;
let store = null;
async function ensure_wasm_init() {
    if (!wasm_ready) {
        await init();
        wasm_ready = true;
    }
    if (!store) store = new RasterStore();
}


self.onmessage = async (a) => {
    const {type, id, args} = a.data;
    //console.log(type, id, args);
    await ensure_wasm_init();
    ///*
    try {
        if (type === "load-array") {
            const {key, buffer, width, height, ntimes} = args;
            const x = new Uint16Array(buffer);
            console.log("loading", key);
            store.add(key, x, ntimes, height, width);
            self.postMessage({
                id:id,
                ok:true,
                result:null,
                error:null,
            });
        } else if (type === "delete-array") {
            console.log("deleting", args.key);
            store.del(args.key);
            self.postMessage({
                id:id,
                ok:true,
                result:null,
                error:null,
            });
        } else if (type === "get-mask") {
            const {mask, time_index, resolution, norm, mask_val} = args;
            const keys = [];
            const mval = [];
            const res = [];
            const norm_min = [];
            const norm_max = [];
            const thresh_min = [];
            const thresh_max = [];
            for (const i in mask) {
                keys.push(mask[i].key);
                mval.push(mask_val);
                res.push(resolution);
                norm_min.push(norm[i].min);
                norm_max.push(norm[i].max);
                thresh_min.push(mask[i].min);
                thresh_max.push(mask[i].max);
            }
            /*
            const mask = store.generate_threshold_mask(
                keys,
                time_index,
                Uint16Array.from(mval),
                Uint32Array.from(res),
                Float64Array.from(norm_min),
                Float64Array.from(norm_max),
                Float64Array.from(thresh_min),
                Float64Array.from(thresh_max),
            );
            self.postMessage({
                id:id,
                ok:true,
                result:mask,
                error:null,
            });
            */
        } else if (type === "get-rgb") {
            const {
                key, time_index, cmap, resolution, mask_val,
                norm, cmap_bounds, thresh_bounds
            } = args;
            console.log("getting rgb of ", key, time_index);
            const rgb = store.generate_rgb(
                key,
                time_index,
                cmap,
                mask_val,
                resolution,
                norm.min,
                norm.max,
                cmap_bounds.min,
                cmap_bounds.max,
                thresh_bounds.min,
                thresh_bounds.max,
            );
            self.postMessage({
                id:id,
                ok:true,
                result:rgb,
                error:null,
            });
        } else {
            console.error("unrecognized type:", type);
        }
    } catch (error) {
        self.postMessage({id, ok:false, error:error.toString()});
    }
    //*/
}
