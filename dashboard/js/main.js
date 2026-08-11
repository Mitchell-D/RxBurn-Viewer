import { Map } from "./map.js";
//import { MenuDate } from "./menu_date.js";
//import { MenuFeat } from "./menu_feat.js";
import { ColorBar } from "./ColorBar.js";
import { Menu } from "./Menu.js";
import { DualRangeSlider } from "./DualRangeSlider.js";
import { KeyedTable } from "./KeyedTable.js";
import {GEFSRasterBuffer} from "./GEFSRasterBuffer.js";
//import { MenuPoly } from "./menu_pgroup.js";
//import { MenuRaster } from "./menu_raster.js";
//import { ColorMap } from "./color_map.js";
//import {datestring_to_date,date_to_datestring,format_date} from "./utils.js";
//import { default as init_wasm } from "./wasm/wasm_cmap.js";

// more emphatic variable names for modules
//const RASTER = Raster;
//const COLOR = ColorMap
//const FMENU = MenuFeat;
//const DMENU = MenuDate;
//const PMENU = MenuPoly;
//const RMENU = MenuRaster;

const state = {
    dom:{
        main_map_container:"main_map_container",
        region_menu_container:"dd_region_name",
        region_menu_button:"dd_button_region_name",
        itime_menu_container:"menu_container_itime",
        feat_menu:"menu_container_feat",
        metric_menu_value:"menu_container_metric_value",
        metric_menu_spread:"menu_container_metric_spread",
        cmap_dropdown:"dd_cmap_name",
        cmap_dropdown_button:"dd_button_cmap",
        cbar_container:"cbar_container",
        mask_table:"mask_threshold_table",
        mask_update_button:"button_update_mask",

        cmap_slider_container_id:"cmap_slider_row",
        threshold_slider_container_id:"threshold_slider_row",

        tpl_cbar:"vertical_cbar_template",
        tpl_menu_flex_button:"menu_flex_button_temp",
        tpl_menu_button:"menu_button_temp",
        tpl_menu_dropdown:"dropdown_temp",
        //pgroup_menu:"menu_container_pgroup",
        //date_picker:"buffer_date_range",
    },
    sel:{
        region:8,
        feat:"temp",
        metric:"mean",
        cmin:null, // minimum value bound for color map
        cmax:null, // minimum value bound for color map
        vmin:null, // minimum value bound for threshold
        vmax:null, // minimum value bound for threshold
        cmap:null,
        //pgroup:"fulldomain",
        //poly:"fulldomain_0",
        //t0:null,
        //tf:null,
    },
    urls:{
        raster:"/api/gefs/raster",
        menu:"/api/gefs/menu",
        cmaps:"/api/cmaps",
    },
    labels:{
        regions:null,
        feats:null,
        metrics:null,
        spread_metrics:null,
        itimes:null,
    },
    long_labels:{
        regions:null,
        feats:null,
        metrics:null,
        units:null,
    },
    short_labels:{
        regions:null,
        feats:null,
        metrics:null,
        units:null,
    },
    regions:null, // maps region numbers to dimensions and coord bounds
    nvtimes:null, // number of valid times per forecast run
    norm:{
        bounds:null,
        resolution:null,
        mask:null,
    },
    cmap:{
        arrays:null,
        default_bounds:null,
        options:null,
        resolution:null,
    },

    main_cbar:{
        orientation:"vertical",
        nticks:8,
        tick_size:5,
        tick_padding:2,
    },

    // degree bounds around selected domain within which to allow panning
    map_bounds_buffer:[1, 1],

    mask_table_header_labels:[
        "Feature",
        "Metric",
        "Minimum",
        "Maximum",
        "Delete",
    ],

    // maximum number of arrays to allow in the buffer at once
    max_num_arrays:5,
}

// make a promise for when the DOM is loaded
const dom_ready = new Promise(resolve => {
  if (document.readyState === "loading") {
    document.addEventListener('DOMContentLoaded', resolve);
  } else {
    resolve();
  }
});

let MAP = null; // main map
let MENU_REGION = null; // init time menu
let MENU_ITIME = null; // init time menu
let MENU_FEAT = null; // feature button menu
let MENU_METRIC = null; // metric button menu
let MENU_CSLIDER = null; // color map slider forms
let MENU_TSLIDER = null; // threshold slider forms
let MENU_CMAP = null; // color map name forms
let MAIN_CBAR = null;
let MENU_TTABLE = null; // threshold table
let RASTER_BUFFER = null;

// explicitly unpack metadata so there's no ambiguity
const meta_loaded = fetch(state.urls.menu)
    .then(r => r.json())
    .then(r => {
        state.labels.regions = r["labels"]["regions"];
        state.labels.feats = r["labels"]["feats"];
        state.labels.metrics = r["labels"]["metrics"];
        state.labels.spread_metrics = r["labels"]["spread_metrics"];
        state.labels.itimes = r["labels"]["itimes"];

        state.regions = r["regions"];

        state.nvtimes = r["nvtimes"];

        state.norm.bounds = r["norm_bounds"];
        state.norm.resolution = r["norm_res"];
        state.norm.mask = r["mask_val"];

        state.long_labels.regions = r["long_labels"]["regions"];
        state.long_labels.feats = r["long_labels"]["feats"];
        state.long_labels.metrics = r["long_labels"]["metrics"];
        state.long_labels.units = r["long_labels"]["units"];

        state.short_labels.regions = r["short_labels"]["regions"];
        state.short_labels.feats = r["short_labels"]["feats"];
        state.short_labels.metrics = r["short_labels"]["metrics"];
        state.short_labels.units = r["short_labels"]["units"];

        // go ahead and set the default itime so the first raster request can
        // issue after this promise resolves. Other fields are global defaults.
        const last_ix = state.labels.itimes[state.sel.region].length - 1
        state.sel.itime = state.labels.itimes[state.sel.region][last_ix];
    });

const cmaps_loaded = fetch(state.urls.cmaps)
    .then(r => r.json())
    .then(r => {
        //state.cmap.arrays = r["cmaps"];
        state.cmap.default_bounds = r["default_bounds"];
        state.cmap.default_name = r["default_name"];
        //state.cmap.slices = r["slices"];
        state.cmap.options = r["options"];
        state.cmap.resolution = r["resolution"];

        state.cmap.arrays = {};
        for (const i in r["slices"]) {
            const [ix0,ixf] = r["slices"][i];
            const ck = r["options"][i];
            state.cmap.arrays[ck] = new Uint8ClampedArray(
                r["cmaps"].slice(ix0,ixf));
        }
    });

// initialize the map
const map_started = Promise.all([dom_ready, meta_loaded])
    .then(() => {
        const mcon = document.getElementById(state.dom.main_map_container)
        MAP = new Map({map_container:mcon});
        MAP.set_region({
            bbox:[
                state.regions[state.sel.region]["lon_bounds"][0],
                state.regions[state.sel.region]["lat_bounds"][0],
                state.regions[state.sel.region]["lon_bounds"][1],
                state.regions[state.sel.region]["lat_bounds"][1],
            ],
            bounds_buffer:state.map_bounds_buffer,
            raster_width:state.regions[state.sel.region]["width"],
            raster_height:state.regions[state.sel.region]["height"],
        });
    });

// load the IFS menu and
const menu_forms_initialized = Promise.all([dom_ready, meta_loaded])
    .then(r => {
        // initialize region menu
        MENU_REGION = new Menu({
            container_id:state.dom.region_menu_container,
            button_template_id:state.dom.tpl_menu_dropdown,
            labels:state.labels.regions,
            defaults:state.sel.region,
            initial_conditions:[],
            long_labels:state.long_labels.regions,
            class_active:"btn-primary",
            class_inactive:"btn-secondary",
        });
        const mrbtn = document.getElementById(state.dom.region_menu_button);
        mrbtn.textContent = state.long_labels.regions[state.sel.region];

        const itdef = {};
        for (const r of state.labels.regions) {
            const last_ix = state.labels.itimes[r].length - 1
            itdef[r] = state.labels.itimes[r][last_ix];
        }
        MENU_ITIME = new Menu({
            container_id:state.dom.itime_menu_container,
            button_template_id:state.dom.tpl_menu_flex_button,
            labels:state.labels.itimes,
            defaults:itdef,
            initial_conditions:[state.sel.region],
            class_active:"btn-primary",
            class_inactive:"btn-secondary",
        })

        // initialize feature menu
        MENU_FEAT = new Menu({
            container_id:state.dom.feat_menu,
            button_template_id:state.dom.tpl_menu_button,
            labels:state.labels.feats,
            defaults:state.sel.feat,
            initial_conditions:[],
            long_labels:state.long_labels.feats,
            class_active:"btn-primary",
            class_inactive:"btn-secondary",
        });

        // initialize metric menu
        // for now, assume all feats have all metrics, though the menu
        // class is general enough to handle complex nesting
        const metric_menu_labels = {}
        for (const l of state.labels.feats) {
            metric_menu_labels[l] = state.labels.metrics;
        }
        // condition the container for buttons on whether or not they
        // are spread metrics
        const metric_container_ids = {}
        for (const l of state.labels.metrics) {
            metric_container_ids[l] = state.labels.spread_metrics.includes(l)
                ? state.dom.metric_menu_spread : state.dom.metric_menu_value;
        }
        MENU_METRIC = new Menu({
            container_id:metric_container_ids,
            button_template_id:state.dom.tpl_menu_button,
            labels:metric_menu_labels,
            defaults:state.sel.metric,
            initial_conditions:[state.sel.feat],
            long_labels:state.long_labels.metrics,
            class_active:"btn-primary",
            class_inactive:"btn-secondary",
        });

        MENU_REGION.subscribe((new_region) => {
            state.sel.region = new_region;
            const mrbtn = document.getElementById(
                state.dom.region_menu_button);
            mrbtn.textContent = state.long_labels.regions[state.sel.region];
            console.log("new region:", new_region);
        });

        // subscribe the metric menu to update based on the feat menu
        MENU_FEAT.subscribe((new_feat) => {
            // main state needs to be the first to update so that subscribers
            // to the metric menu can be provided an up-to-date feat state
            state.sel.feat = new_feat;
            console.log("new feat:", new_feat);
            MENU_METRIC.update([new_feat]);
        });
        // set subscriptions to menu (and by extension feat) changes
        MENU_METRIC.subscribe((new_metric) => {
            state.sel.metric = new_metric;
            console.log("new metric:", new_metric);
        });
        MENU_ITIME.subscribe((new_itime) => {
            state.sel.itime = new_itime;
            console.log("new itime", new_itime);
        });
    });

const sliders_initialized = Promise.all([
    dom_ready, menu_forms_initialized, cmaps_loaded])
    .then(() => {
        // initialize the color map slider menu
        MENU_CSLIDER = new DualRangeSlider({
            target_container_id:state.dom.cmap_slider_container_id,
            extrema:state.norm.bounds,
            defaults:state.cmap.default_bounds,
            initial_conditions:[state.sel.feat, state.sel.metric],
        });
        state.sel.cmin = MENU_CSLIDER.min_val_bnd;
        state.sel.cmax = MENU_CSLIDER.max_val_bnd;

        // initialize the threshold slider menu
        MENU_TSLIDER = new DualRangeSlider({
            target_container_id:state.dom.threshold_slider_container_id,
            extrema:state.norm.bounds,
            defaults:structuredClone(state.norm.bounds),
            initial_conditions:[state.sel.feat, state.sel.metric],
        });
        state.sel.tmin = MENU_CSLIDER.min_val_bnd;
        state.sel.tmax = MENU_CSLIDER.max_val_bnd;

        const cmap_options = {};
        for (const fk of state.labels.feats) {
            cmap_options[fk] = {};
            for (const mk of state.labels.metrics) {
                cmap_options[fk][mk] = state.cmap.options;

            }
        }
        MENU_CMAP = new Menu({
            container_id:state.dom.cmap_dropdown,
            button_template_id:state.dom.tpl_menu_dropdown,
            labels:cmap_options,
            defaults:state.cmap.default_name,
            initial_conditions:[state.sel.feat, state.sel.metric],
            long_labels:{},
            class_active:"btn-primary",
            class_inactive:"btn-secondary",
        });
        state.sel.cmap = MENU_CMAP.current_value;

        MENU_TTABLE = new KeyedTable({
            table_id:state.dom.mask_table,
            header_labels:state.mask_table_header_labels,
        });
        const tbtn = document.getElementById(state.dom.mask_update_button);
        tbtn.addEventListener("click", () => {
            MENU_TTABLE.update_row({
                key:[state.sel.feat, state.sel.metric],
                min:state.sel.tmin,
                max:state.sel.tmax,
            });
        });

        MAIN_CBAR = new ColorBar({
            container_id:state.dom.cbar_container,
            template_id:state.dom.tpl_cbar,
            orientation:state.main_cbar.orientation,
            nticks:state.main_cbar.nticks,
            tick_size:state.main_cbar.tick_size,
            tick_padding:state.main_cbar.tick_padding,
        });

        MENU_TSLIDER.subscribe((tmin, tmax) => {
            state.sel.tmin = tmin;
            state.sel.tmax = tmax;
        })

        // set subscriptions to menu (and by extension feat) changes
        MENU_METRIC.subscribe((new_metric) => {
            // new metric runs any time a new feature is selected too since
            // it is conditioned on the feat menu.
            MENU_CSLIDER.set_new_conditions([state.sel.feat,state.sel.metric]);
            MENU_TSLIDER.set_new_conditions([state.sel.feat,state.sel.metric]);
        });

        // set subscriptions to color map bounds changes
        MENU_CSLIDER.subscribe((cmin,cmax) => {
            state.sel.cmin = cmin;
            state.sel.cmax = cmax;
            MAIN_CBAR.draw({
                cbar:state.cmap.arrays[state.sel.cmap].slice(0,-4),
                vmin:state.sel.cmin,
                vmax:state.sel.cmax,
                nticks:state.main_cbar,
                new_image:false,
            });
        });

        // set subscriptions to threshold bounds changes
        MENU_TSLIDER.subscribe((vmin,vmax) => {
            state.sel.vmin = vmin;
            state.sel.vmax = vmax;
        });

        MENU_METRIC.subscribe((new_metric) => {
            MENU_CMAP.update([state.sel.feat, state.sel.metric]);
        });
        const cmap_btn = document.getElementById(
            state.dom.cmap_dropdown_button);
        cmap_btn.textContent = MENU_CMAP.current_value;
        MENU_CMAP.subscribe((new_cmap) => {
            state.sel.cmap = new_cmap;
            cmap_btn.textContent = new_cmap;
            MAIN_CBAR.draw({
                cbar:state.cmap.arrays[state.sel.cmap].slice(0,-4),
                vmin:state.sel.cmin,
                vmax:state.sel.cmax,
                nticks:state.main_cbar.nticks,
                new_image:true,
            });
        });

        MAIN_CBAR.draw({
            cbar:state.cmap.arrays[state.sel.cmap].slice(0,-4),
            vmin:state.sel.cmin,
            vmax:state.sel.cmax,
            nticks:state.main_cbar.nticks,
            new_image:true,
        });

    });

const map_regions_bound = Promise.all([map_started, menu_forms_initialized])
    .then(() => {
        MENU_REGION.subscribe((new_region) => {
            MAP.set_region({
                bbox:[
                    state.regions[state.sel.region]["lon_bounds"][0],
                    state.regions[state.sel.region]["lat_bounds"][0],
                    state.regions[state.sel.region]["lon_bounds"][1],
                    state.regions[state.sel.region]["lat_bounds"][1],
                ],
                bounds_buffer:state.map_bounds_buffer,
                raster_width:state.regions[state.sel.region]["width"],
                raster_height:state.regions[state.sel.region]["height"],
            });
            MENU_ITIME.update([state.sel.region]);
        });
    });

function update_active_array() {
    RASTER_BUFFER.update_array({
        array_request:{
            region:state.sel.region,
            feat:state.sel.feat,
            metric:state.sel.metric,
            itime:state.sel.itime,
        },
        width:state.regions[state.sel.region].width,
        height:state.regions[state.sel.region].height,
        ntimes:state.nvtimes,
    });
}
const buffer_initialized = meta_loaded.then(()=> {
    const rdims = {};
    for (const r in state.regions) {
        const {width, height} = state.regions[r];
        rdims[r] = {width:width, height:height, ntimes:state.nvtimes};
    }
    RASTER_BUFFER = new GEFSRasterBuffer({
        url_formatter:(a) => {
            return state.urls.raster
                + `/${a.region}/${a.feat}/${a.metric}/${a.itime}`;
        },
        max_num_arrays:state.max_num_arrays,
        region_dimensions:rdims,
    });
    update_active_array()
});

const bind_array_requests = Promise.all([
    buffer_initialized, map_regions_bound
]).then(() => {
    MENU_METRIC.subscribe(update_active_array);
    MENU_ITIME.subscribe(update_active_array);
    MENU_REGION.subscribe(update_active_array);
});
