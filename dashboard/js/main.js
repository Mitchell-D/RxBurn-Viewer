import { Map } from "./map.js";
//import { MenuDate } from "./menu_date.js";
//import { MenuFeat } from "./menu_feat.js";
import { Menu } from "./Menu.js";
import { DualRangeSlider } from "./DualRangeSlider.js";
//import { MenuPoly } from "./menu_pgroup.js";
//import { MenuRaster } from "./menu_raster.js";
//import { ColorMap } from "./color_map.js";
//import {datestring_to_date,date_to_datestring,format_date} from "./utils.js";
//import { default as init_wasm } from "./wasm/wasm_cmap.js";

// more emphatic variable names for modules
//const RASTER = Raster;
const MAP = Map;
//const COLOR = ColorMap
//const FMENU = MenuFeat;
//const DMENU = MenuDate;
//const PMENU = MenuPoly;
//const RMENU = MenuRaster;

const state = {
    dom:{
        feat_menu:"menu_container_feat",
        metric_menu:"menu_container_metric",
        button_template_id:"menu_button_temp",

        cmap_slider_container_id:"cmap_slider_row",
        threshold_slider_container_id:"threshold_slider_row",
        //pgroup_menu:"menu_container_pgroup",
        //date_picker:"buffer_date_range",
    },
    sel:{
        feat:"temperature_2m",
        metric:"mean",
        cmin:null, // minimum value bound for color map
        cmax:null, // minimum value bound for color map
        vmin:null, // minimum value bound for threshold
        vmax:null, // minimum value bound for threshold
        //pgroup:"fulldomain",
        //poly:"fulldomain_0",
        //t0:null,
        //tf:null,
    },
    urls:{
        menu:"/api/menu/ens/ifs",
        cmap:"/api/cmaps",
    },
    labels:{
        feats:null,
        metrics:null,
        units:null,
    },
    options:{
        feats:null,
        metrics:null,
        itimes:null,
        htimes:null,
    },
    // hard-coded to prevent the need for a request before map init
    image_shape:[59, 27],
    bbox:[-106.75, 25, -75.25, 39.5],
    map_bounds_buffer:[2.5, 5],
    // days until present displayed on init
    default_day_diff:30,
}

// make a promise for when the DOM is loaded
const dom_ready = new Promise(resolve => {
  if (document.readyState === "loading") {
    document.addEventListener('DOMContentLoaded', resolve);
  } else {
    resolve();
  }
});

let MENU_ITIME = null; // init time menu
let MENU_FEAT = null; // feature button menu
let MENU_METRIC = null; // metric button menu
let MENU_CSLIDER = null; // color map slider forms
let MENU_TSLIDER = null; // threshold slider forms
let MENU_CMAP = null; // color map name forms

// initialize the map
const map_started = dom_ready
    .then(() => MAP.init_map({
        main_map_container:"main_map_container",
        default_selection:state.sel,
        image_shape:state.image_shape,
        bbox:state.bbox,
        bounds_buffer:state.map_bounds_buffer,
    }))
    .then(MAP.style_loaded());
    //.then(() => Promise.all([MAP.update_polys(), MAP.style_loaded()]))
    //.then(() => MAP.set_active_pgroup({ pgroup:state.sel.pgroup }));

// load the IFS menu and
const menu_fetched = fetch(state.urls.menu)
    .then(r => r.json())
    .then(r => {
        console.log(r);
        state.labels.feats = r["long_labels_feats"];
        state.labels.metrics = r["long_labels_metrics"];
        state.labels.units_long = r["long_labels_units"];
        state.labels.units_short = r["short_labels_units"];

        state.options.feats = r["feats"];
        state.options.metrics = r["metrics"];
        state.options.itimes = r["init_times"];
        state.options.htimes = r["horizon_times"];

        // initialize feature menu
        MENU_FEAT = new Menu({
            container_id:state.dom.feat_menu,
            button_template_id:state.dom.button_template_id,
            labels:state.options.feats,
            defaults:state.sel.feat,
            initial_conditions:[],
            long_labels:state.labels.feats,
            class_active:"btn-primary",
            class_inactive:"btn-secondary",
        });

        // initialize metric menu
        // for now, assume all feats have all metrics, though the menu
        // class is general enough to handle complex nesting
        const metric_menu_labels = {}
        for (const l of state.options.feats) {
            metric_menu_labels[l] = state.options.metrics;
        }
        MENU_METRIC = new Menu({
            container_id:state.dom.metric_menu,
            button_template_id:state.dom.button_template_id,
            labels:metric_menu_labels,
            defaults:state.sel.metric,
            initial_conditions:[state.sel.feat],
            long_labels:state.labels.metrics,
            class_active:"btn-primary",
            class_inactive:"btn-secondary",
        });

        // subscribe the metric menu to update based on the feat menu
        MENU_FEAT.subscribe((new_feat) => {
            // main state needs to be the first to update so that subscribers
            // to the metric menu can be provided an up-to-date feat state
            state.sel.feat = new_feat;
            MENU_METRIC.update([new_feat]);
        });

        // initialize the color map slider menu
        MENU_CSLIDER = new DualRangeSlider({
            target_container_id:state.dom.cmap_slider_container_id,
            extrema:r["norm_bounds"],
            defaults:r["cmap_default_bounds"],
            initial_conditions:[state.sel.feat, state.sel.metric],
        });

        // initialize the threshold slider menu
        MENU_TSLIDER = new DualRangeSlider({
            target_container_id:state.dom.threshold_slider_container_id,
            extrema:r["norm_bounds"],
            defaults:structuredClone(r["norm_bounds"]),
            initial_conditions:[state.sel.feat, state.sel.metric],
        });

        // set subscriptions to menu (and by extension feat) changes
        MENU_METRIC.subscribe((new_metric) => {
            state.sel.metric = new_metric;
            // new metric runs any time a new feature is selected too since
            // it is conditioned on the feat menu.
            MENU_CSLIDER.set_new_conditions([state.sel.feat,state.sel.metric]);
            MENU_TSLIDER.set_new_conditions([state.sel.feat,state.sel.metric]);
        });

        // set subscriptions to color map bounds changes
        MENU_CSLIDER.subscribe((cmin,cmax) => {
            state.sel.cmin = cmin;
            state.sel.cmax = cmax;
        });

        // set subscriptions to threshold bounds changes
        MENU_TSLIDER.subscribe((vmin,vmax) => {
            state.sel.vmin = vmin;
            state.sel.vmax = vmax;
        });

        // initialize the color map name forms

        // initialize the time/date selection forms

        // initialize the playback/buffer forms
    });

/*
const cmaps_fetched = fetch(state.urls.cmap)
    .then(r => r.json())
    .then(j => {
        COLOR.set_menu({
            dom_ids:{
                cmap_menu_container:"cmap_menu_container",
                menu_dd_temp:"menu_dropdown_temp",
                split_dd_temp:"split_dropdown_temp",
                dd_cmap_name:"dd_cmap_name",
                dd_button_cmap_name:"dd_button_cmap_name",
                //dd_cmap_res:"dd_cmap_res",
                //dd_button_cmap_res:"dd_button_cmap_res",
                cbar_container:"cbar_container",
                cbar_axis:"cbar_axis",
                cbar_axis_container:"cbar_axis_container",
                cbar_raster_container:"cbar_raster_container",
            },
            cmaps:j["cmaps"],
            defaults:j["defaults"],
            options:j["options"],
            resolution:j["resolution"],
            slices:j["slices"],
            image_shape:state.image_shape,
        });
        COLOR.set_new_feat(state.sel.feat);
    });

const menu_populated = Promise.all([menu_fetched, dom_ready])
    .then(r => {
        const [j,_] = r;
        // determine the default active time range per state.default_day_diff
        const a0 = datestring_to_date(j["time_range"][0]);
        const af = datestring_to_date(j["time_range"][1]);
        let s0 = null;
        let sf = null;
        const dtmp = "{yyyy}{mm}{dd}"
        if (state.sel.t0===null) {
            s0 = new Date(af);
            s0.setDate(s0.getDate() - state.default_day_diff);
            state.sel.t0 = date_to_datestring(s0);
        }
        if (state.sel.tf===null) {
            state.sel.tf = date_to_datestring(af);
        }

        // data feature menu
        FMENU.set_menu({
            container_id:state.dom.feat_menu,
            labels:j["feat_labels"],
            selected:state.sel.feat,
            long_labels:j["long_labels"]["feat"],
        });

        // polygon group menu
        PMENU.set_menu({
            container_id:state.dom.pgroup_menu,
            labels:j["selectable_pgroups"],
            selected:state.sel.pgroup,
            long_labels:j["long_labels"]["pgroup"],
        });

        // date range menu
        DMENU.set_menu({
            dom_id:state.dom.date_picker,
            time_range:j["time_range"],
            t0:state.sel.t0,
            tf:state.sel.tf,
        });


    });

const forms_active = menu_populated.then(() => {
    DMENU.subscribe(s => {
        state.sel.t0 = s.t0;
        state.sel.tf = s.tf;
    });
    FMENU.subscribe(s => {
        state.sel.feat = s.feat;
    });
    PMENU.subscribe(s => {
        state.sel.pgroup = s.pgroup;
        MAP.set_active_pgroup(s);
    });
});

const raster_started = menu_populated
    .then(() => {
        RMENU.init_forms({
            id_ticker_template:"ticker_template",
            id_container_ticker:"main_container_ticker",
            id_buffer_button_prev:"buffer_button_prev",
            id_buffer_button_toggle:"buffer_button_toggle",
            id_buffer_button_next:"buffer_button_next",
            id_buffer_input_framedelay:"buffer_input_framedelay",
        });
        RMENU.set_new_buffer(state.sel.feat, state.sel.t0, state.sel.tf);
        DMENU.subscribe(s => {
            RMENU.set_new_buffer(state.sel.feat, s.t0, s.tf);
        });
        FMENU.subscribe(s => {
            RMENU.set_new_buffer(s.feat, state.sel.t0, state.sel.tf);
        });
    });

const render_ready = Promise.all([cmaps_fetched, raster_started])
    .then(() => {
        RMENU.subscribe((array) => { COLOR.apply_cmap(array) });
        COLOR.subscribe((image) => { MAP.render(image) });
    });
    */
