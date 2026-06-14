const state = {
    sel:{
        // states managed by set_new_buffer
        poly:null,
        pgroup:null,
    },

    active_poly:{
        id:null,
        properties:null,
        source:null,
    },

    dom:{
        main_map_container:"main_map_container",
    },

    // hard-code so that the map doesn't have to wait for a request
    rmeta:{
        width:392,
        height:357,
        bbox:[-91.625, 24.5833, -75.3333, 39.4166],
        bounds_buffer:[2.5, 5],
    },

    urls:{
        poly_fulldomain:"/api/poly/fulldomain",
        poly_states:"/api/poly/states",
        poly_counties:"/api/poly/counties",
        poly_roads:"/api/poly/roads",
        poly_waterways:"/api/poly/waterways",
    },

    pgroups:{}, // managed  by add_poly_group
    lgroups:{}, // managed  by add_line_group

    // cosmetics for each of the polygon layers. Each layer should have a
    // "core" (center line) and a "case" (larger, lower-opacity outer line)
    // so that the polygon is visible on any background color
    ppaint:{
        fulldomain:{
            "core":{
                "line-color":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    "#0d6efd", // enabled
                    "#2a2c2e", // disabled
                ],
                "line-opacity":1,
                "line-width":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    3, // enabled
                    1, // disabled
                ],
            },
            "case":{
                "line-color":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    "#fd9c0d", // enabled
                    "#d0d0d1", // disabled
                ],
                "line-opacity":1,
                "line-width":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    4, // enabled
                    2, // disabled
                ],
            },
        },
        states:{
            "core":{
                "line-color":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    "#0d6efd", // enabled
                    "#2a2c2e", // disabled
                ],
                "line-opacity":1,
                "line-width":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    3, // enabled
                    1, // disabled
                ],
            },
            "case":{
                "line-color":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    "#fd9c0d", // enabled
                    "#d0d0d1", // disabled
                ],
                "line-opacity":1,
                "line-width":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    4, // enabled
                    2, // disabled
                ],
            },
        },
        counties:{
            "core":{
                "line-color":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    "#0d6efd", // enabled
                    "#2a2c2e", // disabled
                ],
                "line-opacity":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    .8, // enabled
                    .4, // disabled
                ],
                "line-width":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    3, // enabled
                    1, // disabled
                ],
            },
            "case":{
                "line-color":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    "#fd9c0d", // enabled
                    "#d0d0d1", // disabled
                ],
                "line-opacity":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    .6, // enabled
                    .2, // disabled
                ],
                "line-width":[
                    "case",
                    ["boolean", ["feature-state", "selected"], false],
                    4, // enabled
                    2, // disabled
                ],
            },
        },
    },
    lpaint:{
        roads:{},
        waterways:{
            "line-color":"#0033cc",
            "line-opacity":0.1,
            "line-width":1.4,
        },
    },

    t_dstr:"{yyyy}{mm}{dd}", // template for date strings

    // used to to simulate same click when polygon scope changes
    last_click:null,

    subscriptions:[],

    //streams:null,
    map:null,
    canvas:null,
    ctx:null,
    //cmap_slider:null,
    //cmap_default_res:null,

    //raster_visible:true,
    //has_cbar:false,
    //cmap_thumbnail_res:64,
    //cbar_nticks:10,
}

// download all the polygon groups
async function update_polys() {
    await Promise.all([
        fetch(state.urls.poly_states)
            .then(r => r.json())
            .then(r => add_poly_group("states", r))
            .catch(e => console.log(e)),
        fetch(state.urls.poly_counties)
            .then(r => r.json())
            .then(r => add_poly_group("counties", r))
            .catch(e => console.log(e)),
        fetch(state.urls.poly_fulldomain)
            .then(r => r.json())
            .then(r => add_poly_group("fulldomain", r))
            .catch(e => console.log(e)),
        fetch(state.urls.poly_roads)
            .then(r => r.json())
            .then(r => add_line_group("roads", r))
            .catch(e => console.log(e)),
        fetch(state.urls.poly_waterways)
            .then(r => r.json())
            .then(r => add_line_group("waterways", r))
            .catch(e => console.log(e)),
    ]);
}

// set the poly group to know which should be activated
function set_active_pgroup({pgroup}) {
    state.sel.pgroup = pgroup;
    if (state.last_click === null) {
        // default to the first feature in the group
        set_active_polygon(state.map.querySourceFeatures("pgroup_"+pgroup)[0]);
    } else {
        // simulate most recent click given the new scope
        state.map.fire("click", state.last_click);
    }
}

function set_active_polygon(poly_obj) {
    // ignore repeated clicks
    if (state.active_poly.id == poly_obj.id) { return; }
    const old_active = state.active_poly;
    state.active_poly = {
        id:poly_obj.id,
        source:"pgroup_"+state.sel.pgroup,
        properties:poly_obj.properties,
    };
    state.sel.poly = poly_obj.id;
    if (old_active.source !== null) {
        state.map.setFeatureState(
            {source:old_active.source, id:old_active.id},
            {selected:false},
        );
    }
    state.map.setFeatureState(
        {source:state.active_poly.source, id:state.active_poly.id},
        {selected:true},
    );
    notify_subscribers();
}

function handle_map_click(click_event) {
    const cpolys = state.map.queryRenderedFeatures(click_event.point)
    const cloc = click_event.lngLat
    state.last_click = click_event;
    if (cpolys.length == 0) { return; }
    console.log("click location: "+cloc);
    cpolys.forEach(p => {
        if (p.id.split("_")[0] == state.sel.pgroup) {
            set_active_polygon(p);
        }
    });
}

// return a promise that the map styles are loaded.
function style_loaded() {
    return new Promise((resolve) => {
        // if the style is already loaded, resolve immediately
        if (state.map.isStyleLoaded()) {
            resolve();
            return;
        }

        // otherwise, wait for the style.load event
        state.map.once("style.load", () => {
            resolve();
        });
    });
}

// intialize the map and canvas used to draw on it.
async function init_map({
    main_map_container, default_selection, image_shape, bbox, bounds_buffer,
}) {
    state.rmeta = {
        bbox:bbox,
        height:image_shape[0],
        width:image_shape[1],
        bounds_buffer:bounds_buffer,
    }
    return new Promise((resolve, reject) => {
        state.map = new maplibregl.Map({
            container: main_map_container,
            center: [
              (state.rmeta.bbox[0] + state.rmeta.bbox[2]) / 2,
              (state.rmeta.bbox[1] + state.rmeta.bbox[3]) / 2
            ],
            zoom: 5,
            minZoom:3,
            maxZoom:7,
            maxBounds:[
                [state.rmeta.bbox[0]-state.rmeta.bounds_buffer[1],
                    state.rmeta.bbox[1]-state.rmeta.bounds_buffer[0]],
                [state.rmeta.bbox[2]+state.rmeta.bounds_buffer[1],
                    state.rmeta.bbox[3]+state.rmeta.bounds_buffer[0]],
            ],
          });

        state.map.on("load", () => { resolve() });
        state.map.on("error", (e) => {
            reject(new Error(`Map failed to load: ${e.error.message}`));
        });
        state.map.on("click", handle_map_click);
        // want to reclaim shift+click for selection
        state.map.boxZoom.disable();
        state.map.doubleClickZoom.disable();
        state.canvas = document.createElement("canvas");
        state.canvas.width = state.rmeta.width;
        state.canvas.height = state.rmeta.height;
        state.ctx = state.canvas.getContext("2d");
        /// don't do any kind of antialiasing
        state.ctx.imageSmoothingEnabled = false;
        state.ctx.webkitImageSmoothingEnabled = false;
        state.ctx.mozImageSmoothingEnabled = false;

        // set default selection
        state.sel.pgroup = default_selection.pgroup;
        state.sel.poly = default_selection.poly;
        resolve();
    });
}

// add the streams to the map
function add_line_group(lgname, ldata) {
    // add line data to the state dict
    if (!state.pgroups.hasOwnProperty(lgname)) {
        state.lgroups[lgname] = ldata;
    }

    const map_src_str = "lgroup_"+lgname

    state.map.addSource(lgname, {
        type:"geojson",
        data:state.lgroups[lgname],
        //promoteId:"DASHID", // treat DASHID property as the ID
    });

    state.map.addLayer({
        id:map_src_str+"_line",
        type:"line",
        source:map_src_str,
        paint:state.lpaint[lgname],
    });
}

// request the geoJSON associated with the selected polygon group, and fetch
// the geojson FeatureCollection-like object, and render the provided polygons
function add_poly_group(pgname, pdata) {
    // add polygon data to the state dict
    if (!state.pgroups.hasOwnProperty(pgname)) {
        state.pgroups[pgname] = pdata;
    }

    // on new group select, default to no polygons displayed
    //state.sel.pgroup = pgname;
    //state.sel.poly = null;

    // construct a unique source name for this polygon group
    const map_src_str = "pgroup_"+pgname

    // create a new data source for this polygon group
    state.map.addSource(map_src_str, {
        type:"geojson",
        data:state.pgroups[pgname],
        promoteId:"DASHID", // treat DASHID property as the ID
    });

    // add the invisible fill layer
    state.map.addLayer({
        id:map_src_str+"_fill",
        type:"fill",
        source:map_src_str,
        paint: {
            "fill-color": "#000",
            "fill-opacity": 0 // invisible but clickable
        }
    });

    // add the outline core layer
    state.map.addLayer({
        id:map_src_str+"_outline_core",
        type:"line",
        source:map_src_str,
        paint:state.ppaint[pgname]["core"],
    });
    // add the outline case layer below the core
    state.map.addLayer({
        id:map_src_str+"_outline_case",
        type:"line",
        source:map_src_str,
        paint:state.ppaint[pgname]["case"],
    }, map_src_str+"_outline_core");
    //state.map.on("click", "poly-volume", (e) => {
    //    toggle_poly(e);
    //});
}

async function render(image_data) {
    state.ctx.putImageData(image_data, 0, 0);
    //console.log(state.canvas);
    //document.getElementById("fig_daily_container").appendChild(state.canvas);

    // check whether the source already exists
    const source = state.map.getSource("raster");
    if (!source) {
        state.map.addSource("raster", {
            type:"canvas",
            canvas:state.canvas,
            coordinates:[
                [state.rmeta.bbox[0], state.rmeta.bbox[3]],
                [state.rmeta.bbox[2], state.rmeta.bbox[3]],
                [state.rmeta.bbox[2], state.rmeta.bbox[1]],
                [state.rmeta.bbox[0], state.rmeta.bbox[1]],
            ],
        });
        state.map.addLayer({
            id:"raster-layer",
            type:"raster",
            source:"raster",
            paint:{
                "raster-opacity":.85, // changeable param in future?
                "raster-resampling":"nearest",
            },
        }, state.map.getStyle().layers[0].id); // add below all other layers
    } else {
        state.map.triggerRepaint();
    }
}

function subscribe(callback)  {
    if (!typeof callback == "function") {
        throw new Error("Must provide a callback function not "+callback);
    }
    state.subscriptions.push(callback);
}

function notify_subscribers() {
    state.subscriptions.forEach(f=>f(state.sel));
}

export const Map = {
    update_polys:update_polys,
    init_map:init_map,
    style_loaded:style_loaded,
    set_active_pgroup:set_active_pgroup,
    subscribe:subscribe,
    render:render,
};

