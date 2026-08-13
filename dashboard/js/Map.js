export class Map {
    #ready;
    #map;
    constructor({
        map_container,
        map_anchors,
    }) {
        this.#ready = new Promise((resolve, reject) => {
            this.#map = new maplibregl.Map({
                container:map_container,
                center:[-92.195082, 37.104743],
                style: {
                    version: 8,
                    sources: {
                        "basemap":{
                            type:"raster",
                            tiles: [
                                //"https://c.tile.opentopomap.org/{z}/{x}/{y}.png"
                                "/basemap/natural_earth_2_shaded_relief.raster"
                                + "/{z}/{x}/{y}",
                            ],
                            minzoom:2,
                            maxzoom:6,
                            tileSize:256,
                            //attribution:"© OpenStreetMap",
                        }
                    },
                    layers:map_anchors,
                },
                zoom: 3,
                minZoom:2,
                maxZoom:8,
            });

            this.#map.once("load", resolve);
            this.#map.once("error", (e) => {
                reject(e.error ?? new Error("map load failed"));
            });
            this.canvas = document.createElement("canvas");
            this.ctx = this.canvas.getContext("2d");
            this.cur_bbox = null;
        });
    }

    // resolve when map initialized.
    ready() {
        return this.#ready;
    }

    async set_region({
        bbox,
        bounds_buffer,
        raster_width,
        raster_height,
    }) {
        // make sure the map is ready
        await this.#ready;

        // update the map view location
        this.#map.setMaxBounds([
            [bbox[0]-bounds_buffer[1], bbox[1]-bounds_buffer[0]],
            [bbox[2]+bounds_buffer[1], bbox[3]+bounds_buffer[0]],
        ]);
        this.#map.setCenter([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]);

        // update the raster canvas
        this.canvas.width = raster_width;
        this.canvas.height = raster_height;
        this.ctx.imageSmoothingEnabled = false;
        this.ctx.webkitImageSmoothingEnabled = false;
        this.ctx.mozImageSmoothingEnabled = false;
        this.cur_bbox = bbox;

        const src = this.#map.getSource("raster");
        const coords = [
            [this.cur_bbox[0], this.cur_bbox[3]],
            [this.cur_bbox[2], this.cur_bbox[3]],
            [this.cur_bbox[2], this.cur_bbox[1]],
            [this.cur_bbox[0], this.cur_bbox[1]],
        ];
        if (!src) {
            this.#map.addSource("raster", {
                type:"canvas",
                canvas:this.canvas,
                coordinates:coords,
                animate:true,
            });
            this.#map.addLayer({
                id:"raster-layer",
                type:"raster",
                source:"raster",
                paint:{
                    "raster-opacity":1.,
                    "raster-resampling":"nearest",
                },
            }, "raster-anchor");
        } else {
            src.setCoordinates(coords);
        }
    }

    async add_geojson({
        name,
        data,
        layers,
        anchor,
    }) {
        await this.#ready;
        return new Promise((resolve, reject) => {
            const map_idle = () => {
                this.#map.off("idle", map_idle);
                resolve();
            }

            const map_error = (e) =>  {
                this.#map.off("idle", map_idle);
                this.#map.off("error", map_error);
                reject(e.error ?? e);
            }

            this.#map.once("idle", map_idle);
            this.#map.on("error", map_error);

            this.#map.addSource(name, {
                type:"geojson",
                data:data,
            });
            let cur_anchor = anchor;
            for (const l of layers) {
                this.#map.addLayer({
                    id:`${name}_${l.name}`,
                    type:l.type,
                    source:name,
                    paint:l.paint,
                }, cur_anchor);
                cur_anchor = `${name}_${l.name}`;
            }
        });
    }

    async render(image_data) {
        this.ctx.putImageData(image_data, 0, 0);
        this.#map.triggerRepaint();
    }

    async set_vector_visibility({
        substring,
        visible,
    }) {
        await this.#ready;
        const vis = visible ? "visible" : "none";

        for (const l of this.#map.getStyle().layers ?? []) {
            if (l.id.includes(substring)) {
                this.#map.setLayoutProperty(l.id, "visibility", vis);
            }
        }
    }
}
