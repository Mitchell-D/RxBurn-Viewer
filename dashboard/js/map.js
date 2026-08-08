export class Map {
    #ready;
    #map;
    constructor({
        map_container
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
                                "https://c.tile.opentopomap.org/{z}/{x}/{y}.png"
                            ],
                            minzoom:3,
                            maxzoom:6,
                            tileSize:256,
                            attribution:"© OpenStreetMap",
                        }
                    },
                    layers: [
                        {
                            id: "background-layer",
                            type: "background",
                            paint: {
                                "background-color": "#000000",
                                "background-opacity":1,
                            },
                        },
                        {
                            id: "bottom-anchor",
                            type: "background",
                            layout: { visibility: "none" },
                        },
                        {
                            id: "basemap-tiles",
                            type: "raster",
                            source: "basemap",
                            paint: {
                                "raster-opacity":.4,
                                "raster-contrast": -0.2,
                                "raster-saturation": -0.4,
                            },
                        },
                        {
                            id: "raster-anchor",
                            type: "background",
                            layout: { visibility: "none" },
                        },
                        {
                            id: "poly-anchor-low",
                            type: "background",
                            layout: { visibility: "none" },
                        },
                        {
                            id: "poly-anchor-high",
                            type: "background",
                            layout: { visibility: "none" },
                        },
                        {
                            id: "top-anchor",
                            type: "background",
                            layout: { visibility: "none" },
                        }
                    ],
                },
                zoom: 5,
                minZoom:3,
                maxZoom:9,
            });

            this.#map.once("load", resolve);
            this.#map.once("error", (e) => {
                reject(e.error ?? new Error("map load failed"));
            });
            this.canvas = document.createElement("canvas");
            this.ctx = this.canvas.getContext("2d");
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
    }
}
