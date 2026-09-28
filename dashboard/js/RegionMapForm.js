export class RegionMapForm {
    constructor({
        canvas_container,
        width,
        height,
        display_array, // 1d array of 32 bit colors (sized width*height*4)
        pixel_ids, // 1d array of integer IDs (sized width*height)
        default_id, // initially selected id
        mask_val=255,
        highlight_color=[46, 59, 52, 255], // RGBA
    }) {
        this.container = document.getElementById(canvas_container);
        this.canvas = document.createElement("canvas");
        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');
        this.ctx.imageSmoothingEnabled = false;
        this.ctx.webkitImageSmoothingEnabled = false;
        this.ctx.mozImageSmoothingEnabled = false;
        this.width = width;
        this.height = height;
        this.pids = pixel_ids;
        this.darr = display_array;
        this.hlcolor = highlight_color;
        this.mask_val = mask_val;

        // Set internal resolution to match raw pixel dimensions
        this.canvas.width = width;
        this.canvas.height = height;

        this.cur_id = default_id;
        this.image = this.ctx.createImageData(width, height);

        // 32-bit view allows direct 1:1 pixel array assignments
        this.pixels = this.image.data;

        this.subscriptions = [];

        this.canvas.addEventListener('click', (e) => this.handle_click(e));
        this.render();
    }

    handle_click(e) {
        const rect = this.canvas.getBoundingClientRect();

        // get internal canvas pixel coordinates
        const x = Math.floor((e.clientX - rect.left)
                * (this.width / rect.width));
        const y = Math.floor((e.clientY - rect.top)
                * (this.height / rect.height));

        if (x >= 0 && x < this.width && y >= 0 && y < this.height) {
            this.set_id(this.pids[y * this.width + x]);
        }
    }

    set_id(new_id) {
        console.log("new id:", new_id, this.mask_val);
        if (new_id === this.mask_val) return;
        if (new_id === this.cur_id) return;
        this.cur_id = new_id;
        this.render();
        this._notify_subscribers(this.cur_id);
    }

    render() {
        const npx = this.width * this.height;

        for (let i = 0; i < npx; i++) {
            const ptr = i*4;
            // Check if alpha channel is 0 (0xAABBGGRR format)
            const is_trans = this.darr[i+3] === 0

            //console.log(is_trans, this.pids[i], this.cur_id);
            if (is_trans && this.pids[i] == this.cur_id) {
                // set active ids to the highlight color
                for (let j = 0 ; j < 4 ; j++ )
                    this.pixels[ptr + j] = this.hlcolor[j];
            } else {
                // reset previously highlighted pixel back to its default
                for (let j = 0 ; j < 4 ; j++ )
                    this.pixels[ptr + j] = this.darr[ptr + j];
            }
        }

        this.ctx.putImageData(this.image, 0, 0);
    }

    subscribe(callback) {
        if (typeof callback !== "function") {
            throw new Error("Must provide a callback function not "+callback);
        }
        this.subscriptions.push(callback);
    }

    _notify_subscribers(args) {
        this.subscriptions.forEach(f => f(args));
    }
}
