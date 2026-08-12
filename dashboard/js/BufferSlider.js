export class BufferSlider {
    constructor({container_id, template_id}) {
        this.container = document.getElementById(container_id);
        this.template = document.getElementById(template_id);
        this.keys = [];
        this.cur = null;
        this.track = null;
        this.thumb = null;
        this.fill = null;
        this.active = false;
        this._build();
    }

    _build() {
        this.track = this.template.cloneNode(true).content;
        this.track = this.track.querySelector(".buffer-slider-track")
            .cloneNode(true);
        this.thumb = this.track.querySelector(".buffer-slider-thumb");
        this.fill = this.track.querySelector(".buffer-slider-fill");
        this.container.appendChild(this.track);
        this._track_click = (event) => {
            if (!this.active || this.keys.length === 0) return;
            const rect = this.track.getBoundingClientRect();
            const frac = (event.clientX - rect.left) / rect.width;
            const ix = Math.round(frac * (this.keys.length - 1));
            this.set_position(this.keys[ix]);
        }
        this.track.addEventListener("click", this._track_click)
    }

    update(new_keys) {
        const prev = this.cur;
        this.keys = [...new_keys];

        if (this.keys.length === 0) {
            this.cur = null;
        } else if (prev && this.keys.includes(prev)) {
            this.cur = prev;
        } else {
            this.cur = this.keys[0];
        }

        this._render();
    }

    set_position(key) {
        if (!this.keys.includes(key)) {
            throw new Error(`key not in slider: ${key}`);
        }
        this.cur = key;
        this._render();
    }

    set_active(is_active) {
        this.active = is_active;
        this._render();
    }

    get_position() {
        return this.cur;
    }

    _render() {
        console.log(this.cur);
        const enabled = this.active && this.keys.length > 0;
        this.track.classList.toggle("buffer-slider-track-inactive", !enabled);
        this.track.classList.toggle("buffer-slider-track-active", enabled);

        if (this.cur === null || this.keys.length <= 1) {
            this.thumb.style.left = "0%";
            this.fill.style.width = "0%";
            return;
        }

        const ix = this.keys.indexOf(this.cur);
        const pct = ix / (this.keys.length - 1) * 100;
        this.thumb.style.left = `${pct}%`;
        this.fill.style.width = `${pct}%`;
    }
}
