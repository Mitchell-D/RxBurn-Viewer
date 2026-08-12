export class KeyedTable {
    constructor({
        table_id, // string <table> element id
        header_labels, // 4-element array of string header labels
    }) {
        this.table = document.getElementById(table_id)
        this.header_labels = header_labels;

        if (!(this.table instanceof HTMLTableElement)) {
            throw new Error("table_id must be a <table> element");
        }

        this.rows = new Map();
        this.subscriptions = [];

        this._create_header();
        this.tbody = this.table.querySelector("tbody");

        if (!this.tbody) {
            this.tbody = document.createElement("tbody");
            this.table.appendChild(this.tbody);
        }
    }

    // add or update a row based on the key
    update_row({
        key, // string keys for [feat, metric]
        min, // minimum threshold value for this entry
        max, // maximum threshold value for this entry
    }) {
        this.validate_key(key);

        min = Number(min);
        max = Number(max);

        if (!Number.isFinite(min) || !Number.isFinite(max)) {
            throw new Error("min and max must be numbers.");
        }

        if (min > max) {
            throw new Error("min cannot be greater than max.");
        }

        const k = this._key_to_string(key);

        let rdata = this.rows.get(k);

        if (rdata) {
            rdata.min = min;
            rdata.max = max;

            rdata.min_input.value = min;
            rdata.max_input.value = max;
        } else {
            rdata = this._create_row(key, min, max);
            this.rows.set(k, rdata);
        }
        this._notify_subscribers();
    }

    delete_row(key) { // string keys for [feat, metric]
        console.log("deleting", key);
        this.validate_key(key);

        const k = this._key_to_string(key);
        const rdata = this.rows.get(k);

        if (!rdata) {
            return false;
        }

        rdata.element.remove();
        this.rows.delete(k);

        this._notify_subscribers();

        return true;
    }

    // return current values for a single row
    get_row({
        key, // string keys for [feat, metric]
    }) {
        this.validate_key(key);

        const rdata = this.rows.get(this._key_to_string(key));

        if (!rdata) {
            return null;
        }

        return {
            min: rdata.min,
            max: rdata.max
        };
    }

    // get current values for all rows
    get_rows() {
        return Array.from(this.rows.values()).map(row => ({
            key: [...row.key],
            min: row.min,
            max: row.max
        }));
    }

    // initialize the header with user-provided labels
    _create_header() {
        // Don't overwrite an existing thead.
        let thead = this.table.querySelector("thead");

        if (!thead) {
            thead = document.createElement("thead");
            this.table.insertBefore(thead, this.table.firstChild);
        }

        if (thead.rows.length > 0) {
            return;
        }

        const row = thead.insertRow();

        this.header_labels.forEach(text => {
            const th = document.createElement("th");
            th.textContent = text;
            row.appendChild(th);
        });
    }

    // create a new row with user-provided minimum and maximum
    _create_row(key, min, max) {
        const tr = this.tbody.insertRow();

        // First key column.
        const k1c = tr.insertCell();
        k1c.textContent = key[0];

        // Second key column.
        const k2c = tr.insertCell();
        k2c.textContent = key[1];

        // Minimum input.
        const min_cell = tr.insertCell();
        const min_input = this._create_number_input(min);
        min_cell.appendChild(min_input);

        // Maximum input.
        const max_cell = tr.insertCell();
        const max_input = this._create_number_input(max);
        max_cell.appendChild(max_input);

        // Delete button.
        const delete_cell = tr.insertCell();
        const delete_button = document.createElement("button");

        delete_button.type = "button";
        delete_button.textContent = "x";

        delete_cell.appendChild(delete_button);

        const k = this._key_to_string(key);

        const rdata = {
            key: [...key],
            min,
            max,
            element: tr,
            min_input,
            max_input
        };

        // Handle user editing of Minimum.
        min_input.addEventListener("change", () => {
            let value = Number(min_input.value);

            if (!Number.isFinite(value)) {
                min_input.value = rdata.min;
                return;
            }

            // Minimum cannot exceed max.
            if (value > rdata.max) {
                value = rdata.max;
                min_input.value = value;
            }

            rdata.min = value;
        });

        // Handle user editing of Maximum.
        max_input.addEventListener("change", () => {
            let value = Number(max_input.value);

            if (!Number.isFinite(value)) {
                max_input.value = rdata.max;
                return;
            }

            // Maximum cannot be below min.
            if (value < rdata.min) {
                value = rdata.min;
                max_input.value = value;
            }

            rdata.max = value;
        });

        // Delete the row.
        const cls = this;
        delete_button.addEventListener("click", () => {
            tr.remove();
            console.log(key);
            cls.delete_row(key);
        });

        return rdata;
    }

    _create_number_input(value) {
        const input = document.createElement("input");

        input.type = "number";
        input.value = value;
        input.step = "any";

        return input;
    }

    validate_key(key) {
        if (
            !Array.isArray(key) ||
            key.length !== 2 ||
            typeof key[0] !== "string" ||
            typeof key[1] !== "string"
        ) {
            throw new Error(
                "Key must be an array containing exactly two strings."
            );
        }
    }

    _key_to_string(key) {
        // JSON encoding avoids collisions that could occur with simple
        // concatenation.
        return JSON.stringify(key);
    }

    subscribe(callback) {
        if (!typeof callback == "function") {
            throw new Error("Must provide a callback function not "+callback);
        }
        this.subscriptions.push(callback);
    }

    _notify_subscribers() {
        const cur_rows = this.get_rows();
        const out = [];
        for (const r of cur_rows) {
            out.push({
                feat:r.key[0],
                metric:r.key[1],
                min:r.min,
                max:r.max,
            });
        }
        this.subscriptions.forEach(f => f(out));
    }
}
