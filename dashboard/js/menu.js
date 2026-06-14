/*
class for managing a button menu with contents that depend on an arbitrary
number of conditions.

start_menu saves the nesting settings, and populates the container with the
    default button configuration.

update provides a new set of conditions to the menu and changes it accordingly.
*/

// expand subtree to match the nesting structure of bigtree, such that
// terminal elements that are nested more deeply in bigtree default to the
// earier-terminated value.
function sync_object_trees(bigtree, subtree) {
    // If the subtree value is terminal (primitive or array), propagate it
    // down through the full depth of bigtree.
    if (!is_array_or_terminal(subtree)) {
        return propagate_terminal(bigtree, subtree);
    }

    // both sides are plain objects; recurse over each key of bigtree.
    const result = {};
    for (const k of Object.keys(bigtree)) {
        const vb = bigtree[k];
        const vs = subtree[k];

        if (vs === undefined) {
            // if key is absent from the subtree, add and propagate undefined
            result[k] = propagate_terminal(vb, undefined);
        } else {
            result[k] = sync_object_trees(vb, vs);
        }
    }
    return result;
}

// built an object matching the parent_tree with provided terminal values
function propagate_terminal(template, fill) {
    if (!is_array_or_terminal(template)) {
        // template is itself a leaf; just return fill directly
        return fill;
    }

    // template is a plain object so keep descending
    const result = {};
    for (const k of Object.keys(template)) {
        result[k] = propagate_terminal(template[k], fill);
    }
    return result;
}

// Returns true only for plain (non-null, non-array) objects.
function is_array_or_terminal(val) {
  return val !== null && typeof val === 'object' && !Array.isArray(val);
}

/*
function sync_object_trees(bigtree, subtree) {
    for (const k in bigtree) {
        // ignore object prototype keys
        if (Object.prototype.hasOwnProperty.call(bigtree, k)) {
            const vb = bigtree[k];
            let vs = null;
            if (typeof(subtree) !== "object" || subtree=== null) {
                vs = subtree;
            } else {
                vs = subtree[k];
            }

            console.log(bigtree, subtree)

            const is_parent = (typeof(vb) === "object" && vb !== null)

            // check if subtree terminated
            const child_terminal = typeof vs !== "object" || vs === null;

            if (is_parent) {
                if (child_terminal) {
                    // subtree terminated; fill it with its terminal value
                    console.log(k, subtree, subtree[k]);
                    subtree[k] = fill_subtree(vb, vs);
                } else {
                    // continue syncing one level deeper
                    sync_object_trees(vb, subtree[k]);
                }
            }
        }
    }
}

// built an object matching the parent_tree with provided terminal values
function fill_subtree(parent_tree, value) {
    const new_tree = {};
    for (const k in parent_tree) {
        if (Object.prototype.hasOwnProperty.call(parent_tree, k)) {
            const target = parent_tree[k];
            if (typeof target === "object" && target !== null) {
                  new_tree[k] = fill_subtree(target, value);
            } else {
                if (typeof value === "object" && value !== null) {
                    new_tree[k] = structuredClone(value);
                } else {
                    new_tree[k] = value;
                }
            }
        }
    }
    return new_tree;
}
*/

// recursive method for determining how deep the labels mapping is.
// Assumes that the depth is uniform among all sub-mappings
function object_depth(labels) {
    if (Array.isArray(labels)) {
        return 0;
    } else {
        return 1+Math.max(...Object.values(labels).map(object_depth));
    }
}


export class Menu {
    // save the state as a private property so it's individual to instances
    #state = {
        subscriptions:[],
        sel:null,
        menu_depth:null,
    }

    // store settings and launch the default version of the menu
    // button_template is expected to have elements with css classes:
    // menu-button containing a single button element.
    constructor({
        container_id, // container where buttons will be generated
        button_template_id, // dom template that will be copied for buttons
        labels, // array or mapping from dependencies to label arrays
        defaults, // string or nested matchin conditions to default selection
        conditions=[], // current array of conditions
        long_labels={}, // mapping from label strings to long label strings
        class_active="btn-primary",
        class_inactive="btn-secondary",
    }){
        this.container = document.getElementById(container_id);
        this.button_template = document.getElementById(button_template_id);
        this.class_active = class_active;
        this.class_inactive = class_inactive;

        this.#state.menu_depth = object_depth(labels);
        this.labels = labels;
        this.long_labels = long_labels;


        if (!(conditions.length == this.#state.menu_depth)) {
            throw new Error("Menu depth doesn't match conditions");
        }

        // make sure the defaults tree matches the labels nesting
        this.defaults = sync_object_trees(this.labels, defaults);
        // set the menu according to the default conditions
        this.current_conditions = conditions;
        this.current_value = this._conditions_to_default(
            this.current_conditions);
        this.update(this.current_conditions);
    }

    // get the labels associated with a series of conditions
    _conditions_to_labels(conditions) {
        if (this.#state.menu_depth == 0) {
            return this.labels;
        }
        let labels = structuredClone(this.labels);
        for (const i in conditions) {
            labels = labels[conditions[i]]
        }
        return labels
    }

    // get the labels associated with a series of defaults
    _conditions_to_default(conditions) {
        let defaults = structuredClone(this.defaults);
        for (const i in conditions) {
            defaults = defaults[conditions[i]]
        }
        return defaults
    }

    // update the defaults property with the most recent selection
    _set_default(conditions, default_value) {
        if (!(conditions.length == this.#state.menu_depth)) {
            throw new Error("Menu depth doesn't match conditions",
                conditions, this.#state.menu_depth);
        }
        return conditions.reduce(
            (cl, k, ix) => {
                if (ix === conditions.length - 1) {
                    cl[k] = default_value;
                    return;
                } else {
                    return cl[k];
                }
            },
            this.defaults
        );
    }

    update(conditions=[]) {
        if (!conditions.length == this.#state.menu_depth) {
            throw new Error("Menu depth doesn't match conditions",
                conditions, this.#state.menu_depth);
        }
        this.current_conditions = conditions;
        this._set_menu(
            this._conditions_to_labels(conditions),
            this._conditions_to_default(conditions),
        );
    }

    _set_menu(label_array, selected) {
        this.container.replaceChildren();
        const buttons = [];
        for (const k of label_array) {
            // grab the button template
            const tbc = this.button_template.content.querySelector(
                ".menu-button-container").cloneNode(true);
            const tb = tbc.querySelector(":scope > button");

            // set the button text and key
            if (this.long_labels.hasOwnProperty(k)) {
                tb.textContent = this.long_labels[k];
            } else {
                tb.textContent = k;
            }
            tb.value = k;

            // add a click callback to swap the state and notify subscribers
            const class_scope = this;
            tb.addEventListener("click", function() {
                if (this.classList.contains(class_scope.class_active)) {
                    return;
                }
                for (const el of class_scope.container.children) {
                    //const btn = el.querySelector(":scope > button");
                    const btn = el.querySelector(":scope > button");
                    // swap button to active if its feature value matches
                    if (this.value==btn.value) {
                        this.classList.remove(class_scope.class_inactive);
                        this.classList.add(class_scope.class_active);
                    }
                    // if this is the currently-selected, deactivate it.
                    else if (class_scope.current_value == btn.value) {
                        btn.classList.remove(class_scope.class_active);
                        btn.classList.add(class_scope.class_inactive);
                    }
                }
                class_scope.current_value = this.value;
                // update the default so when conditions change and change
                // back, the same button is selected.
                class_scope._set_default(
                    class_scope.current_conditions,
                    this.value,
                );
                class_scope._notify_subscribers();
            });

            // add the button to the container
            this.container.append(tbc);

            // simulate clicking this button if it is the selected one,
            // thereby notifying any subscribers of the new value
            if (k==selected) { tb.click(); }
        }
    }

    subscribe(callback)  {
        if (!typeof callback == "function") {
            throw new Error("Must provide a callback function not "+callback);
        }
        this.#state.subscriptions.push(callback);
    }

    _notify_subscribers() {
        this.#state.subscriptions.forEach(f=>f(this.current_value));
    }
}
