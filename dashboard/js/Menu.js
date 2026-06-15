/*
class for managing a button menu with contents that depend on an arbitrary
number of conditions.

start_menu saves the nesting settings, and populates the container with the
    default button configuration.

update provides a new set of conditions to the menu and changes it accordingly.
*/

import { sync_object_trees, object_depth, unpack } from "./utils.js";

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
        initial_conditions=[], // array of string starting conditions
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


        if (!(initial_conditions.length == this.#state.menu_depth)) {
            throw new Error("Menu depth doesn't match initial_conditions");
        }

        // make sure the defaults tree matches the labels nesting
        this.defaults = sync_object_trees(this.labels, defaults);
        // set the menu according to the default conditions
        this.current_conditions = initial_conditions;
        this.current_value = unpack(this.defaults, this.current_conditions);
        this.update(this.current_conditions);
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
            unpack(this.labels, conditions),
            unpack(this.defaults, conditions),
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
