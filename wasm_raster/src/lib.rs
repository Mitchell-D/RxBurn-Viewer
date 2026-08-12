//! WASM module for storing keyed (time, lat, lon) uint16 rasters and
//! generating RGBA renders / multi-key threshold masks from them.
//!
//! Layout convention: each stored array is row-major, time-major, i.e.
//! `flat_index = t * (lat_len * lon_len) + y * lon_len + x`.
//!
//! Data values in the stored arrays are normalized ints in `[0, resolution-1]`,
//! except for a single sentinel `mask_value` (guaranteed outside that range)
//! marking "no data". Real-world float values are recovered via:
//!   `real = norm_min + (raw / (resolution - 1)) * (norm_max - norm_min)`

use js_sys::{Array, Float64Array, Uint16Array, Uint32Array, Uint8Array};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// specifies an array entry and its dimensions
struct RasterEntry {
    data: Vec<u16>,
    time_len: usize,
    lat_len: usize,
    lon_len: usize,
}

impl RasterEntry {
    // dimensions of a single-timestep spatial array
    fn slice_len(&self) -> usize {
        self.lat_len * self.lon_len
    }

    // return a single time step (lat, lon) array
    fn time_slice(&self, tix: usize) -> Result<&[u16], JsValue> {
        if tix >= self.time_len {
            return Err(js_err(format!(
                "time index {} out of range (time_len = {})",
                tix, self.time_len
            )));
        }
        let len = self.slice_len();
        let start = tix * len;
        Ok(&self.data[start..start + len])
    }
}

/// wrap a string as a javascript error for console display
fn js_err(msg: impl Into<String>) -> JsValue {
    JsValue::from_str(&msg.into())
}

/// struct for multiple 3d RasterEntry arrays
#[wasm_bindgen]
pub struct RasterStore {
    arrays: HashMap<String, RasterEntry>,
}

#[wasm_bindgen]
impl RasterStore {
    /// declare the struct for all arrays
    #[wasm_bindgen(constructor)]
    pub fn new() -> RasterStore {
        //#[cfg(feature = "console_error_panic_hook")]
        //console_error_panic_hook::set_once();
        RasterStore {
            arrays: HashMap::new(),
        }
    }

    /// Add or replace the raster under `key`.
    /// `data.length()` must equal `time_len * lat_len * lon_len`.
    pub fn add(
        &mut self,
        key: String,
        data: Uint16Array,
        time_len: usize,
        lat_len: usize,
        lon_len: usize,
    ) -> Result<(), JsValue> {
        // make sure the array size matches the specified dimensions
        let expected = time_len
            .checked_mul(lat_len)
            .and_then(|v| v.checked_mul(lon_len))
            .ok_or_else(|| js_err("time_len * lat_len * lon_len overflows usize"))?;
        let len = data.length() as usize;
        if len != expected {
            return Err(js_err(format!(
                "array length {} does not match time_len*lat_len*lon_len = {}",
                len, expected
            )));
        }

        // declare an empty vector and copy the data into it
        let mut vec = vec![0u16; len];
        data.copy_to(&mut vec[..]);

        // add the new raster to the struct
        self.arrays.insert(
            key,
            RasterEntry {
                data: vec,
                time_len,
                lat_len,
                lon_len,
            },
        );
        Ok(())
    }

    /// Remove the raster stored under `key`. Returns whether a value was
    /// actually removed.
    pub fn del(&mut self, key: String) -> bool {
        self.arrays.remove(&key).is_some()
    }

    /// Whether `key` currently has a raster stored.
    pub fn has(&self, key: String) -> bool {
        self.arrays.contains_key(&key)
    }

    /// Render one time step of `key` to an RGBA `Uint8Array`
    /// (`lat_len * lon_len * 4` bytes, row-major).
    ///
    /// `lut` is a flat RGBA lookup table: all entries except the *last*
    /// linearly span `[cmap_min, cmap_max]`; the last 4 bytes are the fixed
    /// color used for any pixel that is masked, out of threshold range, or
    /// otherwise invalid. `lut.length()` must be a multiple of 4 and at
    /// least 8 (>= 1 color entry + the trailing mask entry).
    pub fn generate_rgb(
        &self,
        key: String,
        time_index: usize,
        lut: Uint8Array,
        mask_value: u16,
        resolution: u32,
        norm_min: f64,
        norm_max: f64,
        cmap_min: f64,
        cmap_max: f64,
        thresh_min: f64,
        thresh_max: f64,
    ) -> Result<Uint8Array, JsValue> {
        // pull the requested raster and extract the provided time step
        let entry = self
            .arrays
            .get(&key)
            .ok_or_else(|| js_err(format!("key '{}' not found", key)))?;
        let slice = entry.time_slice(time_index)?;

        // validate the color map array shape
        let lut_len = lut.length() as usize;
        if lut_len < 8 || lut_len % 4 != 0 {
            return Err(js_err(
                "lut must be a multiple of 4 bytes and contain \
                 at least one color and mask/threshold entry",
            ));
        }
        let mut lut_vec = vec![0u8; lut_len];
        lut.copy_to(&mut lut_vec[..]);

        // last entry reserved for mask color
        let n_color_entries = lut_len / 4 - 1;
        let mask_color = &lut_vec[lut_len - 4..lut_len];

        if resolution < 2 {
            return Err(js_err("resolution must be >= 2"));
        }
        if norm_max <= norm_min {
            return Err(js_err("norm_max must be > norm_min"));
        }
        if cmap_max <= cmap_min {
            return Err(js_err("cmap_max must be > cmap_min"));
        }

        let res_denom = (resolution - 1) as f64;
        let pixel_count = slice.len();

        // iterate over each of the pixels and populate the output RGB
        let mut out = vec![0u8; pixel_count * 4];
        for (i, &raw) in slice.iter().enumerate() {
            let o = i * 4;

            // invalid or masked raw value -> mask color.
            if raw == mask_value || raw as u32 >= resolution {
                out[o..o + 4].copy_from_slice(mask_color);
                continue;
            }

            let r = norm_min+ (raw as f64 / res_denom) * (norm_max - norm_min);

            // Outside the threshold window -> transparent (mask color).
            if r < thresh_min || r > thresh_max {
                out[o..o + 4].copy_from_slice(mask_color);
                continue;
            }

            // Within threshold but possibly outside the (narrower) color-map
            // range: clamp to the nearest end color rather than treating it
            // as transparent.
            let clamped = r.clamp(cmap_min, cmap_max);
            let frac = (clamped - cmap_min) / (cmap_max - cmap_min);
            let idx = if n_color_entries == 1 {
                0
            } else {
                (frac * (n_color_entries - 1) as f64).round() as usize
            }
            .min(n_color_entries - 1);

            let c = idx * 4;
            out[o..o + 4].copy_from_slice(&lut_vec[c..c + 4]);
        }

        Ok(Uint8Array::from(&out[..]))
    }

    /// Build a boolean mask (as a `Uint8Array` of 0/1) that is 1 wherever
    /// *every* listed key's real-valued data at `time_index` falls within
    /// its own `[thresh_min, thresh_max]` window, and 0 otherwise (including
    /// wherever any key is masked/invalid at that pixel).
    ///
    /// `keys` is a JS array of strings; `mask_values`, `resolutions`,
    /// `norm_mins`, `norm_maxes`, `thresh_mins`, `thresh_maxes` are parallel
    /// typed arrays, one entry per key, in the same order as `keys`.
    /// All referenced rasters must share the same `lat_len`/`lon_len` and
    /// have `time_len > time_index`.
    pub fn generate_threshold_mask(
        &self,
        keys: Array,
        time_index: usize,
        mask_values: Uint16Array,
        resolutions: Uint32Array,
        norm_mins: Float64Array,
        norm_maxes: Float64Array,
        thresh_mins: Float64Array,
        thresh_maxes: Float64Array,
    ) -> Result<Uint8Array, JsValue> {
        let n = keys.length() as usize;
        if n == 0 {
            return Err(js_err("keys must not be empty"));
        }
        for (name, len) in [
            ("mask_values", mask_values.length() as usize),
            ("resolutions", resolutions.length() as usize),
            ("norm_mins", norm_mins.length() as usize),
            ("norm_maxes", norm_maxes.length() as usize),
            ("thresh_mins", thresh_mins.length() as usize),
            ("thresh_maxes", thresh_maxes.length() as usize),
        ] {
            if len != n {
                return Err(js_err(format!(
                    "{} has length {} but keys has length {}",
                    name, len, n
                )));
            }
        }

        let mask_values = mask_values.to_vec();
        let resolutions = resolutions.to_vec();
        let norm_mins = norm_mins.to_vec();
        let norm_maxes = norm_maxes.to_vec();
        let thresh_mins = thresh_mins.to_vec();
        let thresh_maxes = thresh_maxes.to_vec();

        // resolve keys to entries and validate shared shape.
        let mut entries: Vec<&RasterEntry> = Vec::with_capacity(n);
        let mut key_strs: Vec<String> = Vec::with_capacity(n);
        for i in 0..n {
            let key = keys
                .get(i as u32)
                .as_string()
                .ok_or_else(|| js_err(format!("keys[{}] is not a string", i)))?;
            let entry = self
                .arrays
                .get(&key)
                .ok_or_else(|| js_err(format!("key '{}' not found", key)))?;
            entries.push(entry);
            key_strs.push(key);
        }

        let (lat_len, lon_len) = (entries[0].lat_len, entries[0].lon_len);
        for (i, e) in entries.iter().enumerate() {
            if e.lat_len != lat_len || e.lon_len != lon_len {
                return Err(js_err(format!(
                    "key '{}' shaped ({},{}) but key '{}' shaped ({},{})",
                    key_strs[i], e.lat_len, e.lon_len,
                    key_strs[0], lat_len, lon_len
                )));
            }
        }

        // grab each key's requested time slice up front.
        let slices: Vec<&[u16]> = entries
            .iter()
            .map(|e| e.time_slice(time_index))
            .collect::<Result<_, _>>()?;

        for r in &resolutions {
            if *r < 2 {
                return Err(js_err("all resolutions must be >= 2"));
            }
        }
        for i in 0..n {
            if norm_maxes[i] <= norm_mins[i] {
                return Err(js_err(format!(
                    "key '{}': norm_max must be > norm_min",
                    key_strs[i]
                )));
            }
        }

        let pixel_count = lat_len * lon_len;
        let mut out = vec![0u8; pixel_count];

        // iterate over pixel indices, continuing as soon as a condition
        // is invalidated for one of the arrays
        'pixel: for p in 0..pixel_count {
            for i in 0..n {
                let raw = slices[i][p];
                let resolution = resolutions[i];

                if raw == mask_values[i] || raw as u32 >= resolution {
                    // masked / invalid -> fails the AND
                    continue 'pixel;
                }

                let res_denom = (resolution - 1) as f64;
                let real = norm_mins[i]
                    + (raw as f64 / res_denom) * (norm_maxes[i] - norm_mins[i]);

                if real < thresh_mins[i] || real > thresh_maxes[i] {
                    continue 'pixel;
                }
            }
            out[p] = 1;
        }

        Ok(Uint8Array::from(&out[..]))
    }
}

// initialize the RasterStore by running its constructor
impl Default for RasterStore {
    fn default() -> Self {
        Self::new()
    }
}
