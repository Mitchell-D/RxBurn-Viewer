"""
Configuration dictionaries that are used by preprocessing.extract_cogs,
as well as directly dumped to a JSON for the frontend to request, and
imported by the backend api.
"""

## options only explicitly used by prep_ifs.py
cfg_ifs = {
    ## netCDF figures to extract
    "get_raw_ifs_feats":["temperature_2m", "relative_humidity_2m", "wspd"],
    ## numpy percentile interpolation method for percentiles falling between
    ## data points (linear, lower, higher, midpoint, or nearest)
    "pctl_interp_method":"linear",
    "temporal_chunks":{"latitude":1, "longitude":1},
    "spatial_chunks":{"lead_time":1, "feat":1},

    ## normalization ruldefault_bounds_cmapes
    "cmap_default_bounds":{
        "temperature_2m":{
            "default":(-50, 50),
            "min":(-50, 50),
            "max":(-50, 50),
            "mean":(-50, 50),
            "stddev":(0,  50),
            "10pct":(-50, 50),
            "25pct":(-50, 50),
            "50pct":(-50, 50),
            "75pct":(-50, 50),
            "90pct":(-50, 50),
            "max-min":(0, 100),
            "90-10pct":(0, 100),
            "75-25pct":(0, 100),
            },
        "relative_humidity_2m":{
            "default":(0, 100),
            "min":(0, 100),
            "max":(0, 100),
            "mean":(0, 100),
            "stddev":(0, 50),
            "10pct":(0, 100),
            "25pct":(0, 100),
            "50pct":(0, 100),
            "75pct":(0, 100),
            "90pct":(0, 100),
            "max-min":(0, 100),
            "90-10pct":(0, 100),
            "75-25pct":(0, 100),
            },
        "wspd":{
            "default":(0, 50),
            "min":(0, 50),
            "max":(0, 50),
            "mean":(0, 50),
            "stddev":(0, 50),
            "10pct":(0, 50),
            "25pct":(0, 50),
            "50pct":(0, 50),
            "75pct":(0, 50),
            "90pct":(0, 50),
            "max-min":(0, 50),
            "90-10pct":(0, 50),
            "75-25pct":(0, 50),
            },
        },
    "norm_bounds":{
        "temperature_2m":{
            "default":(-50, 50),
            "min":(-50, 50),
            "max":(-50, 50),
            "mean":(-50, 50),
            "stddev":(0,  50),
            "10pct":(-50, 50),
            "25pct":(-50, 50),
            "50pct":(-50, 50),
            "75pct":(-50, 50),
            "90pct":(-50, 50),
            "max-min":(0, 100),
            "90-10pct":(0, 100),
            "75-25pct":(0, 100),
            },
        "relative_humidity_2m":{
            "default":(0, 100),
            "min":(0, 100),
            "max":(0, 100),
            "mean":(0, 100),
            "stddev":(0, 50),
            "10pct":(0, 100),
            "25pct":(0, 100),
            "50pct":(0, 100),
            "75pct":(0, 100),
            "90pct":(0, 100),
            "max-min":(0, 100),
            "90-10pct":(0, 100),
            "75-25pct":(0, 100),
            },
        "wspd":{
            "default":(0, 50),
            "min":(0, 50),
            "max":(0, 50),
            "mean":(0, 50),
            "stddev":(0, 50),
            "10pct":(0, 50),
            "25pct":(0, 50),
            "50pct":(0, 50),
            "75pct":(0, 50),
            "90pct":(0, 50),
            "max-min":(0, 50),
            "90-10pct":(0, 50),
            "75-25pct":(0, 50),
            },
        },
    "norm_res":2048,
    "mask_val":65535,
    "invalid_thresh":1e19,

    "spread_metrics":[
        "stddev",
        "max-min",
        "90-10pct",
        "75-25pct",
        ],

    ## long labels
    "long_labels_metrics":{
        "default":"Value",
        "min":"Minimum",
        "max":"Maximum",
        "mean":"Average",
        "stddev":"Std Dev",
        "10pct":"10th Pctl",
        "25pct":"25th Pctl",
        "50pct":"Median",
        "75pct":"75th Pctl",
        "90pct":"90th Pctl",
        "max-min":"Max-Min",
        "90-10pct":"90-10 Pctl",
        "75-25pct":"75-25 Pctl"
        },

    "long_labels_feats":{
        "temperature_2m":"Temperature (2m)",
        "relative_humidity_2m":"Relative Humidity (2m)",
        "wspd":"Wind Speed",
        },

    "long_labels_units":{
        "temperature_2m":"Celsius",
        "relative_humidity_2m":"Percent",
        "wspd":"Miles per Hour",
        },

    "short_labels_units":{
        "temperature_2m":"C",
        "relative_humidity_2m":"%",
        "wspd":"MPH",
        },
    }


## colormaps with which to generate lookup tables via matplotilb.
cfg_cmap = {
    "options":[
        "viridis",
        "viridis_r",
        #"gnuplot",
        "gist_rainbow",
        "gist_earth",
        "gist_earth_r",
        "coolwarm",
        "coolwarm_r",
        #"cmr.chroma",
        #"cmr.pride",
        "cmr.rainforest",
        "cmr.rainforest_r",
        "nipy_spectral",
        "magma",
        ],
    "resolution":256,
    "defaults":{
        "temperature_2m":"coolwarm",
        "relative_humidity_2m":"gist_earth_r",
        "wspd":"magma",
        },
    }







"""
Configuration for frontend line plots. Each entry corresponds to a unique
plotted line or range.

If 2 valid feat labels are provided to "lines", a range will be plotted between
those two features.

If one valid feat label is provided to "lines" and "surround" maps to a valid
feature and, a range will be plotted between that feature minus the "lines"
feature and that feature plus the "lines" feature.

If one valid feat label is provided and "surround" is not defined, a line
will be plotted.

dash configuration alternates length for line,gap,line,gap etc:

"dashes":None, ## solid line
"dashes":"5,5", ## dashed line
"dashes":"2,3", ## dotted line
"dashes":"12,3,3,3", ## alternating long and short
"""
daily_lines = {
    "daily mean":{
        "lines":["daily mean"],
        "name":"Average",
        "dashes":None,
        "color":"#BB4444",
        "show":True,
        "zorder":5,
        },
    "climo mean":{
        "lines":["climo mean"],
        "name":"Average",
        "dashes":None,
        "color":"#338F50",
        "show":True,
        "zorder":4,
        },
    "daily stddev":{
        "lines":["daily stddev"],
        "surround":"daily mean",
        "name":"Standard Deviation",
        "dashes":None,
        "color":"#ba8970",
        "show":True,
        "zorder":3,
        },
    "climo stddev":{
        "lines":["climo stddev"],
        "surround":"climo mean",
        "name":"Standard Deviation",
        "dashes":None,
        "show":False,
        "color":"#568f89",
        "zorder":2,
        },
    "daily vrange":{
        "lines":["daily min", "daily max"],
        "name":"Min/Max",
        "dashes":None,
        "color":"#baac70",
        "show":False,
        "zorder":1,
        },
    }

anom_lines = {
    "daily mean":{
        "lines":["daily mean"],
        "name":"Average",
        "dashes":None,
        "color":"#BB4444",
        "show":True,
        "zorder":5,
        },
    #"daily stddev":{
    #    "lines":["daily stddev"],
    #    "surround":"daily mean",
    #    "name":"Standard Deviation",
    #    "dashes":None,
    #    "color":"#ba8970",
    #    "show":True,
    #    "zorder":4,
    #    },
    "climo p10":{
        "lines":["climo p10"],
        "name":"10th pctl",
        "dashes":"12,3,3,3",
        "color":"#338F50",
        "show":True,
        "zorder":2,
        },
    "climo p25":{
        "lines":["climo p25"],
        "name":"25th pctl",
        "dashes":"2,5",
        "color":"#338F50",
        "show":True,
        "zorder":3,
        },
    "climo p75":{
        "lines":["climo p75"],
        "name":"75th pctl",
        "dashes":"2,5",
        "color":"#338F50",
        "show":True,
        "zorder":3,
        },
    "climo p90":{
        "lines":["climo p90"],
        "name":"90th pctl",
        "dashes":"12,3,3,3",
        "color":"#338F50",
        "show":True,
        "zorder":2,
        },
    }

## meta-information including *maximum* histogram bounds, layer
## names, verbose feature names, and feature units.
lsm_menu = {
    #"tmax": {
    #    "levels": {
    #        "01-00": {"hbounds": [-22.0, 44.0], "name": "Surface"}},
    #    "name": "Maximum Temperature",
    #    "units": "C"},
    "soil-moist": {
        "levels": {
            #"01-00": {"name": "Layer 1 (0-10cm)", "hbounds": [7.0, 54.0]},
            #"02-00": {"name": "Layer 2", "hbounds": [80.0, 432.0]},
            #"03-00": {"name": "Layer 3", "hbounds": [69.0, 877.0]}},
            "01-00": {"name": "Layer 1", "hbounds": [0., 100.]},
            "02-00": {"name": "Layer 2", "hbounds": [0., 100.]},
            "03-00": {"name": "Layer 3", "hbounds": [0., 100.]}},
        "name": "Soil Moisture",
        "units": "kg / m^2"},
    #"tmin": {
    #    "levels": {
    #        "01-00": {"hbounds": [-39, 29], "name": "Surface"}},
    #    "name": "Minimum Temperature",
    #    "units": "C"},
    "net-long": {
        "levels": {
            "01-00": {"hbounds": [-118, 3.5], "name": "Surface"}},
        "name": "Net Longwave Flux",
        "units": "W / m^2"},
    "evap": {
        "levels": {
            "01-00": {"hbounds": [-1.0, 23.0], "name": "Surface"}},
        "name": "Evapotranspiration",
        "units": "kg / m^2"},
    "net-short": {
        "levels": {
            "01-00": {"hbounds": [1.9, 332], "name": "Surface"}},
        "name": "Net Shortwave Flux",
        "units": "W / m^2"},
    "rainf": {
        "levels": {
            "01-00": {"hbounds": [0.0, 90.0], "name": "Surface"}},
        "name": "Rainfall",
        "units": "kg / m^2"},
    "runoff": {
        "levels": {
            "01-00": {"hbounds": [0.0, 60.0], "name": "Surface"}},
        "name": "Surface Runoff",
        "units": "kg / m^2"},
    "baseflow": {
        "levels": {
            "01-00": {"hbounds": [0, 32.0], "name": "Sub-surface"}},
        "name": "Base Flow",
        "units": "kg / m^2"},
    }
