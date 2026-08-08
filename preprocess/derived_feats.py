import numpy as np

def get_vpd(temp_c, rh):
    """ return vpd in kpa """
    es = 0.61078 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    return es * (1 - rh / 100)

## for now, all derived feats must have arguments matching raw feats available
## from the dynamical store, not other derived feats
derived_feats = {
    "wspd":[
        ("wind_u_10m", "wind_v_10m"),
        lambda a,b:(a**2+b**2)**(1/2),
        ],
    "wdir":[
        ("wind_u_10m", "wind_v_10m"),
        lambda a,b:np.rad2deg(np.arctan2(-1*a,-1*b)),
        ],
    "vpd":[
        ("temperature_2m", "relative_humidity_2m"),
        get_vpd,
        ],
    "pamt":[ ## assumes 3-hour time steps!!
        ("precipitation_surface",),
        lambda a:a * 60 * 60 * 3
        ],
    }

if __name__=="__main__":
    pass
