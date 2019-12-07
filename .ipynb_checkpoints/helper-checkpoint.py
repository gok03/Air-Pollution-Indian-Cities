from eolearn.core import EOTask, EOPatch, LinearWorkflow, Dependency, FeatureType
from eolearn.core import OverwritePermission
from eolearn.io import S2L1CWCSInput, SentinelHubWCSInput, SentinelHubOGCInput
from eolearn.core import LoadFromDisk, SaveToDisk

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import geopandas as gpd

from sentinelhub import BBox, CRS, DataSource

from math import pi, log, tan, exp, atan, log2, floor
import urllib.request
from PIL import Image
import io

import os
import imageio

from matplotlib import pyplot
import pandas as pd

from IPython.core.display import display, HTML
import matplotlib.colors
#--------------- image helper -----------

ZOOM0_SIZE = 512

#codes for obtaining base map png

def g2p(lat, lon, zoom):
    return (
        ZOOM0_SIZE * (2 ** zoom) * (1 + lon / 180) / 2,
        ZOOM0_SIZE / (2 * pi) * (2 ** zoom) * (pi - log(tan(pi / 4 * (1 + lat / 90))))
    )

def p2g(x, y, zoom):
    return (
        (atan(exp(pi - y / ZOOM0_SIZE * (2 * pi) / (2 ** zoom))) / pi * 4 - 1) * 90,
        (x / ZOOM0_SIZE * 2 / (2 ** zoom) - 1) * 180,
    )

def ax2mb(left, right, bottom, top):
    return (left, bottom, right, top)

def mb2ax(left, bottom, right, top):
    return (left, right, bottom, top)
def get_map_by_bbox(bbox,h,w):
    token = "pk.eyJ1IjoiZ2swMyIsImEiOiJhMzEwZTIyYWRhZWFjNWE5MTg0MzVkOGU5MjUyNzkxMiJ9.MKrbn4sDFM-oNMc9QupIKg"

    (left, bottom, right, top) = bbox

    assert (-90 <= bottom < top <= 90)
    assert (-180 <= left < right <= 180)
    
    if(h > 1280):
        temp = w/h
        h = 1280
        w = int(temp*1280)
    if(w > 1280):
        temp = h/w
        w = 1280
        h = int(temp*1280)
    
    (w, h) = (w, h)

    (lat, lon) = ((top + bottom) / 2, (left + right) / 2)

    snap_to_dyadic = (lambda a, b: (lambda x, scale=(2 ** floor(log2(abs(b - a) / 4))): (round(x / scale) * scale)))

    lat = snap_to_dyadic(bottom, top)(lat)
    lon = snap_to_dyadic(left, right)(lon)

    assert ((bottom < lat < top) and (left < lon < right)), "Reference point not inside the region of interest"

    for zoom in range(16, 0, -1):
        (x0, y0) = g2p(lat, lon, zoom)

        (TOP, LEFT) = p2g(x0 - w / 2, y0 - h / 2, zoom)
        (BOTTOM, RIGHT) = p2g(x0 + w / 2, y0 + h / 2, zoom)

        if (LEFT <= left < right <= RIGHT):
            if (BOTTOM <= bottom < top <= TOP):
                break

    params = {
        'style': "light-v9",
        'lat': lat,
        'lon': lon,
        'token': token,
        'zoom': zoom,
        'w': w,
        'h': h,
        'retina': "@2x",
    }

    url_template = "https://api.mapbox.com/styles/v1/mapbox/{style}/static/{lon},{lat},{zoom}/{w}x{h}{retina}?access_token={token}&attribution=false&logo=false"
    url = url_template.format(**params)
    #print(url)
    with urllib.request.urlopen(url) as response:
        j = Image.open(io.BytesIO(response.read()))

    (W, H) = j.size
    assert ((W, H) in [(w, h), (2 * w, 2 * h)])

    i = j.crop((
        round(W * (left - LEFT) / (RIGHT - LEFT)),
        round(H * (top - TOP) / (BOTTOM - TOP)),
        round(W * (right - LEFT) / (RIGHT - LEFT)),
        round(H * (bottom - TOP) / (BOTTOM - TOP)),
    ))

    return i
    
# ---------------- image helper end -------------

# ---------------- s5l2 helper ------------------


class S5PL2CWCSInput(SentinelHubWCSInput):
    """
    Task for creating EOPatches and filling them with Sentinel-5P L2 data using Sentinel Hub's WCS request.
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer=layer, data_source=DataSource.SENTINEL5P, **kwargs)
        
# ---------------- s5l2 helper end ---------------


# ---------------- plot helper -------------------

aq_values = {"NO2":[[-0.0005, 0.0045, 0.015, 0.02],0.01,"mol/m2"], 
    "SO2": [[-0.5,0.0593,0.625,2],0.35,"DU"],
    "O3": [[0,0.0675,0.135,0.27], 0.075, "mol/m2"],
    "CO": [[0,0.0675,0.135,0.27], 0.05, "mol/m2" ],
    "AER_AI_354_388": [[-5,-1,1,5],0, "Index Value"]}

def get_bbox(outline):
    dam_nominal = outline.geometry[0]
    inflate_bbox = 0.1
    minx, miny, maxx, maxy = dam_nominal.bounds

    delx = maxx - minx
    dely = maxy - miny
    minx = minx - delx * inflate_bbox
    maxx = maxx + delx * inflate_bbox
    miny = miny - dely * inflate_bbox
    maxy = maxy + dely * inflate_bbox

    return BBox([minx, miny, maxx, maxy], crs=CRS.WGS84)

def ppm(mpm2):
    return((mpm2) * (8.314*273.15) * 1000000 / 101300)

def find_quality(satellite_value,component):
    if(satellite_value>=aq_values[component][0][0] and satellite_value<aq_values[component][0][1]):
        return 0
    elif(satellite_value>aq_values[component][0][1] and satellite_value<aq_values[component][0][2]):
        return 1
    elif(satellite_value>aq_values[component][0][2] and satellite_value<aq_values[component][0][3]):
        return 2
    else:
        return -1

def plot_with_data(patch_data, component, dam_bbox, idx, plot_size):
    components_to_color = {"O3":"Blues","SO2":"Reds", "CH4":"Greys", "NO2":"YlOrBr", "HCHO":"RdPu", "CO":"Purples", "AER_AI_354_388":"Greens"}
    ratio = np.abs(patch_data.bbox.max_x - patch_data.bbox.min_x) / np.abs(patch_data.bbox.max_y - patch_data.bbox.min_y)
    fig, ax = plt.subplots(figsize=(ratio * plot_size, plot_size))
    size = fig.get_size_inches()*fig.dpi
    img = Image.fromarray(np.squeeze(patch_data.data[component][idx]), 'F')
    img = img.resize((int(size[0]), int(size[1])))   
    patch_map = get_map_by_bbox(dam_bbox,int(size[0]), int(size[1]))
    patch_map = patch_map.resize((int(size[0]), int(size[1])))
    ax.set_title(component +" ~ "+ patch_data.timestamp[idx].strftime("%d/%m/%Y, %H:%M:%S"))
    ax.imshow(patch_map.convert('RGB'))
    norm = matplotlib.colors.Normalize(aq_values[component][0][0],aq_values[component][0][-1])
    colors = [[norm(aq_values[component][0][0]), "green"],
              [norm(aq_values[component][0][1]), "yellow"],
              [norm(aq_values[component][0][2]), "red"],
              [norm(aq_values[component][0][3]), "red"]]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    im = ax.imshow(img,cmap=cmap, alpha=0.5, vmin=aq_values[component][0][0], vmax=aq_values[component][0][3])
    
    cax = fig.add_axes([0.90,0.15,0.02,0.70])
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    ax.axis('off')
    
def plot_with_data_save(patch_data, component, dam_bbox, idx, plot_size,filepath):
    plt.ioff()
    components_to_color = {"O3":"Blues","SO2":"Reds", "CH4":"Greys", "NO2":"YlOrBr", "HCHO":"RdPu", "CO":"Purples", "AER_AI_354_388":"Greens"}
    ratio = np.abs(patch_data.bbox.max_x - patch_data.bbox.min_x) / np.abs(patch_data.bbox.max_y - patch_data.bbox.min_y)
    fig, ax = plt.subplots(figsize=(ratio * plot_size, plot_size))
    size = fig.get_size_inches()*fig.dpi
    img = Image.fromarray(np.squeeze(patch_data.data[component][idx]), 'F')
    img = img.resize((int(size[0]), int(size[1])))   
    patch_map = get_map_by_bbox(dam_bbox,int(size[0]), int(size[1]))
    patch_map = patch_map.resize((int(size[0]), int(size[1])))
    ax.set_title(component +" ~ "+ patch_data.timestamp[idx].strftime("%d/%m/%Y, %H:%M:%S"))
    ax.imshow(patch_map.convert('RGB'))
    norm = matplotlib.colors.Normalize(aq_values[component][0][0],aq_values[component][0][-1])
    colors = [[norm(aq_values[component][0][0]), "green"],
              [norm(aq_values[component][0][1]), "yellow"],
              [norm(aq_values[component][0][2]), "red"],
              [norm(aq_values[component][0][3]), "red"]]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    im = ax.imshow(img,cmap=cmap, alpha=0.5, vmin=aq_values[component][0][0], vmax=aq_values[component][0][3])
    
    cax = fig.add_axes([0.90,0.15,0.02,0.70])
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    ax.axis('off')
    plt.savefig(filepath, format='png',transparent = True, bbox_inches = 'tight', pad_inches = 0)
    return(filepath)

def aqi_for_patch(patch):
    components = ["O3","SO2", "NO2", "CO", "AER_AI_354_388"]
    calculated_values = {"O3":[],"SO2":[], "NO2":[], "CO":[], "AER_AI_354_388":[]}
    for i in calculated_values:
        patch.data[i] = np.nan_to_num(patch.data[i],np.nanmin(patch.data[i]))
        quality_value_holder = []
        for j in range(len(patch.timestamp)):
            quality_value_holder += [find_quality(patch.data[i][j].max(),i)]
        #calculated_values[i] = [aq_values[i][0][0]]+quality_value_holder+[aq_values[i][0][-1]]
        calculated_values[i] = [0]+quality_value_holder+[2]
    calculated_values = [calculated_values[i] for i in calculated_values]
    timestamp_holder = [patch.timestamp[i].date() for i in range(len(patch.timestamp))]
    timestamp_holder = [timestamp_holder[0]]+timestamp_holder+[timestamp_holder[-1]]
    df=pd.DataFrame(data=[*zip(*calculated_values)],
                    index=timestamp_holder,
                    columns=components)
    df.plot(subplots=True,colormap="Set1",figsize=(10,5))
    pyplot.show()
    
def plot_patch_rgb(eopatch, idx):
    ratio = np.abs(eopatch.bbox.max_x - eopatch.bbox.min_x) / np.abs(eopatch.bbox.max_y - eopatch.bbox.min_y)
    fig, ax = plt.subplots(figsize=(ratio * 10, 10))
    
    ax.imshow(eopatch.data['1_TRUE_COLOR'][idx])
    ax.axis('off')
    
    


def aqi_for_patch_for_date(patch,idate):
    components = {"O3":"Ozone", "SO2":"Sulfur dioxide", "NO2": "Nitrogen dioxide", "CO":"Carbon monoxide", "AER_AI_354_388":"Aerosol"}
    value_to_text = {0:["Good","green"], 1:["Moderate","yellow"], 2:["Unhealthy","red"]}
    max_qi_value = 0
    build_html = "<style> table {width: 100% !important; } </style> <table> <tr> <td>"
    for i in components:    
        patch.data[i] = np.nan_to_num(patch.data[i],np.nanmin(patch.data[i]))
        qi_value = find_quality(patch.data[i][idate].max(),i)
        max_qi_value = max(max_qi_value,qi_value)
        build_html += '<div style="background-color: '+value_to_text[qi_value][1]+'; padding: 10px;"><center>'+components[i]+' Concentration is '+value_to_text[qi_value][0]+'</center></div><br/>'
    build_html += '</td> <td style="background-color: '+value_to_text[max_qi_value][1]+'; padding: 10px;"><center>Overall Air Quality is '+value_to_text[max_qi_value][0]+'</center></div><br/>'
    display(HTML(build_html))
    
    

# ---------------- plot helper end ---------------