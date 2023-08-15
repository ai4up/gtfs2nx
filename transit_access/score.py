import sys
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import h3pandas

from transit_access import network

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def transit_access(G, loc):
    stops, _ = ox.utils_graph.graph_to_gdfs(G)
    stops['index'] = stops['n_departures'] * stops['centrality'] 
    score = _calculate_score(
                loc=loc,
                stops_loc=stops.geometry,
                stops_score=stops['index'].values,
            )
    return score


def transit_access_for_grid(G, area=None, h3_res=9):
    if area is None:
        logger.info('Calculating TransitAccess index for convex hull with 2km buffer around transit stops.')
        stops, _ = ox.utils_graph.graph_to_gdfs(G)
        area = stops.dissolve().convex_hull.buffer(2000).to_frame('geometry')

    hex_grid = _create_hex_grid(h3_res, area)
    hex_grid['access_score'] = transit_access(G, hex_grid.centroid)
    return hex_grid


def transit_access_for_neighborhood(G, neighborhoods):
    hex_grid = transit_access_for_grid(G, neighborhoods)
    hex_grid['geometry'] = hex_grid.centroid
    area_access = _mean_per_area(neighborhoods, hex_grid, 'access_score')
    return area_access
    

def _calculate_score(loc, stops_loc, stops_score, decay_param=0.5):
    if loc.crs != stops_loc.crs:
        raise Exception(f'Coordinate reference system (CRS) of provided locations ({loc.crs}) and transit stops ({stops_loc.crs}) is not the same.')

    if not loc.crs.is_projected:
        raise Exception(f'Please use projected coordinate reference system (CRS) to ensure accurate distance calculation.')

    distance_matrix = loc.apply(lambda g: stops_loc.distance(g))
    distance_matrix.to_pickle(f'distance-matrix-{len(loc)}-{len(stops_loc)}.pkl')

    score = _calculate_gravity_score(
        distance_matrix=distance_matrix / 1000,
        supply=stops_score,
        decay_param=decay_param
    )
    return score


def _create_hex_grid(h3_res, area):
    ori_crs = area.crs
    hexbin = area.dissolve().to_crs('EPSG:4326')[['geometry']].h3.polyfill_resample(h3_res)
    hexbin = hexbin.rename_axis('ID').reset_index()
    hexbin = hexbin.to_crs(ori_crs)
    return hexbin


def _mean_per_area(area, points, col):
    points = points[[col, 'geometry']].to_crs(area.crs)
    matched = gpd.sjoin(points, area, how='left', predicate='within')
    mean = matched.groupby('index_right')[col].mean().reset_index().set_index('index_right')
    area = pd.merge(area, mean, how='left', left_index=True, right_index=True)
    return area


def _calculate_gravity_score(distance_matrix, supply, decay_param):
    # weight access to each stop by distance using a gaussian decay
    access_to_each_stop = (supply * _gaussian_decay(distance_matrix, decay_param)).T

    # only consider a single stop per route and direction
    # intuition: closeness to two stops of same trip is not better than to one stop with identical distance
    access_to_each_stop['route_id_w_direction'] = access_to_each_stop.index.str.split(network.ID_SEP, 1).str[1] 
    access_to_each_route = access_to_each_stop.groupby('route_id_w_direction').max()

    # summing up the access to all reachable stops
    access = access_to_each_route.sum()
    return access


def _gaussian_decay(distance_array, sigma):
    return np.exp(-(distance_array**2 / (2.0 * sigma**2)))
