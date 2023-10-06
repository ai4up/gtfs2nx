import sys
import random
import logging
import datetime

import pandas as pd
import geopandas as gpd
import networkx as nx
import partridge as ptg
from sklearn.neighbors import KDTree

ID_SEP = '@@'
INDENT = ' ' * 11

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def transit_graph(gtfs_paths, route_types=None, time_window=None, agency_ids=None, boundary=None, frac=None, walk_transfer_max_distance=200, walk_speed_kmph=4, crs=None):
    """
    Create transit network graph from GTFS file(s).

    Nodes correspond to transit stops and edges to transit connections between stops.
    Each node and each edge belongs to only a single transit route.
    If multiple routes serve the same station, they are depicted as multiple nodes.
    Edges for walking transfer between nearby nodes of different routes are added.
    For each node, the global closeness centrality and the number of departures are calculated.


    Parameters
    ----------
    gtfs_paths : str or list
        Paths to GTFS files.
    route_types : list, optional
        List of transit route types to include in the graph. If None, all service types are included.
    time_window : list, optional
        Pair of ISO 8601-formatted times to include services only within a time window.
    agency_ids : list, optional
        List of agencies (according to agency.txt) whose transit services are to be included in the graph. If None, all agencies are included.
    boundary : shapely.geometry.Polygon, optional
        Polygon to filter transit stops by.
    frac : float, optional
        Fraction, allowing to randomly sample a subset of transit routes to be included in the graph.
    walk_transfer_max_distance : int, optional
        Maximum distance in meters to allow walking transfer between transit stops.
    walk_speed_kmph : int, optional
        Assumed walking speed in km/h when calculating walking transfer times.
    crs : str, optional
        Metric coordinate reference system (CRS) to project transit stops to. If None, appropriate CRS UTM zone is inferred from lat lon bounds.


    Returns
    -------
    networkx.DiGraph
        Directional transit network graph.


    Examples
    --------
    >>> transit_graph('some-dir/city-GTFS.zip', 'EPSG:26914')
    <networkx.classes.digraph.DiGraph object at 0x7f9b1c1b6a90>
    """

    logger.info('STEP 1/5 - Loading GTFS feed(s) ...')
    feeds = _get_busiest_feeds(gtfs_paths, agency_ids)

    logger.info('STEP 2/5 - Preprocessing GTFS feeds ...')
    routes, trips, stops, stop_times = _combine_feeds(feeds)
    routes, trips, stops, stop_times = _preprocess(routes, trips, stops, stop_times, crs)
    stops, stop_times = _create_unique_route_stop_ids(routes, trips, stops, stop_times)

    if time_window:
        logger.info(f'{INDENT}Filtering transit service between {time_window[0]} and {time_window[1]}...')
        stops, stop_times = _filter_by_time(stops, stop_times, time_window)

    stop_times = _clean_stop_times(stops, stop_times)

    logger.info('STEP 3/5 - Determining service frequency, transfer waiting & travel times...')
    stops = _calculate_stop_headway(stops, stop_times)
    stops = _calculate_service_frequency(stops, stop_times, time_window)
    stops, stop_times = _filter_na_headway(stops, stop_times)
    segments = _calculate_segment_travel_times(stop_times)

    if boundary:
        logger.info('{INDENT}Filtering by geographical boundary...')
        stops, segments = _filter_by_boundary(stops, segments, boundary)

    if route_types:
        logger.info(f'{INDENT}Filtering by transit service types ({route_types})...')
        stops, segments = _filter_by_type(stops, segments, route_types)

    if frac:
        logger.info(f'{INDENT}Sampling {frac*100}% of all transit routes...')
        stops, segments = _sample_routes(stops, segments, frac)

    logger.info('STEP 4/5 - Creating NetworkX graph...')
    G = _create_graph(stops, segments)

    logger.info(f'STEP 5/5 - Adding edges for walk transfers between stops no more than {walk_transfer_max_distance} m apart (assuming walk speed of {walk_speed_kmph} km/h)...')
    G = _add_walk_transfer_edges(G, max_distance=walk_transfer_max_distance, walk_speed_kmph=walk_speed_kmph)

    return G


def _get_busiest_feeds(gtfs_paths, agency_ids=None):
    if not isinstance(gtfs_paths, (list, tuple)):
        gtfs_paths = [gtfs_paths]

    return [_get_busiest_feed(path, agency_ids) for path in gtfs_paths]


def _get_busiest_feed(gtfs_path, agency_ids=None):
    _, service_ids = ptg.read_busiest_date(gtfs_path)
    view = {
        'trips.txt': {'service_id': service_ids},
        'agency.txt': {'agency_id': agency_ids} if agency_ids else None,
        }
    feed = ptg.load_geo_feed(gtfs_path, view)
    return feed


def _combine_feeds(feeds):
    routes = pd.concat([f.routes for f in feeds])
    trips = pd.concat([f.trips for f in feeds])
    stops = pd.concat([f.stops for f in feeds])
    stop_times = pd.concat([f.stop_times for f in feeds])
    return routes, trips, stops, stop_times


def _preprocess(routes, trips, stops, stop_times, crs=None):
    """Drop unnecessary attributes, clean up duplicates, and project geometries to metric coordinate reference system."""
    routes = routes[['route_id', 'route_type', 'route_short_name']].drop_duplicates('route_id')
    trips = trips[['trip_id', 'route_id']].drop_duplicates('trip_id')
    stops = stops[['stop_id', 'geometry']].drop_duplicates('stop_id')
    stop_times = stop_times[['trip_id', 'stop_id', 'arrival_time', 'stop_sequence']].sort_values(
        by=['trip_id', 'stop_sequence']).drop_duplicates(subset=['trip_id', 'stop_sequence'])

    local_crs = crs or stops.estimate_utm_crs()
    stops = stops.to_crs(local_crs)
    return routes, trips, stops, stop_times


def _create_unique_route_stop_ids(routes, trips, stops, stop_times):
    """Split stops served by multipe routes into separate, uniquely identifiable stops belonging to only one route"""
    trips = pd.merge(trips, routes, on='route_id')


    stop_times = pd.merge(stop_times, trips, on='trip_id')
    route_stops = stop_times[['stop_id', 'route_id', 'route_type', 'route_short_name']].drop_duplicates()  # create unique stop per route
    route_stops = pd.merge(stops, route_stops, on='stop_id')  # add geometry

    route_stops['stop_id'] = route_stops['stop_id'] + ID_SEP + route_stops['route_id']
    stop_times['stop_id'] = stop_times['stop_id'] + ID_SEP + stop_times['route_id']
    route_stops = route_stops.set_index('stop_id')

    return route_stops, stop_times


def _calculate_service_frequency(stops, stop_times, time_window):
    """Calculate average number of departures per hour at each stop."""
    if time_window:
        start, end = _parse_time_window(time_window)
        duration_hours = (end - start) / 60 / 60
    else:
        duration_hours = 24

    stops['frequency'] = stop_times.groupby('stop_id')['arrival_time'].nunique() / duration_hours
    return stops


def _calculate_stop_headway(stops, stop_times):
    """Calculate average waiting time until next trip at each stop."""
    stop_times = stop_times.sort_values(['stop_id', 'arrival_time'])
    stop_times['waiting'] = stop_times.groupby('stop_id')['arrival_time'].shift(-1) - stop_times['arrival_time']
    stops['headway'] = stop_times.groupby('stop_id')['waiting'].mean()
    return stops


def _filter_na_headway(stops, stop_times):
    stops = stops[~stops['headway'].isna()]
    stop_times = stop_times[stop_times['stop_id'].isin(stops.index)]
    return stops, stop_times


def _calculate_segment_euclidean_distance(stop_times, stops):
    """Calculate euclidean distance in meter between all consecutive stops."""
    segments = stop_times[['stop_id', 'next_stop_id']].drop_duplicates()
    geom = stops['geometry']
    segments = pd.merge(segments, geom, on='stop_id')
    segments = pd.merge(segments, geom.rename('next_stop_geometry'), left_on='next_stop_id', right_on='stop_id')
    segments['distance'] = gpd.GeoSeries(segments['geometry']).distance(gpd.GeoSeries(segments['next_stop_geometry']))
    stop_times = pd.merge(stop_times, segments[['stop_id', 'next_stop_id', 'distance']], on=['stop_id', 'next_stop_id'])
    return stop_times


def _calculate_segment_travel_times(stop_times):
    """Calculate average travel time in seconds between all consecutive stops."""
    stop_times = stop_times.sort_values(['trip_id', 'stop_sequence'])
    stop_times['next_stop_id'] = stop_times.groupby('trip_id')['stop_id'].shift(-1)
    stop_times['next_stop_travel_time'] = stop_times.groupby('trip_id')['arrival_time'].shift(-1) - stop_times['arrival_time']
    segments = stop_times.groupby(['stop_id', 'next_stop_id'], dropna=True)['next_stop_travel_time'].mean().reset_index()

    return segments


def _clean_stop_times(stops, stop_times):
    """Fix trips with identical consecutive stops and remove trips with corrupt and unrealistic travel times and speeds."""
    stop_times = stop_times.sort_values(['trip_id', 'stop_sequence'])

    # sometimes the waiting time until the next trip is included as the last stop_time of a trip -> remove it to avoid skewed travel time calculation
    stop_times['next_stop_id'] = stop_times.groupby('trip_id')['stop_id'].shift(-1)
    stop_times = stop_times[stop_times['stop_id'] != stop_times['next_stop_id']]

    # remove trips with identical departures at two different stops and with contradictory stop sequence and departure times
    stop_times['next_stop_travel_time'] = stop_times.groupby('trip_id')['arrival_time'].shift(-1) - stop_times['arrival_time']
    corrupt_trips = stop_times[stop_times['next_stop_travel_time'] <= 0]['trip_id']
    stop_times = stop_times[~stop_times['trip_id'].isin(corrupt_trips)]

    if len(corrupt_trips) > 0:
        logger.info(f'Removed {corrupt_trips.nunique()} trips with contradictory stop sequence and departure times.')

    # remove trips with unrealistic euclidean travel speed of more than 30m/s (108km/h)
    stop_times = _calculate_segment_euclidean_distance(stop_times, stops)
    stop_times['speed'] = stop_times['distance'] / stop_times['next_stop_travel_time']
    corrupt_trips = stop_times[stop_times['speed'] > 30]['trip_id']
    stop_times = stop_times[~stop_times['trip_id'].isin(corrupt_trips)]

    if len(corrupt_trips) > 0:
        logger.info(f'Removed {corrupt_trips.nunique()} trips with unrealistic travel speeds (>108km/h).')

    # raise exception if there are unrealistic trips left after cleaning
    if len(stop_times[(stop_times['speed'] <= 0) | (stop_times['distance'] <= 0) | (stop_times['next_stop_travel_time'] <= 0)]) > 0:
        logger.error('Unrealistic trips left after stop times cleaning.')

    if stop_times[['speed', 'distance', 'next_stop_travel_time']].isna().values.any():
        logger.error('Trips with NaN values left after stop times cleaning.')

    return stop_times


def _filter_by_time(stops, stop_times, time_window):
    start, end = _parse_time_window(time_window)
    stop_times = stop_times[stop_times['arrival_time'].between(start, end)]
    stops = stops.loc[stop_times['stop_id']]
    return stops, stop_times


def _parse_time_window(window):
    start = _to_seconds(datetime.time.fromisoformat(window[0]))
    end = _to_seconds(datetime.time.fromisoformat(window[1]))
    return start, end


def _to_seconds(time):
    return (time.hour * 60 + time.minute) * 60 + time.second


def _filter_by_boundary(stops, segments, boundary):
    stops = stops[stops.within(boundary)]
    segments = segments[segments['stop_id'].isin(stops.index) & segments['next_stop_id'].isin(stops.index)]
    return stops, segments


def _filter_by_type(stops, segments, route_types):
    stops = stops[stops['route_type'].isin(route_types)]
    segments = segments[segments['stop_id'].isin(stops.index) & segments['next_stop_id'].isin(stops.index)]
    return stops, segments


def _sample_routes(stops, segments, frac):
    route_ids = list(stops['route_id'].unique())
    sample_routes = random.sample(route_ids, int(frac * len(route_ids)))
    stops = stops[stops['route_id'].isin(sample_routes)]
    segments = segments[segments['stop_id'].isin(stops.index) & segments['next_stop_id'].isin(stops.index)]
    return stops, segments


def _create_graph(stops, segments):
    weighted_edges = list(segments.itertuples(index=False, name=None))
    stops['x'] = stops.geometry.x
    stops['y'] = stops.geometry.y
    nodes = list(zip(stops.index, stops[['y', 'x', 'headway', 'frequency', 'route_id', 'route_type', 'route_short_name']].to_dict('records')))

    G = nx.DiGraph(crs=stops.crs)
    G.add_weighted_edges_from(weighted_edges)
    G.add_nodes_from(nodes)
    nx.set_edge_attributes(G, 'transit', 'mode')

    return G


def _add_walk_transfer_edges(G, max_distance, walk_speed_kmph):
    """Add edges for walking transfer between nearby stops of different routes."""
    stops = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    stops_location = stops[['x', 'y']].to_numpy()

    btree = KDTree(stops_location)
    indices, distance = btree.query_radius(stops_location, r=max_distance, return_distance=True)

    crow_flies_distance_factor = 1.5
    for from_idx, to_indices in enumerate(indices):
        for to_stop_idx, d in zip(to_indices, distance[from_idx]):

            walk_dis = d * crow_flies_distance_factor
            walk_h = walk_dis / 1000 / walk_speed_kmph
            walk_sec = walk_h * 60 * 60

            f = stops.index[from_idx]
            t = stops.index[to_stop_idx]

            # if transit connection between two stops already exists, it implies they are part of the same route, so that so walk transfer edge is needed
            if f != t and not G.has_edge(f, t):

                if G.nodes[f]['route_id'] != G.nodes[t]['route_id']:
                    # transfer time is defined as the average waiting time for the connection (headway/2) plus the walking time
                    transfer_time = G.nodes[t]['headway'] / 2 + walk_sec
                    G.add_edge(f, t, weight=transfer_time, mode='walk')

    return G
