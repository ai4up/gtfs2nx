import sys
import random
import logging
import datetime

import pandas as pd
import networkx as nx
import momepy as mm
import partridge as ptg
from sklearn.neighbors import KDTree

ID_SEP = '@@'
# ID_SEP = '-'

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def transit_graph(gtfs_paths, local_crs, route_types=None, start_time=None, end_time=None, agency_ids=None, boundary=None, frac=None, walk_transfer_max_distance=200, walk_speed_kmph=4):
    """
    Create transit network graph from GTFS file(s).

    Nodes correspond to transit stops and edges to transit connections between stops.
    Each node and each edge belongs to only a single transit route.
    If multiple routes serve the same station, they are depicted as multiple nodes.
    Edges for walking transfer between nearby nodes of different routes are added.
    For each node, the global closeness centrality and the number of departures are calculated.


    Parameters
    ----------
    gtfs_paths : list
        Paths to GTFS files.
    local_crs : str
        Metric coordinate reference system to project transit stops to.
    route_types : list, optional
        List of transit route types to include in the graph. If None, all service types are included.
    start_time : str, optional
        ISO 8601-formatted start time to consider only services within a time window.
    end_time : str, optional
        ISO 8601-formatted end time to consider only services within a time window.
    boundary : shapely.geometry.Polygon, optional
        Polygon to filter transit stops by.
    frac : float, optional
        Fraction, allowing to randomly sample a subset of transit routes to be included in the graph.
    walk_transfer_max_distance : int, optional
        Maximum distance in meters to allow walking transfer between transit stops.
    walk_speed_kmph : int, optional
        Assumed walking speed in km/h when calculating walking transfer times.


    Returns
    -------
    networkx.DiGraph
        Directional transit network graph.


    Examples
    --------
    >>> transit_graph('some-dir/city-GTFS.zip', 'EPSG:26914')
    <networkx.classes.digraph.DiGraph object at 0x7f9b1c1b6a90>
    """

    logger.info('STEP 1/12 - Loading GTFS feed(s) ...')
    feeds = _get_busiest_feeds(gtfs_paths, agency_ids)

    logger.info('STEP 2/12 - Combining GTFS feeds ...')
    routes, trips, stops, stop_times = _combine_feeds(feeds)

    logger.info('STEP 3/12 - Creating unique stop ids per route...')
    stops, stop_times = _create_unique_route_stop_ids(routes, trips, stops, stop_times)

    if start_time and end_time:
        logger.info(f'STEP 4/12 - Filtering transit service between {start_time} and {end_time}...')
        stops, stop_times = _filter_by_time(stops, stop_times, start_time, end_time)

    logger.info('STEP 5/12 - Approximating transfer waiting times...')
    stops = _calculate_stop_headway(stops, stop_times)
    stops, stop_times = _filter_na_headway(stops, stop_times)

    logger.info('STEP 6/12 - Calculating travel times between stops...')
    segments = _calculate_segment_travel_times(stop_times)

    logger.info(f'STEP 7/12 - Projecting transit stop locations to local, metric coordinate system ({local_crs})...')
    stops = stops.to_crs(local_crs)

    if boundary:
        logger.info('STEP 8a/12 - Filtering by geographical boundary...')
        stops, segments = _filter_by_boundary(stops, segments, boundary)

    if route_types:
        logger.info(f'STEP 8b/12 - Filtering by transit service types ({route_types})...')
        stops, segments = _filter_by_type(stops, segments, route_types)

    if frac:
        logger.info(f'STEP 8c/12 - Sampling {frac*100}% of all transit routes...')
        stops, segments = _sample_routes(stops, segments, frac)

    logger.info('STEP 9/12 - Creating NetworkX graph...')
    G = _create_graph(stops, segments)

    logger.info(f'STEP 10/12 - Adding edges for walk transfers between stops no more than {walk_transfer_max_distance} m apart (assuming walk speed of {walk_speed_kmph} km/h)...')
    G = _add_walk_transfer_edges(G, max_distance=walk_transfer_max_distance, walk_speed_kmph=walk_speed_kmph)

    logger.info('STEP 11/12 - Calculating global closeness centrality based on travel times...')
    # reverse directed graph to calculate closest path to all other nodes instead of from all other nodes
    centrality = nx.closeness_centrality(G.reverse(), distance='weight', wf_improved=True)
    nx.set_node_attributes(G, centrality, 'centrality')

    logger.info('STEP 12/12 - Calculating service frequency...')
    n_departures = _count_stop_departures(stop_times).to_dict()
    nx.set_node_attributes(G, n_departures, 'n_departures')

    return G


def local_closeness_centrality(G, radius=3000):
    return mm.closeness_centrality(G.reverse(), name='local_centrality', radius=radius, distance='weight').reverse()


def _get_busiest_feeds(gtfs_paths, agency_ids=None):
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


def _create_unique_route_stop_ids(routes, trips, stops, stop_times):
    trips = pd.merge(trips, routes[['route_id', 'route_type', 'route_short_name']], on='route_id')
    stop_times = pd.merge(stop_times, trips[['trip_id', 'route_id', 'direction_id', 'route_type', 'route_short_name']], on='trip_id')
    stop_times['new_stop_id'] = stop_times['stop_id'] + ID_SEP + stop_times['route_id'] + ID_SEP + stop_times['direction_id'].astype(str)
    stops = pd.merge(stops, stop_times[['stop_id', 'new_stop_id', 'route_id', 'direction_id', 'route_type', 'route_short_name']].drop_duplicates(), on='stop_id')

    stop_times['stop_id'] = stop_times['new_stop_id']
    stops['stop_id'] = stops['new_stop_id']
    stops = stops.set_index('stop_id')
    return stops, stop_times


def _calculate_stop_headway(stops, stop_times):
    stop_times = stop_times.sort_values(['stop_id', 'arrival_time'])
    stop_times['waiting'] = stop_times.groupby('stop_id')['arrival_time'].shift(-1) - stop_times['arrival_time']
    stops['headaway'] = stop_times.groupby('stop_id')['waiting'].mean()
    return stops


def _filter_na_headway(stops, stop_times):
    stops = stops[~stops['headaway'].isna()]
    stop_times = stop_times[stop_times['stop_id'].isin(stops.index)]
    return stops, stop_times


def _calculate_segment_travel_times(stop_times):
    stop_times = stop_times.sort_values(['trip_id', 'stop_sequence'])
    stop_times['next_stop'] = stop_times.groupby('trip_id')['stop_id'].shift(-1)
    stop_times['next_stop_travel_time'] = stop_times.groupby('trip_id')['arrival_time'].shift(-1) - stop_times['arrival_time']
    # If trip speed varies over the day, use the average travel time. Drop next_stop NA values as they correspond to terminal stations.
    segments = stop_times.groupby(['stop_id', 'next_stop'], dropna=True)['next_stop_travel_time'].mean().reset_index()
    return segments


def _count_stop_departures(stop_times):
    return stop_times.groupby('stop_id')['arrival_time'].nunique().rename('n_departures')


def _filter_by_time(stops, stop_times, start_time, end_time):
    start_time = _to_seconds(datetime.time.fromisoformat(start_time))
    end_time = _to_seconds(datetime.time.fromisoformat(end_time))
    stop_times = stop_times[stop_times['arrival_time'].between(start_time, end_time)]
    stops = stops.loc[stop_times['stop_id']]
    return stops, stop_times


def _to_seconds(time):
    return (time.hour * 60 + time.minute) * 60 + time.second


def _filter_by_boundary(stops, segments, boundary):
    stops = stops[stops.within(boundary)]
    segments = segments[segments['stop_id'].isin(stops.index) & segments['next_stop'].isin(stops.index)]
    return stops, segments


def _filter_by_type(stops, segments, route_types):
    stops = stops[stops['route_type'].isin(route_types)]
    segments = segments[segments['stop_id'].isin(stops.index) & segments['next_stop'].isin(stops.index)]
    return stops, segments


def _sample_routes(stops, segments, frac):
    route_ids = list(stops['route_id'].unique())
    sample_routes = random.sample(route_ids, int(frac * len(route_ids)))
    stops = stops[stops['route_id'].isin(sample_routes)]
    segments = segments[segments['stop_id'].isin(stops.index) & segments['next_stop'].isin(stops.index)]
    return stops, segments


def _create_graph(stops, segments):
    weighted_edges = list(segments.itertuples(index=False, name=None))
    stops['x'] = stops.geometry.x
    stops['y'] = stops.geometry.y
    nodes = list(zip(stops.index, stops[['y', 'x', 'headaway', 'route_id', 'route_type', 'route_short_name']].to_dict('records')))

    G = nx.DiGraph(crs=stops.crs)
    G.add_weighted_edges_from(weighted_edges)
    G.add_nodes_from(nodes)

    return G


def _add_walk_transfer_edges(G, max_distance, walk_speed_kmph):
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
                    # transfer time is defined as the average waiting time for the connection (headaway/2) plus the walking time
                    transfer_time = G.nodes[t]['headaway'] / 2 + walk_sec
                    G.add_edge(f, t, weight=transfer_time, mode='walk')

    return G
