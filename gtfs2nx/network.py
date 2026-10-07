import os
import sys
import random
import logging
import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import partridge as ptg
from sklearn.neighbors import KDTree

ID_SEP = '@@'
FEED_SEP = '::'
INDENT = ' ' * 11

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def transit_graph(gtfs_paths, route_types=None, time_window=None, agency_ids=None, boundary=None,
                  frac=None, walk_transfer_max_distance=200, walk_speed_kmph=4, crs=None,
                  date=None, feed_ids=None, max_headway=None, max_speed_kmph=108,
                  min_segment_travel_time=None, max_speed_repair_factor=1.0,
                  transfer_model='headway', max_transfer_wait=None,
                  transfer_wait_quantile=None, min_transfer_time=0):
    """
    Create transit network graph from GTFS file(s).

    Nodes correspond to transit stops and edges to transit connections between stops.
    Each node and each edge belongs to only a single transit route.
    If multiple routes serve the same station, they are depicted as multiple nodes.
    Edges for walking transfer between nearby nodes of different routes are added.
    For each node, the average headway and the service frequency are calculated.


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
        Polygon to filter transit stops by. Expected in the same CRS as ``crs``.
    frac : float, optional
        Fraction, allowing to randomly sample a subset of transit routes to be included in the graph.
    walk_transfer_max_distance : int, optional
        Maximum distance in meters to allow walking transfer between transit stops.
    walk_speed_kmph : int, optional
        Assumed walking speed in km/h when calculating walking transfer times.
    crs : str, optional
        Metric coordinate reference system (CRS) to project transit stops to. If None, appropriate CRS UTM zone is inferred from lat lon bounds.
    date : str, datetime.date or list, optional
        Calendar date (``YYYY-MM-DD``) to take the timetable from. If None, the busiest date of each
        feed is used. A list of the same length as ``gtfs_paths`` selects a separate date per feed,
        which is useful when feeds cover different validity periods.
    feed_ids : list, optional
        Identifiers used to namespace stop, route and trip IDs of each feed so that feeds with
        overlapping ID spaces can be combined. Defaults to the file name of each feed when more
        than one feed is given, and to no namespacing at all for a single feed.
    max_headway : int, optional
        Upper bound in seconds for the average headway of a stop. Stops served only once within the
        time window have an undefined headway; with ``max_headway`` set they are kept and assigned
        this value, otherwise they are dropped from the graph.
    max_speed_kmph : int or dict, optional
        Trips faster than this between two consecutive stops (as the crow flies) are treated as
        corrupt and removed. Pass a dict keyed by ``route_type`` with an optional ``'default'``
        entry to allow high speed rail its 300 km/h while still catching broken bus timetables.
    max_speed_repair_factor : float, optional
        How far above ``max_speed_kmph`` a segment may be before its whole trip is discarded
        instead of repaired.

        Feeds that round departures to whole minutes make short hops look impossibly fast, and
        losing the entire trip over one of them can take a whole rural bus line out of the graph.
        With a factor of 2, say, a segment up to twice the speed limit has its travel time
        stretched to what the limit allows, while anything beyond that is still corrupt enough to
        take its trip with it. The default of 1.0 repairs nothing and removes every trip with a
        segment over the limit.
    min_segment_travel_time : int, optional
        Seconds to assign to segments whose timetable gives them no travel time at all. Without it,
        every trip containing such a segment is dropped, which is costly on feeds that round
        departures to whole minutes. Trips that genuinely contradict themselves (arrival before
        departure) are always dropped.
    transfer_model : {'headway', 'timetable'}, optional
        How long changing between two routes takes.

        ``'headway'`` charges half the headway of the route being boarded, which assumes the two
        timetables are unrelated. That holds for urban networks but not for regional and long
        distance services, where connections are deliberately scheduled: it makes an hourly train
        cost 30 minutes to board even when it was timed to meet the one just arrived.

        ``'timetable'`` instead reads the wait off the timetable. For every arrival of the first
        route it finds the next departure of the second that is reachable after the walk, and
        averages. Coordinated connections come out short and uncoordinated ones long, with no
        parameter to guess at.
    max_transfer_wait : int, optional
        Upper bound in seconds on the waiting part of a transfer, applied under either model. Acts
        as the safety net for pairs of routes that happen never to connect on the modelled day.
    transfer_wait_quantile : float, optional
        Which of a connection's observed waits to charge, between 0 (the best connection of the
        day) and 1 (the worst); the mean is used when this is None. Only applies to
        ``transfer_model='timetable'``.

        The mean answers "I turn up at a random time". A traveller going somewhere instead picks a
        departure whose connections line up, which a static graph cannot represent directly -- a
        low quantile approximates it. Around 0.2 reproduces published journey times best.
    min_transfer_time : int, optional
        Seconds that must be left between arriving and departing for a connection to count, on top
        of the walk. Guards against modelling platform changes as instantaneous.


    Returns
    -------
    networkx.DiGraph
        Directional transit network graph.


    Examples
    --------
    >>> transit_graph('some-dir/city-GTFS.zip')
    <networkx.classes.digraph.DiGraph object at 0x7f9b1c1b6a90>
    """
    gtfs_paths = _as_list(gtfs_paths)
    feed_ids = _resolve_feed_ids(gtfs_paths, feed_ids)
    dates = _broadcast(date, len(gtfs_paths), 'date')

    logger.info(f'STEP 1/5 - Loading {len(gtfs_paths)} GTFS feed(s) ...')
    feeds = _get_feeds(gtfs_paths, agency_ids, dates)

    logger.info('STEP 2/5 - Preprocessing GTFS feeds ...')
    routes, trips, stops, stop_times = _combine_feeds(feeds, feed_ids)
    routes, trips, stops, stop_times = _preprocess(routes, trips, stops, stop_times, crs)
    stops, stop_times = _create_unique_route_stop_ids(routes, trips, stops, stop_times)

    if time_window:
        logger.info(f'{INDENT}Filtering transit service between {time_window[0]} and {time_window[1]}...')
        stops, stop_times = _filter_by_time(stops, stop_times, time_window)

    stop_times = _clean_stop_times(stops, stop_times, max_speed_kmph, min_segment_travel_time,
                                   max_speed_repair_factor)

    logger.info('STEP 3/5 - Determining service frequency, transfer waiting & travel times...')
    stops = _calculate_stop_headway(stops, stop_times)
    stops = _calculate_service_frequency(stops, stop_times, time_window)
    stops, stop_times = _resolve_na_headway(stops, stop_times, max_headway)
    segments = _calculate_segment_travel_times(stop_times)

    if boundary is not None:
        logger.info(f'{INDENT}Filtering by geographical boundary...')
        stops, segments = _filter_by_boundary(stops, segments, boundary)

    if route_types:
        logger.info(f'{INDENT}Filtering by transit service types ({route_types})...')
        stops, segments = _filter_by_type(stops, segments, route_types)

    if frac:
        logger.info(f'{INDENT}Sampling {frac*100}% of all transit routes...')
        stops, segments = _sample_routes(stops, segments, frac)

    logger.info('STEP 4/5 - Creating NetworkX graph...')
    G = _create_graph(stops, segments)
    logger.info(f'{INDENT}{G.number_of_nodes():,} nodes and {G.number_of_edges():,} transit edges.')

    logger.info(f'STEP 5/5 - Adding edges for walk transfers between stops no more than {walk_transfer_max_distance} m apart (assuming walk speed of {walk_speed_kmph} km/h)...')
    G = _add_walk_transfer_edges(G, stop_times, max_distance=walk_transfer_max_distance,
                                 walk_speed_kmph=walk_speed_kmph, transfer_model=transfer_model,
                                 max_transfer_wait=max_transfer_wait,
                                 transfer_wait_quantile=transfer_wait_quantile,
                                 min_transfer_time=min_transfer_time)
    logger.info(f'{INDENT}{G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges in total.')

    return G


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _broadcast(value, length, name):
    """Expand a scalar to one value per feed, or validate a per-feed list."""
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f'{name} has {len(value)} entries but {length} GTFS feeds were given.')
        return list(value)
    return [value] * length


def _resolve_feed_ids(gtfs_paths, feed_ids):
    if feed_ids is None:
        # a single feed cannot collide with anything, so leave its IDs untouched
        if len(gtfs_paths) == 1:
            return None
        feed_ids = [os.path.splitext(os.path.basename(str(p)))[0] for p in gtfs_paths]
    feed_ids = _as_list(feed_ids)

    if len(feed_ids) != len(gtfs_paths):
        raise ValueError(f'feed_ids has {len(feed_ids)} entries but {len(gtfs_paths)} GTFS feeds were given.')
    if len(set(feed_ids)) != len(feed_ids):
        raise ValueError(f'feed_ids must be unique, got {feed_ids}.')
    if any(FEED_SEP in str(f) or ID_SEP in str(f) for f in feed_ids):
        raise ValueError(f'feed_ids must not contain "{FEED_SEP}" or "{ID_SEP}".')

    return [str(f) for f in feed_ids]


def _get_feeds(gtfs_paths, agency_ids=None, dates=None):
    dates = dates or [None] * len(gtfs_paths)
    return [_get_feed(path, agency_ids, date) for path, date in zip(gtfs_paths, dates)]


def _get_feed(gtfs_path, agency_ids=None, date=None):
    if date is None:
        _, service_ids = ptg.read_busiest_date(gtfs_path)
        logger.info(f'{INDENT}{os.path.basename(str(gtfs_path))}: using busiest date ({len(service_ids):,} services).')
    else:
        service_ids = _service_ids_on_date(gtfs_path, date)
        logger.info(f'{INDENT}{os.path.basename(str(gtfs_path))}: using {date} ({len(service_ids):,} services).')

    view = {
        'trips.txt': {'service_id': service_ids},
        'agency.txt': {'agency_id': agency_ids} if agency_ids else None,
        }
    return ptg.load_geo_feed(gtfs_path, view)


def _service_ids_on_date(gtfs_path, date):
    """Look up the service IDs that are active on a given calendar date."""
    date = _parse_date(date)
    by_date = ptg.read_service_ids_by_date(gtfs_path)
    service_ids = by_date.get(date)

    if not service_ids:
        covered = sorted(by_date)
        span = f'{covered[0]} to {covered[-1]}' if covered else 'nothing'
        raise ValueError(
            f'{gtfs_path} has no service on {date}; the feed covers {span}.')

    return service_ids


def _parse_date(date):
    if isinstance(date, datetime.datetime):
        return date.date()
    if isinstance(date, datetime.date):
        return date
    return datetime.date.fromisoformat(str(date))


def _combine_feeds(feeds, feed_ids=None):
    """Concatenate feeds, namespacing their IDs so that feeds cannot collide."""
    feed_ids = feed_ids or [None] * len(feeds)

    routes, trips, stops, stop_times = [], [], [], []
    for feed, feed_id in zip(feeds, feed_ids):
        prefix = '' if feed_id is None else f'{feed_id}{FEED_SEP}'
        routes.append(feed.routes.assign(route_id=prefix + feed.routes['route_id'].astype(str)))
        trips.append(feed.trips.assign(
            trip_id=prefix + feed.trips['trip_id'].astype(str),
            route_id=prefix + feed.trips['route_id'].astype(str)))
        stops.append(feed.stops.assign(
            stop_id=prefix + feed.stops['stop_id'].astype(str),
            feed_id=feed_id if feed_id is not None else ''))
        stop_times.append(feed.stop_times.assign(
            trip_id=prefix + feed.stop_times['trip_id'].astype(str),
            stop_id=prefix + feed.stop_times['stop_id'].astype(str)))

    return (pd.concat(routes, ignore_index=True),
            pd.concat(trips, ignore_index=True),
            pd.concat(stops, ignore_index=True),
            pd.concat(stop_times, ignore_index=True))


def _preprocess(routes, trips, stops, stop_times, crs=None):
    """Drop unnecessary attributes, clean up duplicates, and project geometries to metric coordinate reference system."""
    routes = routes.reindex(columns=['route_id', 'route_type', 'route_short_name']).drop_duplicates('route_id')
    trips = trips[['trip_id', 'route_id']].drop_duplicates('trip_id')
    stop_cols = [c for c in ['stop_id', 'stop_name', 'feed_id', 'geometry'] if c in stops.columns]
    stops = stops[stop_cols].drop_duplicates('stop_id')
    stop_times = stop_times[['trip_id', 'stop_id', 'arrival_time', 'stop_sequence']].sort_values(
        by=['trip_id', 'stop_sequence']).drop_duplicates(subset=['trip_id', 'stop_sequence'])

    local_crs = crs or stops.estimate_utm_crs()
    stops = stops.to_crs(local_crs)
    return routes, trips, stops, stop_times


def _create_unique_route_stop_ids(routes, trips, stops, stop_times):
    """Split stops served by multiple routes into separate, uniquely identifiable stops belonging to only one route"""
    trips = pd.merge(trips, routes, on='route_id')

    stop_times = pd.merge(stop_times, trips, on='trip_id')
    route_stop_cols = ['stop_id', 'route_id', 'route_type', 'route_short_name']
    route_stops = stop_times[route_stop_cols].drop_duplicates()  # create unique stop per route
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
    stops['frequency'] = stops['frequency'].fillna(0)
    return stops


def _calculate_stop_headway(stops, stop_times):
    """Calculate average waiting time until next trip at each stop."""
    stop_times = stop_times.sort_values(['stop_id', 'arrival_time'])
    stop_times['waiting'] = stop_times.groupby('stop_id')['arrival_time'].shift(-1) - stop_times['arrival_time']
    stops['headway'] = stop_times.groupby('stop_id')['waiting'].mean()
    return stops


def _resolve_na_headway(stops, stop_times, max_headway=None):
    """Drop stops without a defined headway, or cap them at ``max_headway`` if given."""
    # stops whose only trips were thrown out while cleaning have no service left to wait for
    stops = stops[stops.index.isin(stop_times['stop_id'])]

    if max_headway is None:
        n_dropped = int(stops['headway'].isna().sum())
        if n_dropped:
            logger.info(f'{INDENT}Dropped {n_dropped:,} stops served only once (undefined headway). '
                        'Set max_headway to keep them.')
        stops = stops[~stops['headway'].isna()]
    else:
        stops['headway'] = stops['headway'].fillna(max_headway).clip(upper=max_headway)

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
    if not {'next_stop_id', 'next_stop_travel_time'}.issubset(stop_times.columns):
        stop_times = stop_times.sort_values(['trip_id', 'stop_sequence'])
        stop_times['next_stop_id'] = stop_times.groupby('trip_id')['stop_id'].shift(-1)
        stop_times['next_stop_travel_time'] = stop_times.groupby('trip_id')['arrival_time'].shift(-1) - stop_times['arrival_time']

    segments = stop_times.groupby(['stop_id', 'next_stop_id'], dropna=True)['next_stop_travel_time'].mean().reset_index()

    return segments


def _clean_stop_times(stops, stop_times, max_speed_kmph=108, min_segment_travel_time=None,
                      max_speed_repair_factor=1.0):
    """Fix trips with identical consecutive stops and remove trips with corrupt and unrealistic travel times and speeds."""
    stop_times = stop_times.sort_values(['trip_id', 'stop_sequence'])

    # sometimes the waiting time until the next trip is included as the last stop_time of a trip -> remove it to avoid skewed travel time calculation
    stop_times['next_stop_id'] = stop_times.groupby('trip_id')['stop_id'].shift(-1)
    stop_times = stop_times[stop_times['stop_id'] != stop_times['next_stop_id']]

    stop_times['next_stop_travel_time'] = stop_times.groupby('trip_id')['arrival_time'].shift(-1) - stop_times['arrival_time']

    # a trip that arrives before it departed contradicts itself and cannot be repaired
    corrupt_trips = stop_times[stop_times['next_stop_travel_time'] < 0]['trip_id']
    stop_times = stop_times[~stop_times['trip_id'].isin(corrupt_trips)]
    if len(corrupt_trips) > 0:
        logger.info(f'{INDENT}Removed {corrupt_trips.nunique():,} trips with contradictory stop sequence and departure times.')

    # a segment without any travel time is usually a timetable rounded to whole minutes
    stop_times['repaired'] = stop_times['next_stop_travel_time'] == 0
    if min_segment_travel_time is None:
        corrupt_trips = stop_times[stop_times['repaired']]['trip_id']
        stop_times = stop_times[~stop_times['trip_id'].isin(corrupt_trips)]
        if len(corrupt_trips) > 0:
            logger.info(f'{INDENT}Removed {corrupt_trips.nunique():,} trips with segments without travel time. '
                        'Set min_segment_travel_time to keep them.')
    elif stop_times['repaired'].any():
        n_repaired = int(stop_times['repaired'].sum())
        stop_times.loc[stop_times['repaired'], 'next_stop_travel_time'] = min_segment_travel_time
        logger.info(f'{INDENT}Assigned {min_segment_travel_time}s to {n_repaired:,} segments without travel time.')

    # remove trips that cover the distance between two stops unrealistically fast
    stop_times = _calculate_segment_euclidean_distance(stop_times, stops)
    max_speed_mps = _max_speed_mps(stop_times['route_type'], max_speed_kmph)
    stop_times['speed'] = stop_times['distance'] / stop_times['next_stop_travel_time']

    # a repaired segment has an invented travel time, so its speed says nothing about the trip
    corrupt = (stop_times['speed'] > max_speed_mps * max_speed_repair_factor) & ~stop_times['repaired']
    corrupt_trips = stop_times[corrupt]['trip_id']
    stop_times = stop_times[~stop_times['trip_id'].isin(corrupt_trips)]
    max_speed_mps = max_speed_mps.loc[stop_times.index]

    if len(corrupt_trips) > 0:
        logger.info(f'{INDENT}Removed {corrupt_trips.nunique():,} trips with unrealistic travel speeds.')

    # what is left is too fast only because the timetable is imprecise, so slow it down to the limit
    if max_speed_repair_factor > 1:
        too_fast = stop_times['speed'] > max_speed_mps
        if too_fast.any():
            stop_times.loc[too_fast, 'next_stop_travel_time'] = (
                stop_times.loc[too_fast, 'distance'] / max_speed_mps[too_fast])
            stop_times['speed'] = stop_times['distance'] / stop_times['next_stop_travel_time']
            logger.info(f'{INDENT}Stretched {int(too_fast.sum()):,} segments that the timetable made '
                        'look faster than possible.')

    # raise exception if there are unrealistic trips left after cleaning
    if len(stop_times[(stop_times['speed'] < 0) | (stop_times['distance'] < 0) | (stop_times['next_stop_travel_time'] <= 0)]) > 0:
        logger.error('Unrealistic trips left after stop times cleaning.')

    if stop_times[['speed', 'distance', 'next_stop_travel_time']].isna().values.any():
        logger.error('Trips with NaN values left after stop times cleaning.')

    return stop_times


def _max_speed_mps(route_type, max_speed_kmph):
    """Speed limit in m/s for every stop time row, which may depend on the route type."""
    if not isinstance(max_speed_kmph, dict):
        return pd.Series(max_speed_kmph / 3.6, index=route_type.index)

    default = max_speed_kmph.get('default', 108)
    limits = {k: v for k, v in max_speed_kmph.items() if k != 'default'}
    return route_type.map(limits).fillna(default) / 3.6


def _filter_by_time(stops, stop_times, time_window):
    start, end = _parse_time_window(time_window)
    stop_times = stop_times[stop_times['arrival_time'].between(start, end)]
    stops = stops.loc[stop_times['stop_id'].unique()]
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
    # a segment whose endpoints did not survive filtering would silently add attribute-less nodes
    segments = segments[segments['stop_id'].isin(stops.index) & segments['next_stop_id'].isin(stops.index)]

    stops = stops.copy()
    stops['x'] = stops.geometry.x
    stops['y'] = stops.geometry.y
    node_attrs = [c for c in ['y', 'x', 'stop_name', 'feed_id', 'headway', 'frequency',
                              'route_id', 'route_type', 'route_short_name'] if c in stops.columns]
    nodes = list(zip(stops.index, stops[node_attrs].to_dict('records')))

    G = nx.DiGraph(crs=stops.crs)
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(segments.itertuples(index=False, name=None))
    nx.set_edge_attributes(G, 'transit', 'mode')

    return G


def _add_walk_transfer_edges(G, stop_times=None, max_distance=200, walk_speed_kmph=4,
                             transfer_model='headway', max_transfer_wait=None,
                             transfer_wait_quantile=None, min_transfer_time=0,
                             crow_flies_distance_factor=1.5):
    """Add edges for walking transfer between nearby stops of different routes."""
    if G.number_of_nodes() == 0:
        return G

    stops = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    source, target, distance = _transfer_candidates(stops, max_distance)
    if len(source) == 0:
        return G

    walk_sec = distance * crow_flies_distance_factor / 1000 / walk_speed_kmph * 3600
    wait_sec = _transfer_wait(stops, stop_times, source, target, walk_sec, transfer_model,
                              max_transfer_wait, transfer_wait_quantile, min_transfer_time)

    node_id = stops.index.to_numpy()
    # an existing transit edge means both stops are part of the same route, so no walk transfer is needed
    edges = [(node_id[s], node_id[t],
              {'weight': float(walk + wait), 'walk_time': float(walk), 'wait_time': float(wait),
               'mode': 'walk'})
             for s, t, walk, wait in zip(source, target, walk_sec, wait_sec)
             if not G.has_edge(node_id[s], node_id[t])]
    G.add_edges_from(edges)

    return G


def _transfer_candidates(stops, max_distance):
    """Pairs of nodes close enough to walk between that belong to different routes."""
    stops_location = stops[['x', 'y']].to_numpy()
    btree = KDTree(stops_location)
    indices, distances = btree.query_radius(stops_location, r=max_distance, return_distance=True)

    neighbour_counts = np.fromiter((len(i) for i in indices), dtype=np.int64, count=len(indices))
    source = np.repeat(np.arange(len(indices)), neighbour_counts)
    target = np.concatenate(indices).astype(np.int64)
    distance = np.concatenate(distances)

    route_id = stops['route_id'].to_numpy()
    # a transfer only makes sense between two different routes
    keep = (source != target) & (route_id[source] != route_id[target])
    return source[keep], target[keep], distance[keep]


def _transfer_wait(stops, stop_times, source, target, walk_sec, transfer_model, max_transfer_wait,
                   transfer_wait_quantile=None, min_transfer_time=0):
    """Seconds spent waiting for the connecting service, on top of the walk."""
    headway = stops['headway'].to_numpy()
    # a traveller arriving at a random moment waits half a headway on average
    fallback = headway[target] / 2

    if transfer_model == 'headway':
        wait = fallback
    elif transfer_model == 'timetable':
        if stop_times is None:
            raise ValueError("transfer_model='timetable' needs the stop times of the feed.")
        wait = _scheduled_transfer_wait(stops.index, stop_times, source, target,
                                        walk_sec + min_transfer_time, transfer_wait_quantile)
        wait = wait + min_transfer_time
        missing = np.isnan(wait)
        if missing.any():
            logger.info(f'{INDENT}{int(missing.sum()):,} of {len(wait):,} transfers never connect on '
                        'the modelled day; falling back to half the headway for those.')
            wait[missing] = fallback[missing]
    else:
        raise ValueError(f"transfer_model must be 'headway' or 'timetable', got {transfer_model!r}.")

    if max_transfer_wait is not None:
        wait = np.minimum(wait, max_transfer_wait)
    return wait


def _scheduled_transfer_wait(node_ids, stop_times, source, target, lead_sec, quantile=None,
                             chunk_size=4_000_000):
    """Wait for the next departure of the target route, read off the timetable.

    For every time the source route calls at its stop, the next departure of the target route that
    can still be caught is looked up. The resulting waits are then summarised by their mean, or by
    ``quantile`` of them. Returns NaN for pairs that never connect within the modelled service day.
    """
    times, offsets = _sorted_arrival_times(node_ids, stop_times)
    counts = np.diff(offsets)

    # offsetting each node's times into its own band lets one global searchsorted serve every pair
    band = float(times.max() + 1) if len(times) else 1.0
    banded = times + np.repeat(np.arange(len(counts)), counts) * band

    wait = np.full(len(source), np.nan)
    per_pair = counts[source]
    # chunk over pairs so that the expanded arrival times stay a bounded amount of memory
    bounds = _chunk_bounds(per_pair, chunk_size)

    for start, end in bounds:
        pairs = np.arange(start, end)
        lengths = per_pair[pairs]
        if lengths.sum() == 0:
            continue

        arrivals = times[_ragged_indices(offsets[source[pairs]], lengths)]
        pair_of = np.repeat(np.arange(len(pairs)), lengths)
        ready = arrivals + np.repeat(lead_sec[pairs], lengths)

        found = np.searchsorted(banded, ready + np.repeat(target[pairs], lengths) * band, side='left')
        within = found < np.repeat(offsets[target[pairs] + 1], lengths)
        gap = times[np.minimum(found, len(times) - 1)] - ready

        wait[start:end] = (_group_mean(pair_of[within], gap[within], len(pairs)) if quantile is None
                           else _group_quantile(pair_of[within], gap[within], len(pairs), quantile))

    return wait


def _group_mean(group, value, n_groups):
    counts = np.bincount(group, minlength=n_groups)
    totals = np.bincount(group, weights=value, minlength=n_groups)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(counts > 0, totals / counts, np.nan)


def _group_quantile(group, value, n_groups, quantile):
    """Nearest-rank quantile within each group, without materialising per-group arrays."""
    out = np.full(n_groups, np.nan)
    if len(group) == 0:
        return out

    order = np.lexsort((value, group))
    group, value = group[order], value[order]

    counts = np.bincount(group, minlength=n_groups)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    picked = starts + np.rint(quantile * np.maximum(counts - 1, 0)).astype(np.int64)

    present = counts > 0
    out[present] = value[picked[present]]
    return out


def _sorted_arrival_times(node_ids, stop_times):
    """Arrival times grouped by node: one flat sorted array plus the start of each node's block."""
    codes = pd.Categorical(stop_times['stop_id'], categories=node_ids).codes
    known = codes >= 0
    codes, times = codes[known], stop_times['arrival_time'].to_numpy(dtype=float)[known]

    order = np.lexsort((times, codes))
    codes, times = codes[order], times[order]

    counts = np.bincount(codes, minlength=len(node_ids))
    return times, np.concatenate([[0], np.cumsum(counts)])


def _ragged_indices(starts, lengths):
    """Indices of ``lengths[i]`` consecutive positions from ``starts[i]``, concatenated."""
    total = int(lengths.sum())
    block_starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return np.repeat(starts - block_starts, lengths) + np.arange(total)


def _chunk_bounds(per_pair, chunk_size):
    """Split pairs into chunks whose expanded arrival times stay under ``chunk_size`` elements."""
    bounds, start, running = [], 0, 0
    for i, n in enumerate(per_pair):
        if running and running + n > chunk_size:
            bounds.append((start, i))
            start, running = i, 0
        running += n
    bounds.append((start, len(per_pair)))
    return bounds
