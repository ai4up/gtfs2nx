import datetime

import geopandas as gpd
import networkx as nx
import pytest

import gtfs2nx as gx

SAMPLE_GTFS_FEED = 'tests/sample-feed.zip'
# the sample feed's busiest date, used to pin the date-based service lookup
SAMPLE_BUSIEST_DATE = datetime.date(2007, 1, 6)


def test_e2e_transit_graph():
    G = gx.transit_graph(SAMPLE_GTFS_FEED)

    nodes = gx.utils.nodes_to_gdf(G)
    edges = gx.utils.edges_to_gdf(G)

    assert G.graph['crs'] == 'EPSG:32611'
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 6
    assert nodes['frequency'].notna().all()
    assert nodes['headway'].notna().all()
    assert edges['mode'].notna().all()
    assert edges['weight'].notna().all()
    assert 'walk' not in edges['mode'].values
    assert nx.shortest_path_length(G, 'NADAV@@CITY', 'DADAN@@CITY', weight='weight') == 420
    with pytest.raises(nx.NetworkXNoPath):
        # the CITY and AAMV routes are not connected without walk transfers
        nx.shortest_path_length(G, 'NADAV@@CITY', 'AMV@@AAMV', weight='weight')


def test_e2e_transit_graph_walk_edges():
    G = gx.transit_graph(SAMPLE_GTFS_FEED, walk_transfer_max_distance=20000)

    edges = gx.utils.edges_to_gdf(G)

    assert G.number_of_edges() == 12
    assert 'walk' in edges['mode'].values


def test_single_feed_ids_are_not_namespaced():
    G = gx.transit_graph(SAMPLE_GTFS_FEED)

    assert all(gx.network.FEED_SEP not in node for node in G.nodes)


def test_date_selects_same_services_as_busiest_date():
    G_busiest = gx.transit_graph(SAMPLE_GTFS_FEED)
    G_dated = gx.transit_graph(SAMPLE_GTFS_FEED, date=SAMPLE_BUSIEST_DATE)

    assert set(G_dated.nodes) == set(G_busiest.nodes)
    assert set(G_dated.edges) == set(G_busiest.edges)


def test_date_accepts_iso_string():
    G = gx.transit_graph(SAMPLE_GTFS_FEED, date=SAMPLE_BUSIEST_DATE.isoformat())

    assert G.number_of_nodes() == 5


def test_date_outside_feed_coverage_is_rejected():
    with pytest.raises(ValueError, match='no service'):
        gx.transit_graph(SAMPLE_GTFS_FEED, date='1999-01-01')


def test_last_segment_of_a_trip_is_kept():
    # the AAMV route runs two-stop trips, which used to vanish entirely
    G = gx.transit_graph(SAMPLE_GTFS_FEED)

    assert G.has_edge('BEATTY_AIRPORT@@AAMV', 'AMV@@AAMV')
    assert G.has_edge('AMV@@AAMV', 'BEATTY_AIRPORT@@AAMV')


def test_combining_feeds_namespaces_colliding_ids():
    # the very same feed twice: without namespacing the two copies would collapse into one
    G = gx.transit_graph([SAMPLE_GTFS_FEED, SAMPLE_GTFS_FEED], feed_ids=['a', 'b'])

    transit_edges = [e for e in G.edges(data=True) if e[2]['mode'] == 'transit']

    assert G.number_of_nodes() == 10
    assert len(transit_edges) == 12
    assert {G.nodes[n]['feed_id'] for n in G.nodes} == {'a', 'b'}
    assert 'a::NADAV@@a::CITY' in G.nodes
    assert 'b::NADAV@@b::CITY' in G.nodes


def test_combining_feeds_requires_unique_feed_ids():
    with pytest.raises(ValueError, match='unique'):
        gx.transit_graph([SAMPLE_GTFS_FEED, SAMPLE_GTFS_FEED], feed_ids=['a', 'a'])


def test_per_feed_dates():
    G = gx.transit_graph([SAMPLE_GTFS_FEED, SAMPLE_GTFS_FEED], feed_ids=['a', 'b'],
                         date=[SAMPLE_BUSIEST_DATE, SAMPLE_BUSIEST_DATE])

    assert G.number_of_nodes() == 10


def test_per_feed_dates_length_is_validated():
    with pytest.raises(ValueError, match='date'):
        gx.transit_graph([SAMPLE_GTFS_FEED, SAMPLE_GTFS_FEED], feed_ids=['a', 'b'],
                         date=[SAMPLE_BUSIEST_DATE])


def test_max_headway_keeps_stops_served_once():
    G_dropped = gx.transit_graph(SAMPLE_GTFS_FEED, time_window=('06:00', '06:30'))
    G_kept = gx.transit_graph(SAMPLE_GTFS_FEED, time_window=('06:00', '06:30'), max_headway=1800)

    assert G_kept.number_of_nodes() > G_dropped.number_of_nodes()
    assert all(G_kept.nodes[n]['headway'] <= 1800 for n in G_kept.nodes)


def test_nodes_carry_stop_names():
    G = gx.transit_graph(SAMPLE_GTFS_FEED)

    assert G.nodes['NADAV@@CITY']['stop_name'] == 'North Ave / D Ave N (Demo)'


def test_max_speed_kmph_can_be_set_per_route_type():
    # the sample feed's fastest segment averages 58 km/h as the crow flies
    G_strict = gx.transit_graph(SAMPLE_GTFS_FEED, max_headway=3600, max_speed_kmph=50)
    G_lenient = gx.transit_graph(SAMPLE_GTFS_FEED, max_headway=3600, max_speed_kmph={3: 300})

    assert G_strict.number_of_nodes() < G_lenient.number_of_nodes()


def test_min_segment_travel_time_keeps_trips_without_travel_time():
    import pandas as pd

    stop_times = pd.DataFrame({
        'trip_id': ['t', 't', 't'],
        'stop_id': ['a', 'b', 'c'],
        'stop_sequence': [1, 2, 3],
        'arrival_time': [0.0, 0.0, 600.0],
        'route_type': [3, 3, 3],
    })
    geometry = gpd.GeoSeries.from_xy([0, 100, 200], [0, 0, 0], index=['a', 'b', 'c'])
    stop_geom = gpd.GeoDataFrame({'geometry': geometry}, index=geometry.index)
    stop_geom.index.name = 'stop_id'

    dropped = gx.network._clean_stop_times(stop_geom, stop_times.copy())
    kept = gx.network._clean_stop_times(stop_geom, stop_times.copy(), min_segment_travel_time=30)

    assert dropped.empty
    assert (kept['next_stop_travel_time'] > 0).all()
    assert kept['next_stop_travel_time'].iloc[0] == 30


def _transfer_fixture():
    """Two routes meeting at one stop: A arrives hourly, B leaves 5 min later every other hour."""
    import numpy as np
    import pandas as pd

    stops = pd.DataFrame({'x': [0.0, 0.0], 'y': [0.0, 0.0], 'route_id': ['A', 'B'],
                          'headway': [3600.0, 7200.0]}, index=['s@@A', 's@@B'])
    arrivals_a = np.arange(6, 22) * 3600.0                   # every hour, 06:00 to 21:00
    arrivals_b = np.arange(6, 22, 2) * 3600.0 + 5 * 60       # 5 min after every second A
    stop_times = pd.DataFrame({
        'stop_id': ['s@@A'] * len(arrivals_a) + ['s@@B'] * len(arrivals_b),
        'arrival_time': np.concatenate([arrivals_a, arrivals_b]),
    })
    return stops, stop_times


def test_scheduled_transfer_wait_finds_the_timed_connection():
    import numpy as np

    stops, stop_times = _transfer_fixture()
    source, target, walk = np.array([0]), np.array([1]), np.array([0.0])

    best = gx.network._scheduled_transfer_wait(stops.index, stop_times, source, target, walk,
                                               quantile=0.0)
    mean = gx.network._scheduled_transfer_wait(stops.index, stop_times, source, target, walk)

    # the connection it was timed for is 5 minutes; averaging over every arrival is far worse
    assert best[0] == pytest.approx(300)
    assert mean[0] > best[0]
    # and the headway model would charge half of B's two-hour headway
    assert mean[0] < stops['headway'].iloc[1] / 2


def test_scheduled_transfer_wait_is_nan_without_a_connection():
    import numpy as np
    import pandas as pd

    stops, stop_times = _transfer_fixture()
    # push every B departure before the first A arrival, so nothing ever connects
    stop_times.loc[stop_times['stop_id'] == 's@@B', 'arrival_time'] = 3600.0
    source, target, walk = np.array([0]), np.array([1]), np.array([0.0])

    wait = gx.network._scheduled_transfer_wait(stops.index, stop_times, source, target, walk)

    assert np.isnan(wait[0])


def test_min_transfer_time_rules_out_connections_that_are_too_tight():
    import numpy as np

    stops, stop_times = _transfer_fixture()
    source, target, walk = np.array([0]), np.array([1]), np.array([0.0])

    tight = gx.network._transfer_wait(stops, stop_times, source, target, walk, 'timetable',
                                      None, 0.0, min_transfer_time=0)
    loose = gx.network._transfer_wait(stops, stop_times, source, target, walk, 'timetable',
                                      None, 0.0, min_transfer_time=600)

    assert tight[0] == pytest.approx(300)
    # 5 minutes is no longer enough, so the next usable departure is two hours later
    assert loose[0] > 3600


def test_max_transfer_wait_bounds_the_headway_model():
    import numpy as np

    stops, stop_times = _transfer_fixture()
    source, target, walk = np.array([0]), np.array([1]), np.array([0.0])

    uncapped = gx.network._transfer_wait(stops, stop_times, source, target, walk, 'headway', None)
    capped = gx.network._transfer_wait(stops, stop_times, source, target, walk, 'headway', 600)

    assert uncapped[0] == pytest.approx(3600)
    assert capped[0] == pytest.approx(600)


def test_unknown_transfer_model_is_rejected():
    import numpy as np

    stops, stop_times = _transfer_fixture()
    with pytest.raises(ValueError, match='transfer_model'):
        gx.network._transfer_wait(stops, stop_times, np.array([0]), np.array([1]),
                                  np.array([0.0]), 'guess', None)


def test_timetable_transfers_change_the_graph_weights():
    G_headway = gx.transit_graph(SAMPLE_GTFS_FEED, walk_transfer_max_distance=20000,
                                 max_headway=7200)
    G_timetable = gx.transit_graph(SAMPLE_GTFS_FEED, walk_transfer_max_distance=20000,
                                   max_headway=7200, transfer_model='timetable',
                                   transfer_wait_quantile=0.0)

    walk_edges = [(u, v) for u, v, d in G_headway.edges(data=True) if d['mode'] == 'walk']
    assert walk_edges
    assert all('wait_time' in G_timetable[u][v] for u, v in walk_edges)
    assert any(G_timetable[u][v]['weight'] != G_headway[u][v]['weight'] for u, v in walk_edges)


def test_group_quantile_picks_the_right_rank():
    import numpy as np

    group = np.array([0, 0, 0, 0, 1])
    value = np.array([10.0, 20.0, 30.0, 40.0, 99.0])

    assert gx.network._group_quantile(group, value, 3, 0.0)[0] == 10
    assert gx.network._group_quantile(group, value, 3, 1.0)[0] == 40
    assert gx.network._group_quantile(group, value, 3, 0.5)[0] == 30
    assert np.isnan(gx.network._group_quantile(group, value, 3, 0.5)[2])


def _too_fast_stop_times():
    """A three stop trip whose middle hop the timetable rounds into an impossible speed."""
    import pandas as pd

    stop_times = pd.DataFrame({
        'trip_id': ['t', 't', 't'],
        'stop_id': ['a', 'b', 'c'],
        'stop_sequence': [1, 2, 3],
        'arrival_time': [0.0, 600.0, 1200.0],   # 10 minutes per hop
        'route_type': [3, 3, 3],
    })
    # a is 1 km from b, but b is 100 km from c: 600 s for that is 600 km/h
    geometry = gpd.GeoSeries.from_xy([0, 1_000, 101_000], [0, 0, 0], index=['a', 'b', 'c'])
    stops = gpd.GeoDataFrame({'geometry': geometry}, index=geometry.index)
    stops.index.name = 'stop_id'
    return stops, stop_times


def test_a_single_fast_segment_drops_the_whole_trip_by_default():
    stops, stop_times = _too_fast_stop_times()

    out = gx.network._clean_stop_times(stops, stop_times, max_speed_kmph=120)

    assert out.empty


def test_max_speed_repair_factor_stretches_instead_of_dropping():
    stops, stop_times = _too_fast_stop_times()

    # 600 km/h is five times the 120 km/h limit, so a factor of 10 repairs rather than discards
    out = gx.network._clean_stop_times(stops, stop_times, max_speed_kmph=120,
                                       max_speed_repair_factor=10)

    assert len(out) == 2
    assert (out['speed'] <= 120 / 3.6 + 1e-9).all()
    # the slow hop is untouched, the impossible one is stretched to exactly the limit
    assert out['next_stop_travel_time'].iloc[0] == 600
    assert out['next_stop_travel_time'].iloc[1] == pytest.approx(100_000 / (120 / 3.6))


def test_max_speed_repair_factor_still_drops_genuinely_corrupt_trips():
    stops, stop_times = _too_fast_stop_times()

    # 600 km/h is well beyond twice the limit, so the trip is corrupt, not imprecise
    out = gx.network._clean_stop_times(stops, stop_times, max_speed_kmph=120,
                                       max_speed_repair_factor=2)

    assert out.empty
