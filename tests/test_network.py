import networkx as nx
import pytest

import gtfs2nx as gx

SAMPLE_GTFS_FEED = 'tests/sample-feed.zip'


def test_e2e_transit_graph():
    G = gx.transit_graph(SAMPLE_GTFS_FEED)

    nodes = gx.utils.nodes_to_gdf(G)
    edges = gx.utils.edges_to_gdf(G)

    assert G.graph['crs'] == 'EPSG:32611'
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 4
    assert nodes['frequency'].notna().all()
    assert nodes['headway'].notna().all()
    assert edges['mode'].notna().all()
    assert edges['weight'].notna().all()
    assert 'walk' not in edges['mode'].values
    assert nx.shortest_path_length(G, nodes.index[0], nodes.index[1], weight='weight') == 420
    with pytest.raises(Exception):
        nx.shortest_path_length(G, nodes.index[0], nodes.index[-1], weight='weight')


def test_e2e_transit_graph_walk_edges():
    G = gx.transit_graph(SAMPLE_GTFS_FEED, walk_transfer_max_distance=20000)

    edges = gx.utils.edges_to_gdf(G)

    assert G.number_of_edges() == 10
    assert 'walk' in edges['mode'].values
