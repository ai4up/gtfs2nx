import pandas as pd
import geopandas as gpd
import networkx as nx
import momepy as mm
from shapely.geometry import LineString
from shapely.geometry import Point


def nodes_to_df(G):
    nodes, data = zip(*G.nodes(data=True))
    df = pd.DataFrame(data, index=nodes)

    return df


def nodes_to_gdf(G):
    crs = G.graph['crs']
    df = nodes_to_df(G)
    geom = gpd.points_from_xy(df['x'], df['y'])
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=crs)

    return gdf


def edges_to_df(G):
    if G.is_multigraph():
        raise Exception('This function does not support multigraphs.')

    u, v, data = zip(*G.edges(data=True))
    index = pd.MultiIndex.from_arrays([u, v], names=['u', 'v'])
    df = pd.DataFrame(data, index=index)
    
    return df


def edges_to_gdf(G):
    crs = G.graph['crs']
    df = edges_to_df(G)
    nodes_x = nx.get_node_attributes(G, 'x')
    nodes_y = nx.get_node_attributes(G, 'y')
    geom = [LineString((Point((nodes_x[u], nodes_y[u])), Point((nodes_x[v], nodes_y[v])))) for u, v in df.index]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=crs)

    return gdf


def graph_to_gdfs(G):
    return nodes_to_gdf(G), edges_to_gdf(G)


def closeness_centrality(G):
    # reverse directed graph to calculate closest path to all other nodes instead of from all other nodes
    G = G.copy()
    centrality = nx.closeness_centrality(G.reverse(), distance='weight', wf_improved=True)
    nx.set_node_attributes(G, centrality, 'centrality')
    return G


def local_closeness_centrality(G, radius):
    # reverse directed graph to calculate closest path to all other nodes instead of from all other nodes
    return mm.closeness_centrality(G.reverse(), name='local_centrality', radius=radius, distance='weight').reverse()
