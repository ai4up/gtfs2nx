import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.geometry import Point


def nodes_to_gdf(G):
    crs = G.graph['crs']
    nodes, data = zip(*G.nodes(data=True))

    geom = [Point(d['x'], d['y']) for d in data]
    gdf = gpd.GeoDataFrame(data, index=nodes, geometry=geom, crs=crs)

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
