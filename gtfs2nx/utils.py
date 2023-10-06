import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString


def nodes_to_df(G):
    """
    Convert DiGraph nodes to DataFrame.

    Parameters
    ----------
    G : networkx.DiGraph
        Transit network graph.

    Returns
    -------
    pandas.DataFrame
        DataFrame of nodes including node attributes.
    """
    nodes, data = zip(*G.nodes(data=True))
    df = pd.DataFrame(data, index=nodes)

    return df


def nodes_to_gdf(G):
    """
    Convert DiGraph nodes to GeoDataFrame.

    Parameters
    ----------
    G : networkx.DiGraph
        Transit network graph.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame of nodes including node attributes and geometries.
    """
    crs = G.graph['crs']
    df = nodes_to_df(G)
    geom = gpd.points_from_xy(df['x'], df['y'])
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=crs)

    return gdf


def edges_to_df(G):
    """
    Convert DiGraph edges to DataFrame.

    Parameters
    ----------
    G : networkx.DiGraph
        Transit network graph.

    Returns
    -------
    pandas.DataFrame
        DataFrame of edges including edge attributes.
    """
    if G.is_multigraph():
        raise Exception('This function does not support multigraphs.')

    u, v, data = zip(*G.edges(data=True))
    index = pd.MultiIndex.from_arrays([u, v], names=['u', 'v'])
    df = pd.DataFrame(data, index=index)

    return df


def edges_to_gdf(G):
    """
    Convert DiGraph edges to GeoDataFrame.

    Parameters
    ----------
    G : networkx.DiGraph
        Transit network graph.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame of edges including edge attributes and geometries.
    """
    crs = G.graph['crs']
    df = edges_to_df(G)
    nodes_x = nx.get_node_attributes(G, 'x')
    nodes_y = nx.get_node_attributes(G, 'y')
    geom = [LineString((Point((nodes_x[u], nodes_y[u])), Point((nodes_x[v], nodes_y[v])))) for u, v in df.index]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=crs)

    return gdf


def graph_to_gdfs(G):
    """
    Convert DiGraph nodes and edges to GeoDataFrames.

    Parameters
    ----------
    G : networkx.DiGraph
        Transit network graph.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame of nodes including node attributes and geometries.
    geopandas.GeoDataFrame
        GeoDataFrame of edges including edge attributes and geometries.
    """
    return nodes_to_gdf(G), edges_to_gdf(G)


def closeness_centrality(G):
    # reverse directed graph to calculate closest path to all other nodes instead of from all other nodes
    G = G.copy()
    centrality = nx.closeness_centrality(G.reverse(), distance='weight', wf_improved=True)
    nx.set_node_attributes(G, centrality, 'centrality')

    return G


def plot_network(G, attr, inc_walk_edges=False, adjust_linewith=True):
    """
    Plot DiGraph edges with node or edge attribute color coded.

    Parameters
    ----------
    G : networkx.DiGraph
        Transit network graph.
    attr : str
        Node or edge attribute to color code.
    inc_walk_edges : bool
        Include walking transfer edges in plot.
    adjust_linewith : bool
        Adjust edge linewidth based on attribute value.

    Returns
    -------
    matplotlib.axes.Axes
        Axis of plot.

    """
    edges = edges_to_gdf(G)
    nodes = nodes_to_df(G)

    if attr not in edges.columns:
        if pd.api.types.is_numeric_dtype(nodes[attr]):
            edges = _mean_node_attr(edges, nodes, attr)
        else:
            edges = _source_node_attr(edges, nodes, attr)

    numeric_attr = pd.api.types.is_numeric_dtype(edges[attr])

    if adjust_linewith and numeric_attr:
        attr_val = edges[edges['mode'] != 'walk'][attr]
        linewidth = _scale_to_range(attr_val, range=[0.1, 2], quantiles=[.05, .95])
    else:
        linewidth = None

    edges = edges.sort_values(attr, ascending=True)
    ax = edges[edges['mode'] != 'walk'].plot(column=attr, linewidth=linewidth, zorder=1, legend=True, legend_kwds={'shrink': 0.6} if numeric_attr else None)

    if inc_walk_edges:
        edges[edges['mode'] == 'walk'].plot(color='lightgrey', linewidth=0.2, zorder=0, ax=ax)

    ax.set_axis_off()
    ax.set_title(attr)

    return ax


def plot_route(G, from_node, to_node):
    """
    Plot shortest path between two stops and annonate route transfers.

    Parameters
    ----------
    G : networkx.DiGraph
        Transit network graph.
    from_node : str
        Stop ID of origin node.
    from_node : str
        Stop ID of destination node.

    Returns
    -------
    matplotlib.axes.Axes
        Axis of plot.

    """
    nodes, edges = graph_to_gdfs(G)

    route = nx.shortest_path(G, from_node, to_node, weight='weight')
    route_edges = edges.loc[list(zip(route, route[1:]))]
    route_transfers = route_edges[route_edges['mode'] == 'walk'].reset_index()

    ax = edges[edges['mode'] != 'walk'].plot(color='lightgrey')
    route_edges.plot(color='green', linewidth=2, ax=ax)
    route_transfers.plot(color='purple', linewidth=5, ax=ax)

    def _transfer_desc(x, nodes=nodes):
        start = nodes.loc[x['u']]
        end = nodes.loc[x['v']]
        duration = int(x['weight'] / 60)

        desc = "transfer "
        desc += f"from {start['route_short_name']} ({start['route_type']}) "
        desc += f"to {end['route_short_name']} ({end['route_type']}) "
        desc += f"[{duration} min]"

        return desc

    route_transfers.apply(lambda x: ax.annotate(text=_transfer_desc(x), xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
    ax.set_axis_off()

    return ax


def _scale_to_range(s, range, quantiles=None):
    if quantiles:
        vmin, vmax = s.quantile(quantiles)
        s = s.mask(s > vmax, vmax)
        s = s.mask(s < vmin, vmin)

    new_min, new_max = range
    old_min, old_max = s.min(), s.max()
    old_range = old_max - old_min
    new_range = new_max - new_min
    scaled = ((s - old_min) / old_range * new_range) + new_min

    return scaled


def _mean_node_attr(edges, nodes, attr):
    u = edges.index.get_level_values('u')
    v = edges.index.get_level_values('v')
    mean = (nodes.loc[u][attr].values + nodes.loc[v][attr].values) / 2
    edges[attr] = mean

    return edges


def _source_node_attr(edges, nodes, attr):
    u = edges.index.get_level_values('u')
    edges[attr] = nodes.loc[u][attr]

    return edges
