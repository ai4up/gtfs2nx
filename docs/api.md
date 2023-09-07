# Package API docs

* [gtfs2nx](#gtfs2nx)
  * [transit\_graph](#gtfs2nx.transit_graph)
* [gtfs2nx.utils](#gtfs2nx.utils)
  * [nodes\_to\_df](#gtfs2nx.utils.nodes_to_df)
  * [nodes\_to\_gdf](#gtfs2nx.utils.nodes_to_gdf)
  * [edges\_to\_df](#gtfs2nx.utils.edges_to_df)
  * [edges\_to\_gdf](#gtfs2nx.utils.edges_to_gdf)
  * [graph\_to\_gdfs](#gtfs2nx.utils.graph_to_gdfs)
  * [plot\_network](#gtfs2nx.utils.plot_network)
  * [plot\_route](#gtfs2nx.utils.plot_route)


<a id="gtfs2nx"></a>

## gtfs2nx

<a id="gtfs2nx.transit_graph"></a>

### transit\_graph


```python
def transit_graph(gtfs_paths,
                  route_types=None,
                  time_window=None,
                  agency_ids=None,
                  boundary=None,
                  frac=None,
                  walk_transfer_max_distance=200,
                  walk_speed_kmph=4,
                  crs=None)
```

Create transit network graph from GTFS file(s).

Nodes correspond to transit stops and edges to transit connections between stops.
Each node and each edge belongs to only a single transit route.
If multiple routes serve the same station, they are depicted as multiple nodes.
Edges for walking transfer between nearby nodes of different routes are added.
For each node, the global closeness centrality and the number of departures are calculated.


#### Parameters
* **gtfs_paths** : str or list
    > Paths to GTFS files.

* **route_types** : list, optional
    > List of transit route types to include in the graph. If None, all service types are include.

* **time_window** : list, optional
    > Pair of ISO 8601-formatted times to include services only within a time window

* **agency_ids** : list, optional
    > List of agencies (according to agency.txt) whose transit services are to be included n the graph. If None, all agencies are included.

* **boundary** : shapely.geometry.Polygon, optional
    > Polygon to filter transit stops by.
* **frac** : float, optional
    > Fraction, allowing to randomly sample a subset of transit routes to be included i the graph.

* **walk_transfer_max_distance** : int, optional
    > Maximum distance in meters to allow walking transfer between transit stops

* **walk_speed_kmph** : int, optional
    > Assumed walking speed in km/h when calculating walking transfer times

* **crs** : str, optional
    > Metric coordinate reference system (CRS) to project transit stops to. If None, apropriate CRS UTM zone is inferred from lat lon bounds.


#### Returns
* networkx.DiGraph
    > Directional transit network graph.


<a id="gtfs2nx.utils"></a>

## gtfs2nx.utils

<a id="gtfs2nx.utils.nodes_to_df"></a>

### nodes\_to\_df

```python
def nodes_to_df(G)
```

Convert DiGraph nodes to DataFrame.

#### Parameters
G : networkx.DiGraph
    Transit network graph.

#### Returns
* pandas.DataFrame
    > DataFrame of nodes including node attributes.

<a id="gtfs2nx.utils.nodes_to_gdf"></a>

### nodes\_to\_gdf

```python
def nodes_to_gdf(G)
```

Convert DiGraph nodes to GeoDataFrame.

#### Parameters
* **G** : networkx.DiGraph
    > Transit network graph.

#### Returns
* geopandas.GeoDataFrame
    > GeoDataFrame of nodes including node attributes and geometries.

<a id="gtfs2nx.utils.edges_to_df"></a>

### edges\_to\_df

```python
def edges_to_df(G)
```

Convert DiGraph edges to DataFrame.

#### Parameters
* **G** : networkx.DiGraph
    > Transit network graph.

#### Returns
* pandas.DataFrame
    > DataFrame of edges including edge attributes.

<a id="gtfs2nx.utils.edges_to_gdf"></a>

### edges\_to\_gdf

```python
def edges_to_gdf(G)
```

Convert DiGraph edges to GeoDataFrame.

#### Parameters
* **G** : networkx.DiGraph
    > Transit network graph.

#### Returns
* geopandas.GeoDataFrame
    > GeoDataFrame of edges including edge attributes and geometries.

<a id="gtfs2nx.utils.graph_to_gdfs"></a>

### graph\_to\_gdfs

```python
def graph_to_gdfs(G)
```

Convert DiGraph nodes and edges to GeoDataFrames.

#### Parameters
* **G** : networkx.DiGraph
    > Transit network graph.

#### Returns
* geopandas.GeoDataFrame
    > GeoDataFrame of nodes including node attributes and geometries.
geopandas.GeoDataFrame
    GeoDataFrame of edges including edge attributes and geometries.

<a id="gtfs2nx.utils.plot_network"></a>

### plot\_network

```python
def plot_network(G, attr=None, inc_walk_edges=False, adjust_linewith=True)
```

Plot DiGraph edges with node or edge attribute color coded.

#### Parameters
* **G** : networkx.DiGraph
    > Transit network graph.
* **inc_walk_edges** : bool
    > Include walking transfer edges in plot.
* **adjust_linewith** : bool
    > Adjust edge linewidth based on attribute value.

#### Returns
* matplotlib.axes.Axes
    > Axis of plot.

<a id="gtfs2nx.utils.plot_route"></a>

### plot\_route

```python
def plot_route(G, from_node, to_node)
```

Plot shortest path between two stops and annonate route transfers.

#### Parameters
* **G** : networkx.DiGraph
    > Transit network graph.
* **from_node** : str
    > Stop ID of origin node.
* **from_node** : str
    > Stop ID of destination node.

#### Returns
* matplotlib.axes.Axes
    > Axis of plot.
