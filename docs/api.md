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
                  crs=None,
                  date=None,
                  feed_ids=None,
                  max_headway=None,
                  max_speed_kmph=108,
                  min_segment_travel_time=None,
                  max_speed_repair_factor=1.0,
                  transfer_model='headway',
                  max_transfer_wait=None,
                  transfer_wait_quantile=None,
                  min_transfer_time=0)
```

Create transit network graph from GTFS file(s).

Nodes correspond to transit stops and edges to transit connections between stops.
Each node and each edge belongs to only a single transit route.
If multiple routes serve the same station, they are depicted as multiple nodes.
Edges for walking transfer between nearby nodes of different routes are added.
For each node, the average headway and the service frequency are calculated.


#### Parameters
* **gtfs_paths** : str or list
    > Paths to GTFS files.

* **route_types** : list, optional
    > List of transit route types to include in the graph. If None, all service types are included.

* **time_window** : list, optional
    > Pair of ISO 8601-formatted times to include services only within a time window.

* **agency_ids** : list, optional
    > List of agencies (according to agency.txt) whose transit services are to be included in the graph. If None, all agencies are included.

* **boundary** : shapely.geometry.Polygon, optional
    > Polygon to filter transit stops by. Expected in the same CRS as `crs`.

* **frac** : float, optional
    > Fraction, allowing to randomly sample a subset of transit routes to be included in the graph.

* **walk_transfer_max_distance** : int, optional
    > Maximum distance in meters to allow walking transfer between transit stops.

* **walk_speed_kmph** : int, optional
    > Assumed walking speed in km/h when calculating walking transfer times.

* **crs** : str, optional
    > Metric coordinate reference system (CRS) to project transit stops to. If None, appropriate CRS UTM zone is inferred from lat lon bounds.

* **date** : str, datetime.date or list, optional
    > Calendar date (`YYYY-MM-DD`) to take the timetable from. If None, the busiest date of each feed is used. A list of the same length as `gtfs_paths` selects a separate date per feed, which is useful when feeds cover different validity periods.

* **feed_ids** : list, optional
    > Identifiers used to namespace stop, route and trip IDs of each feed so that feeds with overlapping ID spaces can be combined. Defaults to the file name of each feed when more than one feed is given, and to no namespacing at all for a single feed.

* **max_headway** : int, optional
    > Upper bound in seconds for the average headway of a stop. Stops served only once within the time window have an undefined headway; with `max_headway` set they are kept and assigned this value, otherwise they are dropped from the graph.

* **max_speed_kmph** : int or dict, optional
    > Trips faster than this between two consecutive stops (as the crow flies) are treated as corrupt and removed. Pass a dict keyed by `route_type` with an optional `'default'` entry to allow high speed rail its 300 km/h while still catching broken bus timetables.

* **min_segment_travel_time** : int, optional
    > Seconds to assign to segments whose timetable gives them no travel time at all. Without it, every trip containing such a segment is dropped, which is costly on feeds that round departures to whole minutes. Trips that genuinely contradict themselves (arrival before departure) are always dropped.

* **max_speed_repair_factor** : float, optional
    > How far above `max_speed_kmph` a segment may be before its whole trip is discarded instead of repaired.
    >
    > Feeds that round departures to whole minutes make short hops look impossibly fast, and losing the entire trip over one of them can take a whole rural bus line out of the graph. With a factor of 2, say, a segment up to twice the speed limit has its travel time stretched to what the limit allows, while anything beyond that is still corrupt enough to take its trip with it. The default of 1.0 repairs nothing and removes every trip with a segment over the limit.

* **transfer_model** : {'headway', 'timetable'}, optional
    > How long changing between two routes takes.
    >
    > `'headway'` charges half the headway of the route being boarded, which assumes the two timetables are unrelated. That holds for urban networks but not for regional and long distance services, where connections are deliberately scheduled: it makes an hourly train cost 30 minutes to board even when it was timed to meet the one just arrived.
    >
    > `'timetable'` instead reads the wait off the timetable. For every arrival of the first route it finds the next departure of the second that is reachable after the walk, and summarises those waits.

* **max_transfer_wait** : int, optional
    > Upper bound in seconds on the waiting part of a transfer, applied under either model. Acts as the safety net for pairs of routes that happen never to connect on the modelled day.

* **transfer_wait_quantile** : float, optional
    > Which of a connection's observed waits to charge, between 0 (the best connection of the day) and 1 (the worst); the mean is used when this is None. Only applies to `transfer_model='timetable'`.
    >
    > The mean answers "I turn up at a random time". A traveller going somewhere instead picks a departure whose connections line up, which a static graph cannot represent directly -- a low quantile approximates it. Around 0.2 reproduces published journey times best.

* **min_transfer_time** : int, optional
    > Seconds that must be left between arriving and departing for a connection to count, on top of the walk. Guards against modelling platform changes as instantaneous.


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
