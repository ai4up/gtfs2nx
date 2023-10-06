# Analyzing public transportation networks with [`gtfs2nx`](https://github.com/ai4up/gtfs2nx) and [NetworkX](https://github.com/networkx/networkx)

To perform network analysis of public transportation networks, a graph representation of the network is needed. `gtfs2nx` is a small Python package that can create a routable [NetworkX](https://github.com/networkx/networkx) graph for any public transportation network, enabling detailed network and accessibility analysis.

<!-- To perform network analysis of public transportation networks, a graph representation of the network is needed, where stops are represented as nodes and transit routes as edges. To enable realistic routing and travel time computation through the entire network, edge weights should represent average travel times and additional walking edges should be added to allow for transfers between transit routes. `gtfs2nx` is a small Python package that does exactly that and creates a routable [NetworkX](https://github.com/networkx/networkx) graph for any public transportation network. -->


## A bit of context
Public transportation networks are vital to urban mobility. Analyzing them can help us understand how a city functions and reveal pressing issues related to sustainable and equitable mobility. We can pinpoint transport poverty, unequal access to opportunities and network vulnerabilities.

## [`gtfs2nx`](https://github.com/ai4up/gtfs2nx): from [GTFS](https://developers.google.com/transit/gtfs/) feeds to routable [NetworkX](https://github.com/networkx/networkx) graphs
Many transit provider publish their schedules as so-called [GTFS](https://developers.google.com/transit/gtfs/) feeds, a common format for public transportation routes and schedules. While they are great for looking at individual departure times, they don't directly allow to analyze the transit network as a whole including its coverage, frequency and connectivity.

`gtfs2nx` can help here. It converts a [GTFS](https://developers.google.com/transit/gtfs/) feed into a [NetworkX](https://github.com/networkx/networkx) graph where stops are represented as nodes and transit routes as edges. What makes it special is that it adds edges for transfers between routes with weights representing the expected transfer time based on the walking distance and route headway. This enables the calculation of realistic travel times for the entire network.

<!-- In addition, the graph can easily be converted to a  [PyTorch Geometric Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html) object with meaningful edge weights to be used for Graph Neural Network (GNN) applications. -->

<!-- Especially for fostering sustainable and equitable mobility  reveal transport poverty -->


<!-- ## What is `gtfs2nx`?
`gtfs2nx` is a small Python package to create routable [NetworkX](https://github.com/networkx/networkx) graphs from [GTFS](https://developers.google.com/transit/gtfs/) feeds. What makes it special is that it determines realistic transfer times between routes so that average travel times can be calculated across the entire network. -->

## Install
The package can be installed with pip from GitHub:
```bash
pip install git+https://github.com/ai4up/gtfs2nx@v0.1.0
```

## Usage
The package has one main method `gx.transit_graph` which creates a NetworkX graph for the transit network including walking transfers.
```Python
import gtfs2nx as gx

G = gx.transit_graph('path/to/GTFS-feed.zip')
```

If one is interested only in particular operation times, transport modes (e.g. busses), or operators or wants to customize walking transfers, additional options can be specified:
```Python
G = gx.transit_graph(
    gtfs_paths='path/to/GTFS-feed.zip',
    time_window=('06:00', '08:00'),
    agency_ids=[123, 124], # subset of operators within the transport association
    route_types=[700], # only buses
    walk_transfer_max_distance=400, # allow transfers with long walking distance
    walk_speed_kmph=5, # fast walker
)
```

What is happening under the hood when calling `gx.transit_graph`?
* Cleaning & preprocessing
    * Remove trips with unrealistic travel times and speeds
    * Fix trips contradictory stop sequences and departure times
* Enable routability
    * Ensure each node belongs only to single route
    * Calculate average segment travel times
    * Calculate average stop headway and service frequency
    * Add edges for walking transfer between routes with realistic transfer time (walking time + headway / 2)


Please refer to the [API docs](./docs/api.md) for more details and the [getting-started notebook](./docs/getting_started.ipynb) for a small hands-on demo.


## NetworkX graph inspection

Some attributes for each stop and network segment have already been calculated during the graph creation. They can be inspected as follows:
```Python
# access precomputed stop characteristics
frequency = nx.get_node_attributes(G, 'frequency')
headway = nx.get_node_attributes(G, 'headway')
travel_time = nx.get_edge_attributes(G, 'weight')
```

The graph can also easily be converted to a [Geopandas](https://github.com/geopandas/geopandas) GeoDataFrame for further inspection and manipulation:
```Python
nodes = gx.utils.nodes_to_gdf(G)
edges = gx.utils.edges_to_gdf(G)
```

The gtfs2nx package also offers some basic plotting functionality:
```Python
gx.utils.plot_network(G, attr='frequency')
```


## Networkx analysis

Onces a NetworkX graph has been created, one can perform any typical network analysis, such as creating connectivity and centrality metrics, determining the shortest path and travel time between two stops, and much more.

```Python
import networkx as nx

# routing
route = nx.shortest_path(G, from_stop, to_stop, weight='weight')

# travel time to other stops
travel_times = nx.single_source_dijkstra_path_length(G, source=from_stop, weight='weight')

# centrality analysis
centrality = nx.closeness_centrality(G, distance='weight')
```

## Graph neural networks (GNN) with PyTorch Geometric 

For a reasonable message passing in GNNs, the edges weights are important. As `gtfs2nx` allows the computation of realistic edge travel times, it is a great tool for preprocessing transportation networks for graph-based machine learning purposes. We want to highlight two possible approaches how a public transportation NetworkX graph can be converted to a [PyTorch Geometric Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html) object:

1. Using the PyTorch helper function `from_networkx`:
    ```Python
    from torch_geometric.utils import from_networkx

    G.graph = {}
    graph = from_networkx(G, group_edge_attrs=['weight'], group_node_attrs=['frequency', 'headway'])
    graph.y = torch.tensor(nodes['route_type'].cat.codes.values, dtype=torch.long)
    ```

2. Manually, to allow for a more customized PyTorch Geometric Data object:
    ```Python
    from torch_geometric.data import Data

    edges = utils.edges_to_gdf(G)
    nodes = utils.nodes_to_gdf(G)

    # Preprocess route types
    nodes['route_type'] = nodes['route_type'].astype('category')
    nodes = nodes.dropna(subset=['route_type'])

    # Remove walk edges (optionally)
    edges = edges[edges['mode'] != 'walk']

    # Create edge index for torch graph
    edges = edges[edges.index.get_level_values('u').isin(nodes.index) & edges.index.get_level_values('v').isin(nodes.index)]
    mapping = dict(zip(nodes.index, range(len(nodes))))
    edge_index = torch.empty((2, len(edges)), dtype=torch.long)
    edge_weight = torch.tensor(edges['weight'].values, dtype=torch.float32)

    for i, (src, dst) in enumerate(edges.index):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    # Create torch graph
    y = torch.tensor(nodes['route_type'].cat.codes.values, dtype=torch.long)
    x = torch.tensor(nodes[['frequency', 'headway']].values, dtype=torch.float32)
    graph = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight)
    ```
## Take-away
gtfs2nx allows to create routable NetworkX graphs with walk transfers and realistic average travel times from GTFS feeds. Meaningful segment travel times are essential for network analysis, for example, to obtain robust closeness centrality measures.

gtfs2nx allows to create realistic routable NetworkX graphs from GTFS feeds. The distinctive feature is that it supports walk transfers between routes resulting in realistic average travel time estimates. This in turn is important for network analysis, for example, to obtain robust closeness centrality measures.

## Further use cases

* [Using graph neural networks to classify missing GTFS route types](https://gist.github.com/FlorianNachtigall/9df1c9f7417aa512220756a35c36b45f)
* [Validating NetworkX transit graph: Comparing routes and travel times to Google Maps](https://gist.github.com/FlorianNachtigall/3e0d2f5e4fa8b2e893a29445a99dfb4f)

