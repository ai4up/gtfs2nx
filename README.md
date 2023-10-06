# gtfs2nx

Create routable [NetworkX](https://github.com/networkx/networkx) graph with realistic transfer times from [GTFS](https://developers.google.com/transit/gtfs/) feeds. 🚌 🚆 🚡 


## How
* Cleaning & preprocessing
    * Remove trips with unrealistic travel times and speeds
    * Fix trips contradictory stop sequences and departure times
* Enable routability
    * Ensure each node belongs only to single route
    * Calculate average segment travel times
    * Calculate average stop headway and service frequency
    * Add edges for walking transfer between routes with realistic transfer time (walking time + headway / 2)


## Install
```bash
pip install git+https://github.com/ai4up/gtfs2nx@v0.2.0
```

## Usage
```Python
import gtfs2nx as gx

G = gx.transit_graph('path/to/GTFS-feed.zip')
```

Customize transit network:
```Python
G = gx.transit_graph(
    gtfs_paths='path/to/GTFS-feed.zip',
    time_window=('06:00', '08:00'),
    route_types=[700], # only buses
    walk_transfer_max_distance=400, # allow transfers with long walking distance
)
```

See the [API docs](./docs/api.md) for more details and the [getting-started notebook](./docs/getting_started.ipynb) for a small demo.


## Exemplary NetworkX analysis
```Python
import networkx as nx

# access precomputed stop characteristics
frequency = nx.get_node_attributes(G, 'frequency')

# routing
route = nx.shortest_path(G, from_stop, to_stop, weight='weight')

# travel time to other stops
travel_times = nx.single_source_dijkstra_path_length(G, source=from_stop, weight='weight')

# centrality analysis
centrality = nx.closeness_centrality(G, distance='weight')
```

## Further use cases

* [Using graph neural networks to classify missing GTFS route types](https://gist.github.com/FlorianNachtigall/9df1c9f7417aa512220756a35c36b45f)
* [Validating NetworkX transit graph: Comparing routes and travel times to Google Maps](https://gist.github.com/FlorianNachtigall/3e0d2f5e4fa8b2e893a29445a99dfb4f)


## Development
Build from source using [poetry](https://python-poetry.org/):
```
poetry build
pip install dist/gtfs2nx-*.whl
```

Configure post-checkout hook for branch specific .gitignore files:
```
git config --local core.hooksPath .githooks/
```
