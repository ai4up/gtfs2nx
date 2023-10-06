# Preprocessing public transportation networks for Graph Neural Networks (GNN) 

## How can the spatial relationships of public transportation networks be captured for machine learning applications?

In this article, we show how to create a **meaningful graph representation of a public transportation network for graph neural network (GNN) prediction tasks** using the [`gtfs2nx`](https://github.com/ai4up/gtfs2nx) Python package.

<!-- `gtfs2nx` is a Python package to **create realistic transit graphs** suitable for graph neural network (GNN) applications. -->


## Problem background
Graph neural networks (GNNs) are great for exploiting spatial relationships in data for prediction tasks. The underlying mechanism is called *[Message Passing](https://en.wikipedia.org/wiki/Graph_neural_network)*, which enables information exchange and aggregation among nodes in a graph.
When using GNNs in the context of public tranportation networks, we want to encode which transit stops are connected so that message passing follows the actual transit routes. Ideally, we also want to encode how well these stops are connected to differentiate for example between rapid rail and bus routes.

Here, [`gtfs2nx`](https://github.com/ai4up/gtfs2nx) can help. It creates a graph representation of a transit network with consecutively served stops being connected by edges whose weights represent actual average travel times. Optionally, walking transfers can be included to support realistic routing and travel time computation through the entire network.

<!-- > ### What is [`gtfs2nx`](https://github.com/ai4up/gtfs2nx)?
>
> [`gtfs2nx`](https://github.com/ai4up/gtfs2nx) is a small Python package to create routable [NetworkX](https://github.com/networkx/networkx) graphs from [GTFS](https://developers.google.com/transit/gtfs/) feeds. What makes it special is that it determines realistic transfer times between routes so that average travel times can be calculated across the entire network. -->

## Structure
In the following, we show how public transportation networks can be preprocessed for GNN prediction tasks using [`gtfs2nx`](https://github.com/ai4up/gtfs2nx). We illustrate this with a brief example of a route type (bus, metro, train, etc.) classification problem for the city of Toulouse, France.
1. Creating a [NetworkX](https://github.com/networkx/networkx) graph from a [GTFS](https://developers.google.com/transit/gtfs/) feed
1. Converting the [NetworkX](https://github.com/networkx/networkx) graph to a [PyTorch Geometric Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html) object
1. Demo: Classifying route types using [PyTorch GCN](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html)

<!-- It preprocesses transit networks in a way that stops are connected based on actual average travel times including walking transfers. -->
<!-- The result is a more accurate and realistic representation of the transit network. -->

> ## A bit of context
>
> Public transportation networks are vital to urban mobility. Analyzing them can help us understand how a city functions and reveal pressing issues related to sustainable and equitable mobility. Machine learning can help here to predict travel times and mode choice, estimate accessibility, and infer missing data.


## Step 1: Creating a [NetworkX](https://github.com/networkx/networkx) graph from a [GTFS](https://developers.google.com/transit/gtfs/) feed

Many transit provider publish their schedules as so-called [GTFS](https://developers.google.com/transit/gtfs/) feeds, a common format for public transportation routes and schedules.
The [`gtfs2nx`](https://github.com/ai4up/gtfs2nx) Python package can convert these GTFS feeds into [NetworkX](https://github.com/networkx/networkx) graphs that can be used for network analysis, but also for GNN prediction tasks.
<!-- Further below, we will exemplify how the graph can easily be converted to a PyTorch Geometric Data object with meaningful edge weights / attributes. -->


### 1.1 GTFS feed download
GTFS feeds can often be downloaded from the transit provider directly or from archival websites like https://transitfeeds.com that collect and partially harmonize feeds. We here download the transit network of Toulouse, France as an example:
```bash
wget -O gtfs-toulouse-example.zip https://data.toulouse-metropole.fr/api/v2/catalog/datasets/tisseo-gtfs/files/fc1dda89077cf37e4f7521760e0ef4e9
```

### 1.2 Package installation
The package can be installed with pip from GitHub:
```bash
pip install git+https://github.com/ai4up/gtfs2nx@v0.1.0
```

### 1.3 Graph creation
A NetworkX graph for the transit network including walking transfers can be created as follows:
```Python
import gtfs2nx as gx

G = gx.transit_graph('path/to/GTFS-feed.zip')
```

If one is interested only in specific operation times, transport modes (e.g. busses), or operators or wants to customize walking transfers, additional options can be specified:
```Python
G = gx.transit_graph(
    gtfs_paths='path/to/GTFS-feed.zip',
    time_window=('06:00', '08:00'),
    agency_ids=['network:1'], # subset of operators within the transport association
    route_types=[715], # only buses
    walk_transfer_max_distance=400, # allow transfers with long walking distance
    walk_speed_kmph=5, # fast walker
)
```

Please refer to the [API docs](./docs/api.md) for more details and the [getting-started notebook](./docs/getting_started.ipynb) for a small hands-on demo.


> ### What is happening under the hood when calling `gx.transit_graph`?
> * Cleaning & preprocessing
>    * Remove trips with unrealistic travel times and speeds
>   * Fix trips contradictory stop sequences and departure times
> * Enable routability
>     * Ensure each node belongs only to single route
>     * Calculate average segment travel times
>     * Calculate average stop headway and service frequency
>     * Add edges for walking transfer between routes with realistic transfer time (walking time + headway / 2)




## Step 2: Converting the [NetworkX](https://github.com/networkx/networkx) graph to a [PyTorch Geometric Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html) object

<!-- For a reasonable message passing in GNNs, the edges weights are important. As [`gtfs2nx`](https://github.com/ai4up/gtfs2nx) allows the computation of realistic edge travel times, it is a great tool for preprocessing transportation networks for graph-based machine learning purposes. We want to highlight two possible approaches how a public transportation NetworkX graph can be converted to a [PyTorch Geometric Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html) object: -->

The preprocessed [NetworkX](https://github.com/networkx/networkx) graph has node and edge attributes describing the route type, service frequency, headway, and segment travel time. For this example, we want to convert the [NetworkX](https://github.com/networkx/networkx) graph to a [PyTorch Geometric Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html) object using the frequency and headway as node features to predict our target variable, the route type, and using the segment travel times as edge weights for the message passing. This can be done in two ways:
1. Using the PyTorch helper function [`from_networkx`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.from_networkx):
    ```Python
    from torch_geometric.utils import from_networkx

    G.graph = {}
    graph = from_networkx(G, group_edge_attrs=['weight'], group_node_attrs=['frequency', 'headway'])
    graph.y = torch.tensor(nodes['route_type'].cat.codes.values, dtype=torch.long)
    ```

2. Manually, to allow for a more customized [PyTorch Geometric Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html) object:
    ```Python
    from torch_geometric.data import Data

    edges = utils.edges_to_gdf(G)
    nodes = utils.nodes_to_gdf(G)

    # Preprocess route types
    nodes['route_type'] = nodes['route_type'].astype('category')
    nodes = nodes.dropna(subset=['route_type'])

    # Remove walk edges (optionally)
    edges = edges[edges['mode'] != 'walk']

    # Create edge index for torch data graph
    edges = edges[edges.index.get_level_values('u').isin(nodes.index) & edges.index.get_level_values('v').isin(nodes.index)]
    mapping = dict(zip(nodes.index, range(len(nodes))))
    edge_index = torch.empty((2, len(edges)), dtype=torch.long)
    edge_weight = torch.tensor(edges['weight'].values, dtype=torch.float32)

    for i, (src, dst) in enumerate(edges.index):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    # Create torch data graph
    y = torch.tensor(nodes['route_type'].cat.codes.values, dtype=torch.long)
    x = torch.tensor(nodes[['frequency', 'headway']].values, dtype=torch.float32)
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight)
    ```

## Step 3: Demo - Using graph neural networks (GNN) to classify transit route types

To highlight how the [PyTorch Geometric Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html) object can be used for GNN prediction tasks, we classify the route type (bus, metro, train, etc.) of each node. See [GitHub Gist](https://gist.github.com/FlorianNachtigall/9df1c9f7417aa512220756a35c36b45f) for the complete notebook.


```Python
class GCN(nn.Module):
    def __init__(self, n_fts, n_classes):
        super().__init__()
        self.conv1 = GCNConv(n_fts, 16)
        self.conv2 = GCNConv(16, n_classes)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        output = self.conv2(x, edge_index, edge_weight=edge_weight)

        return output
```

```Python
def train_node_classifier(model, graph, optimizer, criterion, n_epochs):
    progress = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        acc = eval_node_classifier(model, graph, graph.val_mask)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')
        
        progress.append((float(loss), acc))

    return model, progress


def eval_node_classifier(model, graph, mask):
    model.eval()
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())

    return acc
```
```Python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_fts = data.x.shape[1]
n_classes = torch.unique(data.y).size(dim=1)

gcn = GCN(n_fts, n_classes).to(device)
optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

gcn, progress = train_node_classifier(gcn, data, optimizer_gcn, criterion, n_epochs=100)
test_acc = eval_node_classifier(gcn, graph, graph.test_mask)
```

## Take-away
For reasonable message passing in GNNs, edge weights are important. As [`gtfs2nx`](https://github.com/ai4up/gtfs2nx) allows the computation of realistic edge travel times, it is a great tool for preprocessing transportation networks for graph-based machine learning purposes. 

## Further use cases

* [Using graph neural networks to classify missing GTFS route types](https://gist.github.com/FlorianNachtigall/9df1c9f7417aa512220756a35c36b45f)
* [Validating NetworkX transit graph: Comparing routes and travel times to Google Maps](https://gist.github.com/FlorianNachtigall/3e0d2f5e4fa8b2e893a29445a99dfb4f)

