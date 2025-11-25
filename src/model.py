import torch
import torch_geometric
from typing import *
import torch.nn.functional as F

class ScorerFunctions:
    def __init__(self, scorer, sckw):
        self.sc = scorer
        self.kwargs = sckw

    @property
    def logistic(self):
        return ConcatAndLogisticScorer(self.sc.feature_size, **self.kwargs)

    @property
    def triplet(self):
        # TODO: Explore the margin
        return TripletLossScorer()

class NoLayersProxy:
    def __init__(self):
        return None
    def __call__(self, **kwargs):
        return kwargs.get('x')

class ConcatAndLogisticScorer(torch.nn.Module):
    def __init__(self, feature_size, use_weights=False, device = 'cuda', **kwargs):
        super(ConcatAndLogisticScorer, self).__init__()
        self.classifier = torch.nn.Sequential(torch.nn.Linear(feature_size * 2, feature_size), torch.nn.ReLU(), torch.nn.Linear(feature_size, 1), torch.nn.Sigmoid())

        num_options = 10
        self.potential_weights = torch.tensor([1 - ((num_options - 1) / num_options), ((num_options - 1) / num_options)], device=device)
        self.bce = torch.nn.BCELoss(reduction = 'mean' if not use_weights else 'none')

        if not use_weights:
            self.loss_f = self.bce
        else: self.loss_f = self._bce_with_weights

    def _bce_with_weights(self, logits, labels):
        """
        Compute BCE-with-logits using optional class weighting.
        Args:
            logits: Tensor of raw model outputs (before sigmoid)
            labels: Tensor of 0/1 labels
        Returns:
            Scalar loss (mean if reduction == 'mean')
        """
        # Apply BCE with logits (numerically stable)
        loss = self.bce(logits, labels)

        # Apply class weights based on labels (0 → potential_weights[0], 1 → potential_weights[1])
        weights = torch.where(labels.bool(),
                              self.potential_weights[1],
                              self.potential_weights[0])

        loss = loss * weights

        return loss.mean()  # Or .sum() depending on your training setup

    def forward(self, positive_anchors, negative_anchors, positive_texts, negative_texts, compute_loss = True):
        # A partir de les parelles si cals dupliques els anchors per seguretat no sé
        positive_vector = torch.concat((positive_anchors, positive_texts), dim=-1)
        neagtive_vector = torch.concat((negative_anchors, negative_texts), dim=-1)

        logits = torch.concat((positive_vector, neagtive_vector), dim=0)
        logits = self.classifier(logits)

        if compute_loss:

            labels = torch.concat([torch.ones(len(positive_vector), dtype=torch.float32), torch.zeros(len(neagtive_vector), dtype=torch.int32)], dim=0)[:, None]
            labels = labels.to(logits.device)

            return positive_vector, neagtive_vector, logits, self.loss_f(logits, labels)

        return positive_vector, neagtive_vector, logits

class TripletLossScorer(torch.nn.Module):
    def __init__(self, margin=1.0, **kwargs):
        super(TripletLossScorer, self).__init__()
        self.margin = margin

    def forward(self, positive_anchors, negative_anchors, positive_texts, negative_texts, compute_loss=True):
        # Compute cosine similarities (only diagonal elements)
        # Normalize embeddings
        positive_anchors_norm = F.normalize(positive_anchors, p=2, dim=1)
        positive_texts_norm = F.normalize(positive_texts, p=2, dim=1)
        negative_anchors_norm = F.normalize(negative_anchors, p=2, dim=1)
        negative_texts_norm = F.normalize(negative_texts, p=2, dim=1)
        
        # Element-wise multiplication and sum along feature dimension
        positive_cosine_similarities = (positive_anchors_norm * positive_texts_norm).sum(dim=1)
        negative_cosine_similarities = (negative_anchors_norm * negative_texts_norm).sum(dim=1)

        if compute_loss:
            # Use triplet loss: anchor, positive, negative
            # We need to align the batch dimensions
            loss = torch.clamp(((1 - positive_cosine_similarities) - (1 - negative_cosine_similarities)) + self.margin, 0).mean()
            
            return (positive_anchors_norm, positive_texts_norm), (negative_anchors, negative_texts_norm), (positive_cosine_similarities, negative_cosine_similarities), loss
        
        return (positive_anchors_norm, positive_texts_norm), (negative_anchors, negative_texts_norm), (positive_cosine_similarities, negative_cosine_similarities)
class ManyGraphLayers(torch.nn.Module):
    def __init__(self, layer_class, init_params: dict, num_layers: int):
        super().__init__()
        self.attr_list: List[str] = []
        for layer_idx in range(num_layers): 
            lname =  f"layer_{layer_idx}"           
            setattr(self, lname, layer_class(**init_params))
            self.attr_list.append(lname)

        
    def forward(self, **kwargs):
        x = kwargs.pop('x')
        for layer_name in self.attr_list:
            x = getattr(self, layer_name)(x, **kwargs)
        return x


class GATReasoningScorer(torch.nn.Module):
    def __init__(self, num_channels, num_layers, scorer_type='logistic', dropout = .1, scorer_kwargs: dict = dict(), graph_layer_type = 'gat'):
        super().__init__()
        self.feature_size = self.num_channels = num_channels
        self.num_layers = self.num_steps = self.num_messages = num_layers

        self.drop_edges = False
        if num_layers > 0:
            if graph_layer_type == 'gat':
                self.gat = torch_geometric.nn.GAT(in_channels=num_channels, hidden_channels=num_channels, out_channels=num_channels, num_layers=num_layers, v2=True, dropout=dropout)
                self.add_edge_types = False
            elif graph_layer_type == 'relational':
                self.gat = ManyGraphLayers(layer_class = torch_geometric.nn.RGATConv, init_params = {'in_channels': num_channels, 'out_channels': num_channels, 'num_relations': 36}, num_layers=num_layers)
                self.add_edge_types = True
                self.attr_name = 'edge_type'
                self.attr_processor = lambda x: x
            
            elif graph_layer_type == 'sage':
                self.gat =  ManyGraphLayers(layer_class = torch_geometric.nn.SAGEConv, init_params = {'in_channels': num_channels, 'out_channels': num_channels}, num_layers=num_layers)
                self.add_edge_types = False
            
            elif graph_layer_type == 'trf':
                heads = 8
                self.gat = ManyGraphLayers(layer_class = torch_geometric.nn.TransformerConv, init_params = {'in_channels': num_channels, 'out_channels': num_channels // heads, 'heads': heads, 'concat': True}, num_layers=num_layers)
                self.add_edge_types = False

            elif graph_layer_type == 'pathfinder':

                self.attr_name = 'edge_attr'
                self.add_edge_types = True
                self.attr_processor = torch.nn.Embedding(8**2, 1032)
                self.gat = ManyGraphLayers(layer_class = torch_geometric.nn.PDNConv, init_params = {'in_channels': num_channels, 'out_channels': num_channels, 'edge_dim': num_channels, 'hidden_channels': num_channels}, num_layers=num_layers)

            elif graph_layer_type == 'pna':
                raise NotImplementedError('PNA requires too many tweaks to work')

            elif graph_layer_type == 'grav':
                self.gat = ManyGraphLayers(layer_class = torch_geometric.nn.GravNetConv, init_params = {'in_channels': num_channels, 'out_channels': num_channels, 'space_dimensions': num_channels, 'propagate_dimensions': num_channels, 'k': 2}, num_layers=num_layers)
                self.add_edge_types = False
                self.drop_edges = True
        else:
            self.gat = NoLayersProxy()
            self.add_edge_types = False

        scorer_builder = ScorerFunctions(self, scorer_kwargs)
        if scorer_type == 'logistic':
            self.scorer = scorer_builder.logistic
        elif scorer_type == 'triplet':
            self.scorer = scorer_builder.triplet
        else:
            raise NotImplementedError(f"Not implemented scorer {scorer_type}")

    def forward(self, graph_data, batch):
        input_dictionary = {'x': graph_data.x, 'edge_index': graph_data.edge_index }
        if self.drop_edges:
            input_dictionary.pop('edge_index')
        if self.add_edge_types:
            input_dictionary[self.attr_name] = self.attr_processor(graph_data.edge_attr)
        post_gat_features = self.gat(
            **input_dictionary
        )

        true_pairs = batch.get('true_pairs')
        false_pairs = batch.get('negative_pairs')

        anchor_index_positives = post_gat_features[true_pairs[:, 0]]
        text_index_positives = post_gat_features[true_pairs[:, 1]]

        anchor_index_negatives = post_gat_features[false_pairs[:, 0]]
        text_index_negatives = post_gat_features[false_pairs[:, 1]]

        return self.scorer(anchor_index_positives, anchor_index_negatives, text_index_positives, text_index_negatives, compute_loss = batch.get('compute_loss'))

if __name__ == '__main__':
    import numpy as np

    # ---- Mock shapes ----
    N = 10  # number of nodes
    F = 4  # feature size
    E = 12  # number of edges
    P = 3  # number of positive/negative pairs
    device = 'cuda'

    # ---- Mock tensors ----
    edges = torch.randint(0, N, (2, E)).to(device)  # [2, E]
    node_features = torch.randn(N, F).to(device)   # [N, F]
    true_pairs = np.random.randint(0, N, (P, 2))  # list of (int, int)
    negative_pairs = np.random.randint(0, N, (P, 2))  # list of (int, int)

    example_batch = {
        "edges": edges,
        "node_features": node_features,
        "true_pairs": true_pairs,
        "negative_pairs": negative_pairs,
        "compute_loss": True
    }

    model = GATReasoningScorer(4, 4, .1)
    model.to(device)

    print(model(**example_batch))
