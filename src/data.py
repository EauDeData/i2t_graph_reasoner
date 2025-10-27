import os.path

import networkx as nx
import torch
import tqdm
import json
import numpy as np
import itertools

import json
import torch
from networkx.classes import edges

class HMMMKGDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for handling multimodal reasoning samples with image graphs and option graphs.

    The dataset expects a JSON file with the following structure:

    {
        "features_paths": [
            {
                "Image_graph_file": "path/to/image_graph.json",
                "option_graph_files": {
                    "option_1": "path/to/option1_graph.json",
                    "option_2": "path/to/option2_graph.json",
                    ...
                },
                "correct_option": {
                    "option_1": true,
                    "option_2": false,
                    ...
                }
            },
            ...
        ],
        "name_conversion": {
            "id_sanitized": "original_id",
            ...
        }
    }

    Each entry in "features_paths" contains:
        - An image graph file
        - A set of option graph files (one per answer option)
        - A dictionary marking which options are correct or incorrect

    This dataset builds training samples as triplets:
        (image_graph_path, correct_option_graph_path, incorrect_option_graph_path)

    For each sample, all incorrect options are paired with the correct option.
    """

    def __init__(self, path_to_training_samples: str, path_to_metadata: str,  blacklist_path: str = "black_list.txt"):
        """
        Initialize the dataset.

        Args:
            path_to_training_samples (str): Path to the JSON file with training samples.
        """
        self.blacklist_path = blacklist_path
        with open(path_to_training_samples, "r") as f:
            self.data = json.load(f)

        with open(path_to_metadata, "r") as f:
            metadata = json.load(f)['questions']

        self.av_options = {os.path.splitext(file_id['file_id'])[0]: set(file_id['options'].values()) for file_id in metadata}
        self.triplets = self._build_triplets()
        self.filter_blacklist()

    def filter_blacklist(self):
        # Filter blacklisted items
        if os.path.exists(self.blacklist_path):
            with open(self.blacklist_path, "r") as f:
                blacklisted = {tuple(line.strip().split(" | ")) for line in f if line.strip()}
            before = len(self.triplets)
            self.triplets = [t for t in self.triplets if t not in blacklisted]
            print(f"⚠️ Filtered out {before - len(self.triplets)} blacklisted triplets ({len(self.triplets)} remain).")


    def _build_triplets(self):
        """
        Build a list of triplets (image_graph, correct_option, incorrect_option).

        Returns:
            list of tuples: Each tuple is (image_graph_path, correct_option_graph_path, incorrect_option_graph_path).
        """
        triplets = []

        for sample in tqdm.tqdm(self.data["features_paths"], desc='building learning triplets....'):

            image_graph = sample["image_graph_file"]

            this_image_id = self.data['name_conversion'][image_graph.split('/')[-3]]
            options = self.av_options[this_image_id]
            self.av_options[image_graph] = options

            # Separate correct and incorrect options
            correct_id = [
                opt_id for opt_id, is_correct in sample["correct_option"].items() if is_correct
            ][0]
            incorrect_options = [
                opt_id for opt_id, is_correct in sample["correct_option"].items() if not is_correct
            ]

            # Build triplets
            if not correct_id in sample["options_graph_files"] or not len(incorrect_options):
                continue

            correct_path = sample["options_graph_files"][correct_id]
            for incorrect_id in incorrect_options:
                incorrect_path = sample["options_graph_files"][incorrect_id]
                triplets.append((image_graph, correct_path, incorrect_path))

        return triplets

    def __len__(self):
        """Return the number of triplets."""
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Retrieve one training triplet by index.

        Each triplet consists of:
            1. An image graph
            2. A correct option text graph
            3. An incorrect option text graph
            4. Two classification pairs:
                - Positive alignment: (image_node_id, text_node_id_correct)
                - Negative alignment: (image_node_id, text_node_id_incorrect)

        Args:
            idx (int):
                The index of the triplet to retrieve.

        Returns:
            tuple:
                - image_graph (nx.Graph):
                    A NetworkX graph representing the image.
                    * Exactly one node must have attribute `node_type="image"`.

                - correct_text_graph (nx.Graph):
                    A NetworkX graph representing the correct option.
                    * Exactly one node must have attribute `node_type="option"`.

                - incorrect_text_graph (nx.Graph):
                    A NetworkX graph representing the incorrect option.
                    * Exactly one node must have attribute `node_type="option"`.

                - nodes_to_classify_positive (tuple[str, str]):
                    A tuple `(image_node_id, text_node_id_correct)` indicating
                    the nodes that should be classified as aligned.

                - nodes_to_classify_negative (tuple[str, str]):
                    A tuple `(image_node_id, text_node_id_incorrect)` indicating
                    the nodes that should be classified as not aligned.
        """

        # Open using read graphml from NX
        image_graph_path, correct_query_path, incorrect_query_path = self.triplets[idx]

        image_graph = nx.read_graphml(image_graph_path)
        correct_text_graph = nx.read_graphml(correct_query_path)
        incorrect_text_graph = nx.read_graphml(incorrect_query_path)

        # Extract node ids
        image_node_id = [n for n, d in image_graph.nodes(data=True) if d.get("node_type") == "image"][0]
        try:
            correct_node_id = [n for n, d in correct_text_graph.nodes(data=True) if d.get("node_type") == "option"]
        except IndexError:
            # En el pitjor dels casos doncs mira node random
            correct_node_id = [n for n, d in correct_text_graph.nodes(data=True) if n in self.av_options[image_graph_path]]
            if not len(correct_node_id):
                correct_node_id =  [n for n, d in correct_text_graph.nodes(data=True)]
        correct_node_id = correct_node_id[0]

        try:
            incorrect_node_id = [n for n, d in incorrect_text_graph.nodes(data=True) if d.get("node_type") == "option"]
        except IndexError:
            # En el pitjor dels casos doncs mira node random
            incorrect_node_id = [n for n, d in incorrect_text_graph.nodes(data=True) if n in self.av_options[image_graph_path]]
            if not len(incorrect_node_id):
                incorrect_node_id =  [n for n, d in incorrect_text_graph.nodes(data=True)]

        incorrect_node_id = incorrect_node_id[0]


        nodes_to_classify_positive = (image_node_id, correct_node_id)
        nodes_to_classify_negative = (image_node_id, incorrect_node_id)

        # Ego graph of image_graph around image_node_id
        #image_graph = nx.ego_graph(image_graph, image_node_id, radius=2)

        # Ego graph of correct_text_graph around correct_node_id
        # correct_text_graph = nx.ego_graph(correct_text_graph, correct_node_id, radius=2)

        # Ego graph of incorrect_text_graph around incorrect_node_id
        # incorrect_text_graph = nx.ego_graph(incorrect_text_graph, incorrect_node_id, radius=2)

        return image_graph, correct_text_graph, incorrect_text_graph, nodes_to_classify_positive, nodes_to_classify_negative

class HMMKFDatasetEval(HMMMKGDataset):
    def filter_blacklist(self):
        # Blacklisting is done onload of the triplets
        return None

    def _build_triplets(self):

        if os.path.exists(self.blacklist_path):
            with open(self.blacklist_path, "r") as f:
                blacklisted = set(sum([list(line.strip().split(" | ")) for line in f if line.strip()], start = []))


        triplets = []

        for sample in tqdm.tqdm(self.data["features_paths"], desc='building learning triplets....'):
            this = {}

            image_graph = sample["image_graph_file"]
            if image_graph in blacklisted:
                continue

            this['image_graph'] = image_graph

            this_image_id = self.data['name_conversion'][image_graph.split('/')[-3]]
            options = self.av_options[this_image_id]
            self.av_options[image_graph] = options

            # Separate correct and incorrect options
            correct_id = [
                opt_id for opt_id, is_correct in sample["correct_option"].items() if is_correct
            ][0]
            incorrect_options = [
                opt_id for opt_id, is_correct in sample["correct_option"].items() if not is_correct
            ]

            # Build triplets
            if not correct_id in sample["options_graph_files"] or not len(incorrect_options):
                continue

            correct_path = sample["options_graph_files"][correct_id]
            if correct_path in blacklisted:
                continue
            this['correct_path'] = correct_path

            incorrects = []
            for incorrect_id in incorrect_options:
                incorrect_path = sample["options_graph_files"][incorrect_id]
                if incorrect_path in blacklisted: continue
                incorrects.append(incorrect_path)
            this['incorrect_paths'] = incorrects
            if len(this['incorrect_paths']):
                triplets.append(this)

        return triplets
    def create_blacklist(self):
        # if os.path.exists(self.blacklist_path):
        #     return None
        black_list = set()
        for i in tqdm.tqdm(range(len(self)), 'creating test blacklist...'):
            try:
                _ = self[i]
            except:
                triplet = self.triplets[i]
                entry = " | ".join(triplet['incorrect_paths'] + [triplet['image_graph'], triplet['correct_path']])
                if entry not in black_list:
                    black_list.add(entry)
                    print('wrote something')
                    with open(self.blacklist_path, "a+") as f:
                        f.write(entry + "\n")



    def __getitem__(self, idx):
        # Open using read graphml from NX
        sample = self.triplets[idx]

        image_graph_path = sample['image_graph']
        correct_query_path = sample['correct_path']

        image_graph = nx.read_graphml(image_graph_path)
        correct_text_graph = nx.read_graphml(correct_query_path)

        incorrect_text_graphs = [nx.read_graphml(incorrect_query_path) for incorrect_query_path in
                                 sample['incorrect_paths']]

        # Extract node ids
        image_node_id = [n for n, d in image_graph.nodes(data=True) if d.get("node_type") == "image"][0]

        # --- Correct text graph ---
        try:
            correct_node_id = [n for n, d in correct_text_graph.nodes(data=True) if d.get("node_type") == "option"]
        except IndexError:
            correct_node_id = [n for n, d in correct_text_graph.nodes(data=True) if
                               n in self.av_options[image_graph_path]]
            if not len(correct_node_id):
                correct_node_id = [n for n, d in correct_text_graph.nodes(data=True)]
        correct_node_id = correct_node_id[0]

        # --- Incorrect text graphs ---
        incorrect_node_ids = []
        for incorrect_text_graph in incorrect_text_graphs:
            try:
                node_ids = [n for n, d in incorrect_text_graph.nodes(data=True) if d.get("node_type") == "option"]
            except IndexError:
                node_ids = [n for n, d in incorrect_text_graph.nodes(data=True) if
                            n in self.av_options[image_graph_path]]
                if not len(node_ids):
                    node_ids = [n for n, d in incorrect_text_graph.nodes(data=True)]
            incorrect_node_ids.append(node_ids[0])

        # Now we have one positive and multiple negatives
        nodes_to_classify_positive = (image_node_id, correct_node_id)
        nodes_to_classify_negatives = [(image_node_id, inc_id) for inc_id in incorrect_node_ids]

        return (
            image_graph,
            correct_text_graph,
            incorrect_text_graphs,  # now a list
            nodes_to_classify_positive,
            nodes_to_classify_negatives  # list of tuples
        )


class Collator:

    def __init__(self, to_directed):

        def _build_edge_type_mapping(num_node_types: int = 8):
            """
            Build deterministic mapping between node type pairs and edge type indices.

            Returns:
                edge_type_dict (dict): { (i, j): edge_type_idx }
                edge_type_list (list): [ (i, j), ... ] in deterministic order
            """
            # symmetric combinations: (0,0), (0,1), ..., (0,7), (1,1), (1,2), ..., (7,7)
            edge_type_pairs = [(i, j) for i in range(num_node_types) for j in range(num_node_types)]

            # deterministic mapping
            edge_type_dict = {pair: idx for idx, pair in enumerate(edge_type_pairs)}

            return edge_type_dict, edge_type_pairs

        def _build_directionality_constraints(edge_type_dict):

            types = {'content': [0, 1], 'observation': [2, 3, 4, 5], 'context': [6, 7]}
            allowed = [('content', 'content'), ('observation', 'content'), ('observation', 'observation'),
                       ('context', 'observation'), ('context', 'context')]
            allowed_with_idxs = []
            for src, dst in allowed:
                for idx_src in types[src]:
                    for idx_dst in types[dst]:
                        allowed_with_idxs.append(edge_type_dict[(idx_src, idx_dst)])
            return allowed_with_idxs

        print('Using directed graph is:', to_directed)
        self.edge_type_dict, self.edge_type_list = _build_edge_type_mapping()
        self.directed_allowed_edges = _build_directionality_constraints(self.edge_type_dict)
        self.convert_to_directed = to_directed


    def collate_fn(self, batch):
        """
        Collate function for batching graph triplets.

        Steps:
            1. Join all graphs from the batch into one large graph via nx.compose.
            2. Collect all nodes into a global ordered list.
            3. Build a feature tensor from the `features` attribute of each node.
            4. Convert the positive and negative alignment pairs into index-based tuples.
            5. Convert the composed NetworkX graph into a PyTorch-Geometric format.

        Args:
            batch (list):
                A list of samples, where each sample is:
                (image_graph, correct_text_graph, incorrect_text_graph,
                 nodes_to_classify_positive, nodes_to_classify_negative)

        Returns:
            dict:
                {
                    "edges": PyG edge format,
                    "features": Tensor [N, F],
                    "positive_idxs": list[(int, int)],
                    "negative_idxs": list[(int, int)]
                }
        """
        # 1. Compose all graphs into one
        composed_graph = nx.Graph()
        for (image_graph, correct_graph, incorrect_graph, _, _) in batch:
            composed_graph = nx.compose(composed_graph, image_graph)
            composed_graph = nx.compose(composed_graph, correct_graph)
            composed_graph = nx.compose(composed_graph, incorrect_graph)



        # 2. Global ordered list of nodes
        node_list = list(composed_graph.nodes())

        # Map node -> index
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        # 3. Collect features
        features = []
        for node in node_list:

            feat = json.loads(composed_graph.nodes[node]['features'])
            features.append(torch.tensor(feat, dtype=torch.float))

        features = torch.stack(features, dim=0)  # [N, F]

        # 4. Collect positive & negative pairs
        positive_idxs = []
        negative_idxs = []

        for (_, _, _, pos_pair, neg_pair) in batch:
            pos_idx = (node_to_idx[pos_pair[0]], node_to_idx[pos_pair[1]])
            neg_idx = (node_to_idx[neg_pair[0]], node_to_idx[neg_pair[1]])
            positive_idxs.append(pos_idx)
            negative_idxs.append(neg_idx)

        edges = [[node_to_idx[nin], node_to_idx[nout]] for nin, nout in composed_graph.edges()]
        edges = edges + [x[::-1] for x in edges] # We need to do it symetric

        edge_types = []
        filtered_edges = []

        for src, dst in edges:
            src_type = torch.argmax(features[src, -8:]).item()
            dst_type = torch.argmax(features[dst, -8:]).item()
            node_pair = (src_type, dst_type)  # ensure symmetry
            edge_type_idx = self.edge_type_dict[node_pair]

            # ✅ Only keep if it's an allowed edge type (when converting to directed)
            if not self.convert_to_directed or edge_type_idx in self.directed_allowed_edges:
                filtered_edges.append([src, dst])
                edge_types.append(edge_type_idx)

        # Replace edges with filtered list
        edges = filtered_edges

        return {
            'edges': torch.tensor(edges).T,           # [2, E]
            'edge_types': torch.tensor(edge_types),   # [E]
            'node_features': features,                # [N, F]
            'true_pairs': np.array(positive_idxs),
            'negative_pairs': np.array(negative_idxs)
        }

    def eval_collate_fn(self, batch):

        def _build_graph_data(graph, pair):
            """Helper to build node features, edges, and pair indices from a merged graph."""
            node_list = list(graph.nodes())
            node_to_idx = {n: i for i, n in enumerate(node_list)}

            # Features
            features = []
            for n in node_list:
                feat = json.loads(graph.nodes[n]['features'])
                features.append(torch.tensor(feat, dtype=torch.float))
            features = torch.stack(features, dim=0)  # [N, F]

            # Edges (make symmetric)
            edges = [[node_to_idx[a], node_to_idx[b]] for a, b in graph.edges()]
            edges = edges + [x[::-1] for x in edges]

            edge_types = []
            filtered_edges = []

            for src, dst in edges:
                src_type = torch.argmax(features[src, -8:]).item()
                dst_type = torch.argmax(features[dst, -8:]).item()
                node_pair = (src_type, dst_type)  # ensure symmetry
                edge_type_idx = self.edge_type_dict[node_pair]

                # ✅ Only keep if it's an allowed edge type (when converting to directed)
                if not self.convert_to_directed or edge_type_idx in self.directed_allowed_edges:
                    filtered_edges.append([src, dst])
                    edge_types.append(edge_type_idx)

            # Replace edges with filtered list
            edges = filtered_edges

            edges = torch.tensor(edges, dtype=torch.long).T  # [2, E]

            # Pair (convert node names to indices)
            pair_idx = (node_to_idx[pair[0]], node_to_idx[pair[1]])

            return {
                "graph": graph,
                "edges": edges,
                "node_features": features,
                "pair": pair_idx,
                "node_to_idx": node_to_idx,
                'edge_types': torch.tensor(edge_types),   # [E]

            }

        # The block of code above is indented; it is a private function; this is the actual return of the collator_fn
        """
        Evaluation collate function — builds per-sample merged graphs:
            [[pos_graph, neg_graph1, neg_graph2, ...], ...]

        Each merged graph contains the image graph + one text graph (correct or incorrect).
        """

        all_results = []

        for image_graph, correct_text_graph, incorrect_text_graphs, pos_pair, neg_pairs in batch:
            sample_graphs = []  # [positive_merged_graph, neg_graph1, ...]

            # --- Positive merged graph ---
            pos_graph = nx.compose(image_graph, correct_text_graph)
            pos_graph_data = _build_graph_data(pos_graph, pos_pair)
            sample_graphs.append(pos_graph_data)

            # --- Negative merged graphs ---
            for incorrect_graph, neg_pair in zip(incorrect_text_graphs, neg_pairs):
                neg_graph = nx.compose(image_graph, incorrect_graph)
                neg_graph_data = _build_graph_data(neg_graph, neg_pair)
                sample_graphs.append(neg_graph_data)

            all_results.append(sample_graphs)

        return all_results

def get_blacklist_from_dataset(dataset, BLACKLIST_PATH):

    # Load previous blacklist if exists
    if os.path.exists(BLACKLIST_PATH):
        with open(BLACKLIST_PATH, "r") as f:
            black_list = set(line.strip() for line in f if line.strip())
    else:
        black_list = set()

    print(f"Loaded {len(black_list)} blacklisted items.")

    for idx in tqdm.tqdm(range(len(dataset)), desc="Checking dataset..."):
        try:
            _ = dataset[idx]
        except Exception as e:
            triplet = dataset.triplets[idx]
            entry = " | ".join(triplet)
            if entry not in black_list:
                black_list.add(entry)
                with open(BLACKLIST_PATH, "a") as f:
                    f.write(entry + "\n")

            print(f"\n⚠️ Error on sample {idx}:\n{triplet}")


    print(f"\n✅ Done. Total blacklisted items: {len(black_list)}")
    return dataset

if __name__ == '__main__':
    data = HMMMKGDataset('/data/users/amolina/hmmkgv2/options/features_tmp.json', '/data/users/amolina/hmmkgv2/options/hmmkg_dataset_harder.json')
    print( data[0] )
    print(len( data ))

    from torch.utils.data import DataLoader
    dl = DataLoader(data, batch_size=6, num_workers=12, collate_fn= Collator(False).collate_fn, shuffle=False)
    for batch in tqdm.tqdm(dl):
        edges, nodes, true_pairs, negative_pairs = batch['edges'], batch['node_features'], batch['true_pairs'], batch['negative_pairs']
        edges.cuda()
        nodes.cuda()
        print(edges.shape, nodes.shape)
        exit()