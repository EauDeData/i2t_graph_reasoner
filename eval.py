import torch
from torch import optim
from torch.utils.data import DataLoader
import tqdm
from src.argparser import get_args
from src.data import HMMKFDatasetEval, Collator # Collator has now an eval function
from src.model import GATReasoningScorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score
import numpy as np
from torch_geometric.data import Data

@torch.no_grad()
def main():
    # -------------------------
    # Parse arguments
    # -------------------------
    args = get_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    # -------------------------
    # Initialize wandb
    # -------------------------
    wandb = None
    if args.use_wandb:
        import wandb
        wandb.init(project="evaluation_for_gat_reasoning", config=vars(args))

    dataset = HMMKFDatasetEval(args.path_to_training_samples, args.path_to_metadata, args.blacklist_path)
    # dataset.create_blacklist()
    col = Collator(to_directed=args.use_directed, bridge_ners_to_ocr = args.bridge_ocr_and_ners)
    print(f"Eval on batched: {args.eval_on_batched}")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn= col.eval_collate_fn if not args.eval_on_batched else col.eval_collate_fn_whole_batch
    )

    scorer_kwargs = {'use_weights': args.use_loss_weights, 'device': device}
    model = GATReasoningScorer(
        num_channels=args.num_channels,
        num_layers=args.num_layers,
        scorer_type=args.scorer_type,
        dropout=args.dropout,
        scorer_kwargs=scorer_kwargs,
        graph_layer_type=args.graph_type
    )
    if not isinstance(args.load_from, str):
        raise AssertionError('No has possat cap state dictionary per evaluar.')
    model.load_state_dict(torch.load(args.load_from))
    model.to(device)
    model.eval()

    results = []

    for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
        # Because batch_size=1, dataloader yields a list with one sample:
        #   batch = [ [pos_graph_dict, neg_graph1_dict, ...] ]

        if not args.eval_on_batched:

            for i in range(len(batch)):

                sample_graphs = batch[i]

                sample_scores = []
                sample_labels = []

                # Process each merged graph independently
                for i, graph_dict in enumerate(sample_graphs):
                    # Build model input dict
                    model_input = dict(edges=graph_dict['edges'].to(device),
                                       node_features=graph_dict['node_features'].to(device),
                                       true_pairs=torch.tensor([graph_dict['pair']], dtype=torch.long, device=device),
                                       negative_pairs=torch.empty((0, 2), dtype=torch.long, device=device),
                                       compute_loss=False,
                                       edge_types=graph_dict['edge_types'])

                    model_input['edges'] = model_input['edges'].to(device)
                    model_input['node_features'] = model_input['node_features'].to(device)
                    model_input['compute_loss'] = False
                    model_input['edge_types'] = model_input['edge_types'].to(device)

                    data = Data(x=model_input.pop('node_features'), edge_index=model_input.pop('edges'),
                                edge_attr=model_input.pop('edge_types'))

                    # Forward pass
                    _, _, score = model(data, model_input)

                    sample_scores.append(score)
                    sample_labels.append(1 if i == 0 else 0)  # first is positive, others are negatives

                results.append({
                    "scores": sample_scores,
                    "labels": sample_labels
                })
        else:


            batch_data = batch

            # Move to device
            node_features = batch_data['node_features'].to(device)
            edges = batch_data['edges'].to(device)
            edge_types = batch_data['edge_types'].to(device)
            pairs = batch_data['pairs'].to(device)
            labels = batch_data['labels'].to(device)

            batch_indices = batch_data['batch'].to(device)

            # Create PyG Data object
            data = Data(
                x=node_features,
                edge_index=edges,
                edge_attr=edge_types,
                batch=batch_indices
            )

            # Prepare model input (all pairs are "positive" since we're just scoring them)
            model_input = {
                'true_pairs': pairs,  # All pairs to score
                'negative_pairs': torch.empty((0, 2), dtype=torch.long, device=device),
                'compute_loss': False
            }

            # Forward pass - get scores for all pairs at once
            _, _, scores = model(data, model_input)


            scores_list = scores.cpu().tolist()
            results.append({
                "scores": scores_list,
                "labels": labels.cpu().tolist()
            })

    metrics_list = []
    for sample in tqdm.tqdm(results, 'calculating metrics...'):
        # Untensorize
        if args.eval_on_batched:
            # Scores and labels are already lists/numpy arrays
            scores = sample['scores']  # Already converted to list in batched mode
            labels = np.array(sample['labels'])
        else:
            # Scores are tensors, need to convert
            scores = [float(s.cpu().item()) for s in sample['scores']]
            labels = np.array(sample['labels'])

        # Sort by descending score
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = labels[sorted_indices]

        # Retrieval metrics
        recall_at_1 = 1.0 if 1 in sorted_labels[:1] else 0.0
        recall_at_3 = 1.0 if 1 in sorted_labels[:3] else 0.0
        recall_at_5 = 1.0 if 1 in sorted_labels[:5] else 0.0

        # Average Precision (using sklearn)
        ap = average_precision_score(labels, scores)

        # Classification metrics (assuming threshold = 0.5)
        preds = [1 if s == max(scores) else 0 for s in scores]
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)

        metrics_list.append({
            "Recall@1": recall_at_1,
            "Recall@3": recall_at_3,
            "Recall@5": recall_at_5,
            "Average Precision": ap,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec
        })

    # --- Average all metrics across samples ---
    avg_metrics = {
        k: np.mean([m[k] for m in metrics_list])
        for k in metrics_list[0].keys()
    }

    # --- Log to wandb ---
    if wandb:
        wandb.log(avg_metrics)

    print("Logged averaged metrics to wandb:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")
    return results


main()
