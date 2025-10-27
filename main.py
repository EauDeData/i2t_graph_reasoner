import torch
from torch import optim
from torch.utils.data import DataLoader
import tqdm
from src.argparser import get_args
from src.data import HMMMKGDataset, Collator
from src.model import GATReasoningScorer
from src.trainer import Trainer

def main():
    # -------------------------
    # Parse arguments
    # -------------------------
    args = get_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    # -------------------------
    # Initialize wandb
    # -------------------------
    wandb_logger = None
    if args.use_wandb:
        import wandb
        wandb.init(project="gat_reasoning", config=vars(args))
        wandb_logger = wandb

    # -------------------------
    # Load dataset
    # -------------------------
    dataset = HMMMKGDataset(args.path_to_training_samples, args.path_to_metadata)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=16,
        shuffle=True,
        collate_fn=Collator(to_directed=args.use_directed).collate_fn
    )

    # -------------------------
    # Initialize model
    # -------------------------
    scorer_kwargs = {'use_weights': args.use_loss_weights, 'device': device}
    model = GATReasoningScorer(
        num_channels=args.num_channels,
        num_layers=args.num_layers,
        scorer_type=args.scorer_type,
        dropout=args.dropout,
        scorer_kwargs=scorer_kwargs,
        graph_layer_type=args.graph_type
    )
    if not args.use_single_gpu:
        model = torch.nn.DataParallel(model)
    if isinstance(args.load_from, str):
        model.load_state_dict(torch.load(args.load_from))

    model.to(device)

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # -------------------------
    # Initialize trainer
    # -------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataset=dataloader,
        wandb_logger=wandb_logger,
        device=device
    )

    # -------------------------
    # Training loop
    # -------------------------
    num_epochs = 2000  # you can also make this an argument
    prev_loss = 999

    for epoch in range(1, num_epochs + 1):
        epoch_losses = trainer.iter_epoch(epoch)
        print(f"Epoch {epoch} finished. Last batch loss: {epoch_losses[-1]:.4f}")
        if isinstance(args.save_to, str) and sum(epoch_losses)/len(epoch_losses) <= prev_loss:
            model.to('cpu')
            torch.save(trainer.model.state_dict(), args.save_to)
            model.to(device)
            prev_loss = sum(epoch_losses)/len(epoch_losses)
if __name__ == "__main__":
    main()
