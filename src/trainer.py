import torch
import tqdm
from torch_geometric.data import Data

class Trainer:
    def __init__(self, model, optimizer, dataset, wandb_logger, device = 'cuda'):
        self.optimizer = optimizer
        self.model = model
        self.data = dataset
        self.wandb_logger = wandb_logger
        self.device = device

    def iter_epoch(self, epoch):

        total_loss = []
        for batch in tqdm.tqdm(self.data, total = len(self.data), desc = f'Training loop of epoch {epoch}...'):
            self.optimizer.zero_grad()

            batch['edges'] = batch['edges'].to(self.device)
            batch['node_features'] = batch['node_features'].to(self.device)
            batch['compute_loss'] = True
            batch['edge_types'] = batch['edge_types'].to(self.device)

            data = Data(x = batch.pop('node_features'), edge_index = batch.pop('edges'), edge_attr = batch.pop('edge_types'))
            _, _, _, loss = self.model(data, batch)

            loss.backward()
            self.optimizer.step()

            total_loss.append(loss.item())
            if self.wandb_logger:
                self.wandb_logger.log({
                    'batch_loss': total_loss[-1],
                    'epoch': epoch
                })

        if self.wandb_logger:
            self.wandb_logger.log({
                'epoch_loss': total_loss[-1]
            })
        return total_loss






