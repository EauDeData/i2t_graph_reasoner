import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    # Wandb
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb for logging')

    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the training on (cuda or cpu)')

    # Model parameters
    parser.add_argument('--num_channels', type=int, required=True, help='Number of channels in the model')
    parser.add_argument('--num_layers', type=int, required=True, help='Number of layers in the model')
    parser.add_argument('--scorer_type', type=str, default='logistic', choices=['logistic', 'softmax'], help='Type of scorer')
    parser.add_argument('--graph_type', type=str, default='gat', choices=['grav', 'pna', 'gat', 'relational', 'sage', 'trf', 'pathfinder'], help='Type of message passing')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use_directed', action='store_true', help='Whether to use directed graphs')


    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--use_loss_weights', action='store_true', help='')
    parser.add_argument('--use_single_gpu', action='store_true', help='Whether to use a single gpu')


    # Paths
    parser.add_argument('--path_to_training_samples', type=str, required=True, help='Path to training samples')
    parser.add_argument('--path_to_metadata', type=str, required=True, help='Path to metadata')
    parser.add_argument('--blacklist_path', type=str, default='black_list.txt', help='Path to blacklist file')

    parser.add_argument('--save_to', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--load_from', type=str, default=None, help='Path to checkpoint file')


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
