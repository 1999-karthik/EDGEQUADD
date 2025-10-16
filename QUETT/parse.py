import argparse


def get_args():
    """ create parser """
    parser = argparse.ArgumentParser(description='BQN hyper parameters')
    
    # Device and paths
    parser.add_argument('--device', type=int, default=0, help="cuda:0")
    parser.add_argument('--root_path', type=str, default="/pscratch/sd/s/saik1999/brain_Networks/QUETT")
    parser.add_argument('--data_dir', type=str, default="/pscratch/sd/s/saik1999/brain_Networks/QUETT/data")

    # Dataset and experiment settings - ALTER-style naming
    parser.add_argument('--dataset', default='adhd', 
                       help='brain dataset (use "list" to see available datasets)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', default=5, help='repeat time')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--Train_prop', default=0.7)
    parser.add_argument('--Val_prop', default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)

    # Training hyperparameters
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--adapter_lr', type=float, default=0.0006, help='Learning rate for adapter parameters')
    
    # Model architecture
    parser.add_argument('--layers', type=int, default=3)  # Restored to 3 layers for better performance
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--activation', type=str, default='leaky_relu', choices=['gelu', 'leaky_relu', 'elu'])
    parser.add_argument('--pooling', type=bool, default=True)
    parser.add_argument('--cluster_num', type=int, default=4)
    
    
    parser.add_argument('--droppath', type=float, default=0.05, help='DropPath probability')
    
    
    # CNN correlation parameters
    parser.add_argument('--cnncorr_base_ch', type=int, default=32, 
                        help='Base channels for CNNCorr')
    parser.add_argument('--cnncorr_norm', type=str, default='gn', choices=['bn', 'gn'],
                        help='Normalization type for CNNCorr (bn=batch norm, gn=group norm)')
    parser.add_argument('--cnncorr_diag_mode', type=str, default='zero', choices=['zero', 'channel'],
                        help='Diagonal mode for CNNCorr (zero=zero out diagonal, channel=separate channel)')
    parser.add_argument('--cnncorr_aspp_rates', type=int, nargs='+', default=[1,2,4],
                        help='ASPP dilation rates for CNNCorr')
    
    # ASPP Quadratic parameters
    parser.add_argument('--use_aspp_quadratic', type=bool, default=True,
                        help='Use ASPPQuadraticAdapter for quadratic processing')
    parser.add_argument('--use_dilated_conv', type=bool, default=True,
                        help='Use depth-wise dilated convolution gateway')
    parser.add_argument('--quadratic_aspp_rates', type=int, nargs='+', default=[1,2,4],
                        help='ASPP dilation rates for quadratic adapters')
    
    # Low-rank and atrous parameters
    parser.add_argument('--rank', type=int, default=3,
                        help='Rank for low-rank decomposition in QuadraticNeuron (0=ablation, 4=baseline)')
    parser.add_argument('--atrous_rate', type=int, default=2,
                        help='Atrous rate for dilated convolutions')

    """ The command line reads the parameters """
    return parser.parse_args()