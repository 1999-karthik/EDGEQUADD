import time
import torch
import numpy as np
from data_utils import load_data, init_stratified_dataloader, get_available_datasets
from train_test import train, val_test
from utils import hyper_para_load, count_param, fix_seed, initialize_logger
from model import QuETT, make_param_groups
from parse import get_args


def run(args, dataset, train_loader, val_loader, test_loader):

    (node_sz, node_feature_sz, layers, dropout,
     pooling, cluster_num, num_classes) = hyper_para_load(args=args, dataset=dataset)

    model = QuETT(args=args,
                                node_sz=node_sz,
                                corr_pearson_sz=node_feature_sz,
                                layers=layers,
                                dropout=dropout,
                                cluster_num=cluster_num,
                                pooling=pooling,
                                droppath=args.droppath,
                                cnncorr_base_ch=args.cnncorr_base_ch,
                                cnncorr_norm=args.cnncorr_norm,
                                cnncorr_diag_mode=args.cnncorr_diag_mode,
                                cnncorr_aspp_rates=args.cnncorr_aspp_rates,
                                quadratic_aspp_rates=args.quadratic_aspp_rates,
                                rank=args.rank,
                                num_classes=num_classes)

    # Move model to GPU if available
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    # Reset learnable gates to ensure reproducibility
    model.reset_learnable_gates()
    print(f"Using device: {device}")
    
    total_parameters = count_param(model)
    logger = initialize_logger()

    epoch_val_roc_list, epoch_val_loss_list = [], []
    epoch_test_roc_list, epoch_test_acc_list = [], []
    epoch_test_sen_list, epoch_test_spec_list = [], []

    # === STANDARD TRAINING ===
    print(f"\n Starting STANDARD TRAINING")
    param_groups = make_param_groups(model, base_lr=args.base_lr, adapter_lr=args.adapter_lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(param_groups)

    for epoch in range(args.epochs):
        result_train = train(model=model, optimizer=optimizer, args=args, train_loader=train_loader, epoch=epoch)
        result_val_test = val_test(model=model, args=args, val_loader=val_loader, test_loader=test_loader)

        # Log metrics based on dataset type
        if args.dataset.lower().startswith('ppmi'):
            logger.info(" | ".join([
                f'Epoch[{epoch}/{args.epochs}]',
                f'Train Loss:{result_train["train_loss"]: .3f}',
                f'Train Accuracy:{result_train["train_acc"]: .4f}',
                f'Val Loss:{result_val_test["val_loss"]:.3f}',
                f'Val Accuracy:{result_val_test["val_acc"]:.4f}',
                f'Val Macro AUC:{result_val_test["val_roc"]:.4f}',
                f'Test Accuracy:{result_val_test["test_acc"]: .4f}',
                f'Test Macro AUC:{result_val_test["test_roc"]:.4f}',
                f'Test Macro Sen:{result_val_test["test_sensitivity"]:.4f}',
                f'Test Macro Spec:{result_val_test["test_specificity"]:.4f}'
            ]))
        else:
            logger.info(" | ".join([
                f'Epoch[{epoch}/{args.epochs}]',
                f'Train Loss:{result_train["train_loss"]: .3f}',
                f'Train Accuracy:{result_train["train_acc"]: .4f}',
                f'Val Loss:{result_val_test["val_loss"]:.3f}',
                f'Val Accuracy:{result_val_test["val_acc"]:.4f}',
                f'Val AUC:{result_val_test["val_roc"]:.4f}',
                f'Test Accuracy:{result_val_test["test_acc"]: .4f}',
                f'Test AUC:{result_val_test["test_roc"]:.4f}',
                f'Test Sen:{result_val_test["test_sensitivity"]:.4f}',
                f'Test Spec:{result_val_test["test_specificity"]:.4f}'
            ]))

        epoch_val_loss_list.append(result_val_test['val_loss'])
        epoch_val_roc_list.append(result_val_test['val_roc'])
        epoch_test_roc_list.append(result_val_test['test_roc'])
        epoch_test_acc_list.append(result_val_test['test_acc'])
        epoch_test_sen_list.append(result_val_test['test_sensitivity'])
        epoch_test_spec_list.append(result_val_test['test_specificity'])

    # Model selection: Use validation Macro AUC for PPMI, validation loss for others
    if args.dataset.lower().startswith('ppmi'):
        # For PPMI: Select epoch with highest validation Macro AUC
        index_max = epoch_val_roc_list.index(max(epoch_val_roc_list))
        print(f"Selected epoch {index_max} with highest validation Macro AUC: {epoch_val_roc_list[index_max]:.4f}")
    else:
        # For other datasets: Select epoch with lowest validation loss
        index_max = epoch_val_loss_list.index(min(epoch_val_loss_list))
        print(f"Selected epoch {index_max} with lowest validation loss: {epoch_val_loss_list[index_max]:.4f}")
    
    return epoch_test_acc_list[index_max], epoch_test_roc_list[index_max], epoch_test_sen_list[index_max], epoch_test_spec_list[index_max]


def main(args):
    fix_seed(args.seed)

    # Show available datasets if requested
    if args.dataset == 'list':
        available_datasets = get_available_datasets(args.data_dir)
        print("Available datasets:")
        for dataset in available_datasets:
            print(f"  - {dataset}")
        return

    # load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_data(args)
    
    # Handle different dataset formats
    if len(dataset) == 4:
        # Datasets with timeseries (ABIDE, ADNI, PPMI)
        timeseries, final_pearson, labels, site = dataset
    elif len(dataset) == 3:
        # Datasets without timeseries (ADHD)
        final_pearson, labels, site = dataset
        timeseries = None
    else:
        raise ValueError(f"Unexpected dataset format: {len(dataset)} values returned")
    
    # Create data splits once and reuse for all runs
    dataloaders = init_stratified_dataloader(args, final_pearson, labels, site, timeseries)
    train_loader, val_loader, test_loader = \
        dataloaders["train_dataloader"], dataloaders["val_dataloader"], dataloaders["test_dataloader"]

    runs = int(args.runs)
    run_acc_list, run_roc_list = [], []
    run_sen_list, run_spec_list = [], []
    # Generate list of seeds that will be used
    seeds_used = [args.seed + i * 1000 for i in range(runs)]
    print(f"seeds:{seeds_used}")
    
    for i in range(runs):
        print(f'run: {i} start')
        # Fix seed before each run to ensure reproducibility
        run_seed = args.seed + i * 1000  # Offset each run by 1000
        fix_seed(run_seed)
        print(f"Run {i} using seed: {run_seed}")
        # Recreate data loaders for each run to ensure fresh data splits
        dataloaders = init_stratified_dataloader(args, final_pearson, labels, site, timeseries)
        train_loader, val_loader, test_loader = \
            dataloaders["train_dataloader"], dataloaders["val_dataloader"], dataloaders["test_dataloader"]
        acc, roc, sen, spec = run(args, dataset, train_loader, val_loader, test_loader)
        print(f'run: {i} is over')
        run_acc_list.append(acc)
        run_roc_list.append(roc)
        run_sen_list.append(sen)
        run_spec_list.append(spec)

    acc_mean, acc_std = np.mean(run_acc_list), np.std(run_acc_list)
    roc_mean, roc_std = np.mean(run_roc_list), np.std(run_roc_list)
    sen_mean, sen_std = np.mean(run_sen_list), np.std(run_sen_list)
    spec_mean, spec_std = np.mean(run_spec_list), np.std(run_spec_list)
    
    print("After ", args.runs, "runs on ", args.dataset, "!")
    
    # Print results based on dataset type
    if args.dataset.lower().startswith('ppmi'):
        print("=== PPMI MACRO METRICS ===")
        print("Macro AUC ± std: {:.2f}%±{:.2f}".format(roc_mean * 100, roc_std * 100))
        print("Macro Recall ± std: {:.2f}%±{:.2f}".format(sen_mean * 100, sen_std * 100))
        print("Macro Specificity ± std: {:.2f}%±{:.2f}".format(spec_mean * 100, spec_std * 100))
        print("Accuracy ± std: {:.2f}%±{:.2f}".format(acc_mean * 100, acc_std * 100))
    else:
        print("roc_auc ± std: {:.2f}%±{:.2f}".format(roc_mean * 100, roc_std * 100),
              "mean ± std: {:.2f}%±{:.2f}".format(acc_mean * 100, acc_std * 100))
    result_dir = args.root_path + "/result"
    import os
    os.makedirs(result_dir, exist_ok=True)
    
    result_file_path = result_dir + "/" + args.dataset + ".csv"
    print(f"Saving results to the'{result_file_path}'")
    
    with open(f"{result_file_path}", 'a+') as write_obj:
        if args.dataset.lower().startswith('ppmi'):
            # Save macro metrics for PPMI
            write_obj.write(f"macro_auc:{roc_mean * 100:.2f} ± {roc_std * 100:.2f},"
                            + f"macro_recall:{sen_mean * 100:.2f} ± {sen_std * 100:.2f},"
                            + f"macro_specificity:{spec_mean * 100:.2f} ± {spec_std * 100:.2f},"
                            + f"accuracy:{acc_mean * 100:.2f} ± {acc_std * 100:.2f},"
                            + f"seeds:{seeds_used},"
                            + f"runs:{args.runs},"
                            + f"epochs:{args.epochs},"
                            + f"batch_size:{args.batch_size},"
                            + f"base_lr:{args.base_lr},"
                            + f"adapter_lr:{args.adapter_lr},"
                            + f"wd:{args.weight_decay},"
                            + f"layers:{args.layers},"
                            + f"activation:{args.activation},"
                            + f"dropout:{args.dropout},"
                            + f"pooling:{args.pooling},"
                            + f"cluster_num:{args.cluster_num},"
                            + f"droppath:{args.droppath},"
                            + f"rank:{args.rank},"
                            + f"cnncorr_base_ch:{args.cnncorr_base_ch},"
                            + f"cnncorr_aspp_rates:{args.cnncorr_aspp_rates},"
                            + f"quadratic_aspp_rates:{args.quadratic_aspp_rates},"
                            + f"atrous_rate:{args.atrous_rate}"
                            + "\n"
                            )
        else:
            # Save binary metrics for other datasets
            write_obj.write(f"roc:{roc_mean * 100:.2f} ± {roc_std * 100:.2f},"
                            + f"acc:{acc_mean * 100:.2f} ± {acc_std * 100:.2f},"
                            + f"sen:{sen_mean * 100:.2f} ± {sen_std * 100:.2f},"
                            + f"spec:{spec_mean * 100:.2f} ± {spec_std * 100:.2f},"
                            + f"seeds:{seeds_used},"
                            + f"runs:{args.runs},"
                            + f"epochs:{args.epochs},"
                            + f"batch_size:{args.batch_size},"
                            + f"base_lr:{args.base_lr},"
                            + f"adapter_lr:{args.adapter_lr},"
                            + f"wd:{args.weight_decay},"
                            + f"layers:{args.layers},"
                            + f"activation:{args.activation},"
                            + f"dropout:{args.dropout},"
                            + f"pooling:{args.pooling},"
                            + f"cluster_num:{args.cluster_num},"
                            + f"droppath:{args.droppath},"
                            + f"rank:{args.rank},"
                            + f"cnncorr_base_ch:{args.cnncorr_base_ch},"
                            + f"cnncorr_aspp_rates:{args.cnncorr_aspp_rates},"
                            + f"quadratic_aspp_rates:{args.quadratic_aspp_rates},"
                            + f"atrous_rate:{args.atrous_rate}"
                            + "\n"
                            )
    print()


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)