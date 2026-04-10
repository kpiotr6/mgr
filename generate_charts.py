import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def generate_charts(INPUT_CHUNK_LENGTHS, OUTPUT_CHUNK_LENGTHS, model_names):
    for INPUT_CHUNK_LENGTH in INPUT_CHUNK_LENGTHS:
        for OUTPUT_CHUNK_LENGTH in OUTPUT_CHUNK_LENGTHS:
            print(f"\nGenerating training loss chart for I{INPUT_CHUNK_LENGTH}_O{OUTPUT_CHUNK_LENGTH}...")
            plt.figure(figsize=(10, 6))

            for name in model_names:
                try:
                    metrics_paths = glob.glob(f"outputs/logs/I{INPUT_CHUNK_LENGTH}_O{OUTPUT_CHUNK_LENGTH}/{name}/*/metrics.csv")
                    if metrics_paths:
                        # Sort by creation time to get the latest
                        metrics_paths.sort(key=os.path.getmtime)
                        metrics_df = pd.read_csv(metrics_paths[-1])

                        # Check for train_loss_epoch first (whole epoch loss)
                        if 'train_loss_epoch' in metrics_df.columns:
                            epoch_loss = metrics_df['train_loss_epoch'].dropna()
                            plt.plot(range(len(epoch_loss)), epoch_loss.values, label=f"{name} (epoch)")
                        elif 'train_loss' in metrics_df.columns and 'epoch' in metrics_df.columns:
                            epoch_loss = metrics_df.groupby('epoch')['train_loss'].mean().dropna()
                            plt.plot(epoch_loss.index, epoch_loss.values, label=f"{name} (avg/epoch)")
                        elif 'train_loss' in metrics_df.columns:
                            loss_series = metrics_df['train_loss'].dropna()
                            plt.plot(loss_series.values, label=f"{name} (steps)")

                except Exception as e:
                    print(f"Could not load/plot loss for {name}: {e}")

            plt.title(f"Training Loss per Epoch (I={INPUT_CHUNK_LENGTH}, O={OUTPUT_CHUNK_LENGTH})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

            chart_path = f"outputs/training_loss_I{INPUT_CHUNK_LENGTH}_O{OUTPUT_CHUNK_LENGTH}.png"
            plt.savefig(chart_path)
            plt.close()
            print(f"Training loss chart saved to {chart_path}")

            print(f"Generating MAE and MSE charts for I{INPUT_CHUNK_LENGTH}_O{OUTPUT_CHUNK_LENGTH}...")
            for metric_name in ['MeanAbsoluteError', 'MeanSquaredError']:
                for prefix, phase in zip(['train_', 'val_'], ['Training', 'Validation']):
                    plt.figure(figsize=(10, 6))
                    col_name = f'{prefix}{metric_name}'
                    col_name_epoch = f'{prefix}{metric_name}_epoch'

                    plotted = False
                    for name in model_names:
                        try:
                            metrics_paths = glob.glob(f"outputs/logs/I{INPUT_CHUNK_LENGTH}_O{OUTPUT_CHUNK_LENGTH}/{name}/*/metrics.csv")
                            if metrics_paths:
                                metrics_paths.sort(key=os.path.getmtime)
                                metrics_df = pd.read_csv(metrics_paths[-1])

                                if col_name_epoch in metrics_df.columns:
                                    epoch_metric = metrics_df[col_name_epoch].dropna()
                                    plt.plot(range(len(epoch_metric)), epoch_metric.values, label=f"{name}")
                                    plotted = True
                                elif col_name in metrics_df.columns and 'epoch' in metrics_df.columns:
                                    epoch_metric = metrics_df.groupby('epoch')[col_name].mean().dropna()
                                    plt.plot(epoch_metric.index, epoch_metric.values, label=f"{name}")
                                    plotted = True
                        except Exception as e:
                            pass

                    if plotted:
                        plt.xlabel('Epoch')
                        plt.ylabel(metric_name)
                        plt.title(f'{phase} {metric_name} (I={INPUT_CHUNK_LENGTH}, O={OUTPUT_CHUNK_LENGTH})')
                        plt.legend()
                        plt.grid(True)
                        chart_path_metric = f'outputs/{phase.lower()}_{metric_name.lower()}_I{INPUT_CHUNK_LENGTH}_O{OUTPUT_CHUNK_LENGTH}.png'
                        plt.savefig(chart_path_metric)
                        print(f"{phase} {metric_name} chart saved to {chart_path_metric}")
                    plt.close()
