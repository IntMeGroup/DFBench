import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import os
try:
    from proplot import rc # Using proplot for styling
except ImportError:
    print("Proplot not found, using default matplotlib styling.")
    rc = {} 

# --- Main Plotting Logic ---
def plot_features_from_csv():
    output_base_dir = r"DFBench/features"
    
    csv_files_to_combine = [
        "kadid10k_image_features.csv",
        "clive_image_features.csv",
        "koniq10k_image_features.csv",
        "csiq_image_features.csv",
        "TID2013_image_features.csv"
    ]
    # csv_files_to_combine = [
    #     "siti_features_Infinity.csv",
    #     "siti_features_Janus.csv",
    #     "siti_features_Kandinsky-3.csv",
    #     "siti_features_Kolors_test.csv",
    #     "siti_features_NOVA.csv",
    #     "siti_features_sd3_medium_test.csv"
    # ]

    all_dfs = []
    print("Reading and combining CSV files...")
    for csv_file in csv_files_to_combine:
        features_csv_path = os.path.join(output_base_dir, csv_file)
        try:
            df_temp = pd.read_csv(features_csv_path)
            all_dfs.append(df_temp)
            print(f"Successfully read and added {csv_file}")
        except FileNotFoundError:
            print(f"Warning: The file {features_csv_path} was not found. It will be skipped.")
        except Exception as e:
            print(f"Warning: Error reading {features_csv_path}: {e}. It will be skipped.")

    if not all_dfs:
        print("No dataframes were loaded. Aborting plot generation.")
        return
    
    df_features = pd.concat(all_dfs, ignore_index=True)
    print(f"All specified CSV files have been combined. Total rows: {len(df_features)}")

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir) # Should ideally be checked before trying to read files but ok for now
        print(f"Created output directory: {output_base_dir}")

    if df_features.empty or len(df_features) < 2:
        print("Not enough data in the combined CSV to plot density distributions.")
        return

    print("Generating a single combined density plot (x-axis normalized) for the aggregated data...")
    
    metrics_to_plot = ['Brightness', 'Contrast', 'Colorfulness', 'SI']
    plot_colors = ['#C00000', 'gold', '#4874CB', '#588E32'] 
    
    try:
        if rc and "font.family" in rc:
             rc["font.family"] = "TeX Gyre Schola"
        else:
            plt.rc('font', family='sans-serif')

        plt.rc('xtick', labelsize='large') 
        plt.rc('ytick', labelsize='large')
        plt.rc('axes', labelsize='large') 

        fig, ax = plt.subplots(figsize=(6, 4))
        
        plotted_at_least_one = False

        for i, metric_name in enumerate(metrics_to_plot):
            current_color = plot_colors[i % len(plot_colors)]

            if metric_name not in df_features.columns:
                print(f"Metric {metric_name} not found in combined CSV. Skipping this metric.")
                continue
                
            data_values_original = df_features[metric_name].dropna().values
            if len(data_values_original) < 2:
                 print(f"Not enough data points for {metric_name} in combined data to plot. Skipping this metric.")
                 continue
            
            min_val = data_values_original.min()
            max_val = data_values_original.max()
            range_val = max_val - min_val

            if range_val < 1e-9: 
                print(f"Warning: All values for {metric_name} in combined data are (nearly) identical. Skipping KDE plot.")
                continue 
            
            data_values_normalized_x = (data_values_original - min_val) / range_val
            
            try:
                density_func = gaussian_kde(data_values_normalized_x)
                x_vals_normalized = np.linspace(0, 1, 300) 
                density_values_y = density_func(x_vals_normalized)

                ax.plot(x_vals_normalized, density_values_y, color=current_color, linestyle='-', label=metric_name, linewidth=2.5, alpha=0.8)
                plotted_at_least_one = True
            except Exception as kde_e:
                print(f"Could not generate KDE plot for {metric_name} from combined data. Error: {kde_e}. Skipping.")
                continue
        
        if not plotted_at_least_one:
            print("No metrics could be plotted from the combined data. Aborting plot generation.")
            plt.close(fig)
            return

        ax.set_xlabel('Normalized Metric Value', fontsize='large')
        ax.set_ylabel('Density', fontsize='large')
        ax.set_ylim(bottom=0) 
        ax.set_xlim(0, 1) 
        ax.grid(True, linestyle='--')
        ax.legend(fontsize='large')
        
        fig.tight_layout(pad=1.5)
        plot_save_path = os.path.join(output_base_dir, "combined_distortion_x_normalized_metrics_density.svg") 
        plt.savefig(plot_save_path, format='svg')
        print(f"Combined density plot (x-normalized) for aggregated datasets saved to {plot_save_path}")
        plt.close(fig)
            
    except Exception as e_plot:
        print(f"Error during plotting: {e_plot}")

if __name__ == '__main__':
    plot_features_from_csv()
