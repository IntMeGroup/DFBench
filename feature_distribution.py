import cv2
from PIL import Image, ImageFilter
import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde
from proplot import rc # Using proplot for styling as in plot_siti.py
from siti_tools.siti_tools.siti import SiTiCalculator
from tqdm import tqdm
#env_SEEDX
# --- Feature Calculation Functions (from at22.py and new) ---
def image_colorfulness(image_cv2):
    (B, G, R) = cv2.split(image_cv2.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)

def compute_contrast(image_cv2):
    gray_frame = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    return np.std(gray_frame)

def get_pixel_brightness(pixel):
    red, green, blue = pixel
    return (red * 0.2126) + (green * 0.7152) + (blue * 0.0722)

def get_brightness_pil(image_pil):
    image_rgba = image_pil.convert('RGBA')
    pixel_matrix = image_rgba.load()
    width, height = image_rgba.size
    total_brightness = 0
    pixel_count = 0
    for x in range(width):
        for y in range(height):
            pixel = pixel_matrix[x, y]
            if pixel[3] > 0: # Check transparency
                total_brightness += get_pixel_brightness(pixel[:3])
                pixel_count += 1
    return round(total_brightness / pixel_count, 3) if pixel_count > 0 else 0

# --- Main Processing Logic ---
def process_and_plot_clive_images():
    input_image_dir = r"DFBench/Flick8kimg/Images"
    output_base_dir = r"DFBench/features"
    #names = ["sd3_medium_test","Kandinsky-3","Infinity","Janus","NOVA","Playground_test""PixArt-sigma","LaVi-Bridge","sd3_5_large_test","ali_flux_dev_test","ali_flux_schnell"]
    #dataset_names = [,"Kandinsky-3","Infinity","Janus","NOVA","Playground_test","PixArt-sigma","Kolors_test","LaVi-Bridge","sd3_5_large_test","ali_flux_dev_test","ali_flux_schnell"]
    # dataset_names = ["Playground_test","PixArt-sigma"]
    # names = [""]
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    features_csv_path = os.path.join(output_base_dir, "Flick8kimg_image_features.csv")
    
    image_files = [f for f in os.listdir(input_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    all_features_data = []

    print(f"Processing {len(image_files)} images from {input_image_dir}...")
    for image_filename in tqdm(image_files, desc="Calculating features"):
        try:
            image_path = os.path.join(input_image_dir, image_filename)
            
            # For OpenCV based functions (Colorfulness, Contrast)
            image_cv2 = cv2.imread(image_path)
            if image_cv2 is None:
                print(f"Warning: Could not read image {image_filename} with OpenCV. Skipping.")
                continue

            # For PIL based functions (Brightness) and SI/TI (needs numpy array)
            try:
                image_pil = Image.open(image_path)
            except Exception as e_pil:
                print(f"Warning: Could not read image {image_filename} with PIL: {e_pil}. Skipping.")
                continue
            
            # Calculate at22.py features
            colorfulness_val = image_colorfulness(image_cv2)
            contrast_val = compute_contrast(image_cv2)
            brightness_val = get_brightness_pil(image_pil)

            # Calculate SI/TI features
            # SI/TI tools expect a grayscale numpy array (float32)
            image_gray_pil = image_pil.convert('L') # Convert to grayscale
            frame_data = np.array(image_gray_pil).astype('float32')
            
            si_val = SiTiCalculator.si(frame_data)
            
            # For TI with still images, create a synthetic previous frame (e.g., blurred version or fixed gray)
            # Option 1: Blurred version of the same image
            # blurred_image_pil = image_gray_pil.filter(ImageFilter.GaussianBlur(radius=2))
            # previous_frame_data = np.array(blurred_image_pil).astype('float32')
            # Option 2: Fixed gray frame (more consistent for comparing single images to a baseline)
            gray_level = 128
            previous_frame_data = np.full_like(frame_data, gray_level, dtype='float32')
            
            ti_val = SiTiCalculator.ti(frame_data, previous_frame_data)
            
            all_features_data.append([
                image_filename, brightness_val, contrast_val, 
                colorfulness_val, si_val, ti_val
            ])
            image_pil.close()

        except Exception as e:
            print(f"Error processing image {image_filename}: {e}")
            continue # Skip to next image
            
    # Save features to CSV
    df_features = pd.DataFrame(all_features_data, columns=['image_filename', 'Brightness', 'Contrast', 'Colorfulness', 'SI', 'TI'])
    df_features.to_csv(features_csv_path, index=False)
    print(f"Features saved to {features_csv_path}")

    # --- Plotting Density Distributions (adapted from plot_siti.py) ---
    # if df_features.empty or len(df_features) < 2: # Need at least 2 points for KDE
    #     print("Not enough data to plot density distributions.")
    #     return

    # print("Generating density plots...")
    # try:
    #     rc["font.family"] = "TeX Gyre Schola" # Proplot style setting
    #     plt.rc('xtick', labelsize='medium')
    #     plt.rc('ytick', labelsize='medium')
    #     plt.rc('axes', labelsize='x-small')

    #     for metric_to_plot in ['SI', 'TI']:
    #         if metric_to_plot not in df_features.columns:
    #             print(f"Metric {metric_to_plot} not found in CSV. Skipping plot.")
    #             continue
                
    #         data_values = df_features[metric_to_plot].dropna().values
    #         if len(data_values) < 2:
    #              print(f"Not enough data points for {metric_to_plot} to plot density. Skipping.")
    #              continue
    #         if np.std(data_values) < 1e-6: # KDE fails if all values are the same
    #              print(f"All values for {metric_to_plot} are identical. Skipping density plot.")
    #              plt.figure(figsize=(2.36, 1.7))
    #              plt.hist(data_values, bins=1, color='#3CA0CE', edgecolor='black')
    #              plt.title(f'Histogram of {metric_to_plot} (Single Value)')
    #              plt.xlabel(metric_to_plot, fontsize='large')
    #              plt.ylabel('Frequency', fontsize='medium')
    #         else:
    #             density = gaussian_kde(data_values)
    #             x_vals = np.linspace(data_values.min(), data_values.max(), 500)
                
    #             plt.figure(figsize=(2.36, 1.7)) # Size from plot_siti.py
    #             plt.plot(x_vals, density(x_vals), color='#3CA0CE', label=f'CLIVE ({metric_to_plot})', linestyle='-')
    #             plt.xlabel(metric_to_plot, fontsize='large')
    #             plt.ylabel('Density', fontsize='medium')
            
    #         plt.ylim(bottom=0)
    #         plt.grid(True, linestyle='--')
    #         # plt.legend(fontsize='small') # Legend might be redundant for single dataset plot
    #         plt.tight_layout()
    #         plot_save_path = os.path.join(output_base_dir, f"clive_{metric_to_plot.lower()}_density.png")
    #         plt.savefig(plot_save_path, dpi=300)
    #         print(f"Density plot for {metric_to_plot} saved to {plot_save_path}")
    #         plt.close() # Close plot to free memory
            
    # except Exception as e_plot:
    #     print(f"Error during plotting: {e_plot}")

if __name__ == '__main__':
    process_and_plot_clive_images()
