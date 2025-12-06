import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, classification_report, confusion_matrix
import scipy.stats as stats

BASELINE_DIR = "experiments/base_model"
EXPERIMENT_DIR = "experiments/experiment_model"

SAVE_DIR = "analysis_results"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_outputs(model_dir):
    y_true = np.load(os.path.join(model_dir, "y_true.npy"))
    y_pred = np.load(os.path.join(model_dir, "y_pred.npy"))
    # The metrics.csv file is loaded for existing metrics (MSE, RMSE, CI)
    metrics = pd.read_csv(os.path.join(model_dir, "metrics.csv"))
    return y_true, y_pred, metrics

def calculate_extra_metrics(y_true, y_pred):
    """Calculates R2, MAE, and Max Error."""
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Max_Error": np.max(np.abs(y_true - y_pred))
    }

def plot_rank_order(y_true, y_pred, model_name, save_dir):
    """Generates a Rank-Order Heatmap, directly visualizing CI performance."""
    df = pd.DataFrame({'True': y_true, 'Pred': y_pred})
    df_sorted_by_true = df.sort_values(by='True').reset_index(drop=True)
    
    true_rank = df_sorted_by_true.index.values
    pred_rank = df_sorted_by_true['Pred'].rank(method='average').values - 1 

    plt.figure(figsize=(7, 7))
    # Use density plot (kdeplot) to show concentration along the diagonal
    sns.kdeplot(x=true_rank, y=pred_rank, cmap="magma", fill=True, clip=((0, len(y_true)), (0, len(y_true))))
    
    # Plot the perfect rank line (y=x)
    plt.plot([0, len(y_true)], [0, len(y_true)], '--', color='cyan', alpha=0.8, linewidth=2, label='Perfect Rank')
    
    plt.xlabel("True Rank (Sorted by True pKd)")
    plt.ylabel("Predicted Rank (Sorted by Predicted pKd)")
    plt.title(f"{model_name} Rank-Order Density (CI Visual)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"rank_order_heatmap_{model_name.lower()}.png"), dpi=300)
    plt.close()

y_true_b, y_pred_b, metrics_b = load_outputs(BASELINE_DIR)
y_true_e, y_pred_e, metrics_e = load_outputs(EXPERIMENT_DIR)

#Calculate additional metrics
extra_b = calculate_extra_metrics(y_true_b, y_pred_b)
extra_e = calculate_extra_metrics(y_true_e, y_pred_e)

summary_df = pd.DataFrame({
    "Model": ["Baseline", "Experiment"],
    "MSE": [metrics_b["mse"][0], metrics_e["mse"][0]],
    "RMSE": [metrics_b["rmse"][0], metrics_e["rmse"][0]],
    "CI": [metrics_b["ci"][0], metrics_e["ci"][0]],
    "R2": [extra_b["R2"], extra_e["R2"]],
    "MAE": [extra_b["MAE"], extra_e["MAE"]],
    "Max_Error": [extra_b["Max_Error"], extra_e["Max_Error"]],
})
summary_df.to_csv(f"{SAVE_DIR}/model_summary_full.csv", index=False)
print("\nFull Metric Summary:\n", summary_df.round(4))

res_b = y_pred_b - y_true_b
res_e = y_pred_e - y_true_e


#Predicted vs True Affinity
plt.figure(figsize=(6,6))
plt.scatter(y_true_b, y_pred_b, alpha=0.4, label="Baseline", color='skyblue')
plt.scatter(y_true_e, y_pred_e, alpha=0.4, label="Experiment", color='salmon')
plt.plot([min(y_true_b), max(y_true_b)], 
         [min(y_true_b), max(y_true_b)], 
         '--', color='black')

plt.xlabel("True pKd")
plt.ylabel("Predicted pKd")
plt.legend()
plt.title("Predicted vs True Affinity")
plt.savefig(f"{SAVE_DIR}/scatter_pred_vs_true.png", dpi=300)
plt.close()

#Residual Histogram
plt.figure(figsize=(6,6))
sns.histplot(res_b, bins=30, alpha=0.5, label="Baseline", color='skyblue')
sns.histplot(res_e, bins=30, alpha=0.5, label="Experiment", color='salmon')
plt.legend()
plt.title("Error Distribution")
plt.savefig(f"{SAVE_DIR}/error_hist.png", dpi=300)
plt.close()

# print("THIS IS FOR CI VISUAL")

# plot_rank_order(y_true_b, y_pred_b, "Baseline", SAVE_DIR)
# plot_rank_order(y_true_e, y_pred_e, "Experiment", SAVE_DIR)

plt.figure(figsize=(8,6))
plt.scatter(y_true_b, res_b, alpha=0.4, label=f"Baseline (CI: {metrics_b['ci'][0]:.4f})", color='skyblue')
plt.scatter(y_true_e, res_e, alpha=0.4, label=f"Experiment (CI: {metrics_e['ci'][0]:.4f})", color='salmon')
plt.axhline(0, linestyle="--", color="black")
plt.xlabel("True pKd")
plt.ylabel("Residual (Prediction - True)")
plt.title("Residual Plot Comparison")
plt.legend()
plt.savefig(f"{SAVE_DIR}/residual_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
stats.probplot(res_b, dist="norm", plot=plt)
plt.title("Baseline Model Q-Q Plot")

plt.subplot(1, 2, 2)
stats.probplot(res_e, dist="norm", plot=plt)
plt.title("Experiment Model Q-Q Plot")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/qq_plot.png", dpi=300)
plt.close()


#BOX PLOTS
bins = [5.0, 6.0, 7.0, 8.0, np.inf]
labels = ['5.0-6.0 (Weak)', '6.0-7.0', '7.0-8.0 (Strong)', '8.0+ (Very Strong)']

abs_err_b = np.abs(res_b)
abs_err_e = np.abs(res_e)

df_b = pd.DataFrame({'Abs Error': abs_err_b, 'True pKd': y_true_b, 'Model': 'Baseline'})
df_e = pd.DataFrame({'Abs Error': abs_err_e, 'True pKd': y_true_e, 'Model': 'Experiment'})

combined_df = pd.concat([df_b, df_e])
combined_df['Affinity Group'] = pd.cut(combined_df['True pKd'], bins=bins, labels=labels, right=False)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Affinity Group', y='Abs Error', hue='Model', data=combined_df)
plt.title('Absolute Error by True Affinity Group')
plt.savefig(f"{SAVE_DIR}/error_boxplot_by_affinity.png", dpi=300)
plt.close()


print("\nAll results saved in:", SAVE_DIR)