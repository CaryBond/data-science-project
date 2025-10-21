import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

df = pd.read_csv('R2_0.9384/test_results.csv')

y_true = df['Actual']
y_pred = df['Predicted']

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)
medae = median_absolute_error(y_true, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MedAE: {medae: 4f}")

save_dir = "R2_0.9384"
os.makedirs(save_dir, exist_ok=True)

plt.figure(figsize=(6,6))
hb = plt.hexbin(y_true, y_pred, gridsize=30, cmap='viridis', norm='log', mincnt=1)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', linewidth=1, alpha=0.6)
cb = plt.colorbar(hb)
cb.set_label('Count', fontsize=14)

plt.xlabel("Actual", fontsize=16)
plt.ylabel("Predicted", fontsize=16)

plt.text(0.02, 0.98,
         f"MAE: {mae:.4f}\n$R^2$: {r2:.4f}\nRMSE: {rmse:.4f}",
         fontsize=16,
         ha='left', va='top',
         transform=plt.gca().transAxes,
         fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "parity_plot_hexbin.png"), dpi=300)
# plt.show()