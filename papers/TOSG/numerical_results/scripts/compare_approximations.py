import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load both datasets
zonotope_data = pd.read_csv('papers/TOSG/numerical_results/data/zonotope_approx_metrics.csv')
gpolymatroid_data = pd.read_csv('papers/TOSG/numerical_results/data/approx_metrics.csv')

# Create comparison plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, T in enumerate([5, 6, 7]):
    ax = axes[idx]
    
    # Filter data for this time horizon
    zono_T = zonotope_data[zonotope_data['T'] == T]
    gpoly_T = gpolymatroid_data[gpolymatroid_data['T'] == T]
    
    # Plot both methods
    ax.plot(zono_T['lmda'], zono_T['metric'], 'o-', label='Zonotope', linewidth=2, markersize=6)
    ax.plot(gpoly_T['lmda'], gpoly_T['metric'], 's-', label='g-Polymatroid', linewidth=2, markersize=6)
    
    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Approximation Quality (Volume Ratio)', fontsize=12)
    ax.set_title(f'T = {T}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

plt.tight_layout()
plt.savefig('papers/TOSG/numerical_results/figures/approximation_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison plot saved to papers/TOSG/numerical_results/figures/approximation_comparison.png")

# Print detailed comparison statistics
print("\n" + "="*80)
print("DETAILED COMPARISON: Zonotope vs g-Polymatroid Approximations")
print("="*80)

for T in [5, 6, 7]:
    print(f"\n{'─'*80}")
    print(f"Time Horizon T = {T}")
    print(f"{'─'*80}")
    
    zono_T = zonotope_data[zonotope_data['T'] == T]
    gpoly_T = gpolymatroid_data[gpolymatroid_data['T'] == T]
    
    print(f"\nZonotope Approximation:")
    print(f"  Mean quality:   {zono_T['metric'].mean():.4f}")
    print(f"  Median quality: {zono_T['metric'].median():.4f}")
    print(f"  Min quality:    {zono_T['metric'].min():.4f}")
    print(f"  Max quality:    {zono_T['metric'].max():.4f}")
    print(f"  Std deviation:  {zono_T['metric'].std():.4f}")
    
    print(f"\ng-Polymatroid Approximation:")
    print(f"  Mean quality:   {gpoly_T['metric'].mean():.4f}")
    print(f"  Median quality: {gpoly_T['metric'].median():.4f}")
    print(f"  Min quality:    {gpoly_T['metric'].min():.4f}")
    print(f"  Max quality:    {gpoly_T['metric'].max():.4f}")
    print(f"  Std deviation:  {gpoly_T['metric'].std():.4f}")
    
    # Calculate improvement ratio
    improvement = gpoly_T['metric'].mean() / zono_T['metric'].mean() if zono_T['metric'].mean() > 0 else float('inf')
    print(f"\nRelative Performance:")
    print(f"  g-Polymatroid is {improvement:.2f}x better on average")

print("\n" + "="*80)
