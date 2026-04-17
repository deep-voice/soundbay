import matplotlib.pyplot as plt

# Data Setup
fractions = [3, 5, 10, 15]

# --- Pearson Data ---
# Values
no_pre_pearson = [0.908, 0.573, 0.835, -0.085]
pre_pearson = [0.366, 0.465, 0.923, 0.929]
# Significance (True = Significant/Asterisk, False = Not Significant)
no_pre_pearson_sig = [True, True, True, False]
pre_pearson_sig = [False, False, True, True]

# --- Spearman Data ---
# Values
no_pre_spearman = [0.910, 0.681, 0.892, 0.140]
pre_spearman = [0.663, 0.781, 0.911, 0.908]
# Significance
no_pre_spearman_sig = [True, True, True, False]
pre_spearman_sig = [True, True, True, True]

# Setup the figure with 2 subplots sharing the Y axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)


# Helper function to plot lines and handle significant markers
def plot_with_significance(ax, x, y, sig_mask, color, label, marker_shape='o'):
    # 1. Plot the connecting line
    ax.plot(x, y, color=color, linestyle='-', linewidth=2, label=label, alpha=0.7)

    # 2. Separate significant (sig) and non-significant (ns) points
    x_sig = [x[i] for i in range(len(x)) if sig_mask[i]]
    y_sig = [y[i] for i in range(len(y)) if sig_mask[i]]

    x_ns = [x[i] for i in range(len(x)) if not sig_mask[i]]
    y_ns = [y[i] for i in range(len(y)) if not sig_mask[i]]

    # 3. Plot Significant points (Filled)
    ax.scatter(x_sig, y_sig, color=color, marker=marker_shape, s=80, zorder=5)

    # 4. Plot Non-Significant points (Hollow/Empty)
    if x_ns:
        ax.scatter(x_ns, y_ns, color=color, marker=marker_shape, s=80, zorder=5, facecolors='white', edgecolors=color,
                   linewidth=2)


# --- Plot 1: Pearson ---
plot_with_significance(ax1, fractions, no_pre_pearson, no_pre_pearson_sig, 'tab:blue', 'No Pretraining')
plot_with_significance(ax1, fractions, pre_pearson, pre_pearson_sig, 'tab:orange', 'Pretrained + FT')

ax1.set_title('Pearson Correlation (Linear)', fontsize=14)
ax1.set_xlabel('Data Fraction (%)', fontsize=12)
ax1.set_ylabel('Correlation Coefficient', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_xticks(fractions)
ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Zero line

# --- Plot 2: Spearman ---
plot_with_significance(ax2, fractions, no_pre_spearman, no_pre_spearman_sig, 'tab:blue', 'No Pretraining')
plot_with_significance(ax2, fractions, pre_spearman, pre_spearman_sig, 'tab:orange', 'Pretrained + FT')

ax2.set_title('Spearman Correlation (Rank)', fontsize=14)
ax2.set_xlabel('Data Fraction (%)', fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.set_xticks(fractions)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')

# Legend and Layout
# We add a custom legend entry to explain the hollow markers
from matplotlib.lines import Line2D

custom_legend = [
    Line2D([0], [0], color='tab:blue', lw=2, label='No Pretraining'),
    Line2D([0], [0], color='tab:orange', lw=2, label='Pretrained + FT'),
    Line2D([0], [0], marker='o', color='k', label='Significant (p<0.05)', markerfacecolor='k', markersize=8,
           linestyle='None'),
    Line2D([0], [0], marker='o', color='k', label='Not Significant', markerfacecolor='white', markersize=8,
           linestyle='None')
]

ax2.legend(handles=custom_legend, loc='lower left', ncol=1, fontsize=11)

plt.tight_layout()
plt.show()
