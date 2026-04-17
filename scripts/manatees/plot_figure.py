import matplotlib.pyplot as plt

# Data from Table 7
# X-axis: Percentage of Placencia data used for fine-tuning
# "Initial Pretrained Model" is treated as 0%
x_labels = ['0% (Initial)', '3%', '5%', '10%', '15%']
x_values = [0, 3, 5, 10, 15]

# Y-axis data
precision = [0.8579, 0.8426, 0.8091, 0.5783, 0.5301]
recall = [0.2708, 0.2917, 0.4824, 0.7276, 0.6074]
f1_score = [0.4117, 0.4333, 0.6044, 0.6444, 0.5661]

plt.figure(figsize=(10, 6))

# Plotting the lines
plt.plot(x_values, f1_score, marker='o', label='F1 Score', linestyle='-', linewidth=2)
plt.plot(x_values, precision, marker='^', label='Precision', linestyle='-.', linewidth=2)
plt.plot(x_values, recall, marker='s', label='Recall', linestyle='--', linewidth=2)


# Customizing the chart
plt.title('Call precision, recall, and F1 score on the Placencia dataset with  and without fine-tuning.', fontsize=14)
plt.xlabel('Percentage of Placencia Data Used for Fine-tuning', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1.0)  # Standardize y-axis from 0 to 1
plt.xticks(x_values, x_labels) # Set specific labels for the x-axis points
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

# Adding value annotations for the peak F1 score (at 10%)
plt.annotate(f'Peak F1: {f1_score[3]}',
             xy=(10, f1_score[3]),
             xytext=(10, f1_score[3] + 0.05),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             ha='center')

plt.tight_layout()
plt.show()
