import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file and select first 7 columns
df = pd.read_csv('ft_idealpd.csv', usecols=range(14, 28))

# Set smoothing factor (adjust this between 0-1 as needed)
alpha = 0.04

# Calculate exponential smoothing
smoothed = df.ewm(alpha=alpha, adjust=False).mean()

# Create plots for each column
for col in df.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(df[col], label='Original', alpha=0.7)
    plt.plot(smoothed[col], label=f'Smoothed (α={alpha})', linewidth=2)
    plt.title(f'Exponential Smoothing: {col}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()