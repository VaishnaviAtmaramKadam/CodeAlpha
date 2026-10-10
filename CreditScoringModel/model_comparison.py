
import pandas as pd
import matplotlib.pyplot as plt

# Load model results
results = pd.read_csv("model_results.csv")

# Display results
print("\nModel Comparison:")
print(results.to_string(index=False))

# Plot model accuracy
plt.figure(figsize=(9, 5))
plt.bar(results["Model"], results["Accuracy"])

plt.title("Credit Scoring Model Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("model_comparison.png")
plt.show()

print("\nGraph saved as model_comparison.png")