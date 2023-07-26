import numpy as np

class RiskProcessor:
    def __init__(self, labels):
        self.labels = labels

    def process_risk_array(self, risk_array, risk_table):
        result_array = risk_array.copy()  # Create a copy of risk_array to store the result
        for label in self.labels:
            result_array = np.where(result_array == self.labels[label], risk_table[label], result_array)
        return result_array

# Example data
risk_array = np.array([0, 2, 1, 3, 0, 1, 2])
risk_table = {
    'person': [10, 20, 30, 40, 50, 60, 70],
    'bicycle': [100, 200, 300, 400, 500, 600, 700],
    'car': [1000, 2000, 3000, 4000, 5000, 6000, 7000],
    'motorcycle': [10000, 20000, 30000, 40000, 50000, 60000, 70000]
}
labels = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3}

# Create an instance of RiskProcessor and process the risk_array
processor = RiskProcessor(labels)
processed_array = processor.process_risk_array(risk_array, risk_table)

# Print the original and processed arrays
print("Original Risk Array:")
print(risk_array)
print("\nProcessed Risk Array:")
print(processed_array)
