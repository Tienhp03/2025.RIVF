# Vẽ công suất phát theo khoảng cách truyền dẫn với lambda và bán kính máy thu khác nhau

import matplotlib.pyplot as plt
import numpy as np

# Data
distance = [25, 30, 35, 40, 45, 50] 
result_1_W = [0.46, 0.5, 0.72, 0.75, 0.8, 0.94] # Lambda = 200, a = 0.1
result_2_W = [0.35, 0.47, 0.65, 0.72, 0.76, 0.8] # Lambda = 200, a = 0.15
result_3_W = [0.36, 0.4, 0.56, 0.6, 0.63, 0.73] # Lambda = 150, a = 0.1
result_4_W = [0.3, 0.37, 0.5, 0.58, 0.6, 0.68] # Lambda = 150, a = 0.15

result_1 = 10 * np.log10(np.array(result_1_W) * 1000)
result_2 = 10 * np.log10(np.array(result_2_W) * 1000)
result_3 = 10 * np.log10(np.array(result_3_W) * 1000)
result_4 = 10 * np.log10(np.array(result_4_W) * 1000)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(distance, result_1, marker='o', label='lambda = 200, a = 0.1', linestyle='-', linewidth=2)
plt.plot(distance, result_2, marker='o', label='lambda = 200, a = 0.15', linestyle='--', linewidth=2)
plt.plot(distance, result_3, marker='x', label='lambda = 150, a = 0.1', linestyle='-', linewidth=2)
plt.plot(distance, result_4, marker='x', label='lambda = 150, a = 0.15', linestyle='--', linewidth=2)

# Adding labels and title
plt.xlabel('Transmit Distance(m)', fontsize=20)
plt.ylabel('Average Transmit Power (dBm)', fontsize=20)
# plt.title('Average Transmit Power vs. Average Arrival Rate for HARQ Schemes', fontsize=14)
plt.grid(True)
plt.legend(fontsize=16)

# Set x-axis ticks to only show 150, 200, 250, 300
plt.xticks(distance, fontsize=16)

# Display the plot
plt.show()