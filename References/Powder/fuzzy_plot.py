import matplotlib.pyplot as plt
import numpy as np

# Create the x-axis data
x = np.arange(0.01, 1.01, 0.01)

# Create the y-axis data for Extensive Dosing
y_extensive = x * x - 0.015

# Create the y-axis data for Accurate Control
y_accurate = -x * x + 0.5 + 0.04

# Create a subplot with two curves
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y_extensive, label='Extensive Dosing', color='blue')
ax.plot(x, y_accurate, label='Accurate Control', color='red')

# Add labels and title
ax.set_xlabel('Dosing speed (g/s)')
ax.set_ylabel('Weight difference (g)')
ax.set_title('Weight Difference between Two Objects')
ax.legend()

# Set limits
ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 1.1)

# Add grid
ax.grid(True)

# Show the plot
plt.show()
