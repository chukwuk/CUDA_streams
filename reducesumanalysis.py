import matplotlib.pyplot as plt
import numpy as np


# Define start and end points
start = 200000
end = 1000000000
num_points = 500

# Generate discrete points
points = np.linspace(start, end, num_points)
print(points)

# Data for plotting

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create the plot
plt.plot(x, y, label='y = 2x', color='blue', marker='o')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')

# Add a legend
plt.legend()

# Show the plot
plt.show()
