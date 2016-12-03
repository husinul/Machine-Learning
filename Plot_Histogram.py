
import matplotlib.pyplot as plt
import numpy as np

# Create population of 1000 dog, 50/50 greyhound/labrador
greyhounds = 500
labs = 500

# Assume greyhounds are normally 28" tall
# Assume labradors are normally 24" tall
# Assume normal distribution of +/- 4"
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

# Greyounds - red, labradors - blue
plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()
