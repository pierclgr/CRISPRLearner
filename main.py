# Import libraries and modules
from source.cli import print_menu
import numpy as np

# Necessary for starting Numpy generated random numbers in a well-defined initial state
# in order to obtain reproducible results
np.random.seed(15)

# Print main menu
print_menu()
