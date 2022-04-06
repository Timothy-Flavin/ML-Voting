import numpy as np
import random 

tr = np.arange(10,20)

choices = random.choices(k=len(tr),population=range(len(tr)))
tr = tr[choices]
print(choices)
print(tr)