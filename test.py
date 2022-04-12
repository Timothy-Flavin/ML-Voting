import numpy as np
import random 

tr = np.arange(10,20)

choices = random.choices(k=len(tr),population=range(len(tr)))
tr = tr[choices]
#print(choices)
#print(tr)

arr = np.array([
  [1,2,3],
  [4,5,6],
  [7,8,9],
  [10,11,12],
  [13,14,15],
  [16,17,18],
  [19,20,21],
])

arr = np.apply_along_axis(lambda a_1d: a_1d[0:2], 1, arr)
print(arr)