import pickle as pkl
import numpy as np

step = 8
permutations = []
# permutations.append(np.array([[0,0],
#                               [0,1],
#                               [1,1],
#                               [1,0]]))
# permutations.append(np.array([[0,0],
#                               [0,2],
#                               [2,2],
#                               [2,0]]))
# permutations.append(np.array([[0,0],
#                               [0,3],
#                               [3,3],
#                               [3,0]]))
# permutations.append(np.array([[0,0],
#                               [0,-1],
#                               [-1,-1],
#                               [-1,0]]))
# permutations.append(np.array([[0,0],
#                               [0,-2],
#                               [-2,-2],
#                               [-2,0]]))
# permutations.append(np.array([[0,0],
#                               [0,-3],
#                               [-3,-3],
#                               [-3,0]]))
# permutations.append(np.array([[0,0],
#                               [0,1],
#                               [0,2],
#                               [0,3]]))
# permutations.append(np.array([[0,0],
#                               [0,-1],
#                               [0,-2],
#                               [0,-3]]))

# permutations.append(np.array([[0,0],
#                               [0,1],
#                               [0,2],
#                               [0,3],
#                               [0,-1],
#                               [0,-2]]))
# permutations.append(np.array([[0,0],
#                               [0,1],
#                               [0,2],
#                               [1,0],
#                               [1,1],
#                               [1,2]]))
# permutations.append(np.array([[0,0],
#                               [0,1],
#                               [0,2],
#                               [1,2],
#                               [2,2],
#                               [1,1]]))
permutations.append(np.array([[0,0],
                              [0,1],
                              [0,2],
                              [0,3],
                              [1,0],
                              [1,1],
                              [1,2],
                              [1,3]]))
permutations.append(np.array([[0,0],
                              [0,1],
                              [0,2],
                              [0,3],
                              [0,-1],
                              [0,-2],
                              [0,-3],
                              [0,4]]))
permutations.append(np.array([[0,0],
                              [0,1],
                              [0,2],
                              [0,3],
                              [2,0],
                              [2,1],
                              [2,2],
                              [2,3]]))
permutations.append(np.array([[0,0],
                              [0,1],
                              [0,2],
                              [0,3],
                              [3,0],
                              [3,1],
                              [3,2],
                              [3,3]]))

while len(permutations)!=1000:
    p = np.random.randint(-3, 4, size=(step,2))
    p[0,0]=0
    p[0,1]=0
    a=p[1:, :]
    a=a[np.argsort(a[:, 0])]
    p[1:, :] = a
    found = False
    for j in permutations:
        if np.allclose(j,p):
            found = True
            break
    if found:
        continue
    else:
        permutations.append(p)
with open("trajectory_step-8_range-4.pkl", 'wb') as f:
    pkl.dump(permutations, f)