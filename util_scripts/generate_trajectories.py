import pickle as pkl
import numpy as np

step = 4
permutations = []
permutations.append(np.array([[0,0],
                              [0,2],
                              [2,2],
                              [2,0]]))
permutations.append(np.array([[0,0],
                              [0,2.1],
                              [2,2],
                              [2,0]]))
permutations.append(np.array([[0,0],
                              [0,2],
                              [2.1,2.1],
                              [2,0]]))
permutations.append(np.array([[0,0],
                              [0,2],
                              [2,2],
                              [2.1,0]]))
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

# permutations.append(np.array([[0,0],
#                               [0,1],
#                               [0,2],
#                               [0,3],
#                               [1,0],
#                               [1,1],
#                               [1,2],
#                               [1,3]]))
# permutations.append(np.array([[0,0],
#                               [0,1],
#                               [0,2],
#                               [0,3],
#                               [2,0],
#                               [2,1],
#                               [2,2],
#                               [2,3]]))
# permutations.append(np.array([[0,0],
#                               [0,1],
#                               [0,2],
#                               [0,3],
#                               [3,0],
#                               [3,1],
#                               [3,2],
#                               [3,3]]))
# permutations.append(np.array([[0,0],
#                               [0,1],
#                               [0,2],
#                               [0,3],
#                               [0,4],
#                               [0,5],
#                               [0,6],
#                               [0,7]]))
# permutations.append(np.array([[0,0],
#                               [0,2],
#                               [0,6],
#                               [0,10],
#                               [0,14],
#                               [0,18],
#                               [0,22],
#                               [0,23]]))
# permutations.append(np.array([[0,0],
#                               [1,1],
#                               [2,2],
#                               [3,3],
#                               [4,4],
#                               [5,5],
#                               [6,6],
#                               [7,7]]))


while len(permutations)!=300:
    p = np.random.randint(0, 4, size=(step,2))
    p[0,0]=0
    p[0,1]=0
    a=p[1:, :]
    a=a[np.lexsort((a[:, 1], a[:, 0]))]
    p[1:, :] = a
    found = False
    for j in permutations:
        if np.allclose(j,p):
            found = True
            break
    if found:
        continue
    elif len(p) != len(np.unique(p, axis=0)):
        continue
    else:
        print("now we have %s" % len(permutations))
        permutations.append(p)
with open("zurich_trajectory_step-4_range-4.pkl", 'wb') as f:
    pkl.dump(permutations, f)