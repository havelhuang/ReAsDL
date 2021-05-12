import numpy as np

op = np.load('op_cell.npy')
pfd = np.load('pfd_cell.npy')
op = op[:len(pfd)]

op = op /sum(op)

pfd_model = sum(op * pfd)
acu = sum(pfd)/len(pfd)


print('model ACU: ', acu)
print('model pfd: ', pfd_model)