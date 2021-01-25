
CUDA = False
import pickle
import numpy as np
if CUDA:
  cuda_id = '0'
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
  os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(cuda_id)
import torch


file = open("data/cell_symb.pkl", 'rb')
cell_symb = pickle.load(file)
file.close()

classes = 10

class_count = np.zeros(classes)
class_relia = np.zeros(classes)

for key in cell_symb:
    n = cell_symb[key][0]
    class_count[n] += cell_symb[key][2]
    class_relia[n] += 10 ** (cell_symb[key][1])

for i in range(classes):
    class_relia[i] /= class_count[i]

np.save('class_reliability.npy', class_relia)
