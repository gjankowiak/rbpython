import numpy as np

def ml_iterator(lists, values=False):
    idc = [0 for s in lists]
    lens = map(len, lists)
    k = len(lists) - 1
    while idc < lens:
        yield idc, [lists[n][idc[n]] for n in range(len(lists))]
        while True:
            if idc[k] < lens[k] - 1:
                idc[k] += 1
                k = len(lists) - 1
                break
            else:
                if k == 0:
                    raise StopIteration
                else:
                    idc[k] = 0
                    k -= 1

def intc_array(k):
    return np.array(k, dtype=np.intc)
def f64_array(k):
    return np.array(k, dtype=np.float64)
def add_mat(dest, m1, m2):
    n = dest.size(0)
    dest.zero()
    dest.add((m1+m2).array(), intc_array(range(n)), intc_array(range(n)))

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

