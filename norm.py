import numpy as np 


X = np.array([4,3])
#l0norm: so luong phan tu khac 0 
l0norm = np.linalg.norm(X, ord = 0)
print(l0norm)
#l1norm: khoang cach mahatan 
l1norm = np.linalg.norm(X, ord = 1)
print(l1norm)
#l2norm: khoang cach euclid 
l2norm = np.linalg.norm(X, ord = 2)
print(l2norm)