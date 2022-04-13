import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot

print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)


#X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T ,columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
X = pd.read_csv("nhefs.csv")
X = X.fillna(-1).reset_index(drop=True)
X=(X-X.mean())/X.std()
print(X.head())
model = lingam.DirectLiNGAM()
model.fit(X)
dot = make_dot(model.adjacency_matrix_)

# Save png
dot.format = 'png'
dot.render('dag')

