import gzip
import cPickle as pickle
from scipy import stats
from collections import Counter
import numpy as np

def create_dist(arr):
    cc = Counter(arr)
    keys = []
    vals = []
    for key in cc:
        keys.append( int(key) )
        vals.append( cc[key] )
    sum_vals = sum(vals)
    for i in range(0, len(vals)):
        vals[i] = float(vals[i]) / sum_vals

    return keys, vals

def train(args):
    y_train = args["y_train"].flatten().tolist()
    class_indices, class_probs = create_dist(y_train)
    return (class_indices, class_probs)

def describe(args, model):
    return str(model[0]) + "\n" + str(model[1]) + "\n" + str(args["seed"])

def test(args, model):

    dist = stats.rv_discrete(name='dist', values=model)
    num_instances = args["X_test"].shape[0]
    identity_matrix = np.eye(args["num_classes"])
    predictions = []
    for i in range(0, num_instances):
        random_label = dist.rvs(size=1)[0]
        predictions.append( identity_matrix[random_label].tolist() )
    return predictions

if __name__ == "__main__":
    f = gzip.open("iris.pkl.gz")
    args = pickle.load(f)
    f.close()
    model = train(args)
    args["X_test"] = args["X_train"]
    predictions = test(args, model)
    for pred in predictions:
        print pred