import gzip
import cPickle as pickle
from scipy import stats
from collections import Counter
import numpy as np
import helper

def train(args):
    y_train = args["y_train"].flatten().tolist()
    class_indices, class_probs = helper.create_dist(y_train)
    return (class_indices, class_probs)

def describe(args, model):
    return str(model[0]) + "\n" + str(model[1]) + "\n"

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