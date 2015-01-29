from sklearn.datasets import load_iris
from FS_package.function.statistics_based import chi_square


def main():
    # load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # feature selection
    num_fea = 2
    features = chi_square.chi_square(X, y, num_fea)
    print features.shape


if __name__ == '__main__':
    main()