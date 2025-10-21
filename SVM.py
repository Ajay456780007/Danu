from sklearn.neighbors import KNeighborsClassifier

from Sub_Functions.Evaluate import main_est_parameters


def SVM(x_train, x_test, y_train, y_test, epochs, n_neighbors=3):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])
    # Initialize the K-NN classifier with the number of neighbors you want
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the classifier
    knn.fit(x_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(x_test)

    # Calculate accuracy
    metrics = main_est_parameters(y_pred, y_test)

    return metrics
