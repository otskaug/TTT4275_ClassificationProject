import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as disti
from sklearn.cluster import KMeans
import keras.datasets.mnist as mnist
import time
from collections import Counter
from numba import numba, jit
train_data = 0
test_data = 0
train_size = 60000
test_size = 10000
k_num = [1, 2, 3, 4, 5, 6, 7]
img_size_h = 28
img_size_v = 28
num_pictures = 3
num_clusters = [64, 512, 1024, 2048] #[32, 64, 128]

#---------------------- data loading  ---------------------------------#

def load_dataset():
    (train_data_xval, train_data_yval), (test_data_xval, test_data_yval) = mnist.load_data(path="'mnist.npz'")
    return (train_data_xval, train_data_yval), (test_data_xval, test_data_yval)

#----------------------  Calculations  ---------------------------------#

@jit(nopython=True)
def euclidian_distance(img1, img2):
    return np.linalg.norm(img1-img2)

def slow_euclidian_distance(img1, img2):
    return np.linalg.norm(img1-img2)

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

@jit(nopython=True)
def nearest_neighbour(data, labels, img, k_neighbours):
    neighbour_list = []
    return_labels = []
    for i in range(len(data)):
        dist = euclidian_distance(data[i], img)
        neighbour_list.append((dist, labels[i]))

    sorted_neighbour_list = sorted(neighbour_list, key=lambda dist: dist[0])

    for x in sorted_neighbour_list[0:k_neighbours]:
        return_labels.append(x[1])

    return return_labels

def slow_nearest_neighbour(data, labels, img, k_neighbours):
    neighbour_list = []
    return_labels = []
    for i in range(len(data)):
        dist = slow_euclidian_distance(data[i], img)
        neighbour_list.append((dist, labels[i]))

    sorted_neighbour_list = sorted(neighbour_list, key=lambda dist: dist[0])

    for x in sorted_neighbour_list[0:k_neighbours]:
        return_labels.append(x[1])

    return return_labels

def cluster(training_data_xval, training_data_yval, n_clusters):
    clusters = []
    classes = range(10)
    for clas in classes:
        indice = np.where(training_data_yval == clas)[0]
        data_of_class = training_data_xval[indice]
        num, nx, ny = data_of_class.shape
        d2_data_of_class = data_of_class.reshape((num, nx*ny))
        cluster = KMeans(n_clusters = n_clusters, n_init="auto").fit(d2_data_of_class).cluster_centers_
        clusters.append((clas, cluster))

    training_answers = [[clas]*n_clusters for clas in classes]
    training_answers = np.array(training_answers).reshape(n_clusters*len(classes))
    training_data = []

    for(clas, cluster) in clusters:
        training_data.extend(cluster)

    return(training_data, training_answers)

def confusion_matrix(predictions, test_data_yval, name, plot_bool):
    confusion_matrix = np.zeros((10, 10))
    for i, x in enumerate(predictions):
        confusion_matrix[test_data_yval[i], x] += 1
    if plot_bool:
        plot_confusionmatrix(name, confusion_matrix)
    return confusion_matrix


#------------------------- Plotting  ------------------------------------#
def plot_confusionmatrix(name, conf_matrix):
    fig, ax = plt.subplots()
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')

    plt.xlabel('Predictions')
    plt.ylabel('Actuals')
    plt.title(name + ' - Confusion Matrix')
    #plt.savefig("Result-Cmatrix_" + name, dpi=150)
    plt.show()

def plot_misclassified_img(number_of_images, test_data_xval, test_data_yval, predicted_errors, title):
    print("Plotting misclassified images")
    for q in range(number_of_images):
        img_err = np.array(test_data_xval[predicted_errors[q][0]], dtype='float').reshape((img_size_h, img_size_v))
        plt.imshow(img_err, cmap="gray")
        if np.array(predicted_errors[q][1]).size > 1:
            plt.title(f"{title} nr: {q+1}, Predicted: {Most_Common(predicted_errors[q][1])}, \n Complete K-list: {predicted_errors[q][1]}, Actual number: {test_data_yval[predicted_errors[q][0]]}")
        else:
            plt.title(f"{title} nr: {q+1}, Predicted: {predicted_errors[q][1]}, Actual number: {test_data_yval[predicted_errors[q][0]]}")
        plt.show()


def plot_classified_img(number_of_images, test_data_xval, test_data_yval, predicted_true, title):
    print("Plotting correctly classified images")
    for q in range(number_of_images):
        img_true = np.array(test_data_xval[predicted_true[q][0]], dtype='float').reshape((img_size_h, img_size_v))
        plt.imshow(img_true, cmap="gray")
        if np.array(predicted_true[q][1]).size > 1:
            plt.title(f"{title} nr: {q+1}, Predicted: {Most_Common(predicted_true[q][1])}, \n Complete K-list: {predicted_true[q][1]}, Actual number: {test_data_yval[predicted_true[q][0]]}")
        else:
            plt.title(f"{title} nr: {q+1}, Predicted: {predicted_true[q][1]}, Actual number: {test_data_yval[predicted_true[q][0]]}")
        plt.show()

# ------------------------- Running ------------------------------------ #
def run_NN(train_data_xval, train_data_yval, test_data_xval, test_data_yval, num_pictures, plot_bool, progress_update):
    print("Running Nearest Neighbour classifier")
    reg_errors = 0
    predicted = []
    predicted_errors = []
    predicted_true = []
    start_time = time.time()
    m = progress_update
    for i in range(len(test_data_xval)):
        dist = []
        if i / len(test_data_xval) * 100 > m:
            print(m, " Percent done")
            m += progress_update
        for j in range(len(test_data_xval)):
            dist.append(slow_euclidian_distance(test_data_xval[i], train_data_xval[j]))
        nn = np.argmin(dist)
        predicted_num = train_data_yval[nn]
        predicted.append(predicted_num)
        if predicted_num != test_data_yval[i]:
            reg_errors += 1
            predicted_errors.append([i, predicted_num])
        else:
            predicted_true.append([i, predicted_num])

    end_time = time.time()
    print("Time used for run_NN function: ", end_time - start_time)
    print("Error rate: ", reg_errors/len(test_data_xval)*100, "%")
    if plot_bool:
        plot_misclassified_img(num_pictures, test_data_xval, test_data_yval, predicted_errors, "Run NN Plot of Error")
        plot_classified_img(num_pictures, test_data_xval, test_data_yval, predicted_true, "Run NN Plot of Success")
    confusion_matrix(predicted, test_data_yval, "Run NN", plot_bool)

def run_KNN(train_data_xval, train_data_yval, test_data_xval, test_data_yval, k_neighbours, num_pictures, plot_bool, progress_update):
    print(f"Running K={k_neighbours} - Nearest Neighbour classifier")
    reg_errors = 0
    predicted = []
    predicted_errors = []
    predicted_true = []
    start_time = time.time()
    m = progress_update
    for i in range(len(test_data_xval)):
        dist = []
        if i / len(test_data_xval) * 100 > m:
            print(m, " Percent done")
            m += progress_update
        for j in range(len(test_data_xval)):
            dist.append(slow_euclidian_distance(test_data_xval[i], train_data_xval[j]))
        dist = np.argsort(dist)[0:k_neighbours]
        pred_num = [train_data_yval[dist[n]] for n in range(k_neighbours)]
        z = Most_Common(pred_num)
        predicted_num = z
        predicted.append(predicted_num)
        if predicted_num != test_data_yval[i]:
            reg_errors += 1
            predicted_errors.append([i, pred_num])
        else:
            predicted_true.append([i, pred_num])

    end_time = time.time()
    print("Time used for run_KNN function: ", end_time - start_time)
    print("Error rate: ", reg_errors/len(test_data_xval)*100, "%")
    if plot_bool:
        plot_misclassified_img(num_pictures, test_data_xval, test_data_yval, predicted_errors, "Run KNN, plot of Error")
        plot_misclassified_img(num_pictures, test_data_xval, test_data_yval, predicted_true, "Run KNN plot of Success")
    confusion_matrix(predicted, test_data_yval, "Run KNN", plot_bool)

def run_KCNN(train_data_xval, train_data_yval, test_data_xval, test_data_yval, n_clusters, k_neighbours, num_pictures, plot_bool, progress_update):
    print(f"Running K={k_neighbours} - cluster classifier. Cluster size = {n_clusters}")
    reg_errors = 0
    predicted = []
    predicted_errors = []
    predicted_true = []
    training_cluster = numba.typed.List()
    start_cluster_time = time.time()
    temp, train_labels = cluster(train_data_xval, train_data_yval, n_clusters)
    end_cluster_time = time.time()
    print("Clustering time: ", end_cluster_time - start_cluster_time)
    [training_cluster.append(x) for x in temp]
    m = progress_update
    reshape_test = test_data_xval.flatten().reshape(len(test_data_xval), img_size_v*img_size_h)
    start_time = time.time()

    for i in range(len(test_data_yval)):
        if i / len(test_data_xval) * 100 > m:
            print(m, " Percent done")
            m += progress_update
        k = nearest_neighbour(training_cluster, train_labels, reshape_test[i], k_neighbours)
        most_k = Most_Common(k)
        predicted_num = most_k
        predicted.append(predicted_num)
        if predicted_num != test_data_yval[i]:
            reg_errors += 1
            predicted_errors.append([i, k])
        else:
            predicted_true.append([i, k])

    end_time = time.time()
    print("Cluster time: ", end_time - start_time)
    print("Error rate for cluster NN: ", reg_errors / len(test_data_xval) * 100, "%")
    print("Number of errors: ", reg_errors)
    if plot_bool:
        plot_misclassified_img(num_pictures, test_data_xval, test_data_yval, predicted_errors, "Run KCNN, plot of Error")
        plot_classified_img(num_pictures, test_data_xval, test_data_yval, predicted_true, "Run KCNN, plot of Success")
    confusion_matrix(predicted, test_data_yval, "Run KCNN", plot_bool)


def run_CNN(train_data_xval, train_data_yval, test_data_xval, test_data_yval, n_clusters, num_pictures, plot_bool, progress_update):
    print(f"Running cluster classifier. Cluster size = {n_clusters}")
    reg_errors = 0
    predicted = []
    predicted_errors = []
    predicted_true = []
    classes = range(10)
    start_cluster_time = time.time()
    training_cluster = cluster(train_data_xval, train_data_yval, n_clusters)
    end_cluster_time = time.time()
    print("Clustering Time: ", end_cluster_time - start_cluster_time)
    reshape_test = test_data_xval.flatten().reshape(len(test_data_xval), img_size_v*img_size_h)
    start_time = time.time()
    m = progress_update
    for i in range(len(test_data_xval)):
        dist = []
        if i / len(test_data_xval) * 100 > m:
            print(m, " Percent done")
            m += progress_update
        for k in range(len(training_cluster)):
            for j in range(len(training_cluster[0])):
                dist.append(slow_euclidian_distance(reshape_test[i], training_cluster[k][j]))

        nn = np.argmin(dist)-1
        predicted_num = classes[int(nn//n_clusters)]
        predicted.append(predicted_num)
        if predicted_num != test_data_yval[i]:
            reg_errors += 1
            predicted_errors.append([i, predicted_num])
        else:
            predicted_true.append([i, predicted_num])

    end_time = time.time()
    print("Cluster time: ", end_time - start_time)
    print("Error rate for cluster NN: ", reg_errors / len(test_data_xval) * 100, "%")
    print("Number of errors: ", reg_errors)
    if plot_bool:
        plot_misclassified_img(num_pictures, test_data_xval, test_data_yval, predicted_errors, "Run CNN, plot of Error")
        plot_classified_img(num_pictures, test_data_xval, test_data_yval, predicted_true, "Run CNN, plot of Success")
    confusion_matrix(predicted, test_data_yval, "Run CNN", plot_bool)


def main():
    print("Main stuff")
    plot_bool = False
    (train_data_xval, train_data_yval), (test_data_xval, test_data_yval) = load_dataset()

    train_data_xval = np.array(train_data_xval)
    train_data_yval = np.array(train_data_yval)
    test_data_xval = np.array(test_data_xval)
    test_data_yval = np.array(test_data_yval)

    show_progress_percent = 101 #Over 100 will not show progress bar. If show_progress_percent < 100, it will print for each n*show_progress_percent it is done

    #run_NN(train_data_xval[0:train_size], train_data_yval[0:train_size], test_data_xval[0:test_size], test_data_yval[0:test_size], num_pictures, plot_bool, show_progress_percent)
    #run_KNN(train_data_xval[0:train_size], train_data_yval[0:train_size], test_data_xval[0:test_size],test_data_yval[0:test_size], 4, num_pictures, plot_bool, show_progress_percent)
    run_CNN(train_data_xval[0:train_size], train_data_yval[0:train_size], test_data_xval[0:test_size], test_data_yval[0:test_size], num_clusters[1], num_pictures, plot_bool, show_progress_percent)
    run_KCNN(train_data_xval[0:train_size], train_data_yval[0:train_size], test_data_xval[0:test_size], test_data_yval[0:test_size], num_clusters[1], 1, num_pictures, plot_bool, show_progress_percent)
    '''
    for i in range(len(k_num)):
        run_KNN(train_data_xval[0:train_size], train_data_yval[0:train_size], test_data_xval[0:test_size], test_data_yval[0:test_size], k_num[i], num_pictures, plot_bool, show_progress_percent)

    for i in range(len(k_num)):
        run_KCNN(train_data_xval[0:train_size], train_data_yval[0:train_size], test_data_xval[0:test_size], test_data_yval[0:test_size], num_clusters[1], k_num[i], num_pictures, plot_bool, show_progress_percent)

    '''
main()
