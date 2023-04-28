import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as disti
from sklearn.cluster import KMeans
import keras.datasets.mnist as mnist
import time
from collections import Counter
train_data = 0
test_data = 0
train_size = 60000
test_size = 10000
k_num = [1, 2, 4]
img_size_h = 28
img_size_v = 28
num_clusters = [512, 1024, 2048]

#---------------------- data loading  ---------------------------------#

def load_dataset():
    (train_data_xval, train_data_yval), (test_data_xval, test_data_yval) = mnist.load_data(path="'mnist.npz'")
    return (train_data_xval, train_data_yval), (test_data_xval, test_data_yval)

#----------------------  Calculations  ---------------------------------#

def euclidian_distance(img1, img2):
    return np.linalg.norm(img1-img2)

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


def prediction_img_classify(neighbours):
    predicted_number = -1
    nearest_num = np.zeros(10)
    for i in range(len(neighbours)):
        nearest_num[neighbours[i]] += 1
    for i in range(len(nearest_num)):
        if nearest_num[i] > 0:
            predicted_number = i
    return predicted_number
'''
#vet funker
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
'''
#vet ikke om den funker:
def nearest_neighbour(data, labels, img, k_neighbours):
    neighbour_list = []
    for i in range(len(data)):
        dist = euclidian_distance(data[i], img)
        neighbour_list.append((dist, labels[i]))

    neighbour_list = sorted(neighbour_list, key=lambda dist: dist[0])

    return_labels = [label[1] for label in neighbour_list[:k_neighbours]]

    return return_labels

def confusion_matrix(predictions, test_data_yval):
    confusion_matrix = np.zeros((10, 10))
    for i, x in enumerate(predictions):
        confusion_matrix[test_data_yval[i], x] += 1
    plt.imshow(confusion_matrix)
    plt.show()
    return confusion_matrix

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

#------------------------- Plotting  ------------------------------------#
def plot_confusion_matix():
    print("Plotting confusion matrix")

def plot_misclassified_img():
    print("Plotting misclassified images")

def plot_classified_img():
    print("Plotting correctly classified images")

# ------------------------- Running ------------------------------------ #
def run_NN(train_data_xval, train_data_yval, test_data_xval, test_data_yval):
    print("Running Nearest Neighbour classifier")
    reg_errors = 0
    predicted = []
    k_neighbours = 1
    start_time = time.time()
    print(start_time)

    for i in range(len(test_data_xval)):
       nn = nearest_neighbour(train_data_xval, train_data_yval, test_data_xval[i], k_neighbours)
       predicted_num = nn #prediction_img_classify(nn)
       predicted.append(predicted_num)
       if predicted_num != test_data_yval[i]:
           reg_errors += 1

    end_time = time.time()
    print("Time used for run_NN function: ", end_time - start_time)
    print("Error rate: ", reg_errors/len(test_data_xval)*100, "%")
    confusion = confusion_matrix(predicted, test_data_yval)
    #print(confusion)

def run_KNN(train_data_xval, train_data_yval, test_data_xval, test_data_yval, k):
    print("Running K - Nearest Neighbour classifier")
    reg_errors = 0
    reg_errors_most = 0
    predicted = []
    k_neighbours = k
    start_time = time.time()

    for i in range(len(test_data_xval)):
       nn = nearest_neighbour(train_data_xval, train_data_yval, test_data_xval[i], k_neighbours)
       predicted_num = prediction_img_classify(nn)
       most_n = Most_Common(nn)
       predicted.append(predicted_num)

       if predicted_num != test_data_yval[i]:
           reg_errors += 1
       if most_n != test_data_yval[i]:
           reg_errors_most += 1

    end_time = time.time()
    print("Time used for run_KNN function: ", end_time - start_time)
    print("Error rate: ", reg_errors/len(test_data_xval)*100, "%")
    print("Error rate alt: ", reg_errors_most/len(test_data_xval)*100, "%")
    confusion = confusion_matrix(predicted, test_data_yval)
    #print(confusion)

def run_KCNN(train_data_xval, train_data_yval, test_data_xval, test_data_yval, n_clusters, k_num):
    print("Running cluster classifier")
    reg_errors = 0
    predicted = []
    training_cluster, train_labels = cluster(train_data_xval, train_data_yval, n_clusters)
    reshape_test = test_data_xval.flatten().reshape(len(test_data_xval), 784)
    start_time = time.time()

    for i in range(len(test_data_yval)):
        k = nearest_neighbour(training_cluster, train_labels, reshape_test[i], k_num)
        most_k = Most_Common(k)
        predicted_num = most_k
        predicted.append(predicted_num)
        if predicted_num != test_data_yval[i]:
            reg_errors += 1

    end_time = time.time()
    print("Cluster time: ", end_time - start_time)
    print("Error rate for cluster NN: ", reg_errors / len(test_data_xval) * 100, "%")
    print("Number of errors: ", reg_errors)

    confusion = confusion_matrix(predicted, test_data_yval)
    #print(confusion)

def run_CNN(train_data_xval, train_data_yval, test_data_xval, test_data_yval, n_clusters):
    print("Running cluster classifier")
    reg_errors = 0
    predicted = []
    classes = range(10)
    training_cluster = cluster(train_data_xval, train_data_yval, n_clusters)
    reshape_test = test_data_xval.flatten().reshape(len(test_data_xval), 784)
    start_time = time.time()
    print(start_time)

    for i in range(len(test_data_xval)):
        dist = []
        for k in range(len(training_cluster)):
            for j in range(len(training_cluster[0])):
                dist.append(euclidian_distance(reshape_test[i], training_cluster[k][j]))

        nn = np.argmin(dist)-1
        predicted_num = classes[int(nn//n_clusters)]
        predicted.append(predicted_num)
        if predicted_num != test_data_yval[i]:
            reg_errors += 1

    end_time = time.time()
    print("Cluster time: ", end_time - start_time)
    print("Error rate for cluster NN: ", reg_errors / len(test_data_xval) * 100, "%")
    print("Number of errors: ", reg_errors)
    confusion = confusion_matrix(predicted, test_data_yval)
    print(confusion)


def main():
    print("Main stuff")

    (train_data_xval, train_data_yval), (test_data_xval, test_data_yval) = load_dataset()
    train_data_xval = np.array(train_data_xval)
    train_data_yval = np.array(train_data_yval)
    test_data_xval = np.array(test_data_xval)
    test_data_yval = np.array(test_data_yval)

    run_KNN(train_data_xval[0:train_size], train_data_yval[0:train_size], test_data_xval[0:test_size],test_data_yval[0:test_size], k_num[2])
    '''
    #for i in range(len(k_num)):
    for j in range(len(num_clusters)):
        print("K_num: ", k_num[2], " --------------------------- Cluster_num: ", num_clusters[j])
        run_KCNN(train_data_xval, train_data_yval, test_data_xval, test_data_yval, num_clusters[j], k_num[2])
   
    run_NN(train_data_xval[0:train_size], train_data_yval[0:train_size], test_data_xval[0:test_size], test_data_yval[0:test_size])

 
    for i in range(len(k_num)):
        print("----------------------------------------------- K: ", k_num[i], "-------------------------------------------------------")
        run_CNN(train_data_xval[0:train_size], train_data_yval[0:train_size], test_data_xval[0:test_size], test_data_yval[0:test_size], num_clusters, k_num[i])
   
 run_KNN(train_data_xval[0:train_size], train_data_yval[0:train_size], test_data_xval[0:test_size], test_data_yval[0:test_size], k_num[i])

    training_sizes = [100, 1000, 2000, 5000, 10000, 60000]

    for i in range(len(training_sizes)):
        print("Training size: ", training_sizes[i])
        run_KNN(train_data_xval[0:training_sizes[i]], train_data_yval[0:training_sizes[i]], test_data_xval[0:test_size],
                test_data_yval[0:test_size], k_num[1])
    '''

    '''
    training_cluster = cluster(train_data_xval, train_data_yval, num_clusters)
    reshape_test = test_data_xval.flatten().reshape(len(test_data_xval), 784)
 
    for i in range(10):
        for k in range(len(training_cluster)):
            print(nearest_neighbour(reshape_test[i], train_data_yval, training_cluster[k], 1))
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(nearest_neighbour(train_data_xval, train_data_yval, test_data_xval[k], 1))
            print("-----------------------------------------------------------------------")
            
    '''


main()
