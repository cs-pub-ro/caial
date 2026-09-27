// KMEANS
// https://www.geeksforgeeks.org/k-means-clustering-introduction/
// Ciocirlan Stefan-Dan 14.05.2019

#include <stdlib.h>
#include <stdio.h>
#include "../common/common.h"

#define K_VALUE 7
#define MAX_ITERATIONS 40


//cluster
size_t g_training_position_cluster[TRAINING_DATA_LENGTH];
element_t g_cluster_mean[K_VALUE][INPUT_LENGTH];
size_t g_cluster_len[K_VALUE];
element_t g_cluster_value[K_VALUE];

/* Calculate the euclidian distance between vectors X and Y with a length given
*/
element_t euclidian_distance_square (element_t X[], element_t Y[], size_t length) {
    size_t index;
    element_t sum = 0;
    for(index = 0; index < length; index++) {
        sum = sum + (X[index]-Y[index])*(X[index]-Y[index]);
    }
    return sum;
}

/*
* Initialize the cluster means and training data point position in cluster
*/
void initialise_cluster(element_t training_data_X[][INPUT_LENGTH], size_t training_position_cluster[],
                        size_t training_data_length, size_t input_length,
                        element_t cluster_mean[][INPUT_LENGTH], size_t cluster_len[],
                        size_t k) {
    size_t index, jindex, kindex;
    element_t min;
    element_t max;
    
    // position in no cluster at the begining
    for(jindex = 0; jindex < training_data_length; jindex++) {
        training_position_cluster[jindex] = -1;
    }

    //every cluster does'n have any points inside
    for(kindex = 0; kindex < k; kindex++) {
        cluster_len[kindex] = 0;
    }

    /*
    * Calculate the minimum and maximum value for every column of input
    * and then generates k intervals of the same size between minimum
    * value and maximum value. This right value of the interval will
    * become de mean for a cluster
    */
    for(index = 0; index < input_length; index++) {
        min = training_data_X[0][index];
        max = training_data_X[0][index];
        for(jindex = 1; jindex < training_data_length; jindex++) {
            if(min > training_data_X[jindex][index]) {
                min = training_data_X[jindex][index];
            }
            if(max < training_data_X[jindex][index]) {
                max = training_data_X[jindex][index];
            }
        }
        for(kindex = 0; kindex < k; kindex++) {
            cluster_mean[kindex][index] = min + (max - min) * kindex / k;
        }
    }
}
/*
* It calculates the colsest cluster mean to the given point X and return
* the cluster index
*/
size_t classify_cluster(element_t X[], element_t cluster_mean[][INPUT_LENGTH],
                        size_t input_length, size_t k) {
    size_t kindex;
    element_t min;
    element_t aux;
    size_t return_index;
    min = euclidian_distance_square(X, cluster_mean[0], input_length);
    return_index = 0;
    for(kindex = 1; kindex < k; kindex++) {
        aux = euclidian_distance_square(X, cluster_mean[kindex], input_length);
        if(min > aux) {
            min = aux;
            return_index = kindex;
        }
    }
    return return_index;
}

/*
* it deletes the X point from the cluster and recalculate the mean
*/
void update_mean_minus(element_t X[], element_t cluster_mean[][INPUT_LENGTH],
                       size_t cluster_len[], size_t input_length,
                       size_t kindex) {
    size_t index;
    for(index = 0; index < input_length; index++) {
        cluster_mean[kindex][index] = (cluster_mean[kindex][index] * cluster_len[kindex] - X[index])
                                       / (cluster_len[kindex] - 1);
    }
    cluster_len[kindex] = cluster_len[kindex] - 1;
}

/*
* it add the X point to the cluster and recalculate the mean
*/
void update_mean_plus(element_t X[], element_t cluster_mean[][INPUT_LENGTH],
                       size_t cluster_len[], size_t input_length,
                       size_t kindex)  {
    size_t index;
    for(index = 0; index < input_length; index++) {
        cluster_mean[kindex][index] = (cluster_mean[kindex][index] * cluster_len[kindex] + X[index])
                                      / (cluster_len[kindex] + 1);
    }
    cluster_len[kindex] = cluster_len[kindex] + 1;
}

/*
* It is training the a kmeans clusters with training_data_X
*/
void training_kmeans(element_t training_data_X[][INPUT_LENGTH], size_t training_position_cluster[],
                        size_t training_data_length, size_t input_length,
                        element_t cluster_mean[][INPUT_LENGTH], size_t cluster_len[],
                        size_t k, size_t max_iterations) {
    size_t index, jindex, kindex, old_kindex;
    size_t has_changed;
    size_t rindex;
    //do a maximum number of iterations to every node
    for(rindex = 0; rindex < max_iterations; rindex++) {
        has_changed = 0;
        /*
        *for every point from the training data calculates the cluster index
        *if is different from its old cluster than update both clusters
        */
        for(jindex = 0; jindex < training_data_length; jindex++) {
            old_kindex = training_position_cluster[jindex];
            kindex = classify_cluster(training_data_X[jindex], cluster_mean,
                                      input_length, k);
            if(kindex != old_kindex) {
                if(old_kindex != -1) {
                    update_mean_minus(training_data_X[jindex], cluster_mean,
                                      cluster_len, input_length, old_kindex);
                }
                update_mean_plus(training_data_X[jindex], cluster_mean,
                                    cluster_len, input_length, kindex);
                training_position_cluster[jindex] = kindex;
                has_changed = 1;
            }
        }
        if(has_changed == 0) {
            break;
        }
    }
}

/*
* Find the values with the higher frecuency in every cluster
*/
void kmeans_clusters_value(element_t cluster_value[],
                              element_t training_data_Y[], size_t training_position_cluster[],
                              size_t training_data_length, size_t k) {
    size_t index, jindex, kindex;
    size_t current_frequency;
    size_t max_frequency;
    for(kindex = 0; kindex < k; kindex++) {
        max_frequency = 0;
        for(jindex = 0; jindex < training_data_length; jindex++) {
            if(training_position_cluster[jindex] == kindex) {
                current_frequency = 1;
                /*
                * Search the same value in the next points in training data,
                * because if the values has also before it was already counted
                */
                for(index = jindex + 1; index < training_data_length; index++) {
                    if(training_position_cluster[index] == kindex) {
                        if(training_data_Y[jindex] == training_data_Y[index]) {
                            current_frequency = current_frequency + 1;
                        }
                    }
                }
                //if it is the the new max frequency value
                if(current_frequency > max_frequency) {
                    cluster_value[kindex] = training_data_Y[jindex];
                    max_frequency = current_frequency;
                }
            }
        }
    }

    
}

/*
* Given a test_data it finds the cluster and return the value of the cluster
*/
element_t classify_kmeans(element_t test_data[], element_t cluster_value[],
                          size_t input_length,
                          element_t cluster_mean[][INPUT_LENGTH], size_t k) {
    size_t kindex;
    element_t return_value;
    kindex = classify_cluster(test_data, cluster_mean, input_length, k);
    return_value = cluster_value[kindex];
    return return_value;
}


int main(void)
{

    //initialise the cluster
    initialise_cluster(g_training_data_X, g_training_position_cluster,
                       TRAINING_DATA_LENGTH, INPUT_LENGTH,
                       g_cluster_mean, g_cluster_len,
                       K_VALUE);
    training_kmeans(g_training_data_X, g_training_position_cluster,
                    TRAINING_DATA_LENGTH, INPUT_LENGTH,
                    g_cluster_mean, g_cluster_len,
                    K_VALUE, MAX_ITERATIONS);
    kmeans_clusters_value(g_cluster_value,
                          g_training_data_Y, g_training_position_cluster,
                          TRAINING_DATA_LENGTH, K_VALUE);
    element_t answer = classify_kmeans(g_test_data, g_cluster_value,
                                       INPUT_LENGTH,
                                       g_cluster_mean, K_VALUE);
    #ifdef DEBUG
        printf("Result: %f\n", answer);
    #endif
    return 0;
}
