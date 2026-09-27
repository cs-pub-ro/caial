//DNN L1
// https://cognitivedemons.wordpress.com/2017/07/06/a-neural-network-in-10-lines-of-c-code/
// https://www.geeksforgeeks.org/program-to-efficiently-calculate-ex/
// Ciocirlan Stefan-Dan 15.05.2019

#include <stdlib.h>
#include <stdio.h>
#include "../common/common.h"
#include "../common/l_alg.h"

#define MAX_ITERATIONS 40

element_t g_weights[INPUT_LENGTH];
element_t g_prediction[TRAINING_DATA_LENGTH];
element_t g_prediction_error[TRAINING_DATA_LENGTH];
element_t g_prediction_delta[TRAINING_DATA_LENGTH];
element_t g_weights_delta[INPUT_LENGTH];
element_t g_aux1[TRAINING_DATA_LENGTH];
element_t g_aux2[TRAINING_DATA_LENGTH];

void initialise_dnn_1(element_t weights[], size_t input_length) {
    size_t index;
    for(index = 0; index < input_length; index++) {
        weights[index] = 0.5;
    }
}
/*
* training a dnn with 1 layer
*/
void training_dnn_1(element_t training_data_X[][INPUT_LENGTH], element_t training_data_Y[],
                    size_t training_data_length, size_t input_length,
                    size_t max_iterations,
                    element_t weights[], element_t prediction[], element_t prediction_error[],
                    element_t weights_delta[], element_t prediction_delta[]) {
    size_t index;
    for (index = 0; index < max_iterations; index++) {
        /*
        * training_data_X training_data_length x input_length
        * weights input_length x 1
        * prediction training_data_length x 1
        * prediction = training_data_X x theta
        */
        matrix_multiply_vector(&training_data_X[0][0], training_data_length, input_length,
                               weights, prediction);
        
        /*
        * prediction training_data_length x 1
        * f sigmoid function f(x) = 1/(1 + e^-x)
        * prediction = f(prediction)
        */
        vector_sigmoid(prediction, training_data_length, prediction);
        /*
        * training_data_Y training_data_length x 1
        * prediction training_data_length x 1
        * prediction_error training_data_length x 1
        * prediction_error = training_data_Y - prediction
        */
        vector_difference(training_data_Y, prediction, training_data_length,
                          prediction_error);
        /*
        * prediction training_data_length x 1
        * f sigmoid function f'(x) = f(x)(1 - f(x)) g(x) = x(1-x)
        * prediction = g(prediction)
        */
        vector_sigmoid_derivate(prediction, training_data_length, prediction);
        /*
        * prediction_delta training_data_length x 1
        * prediction training_data_length x 1
        * prediction_error training_data_length x 1
        * prediction_delta = prediction_error .* prediction
        */
        vector_multiply_elementwise(prediction_error, prediction, training_data_length,
                                    prediction_delta);
        /*
        * training_data_X training_data_length x input_length
        * prediction_delta training_data_length x 1
        * weights_delta input_length x 1
        * weights_delta = training_data_X.T x prediction_delta
        */
        matrix_T_multiply_vector(&training_data_X[0][0], training_data_length, input_length,
                                 prediction_delta, weights_delta);
        /*
        * weights input_length x 1
        * weights_delta input_length x 1
        * weights = weights + weights_delta
        */
        vector_sum(weights, weights_delta, input_length, weights);
    };
}

/*
* classify test data with a dnn with 1 layer
*/
element_t classify_dnn_1(element_t test_data[], size_t input_length,
                         element_t weights[]) {
    element_t return_value;
    /*
    * test_data input_length x 1
    * weights input_length x 1
    * return_value 1 x 1
    * return_value = test_data.T x weights
    */
    vector_T_multiply_vector(test_data, weights, input_length,
                             &return_value);
    /*
    * f sigmoid function
    * return_value = f(return_value)
    */
    number_sigmoid(return_value, &return_value);
    return return_value;
}

 
int main(void) {
    initialise_dnn_1(g_weights, INPUT_LENGTH);
    training_dnn_1(g_training_data_X, g_training_data_Y,
                   TRAINING_DATA_LENGTH, INPUT_LENGTH,
                   MAX_ITERATIONS,
                   g_weights, g_prediction, g_prediction_error,
                   g_weights_delta, g_prediction_delta);
    element_t answer = classify_dnn_1(g_test_data, INPUT_LENGTH,
                                      g_weights);
    #ifdef DEBUG
        printf("Result: %f\n", answer);
    #endif
    return 0;
}