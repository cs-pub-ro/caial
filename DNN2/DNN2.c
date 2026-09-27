//DNN L2
// https://cognitivedemons.wordpress.com/2017/07/06/a-neural-network-in-10-lines-of-c-code/
// https://www.geeksforgeeks.org/program-to-efficiently-calculate-ex/
// Ciocirlan Stefan-Dan 16.05.2019

#include <stdlib.h>
#include <stdio.h>
#include "../common/common.h"
#include "../common/l_alg.h"

#define MDIM 4


#define MAX_ITERATIONS 40
#define WEIGHTS_L0_LENGTH 3

element_t g_weights_l0[INPUT_LENGTH][WEIGHTS_L0_LENGTH];
element_t g_weights_l0_delta[INPUT_LENGTH][WEIGHTS_L0_LENGTH];
element_t g_weights_l1[WEIGHTS_L0_LENGTH];
element_t g_weights_l1_delta[WEIGHTS_L0_LENGTH];
element_t g_l1[TRAINING_DATA_LENGTH][WEIGHTS_L0_LENGTH];
element_t g_l1_delta[TRAINING_DATA_LENGTH][WEIGHTS_L0_LENGTH];
element_t g_l1_error[TRAINING_DATA_LENGTH][WEIGHTS_L0_LENGTH];
element_t g_l2[TRAINING_DATA_LENGTH];
element_t g_l2_error[TRAINING_DATA_LENGTH];
element_t g_l2_delta[TRAINING_DATA_LENGTH];

void initialise_dnn_2(element_t weights_l0[][WEIGHTS_L0_LENGTH], element_t weights_l1[],
                      size_t input_length, size_t weights_l0_length) {
    size_t index, jindex;
    for(index = 0; index < weights_l0_length; index++) {
        weights_l1[index] = 0.5;
        for(jindex = 0; jindex < input_length; jindex++) {
            weights_l0[jindex][index] = (index+1) + jindex*input_length;
        }
    }
}



/*
* training a dnn with 2 layer
* l0 = training_data_X
* l1 = f(training_data_X x weights_l0)
* l2 = l1 x weights_l1
*/
void training_dnn_2(element_t training_data_X[][INPUT_LENGTH], element_t training_data_Y[],
                    size_t training_data_length, size_t input_length,
                    size_t max_iterations,
                    element_t weights_l0[][WEIGHTS_L0_LENGTH], element_t weights_l1[],
                    size_t weights_l0_length,
                    element_t l1[][WEIGHTS_L0_LENGTH], element_t l1_error[][WEIGHTS_L0_LENGTH],
                    element_t l1_delta[][WEIGHTS_L0_LENGTH],
                    element_t l2[], element_t l2_error[],
                    element_t l2_delta[],
                    element_t weights_l0_delta[][WEIGHTS_L0_LENGTH], element_t weights_l1_delta[]) {
    size_t index;
    for (index = 0; index < max_iterations; index++) {
        //l1 = nonlin(np.dot(l0,syn0))
        /*
        * training_data_X training_data_length x input_length
        * weights_l0 input_length x weights_l0_length
        * l1 training_data_length x weights_l0_length
        * l1 = training_data_X x weights_l0
        */
        matrix_multiply_matrix(&training_data_X[0][0], &weights_l0[0][0], training_data_length,
                               input_length, weights_l0_length, &l1[0][0]);
        /*
        * l1 training_data_X x weights_l0_length
        * f sigmoid function f(x) = 1/(1 + e^-x)
        * l1 = f(l1)
        */
        matrix_sigmoid(&l1[0][0], training_data_length, weights_l0_length, &l1[0][0]);

        //l2 = nonlin(np.dot(l1,syn1))
        /*
        * l1 training_data_X x weights_l0_length
        * weights_l1 weights_l0_length x 1
        * l2 training_data_length x 1
        * l2 = l1 x weights_l1
        */
        matrix_multiply_vector(&l1[0][0], training_data_length, weights_l0_length,
                               weights_l1, l2);
        /*
        * l2 training_data_length x 1
        * f sigmoid function f(x) = 1/(1 + e^-x)
        * l2 = f(l2)
        */
        vector_sigmoid(l2, training_data_length, l2);



        //l2_error = y - l2
        /*
        * training_data_Y training_data_length x 1
        * l2 training_data_length x 1
        * l2_error training_data_length x 1
        * l2_error = training_data_Y - l2
        */
        vector_difference(training_data_Y, l2, training_data_length,
                          l2_error);
        
        //l2_delta = l2_error*nonlin(l2,deriv=True)
        /*
        * l2 training_data_length x 1
        * f sigmoid function f'(x) = f(x)(1 - f(x)) g(x) = x(1-x)
        * l2 = g(l2)
        */
        vector_sigmoid_derivate(l2, training_data_length, l2);
        /*
        * l2_delta training_data_length x 1
        * l2 training_data_length x 1
        * l2_error training_data_length x 1
        * l2_delta = l2_error .* l2
        */
        vector_multiply_elementwise(l2_error, l2, training_data_length,
                                    l2_delta);

        //l1_error = l2_delta.dot(syn1.T)
        /*
        * l2_delta training_data_length x 1
        * weights_l1 weights_l0_length x 1
        * l1_error training_data_length x weights_l0_length
        * l1_error = l2_delta * weights_l1.T
        */
        vector_multiply_vector_T(l2_delta, weights_l1,
                                 training_data_length, weights_l0_length,
                                 &l1_error[0][0]);


        //l1_delta = l1_error * nonlin(l1,deriv=True)
        /*
        * f sigmoid function f(x) = 1/(1 + e^-x)
        * l1 training_data_length x weights_l0_length
        * l1_delta training_data_length x weights_l0_length
        * l1_delta = f(l1)
        */        
        matrix_sigmoid(&l1[0][0], training_data_length,
                       weights_l0_length, &l1_delta[0][0]);
        /*
        * l1_error training_data_length x weights_l0_length
        * l1_delta training_data_length x weights_l0_length
        * l1_delta = l1_error .* l1_delta
        */
        matrix_multiply_elementwise(&l1_error[0][0], &l1_delta[0][0],
                                    training_data_length, weights_l0_length,
                                    &l1_delta[0][0]);


        //syn1 += l1.T.dot(l2_delta)
        /*
        * l1 training_data_length x weights_l0_length
        * l2_delta training_data_length x 1
        * weights_l1_delta weights_l0_length x 1
        * weights_l1_delta = l1.T .* weights_l1_delta
        */        
        matrix_T_multiply_vector(&l1[0][0], training_data_length, weights_l0_length,
                                l2_delta, weights_l1_delta);
        /*
        * weights_l1 weights_l0_length x 1
        * weights_l1_delta weights_l0_length x 1
        * weights_l1 = weights_l1 + weights_l1_delta
        */
        vector_sum(weights_l1, weights_l1_delta, weights_l0_length, weights_l1);

        //syn0 += l0.T.dot(l1_delta)
        /*
        * training_data_X training_data_length x input_length
        * l1_delta training_data_length x weights_l0_length
        * weights_l0_delta input_length x weights_l0_length
        * weights_l0_delta = training_data_X.T x l1_delta
        */
        matrix_T_multiply_matrix(&training_data_X[0][0], &l1_delta[0][0],
                                 input_length, training_data_length, weights_l0_length,
                                 &weights_l0_delta[0][0]);
        /*
        * weights_l0 input_length x weights_l0_length
        * weights_l0_delta input_length x weights_l0_length
        * weights_l0 = weights_l0 + weights_l0_delta
        */
        matrix_sum(&weights_l0[0][0], &weights_l0_delta[0][0],
                   input_length, weights_l0_length,
                   &weights_l0[0][0]);
    };
}

/*
* classify test data with a dnn with 1 layer
*/
element_t classify_dnn_2(element_t test_data[], size_t input_length,
                         element_t weights_l0[][WEIGHTS_L0_LENGTH], element_t weights_l1[],
                         element_t l1[],
                         size_t weights_l0_length) {
    //l1 = nonlin(np.dot(l0,syn0))
    /*
    * test_data input_length x 1
    * weights_l0 input_length x weights_l0_length
    * l1 1 x weights_l0_length
    * l1 = test_data.T x weights_l0
    */
    vector_T_multiply_matrix(test_data, &weights_l0[0][0],
                             input_length, weights_l0_length,
                             l1);
    /*
    * l1 weights_l0_length x 1
    * f sigmoid function f(x) = 1/(1 + e^-x)
    * l1 = f(l1)
    */
    vector_sigmoid(l1, weights_l0_length, l1);

    //l2 = nonlin(np.dot(l1,syn1))
    element_t return_value;
    /*
    * l1 weights_l0_length x 1
    * weights_l1 weights_l0_length x 1
    * return_value 1 x 1
    * return_value = l1.T x weights_l1
    */
    vector_T_multiply_vector(l1, weights_l1, weights_l0_length,
                             &return_value);
    /*
    * f sigmoid function
    * return_value = f(return_value)
    */
    number_sigmoid(return_value, &return_value);

    return return_value;
}

 
int main(void) {
    initialise_dnn_2(g_weights_l0, g_weights_l1,
                     INPUT_LENGTH, WEIGHTS_L0_LENGTH);
    training_dnn_2(g_training_data_X, g_training_data_Y,
                   TRAINING_DATA_LENGTH, INPUT_LENGTH,
                   MAX_ITERATIONS,
                   g_weights_l0, g_weights_l1,
                   WEIGHTS_L0_LENGTH,
                   g_l1, g_l1_error, g_l1_delta,
                   g_l2, g_l2_error, g_l2_delta,
                   g_weights_l0_delta, g_weights_l1_delta);
    element_t answer = classify_dnn_2(g_test_data, INPUT_LENGTH,
                                      g_weights_l0, g_weights_l1,
                                      g_weights_l1_delta, //for using less memory
                                      WEIGHTS_L0_LENGTH);
    #ifdef DEBUG
        printf("Result: %f\n", answer);
    #endif
    return 0;
}