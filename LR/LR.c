//LR
// https://www.geeksforgeeks.org/multivariate-regression/
// https://www.ritchieng.com/multi-variable-linear-regression/
// Ciocirlan Stefan-Dan 15.05.2019

#include <stdlib.h>
#include <stdio.h>
#include "../common/common.h"
#include "../common/l_alg.h"

#define MAX_ITERATIONS 15

element_t g_theta[INPUT_LENGTH+1];
element_t g_alpha = 1;
element_t g_h_theta[TRAINING_DATA_LENGTH];
element_t g_error[TRAINING_DATA_LENGTH];
element_t g_delta[INPUT_LENGTH+1];
element_t g_training_data_X_LR[TRAINING_DATA_LENGTH][INPUT_LENGTH+1];
element_t g_test_data_LR[INPUT_LENGTH+1];

/*
* Converts training data adding the first column only with value 1
*/
void convert_LR(element_t training_data_X[][INPUT_LENGTH], element_t test_data[],
                size_t training_data_length, size_t input_length,
                element_t training_data_X_LR[][INPUT_LENGTH+1], element_t test_data_LR[]) {
    size_t index, jindex;
    for(index = 0; index < training_data_length; index++) {
        training_data_X_LR[index][0] = 1;
    }
    test_data_LR[0] = 1;
    for(index = 0; index < training_data_length; index++) {
        for(jindex = 0; jindex < input_length; jindex++) {
            training_data_X_LR[index][jindex+1] = training_data_X[index][jindex];
        }
    }
    for(jindex = 0; jindex < input_length; jindex++) {
        test_data_LR[jindex+1] = test_data[jindex];
    }

}

/*
* Works on all type of data vs normal quation where X.T*X must be invertible
* Train the linear regresion gausian descendet
*/
void training_gd_LR(element_t training_data_X[][INPUT_LENGTH+1], element_t training_data_Y[],
                    size_t training_data_length, size_t input_length,
                    element_t theta[], size_t max_iterations,
                    element_t alpha, element_t h_theta[], element_t error[],
                    element_t delta[]) {
    size_t index;
    for(index = 0; index < max_iterations; index++) {
        /*
        * training_data_X training_data_length x input_length
        * theta input_length x 1
        * h_theta training_data_length x 1
        * h_theta = training_data_X x theta (same as theta x training_data_X.T)
        */
        matrix_multiply_vector(&training_data_X[0][0], training_data_length,
                               input_length, theta, h_theta);
        /*
        * h_theta training_data_length x 1
        * training_data_Y training_data_length x 1
        * error training_data_length x 1
        * error = h_theta - training_data_Y 
        */
        vector_difference(h_theta, training_data_Y, training_data_length,
                          error);
        /*
        * error training_data_length x 1
        * training_data_X training_data_length x input_length
        * delta 1 x input_length -> input_length x 1
        * delta = error.T * training_data_X
        */
        vector_T_multiply_matrix(error, &training_data_X[0][0], training_data_length,
                                 input_length, delta);
        /*
        * delta input_length x 1
        * delta = (alpha/training_data_length) * delta
        */
        vector_multiply_scalar(delta, input_length,
                               alpha/training_data_length, delta);
        vector_difference(theta, delta, input_length, theta);
        /*
        * theta input_length x 1
        * delta input_length x 1
        * theta = theta - delta 
        */
        vector_difference(theta, delta, training_data_length,
                          theta);
    }
}



element_t g_inverse_matrix[INPUT_LENGTH+1][INPUT_LENGTH+1];
element_t g_aux1[INPUT_LENGTH+1][INPUT_LENGTH+1];
element_t g_aux2[INPUT_LENGTH+1];
element_t g_aux3[INPUT_LENGTH+1][INPUT_LENGTH+1];
element_t g_aux4[INPUT_LENGTH+1][INPUT_LENGTH+1];
element_t g_aux5[INPUT_LENGTH+1][INPUT_LENGTH+1];
/*
* normal equation
* theta = ((X.T*X)^-1) * X.T * y
*/
size_t training_normal_LR(element_t **training_data_X, element_t training_data_Y[],
                    size_t training_data_length, size_t input_length,
                    element_t theta[],
                    element_t **inverse_matrix, element_t **aux1,
                    element_t aux2[],  element_t **aux3,  element_t **aux4,
                    element_t **aux5) {
    size_t return_value;
    /*
    * training_data_X training_data_length x input_length
    * aux1 input_length x input_length
    * aux1 = X.T*X
    */
    matrix_T_multiply_matrix(&training_data_X[0][0], &training_data_X[0][0],
                             input_length, training_data_length, input_length,
                             &aux1[0][0]);
    /*
    * training_data_X training_data_length x input_length
    * training_data_Y training_data_length x 1
    * aux2 input_length x 1
    * aux2 = X.T*Y
    */
    matrix_T_multiply_vector(&training_data_X[0][0], training_data_length, input_length,
                             training_data_Y, aux2);
    

    /*
    * aux1 input_length x input_length
    * aux3 input_length x input_length
    * aux4 input_length x input_length
    * aux5 input_length x input_length
    * inverse_matrix input_length x input_length
    * inverse_matrix= aux^-1
    */
    return_value = matrix_inverse(&aux1[0][0], input_length,
                                  &aux3[0][0], &aux4[0][0], &aux5[0][0], &inverse_matrix[0][0]);
    if(return_value != 0) {
        return return_value;
    }

    /*
    * inverse_matrix input_length x input_length
    * aux2 input_length x 1
    * theta input_length x 1
    * theta = inverse_matrix * aux2
    */
    matrix_multiply_vector(&inverse_matrix[0][0], input_length, input_length,
                           aux2, theta);

}

/*
* y = theta.T*X
*/
element_t classify_LR(element_t test_data[],  element_t theta[],
                      size_t input_length) {
    element_t return_value;
    /*
    * theta input_length x 1
    * test_data input_length x 1
    * return_value 1 x 1
    * theta = theta.T x test_data
    */
    vector_T_multiply_vector(theta, test_data, input_length, &return_value);
    return return_value;
}

int main(void)
{
    convert_LR(g_training_data_X, g_test_data,
               TRAINING_DATA_LENGTH, INPUT_LENGTH,
               g_training_data_X_LR, g_test_data_LR);
    training_gd_LR(g_training_data_X_LR, g_training_data_Y,
                   TRAINING_DATA_LENGTH, INPUT_LENGTH+1,
                   g_theta, MAX_ITERATIONS,
                   g_alpha, g_h_theta, g_error,
                   g_delta);
    /* for normal ecuation
    size_t return_value;
    return_value = training_normal_LR(g_training_data_X_LR, g_training_data_Y,
                                      TRAINING_DATA_LENGTH, INPUT_LENGTH+1,
                                      g_theta,
                                      g_inverse_matrix, g_aux1,
                                      g_aux2, g_aux3, g_aux4,
                                      g_aux5);
    if(return_value != 0) {
        return return_value;
    }
    */
    element_t answer = classify_LR(g_test_data_LR, g_theta,
                                   INPUT_LENGTH+1);
    #ifdef DEBUG
        printf("Result: %f\n", answer);
    #endif
    return 0;
}
