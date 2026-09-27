//NB
// https://msdn.microsoft.com/en-us/magazine/jj891056.aspx
// https://www.machinelearningplus.com/predictive-modeling/how-naive-bayes-algorithm-works-with-example-and-full-code/
// Ciocirlan Stefan-Dan 14.05.2019
#include <stdlib.h>
#include <stdio.h>
#include "../common/common.h"

#define INPUT_GROUPS_LENGTH 5
#define OUTPUT_GROUPS_LENGTH 3

size_t g_training_data_X_BNB [TRAINING_DATA_LENGTH][INPUT_LENGTH];
size_t g_training_data_Y_BNB [TRAINING_DATA_LENGTH];
size_t g_test_data_BNB[INPUT_LENGTH];
size_t g_joints[INPUT_LENGTH][INPUT_GROUPS_LENGTH][OUTPUT_GROUPS_LENGTH];
size_t g_dep_joints[OUTPUT_GROUPS_LENGTH];
real_t g_partial_probability[OUTPUT_GROUPS_LENGTH];
real_t g_probability[OUTPUT_GROUPS_LENGTH];


    
/*
* converts to a format that works with BNB by grouping interval of values
* at the same value
*/
void convert_to_BNB_memory(element_t training_data_X[][INPUT_LENGTH], element_t training_data_Y[],
                    element_t test_data[],
                    size_t training_data_length, size_t input_length,
                    size_t training_data_X_BNB[][INPUT_LENGTH], size_t training_data_Y_BNB[],
                    size_t test_data_BNB[],
                    size_t input_groups_length, size_t output_groups_length) {
    size_t index, jindex, kindex;
    element_t min;
    element_t max;
    element_t value;

    //Converting traininf_data_X and test_data
    for(index = 0; index < input_length; index++) {
        // Find the minimum and maximum value for the current input column
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
        //setting all to the biggest group value
        for(jindex = 0; jindex < training_data_length; jindex++) {
            training_data_X_BNB[jindex][index] = input_groups_length - 1;
        }
        test_data_BNB[index] = input_groups_length - 1;

        /* going trough every group if the input value is lower
        * than the maximum value of the group add the current input to
        * to the group
        */
        for(kindex = input_groups_length - 2; kindex > -1; kindex--) {
            value = min + ((max - min) * (kindex+1)) / input_groups_length;
            for(jindex = 0; jindex < training_data_length; jindex++) {
                if(training_data_X[jindex][index] <= value) {
                    training_data_X_BNB[jindex][index] = kindex;
                }
            }
            if(test_data[index] <= value) {
                test_data_BNB[index] = kindex;
            }
        }
    }

    //doing the same thing but for output
    min = training_data_Y[0];
    max = training_data_Y[0];
    for(jindex = 1; jindex < training_data_length; jindex++) {
        if(min > training_data_Y[jindex]) {
            min = training_data_Y[jindex];
        }
        if(max < training_data_Y[jindex]) {
            max = training_data_Y[jindex];
        }
    }

    for(jindex = 0; jindex < training_data_length; jindex++) {
        training_data_Y_BNB[jindex] = output_groups_length - 1;
    }

    for(kindex = output_groups_length - 2; kindex > -1; kindex--) {
        value = min + ((max - min) * (kindex+1)) / output_groups_length;
        for(jindex = 0; jindex < training_data_length; jindex++) {
            if(training_data_Y[jindex] <= value) {
                training_data_Y_BNB[jindex] = kindex;
            }
        }
    }

}

/* Training the BNB
* joints [index][kindex][vindex] - how many elements of traning_data_X has on column
* index the value kindex and on training_data_Y has value vindex
* dep_joints[vindex] - how many elements in training_data_Y has value vindex
*/
void training_BNB(size_t training_data_X[][INPUT_LENGTH], size_t training_data_Y[],
                    size_t training_data_length, size_t input_length,
                    size_t input_groups_length, size_t output_groups_length,
                    size_t joints[][INPUT_GROUPS_LENGTH][OUTPUT_GROUPS_LENGTH], size_t dep_joints[]) {
    size_t index, jindex, kindex, vindex;

    /*
    Initialize joints
    1 - if you want Laplacian Smoothing
    0 - otherwise
    */
    for(index = 0; index < input_length; index++) {
        for(kindex = 0; kindex < input_groups_length; kindex++) {
            for(vindex = 0; vindex < output_groups_length; vindex++) {
                joints[index][kindex][vindex] = 1;
            }
        }
    }

    //calculate the joints
    for(jindex = 0; jindex < training_data_length; jindex++) {
        for(index = 0; index < input_length; index++) {
            kindex = training_data_X[jindex][index];
            vindex = training_data_Y[jindex];
            joints[index][kindex][vindex] = joints[index][kindex][vindex] + 1;
        }
    }

    //calculate the dep_joints
    for(vindex = 0; vindex < output_groups_length; vindex++) {
        dep_joints[vindex] = 0;
    }
    for(jindex = 0; jindex < training_data_length; jindex++) {
        vindex = training_data_Y[jindex];
        dep_joints[vindex] = dep_joints[vindex] + 1;
    }
}

size_t classify_BNB(size_t test_data[], size_t input_length,
                    size_t input_groups_length, size_t output_groups_length,
                    size_t joints[][INPUT_GROUPS_LENGTH][OUTPUT_GROUPS_LENGTH], size_t dep_joints[],
                    real_t partial_probability[], real_t probability[]) {

    size_t index, jidenx, kindex, pindex;
    size_t elements_number;
    real_t partial_probability_sum;
    real_t max_probability;
    size_t return_value;
    
    //initialise the partial probabilities with 1.0
    for(pindex = 0; pindex < output_groups_length; pindex++) {
        partial_probability[pindex] = 1.0;
    }

    /* Caculate de partial probability by multipling with probability
    * every input to happen for the given output
    * pp[Y|X] = prod for i in X_length of p(Y|Xi)
                /total_number_of_elements
    * p(Y/Xi) = joints with value Xi on column I and value Y on training_data_Y are
                / number of elements in training_data_Y with value Y
    */
    for(index = 0; index < input_length; index++) {
        for(pindex = 0; pindex < output_groups_length; pindex++) {
            partial_probability[pindex] = partial_probability[pindex]
            * joints[index][test_data[index]][pindex] / dep_joints[pindex];
        }
    }
    /* calcualte the total_number_of_elements and find every partial probability
    */
    elements_number = 0;
    for(pindex = 0; pindex < output_groups_length; pindex++) {
        elements_number = elements_number + dep_joints[pindex];
    }
    for(pindex = 0; pindex < output_groups_length; pindex++) {
        partial_probability[pindex] = partial_probability[pindex]
        * dep_joints[pindex] / elements_number;
    }

    partial_probability_sum = 0;
    /* calcualte the sum of partial probabilities
    */
    for(pindex = 0; pindex < output_groups_length; pindex++) {
        partial_probability_sum = partial_probability_sum + partial_probability[pindex];
    }
    /*
    * calculate every probability dor output
    * p(Y|Xi) = pp(Y|Xi) / sum for j in Y_VALUES of pp(yj|Xi)
    */
    for(pindex = 0; pindex < output_groups_length; pindex++) {
        probability[pindex] = partial_probability[pindex] / partial_probability_sum;
    }
    
    /* chose the biggest probability
    */
    max_probability = probability[0];
    return_value = 0;
    for(pindex = 1; pindex < output_groups_length; pindex++) {
        if(probability[pindex] > max_probability) {
            max_probability = probability[pindex];
            return_value = pindex;
        }
    }
    return return_value;
}
int main(void)
{
    convert_to_BNB_memory(g_training_data_X, g_training_data_Y,
                    g_test_data,
                    TRAINING_DATA_LENGTH, INPUT_LENGTH,
                    g_training_data_X_BNB, g_training_data_Y_BNB,
                    g_test_data_BNB,
                    INPUT_GROUPS_LENGTH, OUTPUT_GROUPS_LENGTH);
    training_BNB(g_training_data_X_BNB, g_training_data_Y_BNB,
                    TRAINING_DATA_LENGTH, INPUT_LENGTH,
                    INPUT_GROUPS_LENGTH, OUTPUT_GROUPS_LENGTH,
                    g_joints, g_dep_joints);
    size_t answer = classify_BNB(g_test_data_BNB, INPUT_LENGTH,
                    INPUT_GROUPS_LENGTH, OUTPUT_GROUPS_LENGTH,
                    g_joints, g_dep_joints,
                    g_partial_probability, g_probability);
   
    #ifdef DEBUG
        printf("Result: %d\n", answer);
    #endif
    return 0;
}