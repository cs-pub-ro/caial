//CNN 
// https://cognitivedemons.wordpress.com/2017/07/06/a-neural-network-in-10-lines-of-c-code/
// https://www.geeksforgeeks.org/program-to-efficiently-calculate-ex/
// Ciocirlan Stefan-Dan 17.05.2019

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "../common/common.h"
#include "../common/l_alg.h"

#define MAX_ITERATIONS 40


element_t g_input_layer[290400]; // 55x55x96
element_t g_conv_output[3025]; // 55x55
element_t g_output_layer[290400]; // 55x55x96
element_t g_kernel[9216]; //6x6x256

size_t max(size_t a, size_t b) {
    return (a > b) ? a : b;
}

void convolution_3D(element_t *i_matrix_A, size_t i_rows, size_t i_columns,
                    size_t i_start_depth, size_t i_end_depth, size_t i_total_depth,
                    element_t *i_kernel_B, size_t i_kernel_size,
                    size_t i_padding_type, size_t i_padding[],
                    size_t i_stride_row, size_t i_stride_column,
                    element_t *o_matrix_C, size_t *o_rows, size_t *o_columns) {
    size_t index, jindex;
    size_t rindex, pindex, kindex;
    element_t sum;
    /*
    * Setring the padding
    */
    if(i_padding_type == 0) {
        for(index = 0; index < 4; index++) {
            i_padding[index] = 0;
        }
    } else {
        if(i_padding_type == 1) {
            for(index = 0; index < 4; index++) {
                i_padding[index] = (i_kernel_size - 1) / 2;
            }
        }
    }
    *o_rows = (i_rows + i_padding[0] + i_padding[1] - i_kernel_size) / i_stride_row + 1;
    *o_columns = (i_columns + i_padding[2] + i_padding[3] - i_kernel_size) / i_stride_column + 1;

    size_t matrix_row, matrix_column;
    for(index = 0; index < *o_rows; index++) {
        for(jindex = 0; jindex < *o_columns; jindex++) {
            sum = 0;
            for(rindex = 0; rindex < i_kernel_size; rindex++) {
                for(pindex = 0; pindex < i_kernel_size; pindex++) {
                    for(kindex = i_start_depth; kindex < i_end_depth; kindex++) {
                        matrix_row = index * i_stride_row + rindex;
                        matrix_column = jindex * i_stride_column + pindex;
                        if( matrix_row < i_padding[0] || matrix_row >= i_rows + i_padding[0]) {
                            sum = sum + 0;
                        } else {
                            if( matrix_column < i_padding[2] || matrix_column >= i_columns + i_padding[2]) {
                                sum = sum + 0;
                            } else {
                                matrix_row = matrix_row - i_padding[0];
                                matrix_column = matrix_column - i_padding[2];
                                sum = sum + (*(i_matrix_A + matrix_row * i_columns + matrix_column * i_total_depth + kindex))
                                      *  (*(i_kernel_B + rindex * i_kernel_size + pindex * (i_end_depth-i_start_depth) + kindex));
                            }
                        }
                    }
                }
            }
            *(o_matrix_C + index * (*o_columns) + jindex) = sum;
        }
    }
}

void convolution_2D(element_t *i_matrix_A, size_t i_rows, size_t i_columns,
                    element_t *i_kernel_B, size_t i_kernel_size,
                    size_t i_padding_type, size_t i_padding[],
                    size_t i_stride_row, size_t i_stride_column,
                    element_t *o_matrix_C, size_t *o_rows, size_t *o_columns) {
    size_t index, jindex;
    size_t rindex, pindex, kindex;
    element_t sum;
    /*
    * Setring the padding
    */
    if(i_padding_type == 0) {
        for(index = 0; index < 4; index++) {
            i_padding[index] = 0;
        }
    } else {
        if(i_padding_type == 1) {
            for(index = 0; index < 4; index++) {
                i_padding[index] = (i_kernel_size - 1) / 2;
            }
        }
    }
    *o_rows = (i_rows + i_padding[0] + i_padding[1] - i_kernel_size) / i_stride_row + 1;
    *o_columns = (i_columns + i_padding[2] + i_padding[3] - i_kernel_size) / i_stride_column + 1;

    size_t matrix_row, matrix_column;
    for(index = 0; index < *o_rows; index++) {
        for(jindex = 0; jindex < *o_columns; jindex++) {
            sum = 0;
            for(rindex = 0; rindex < i_kernel_size; rindex++) {
                for(pindex = 0; pindex < i_kernel_size; pindex++) {
                    matrix_row = index * i_stride_row + rindex;
                    matrix_column = jindex * i_stride_column + pindex;
                    if( matrix_row < i_padding[0] || matrix_row >= i_rows + i_padding[0]) {
                        sum = sum + 0;
                    } else {
                        if( matrix_column < i_padding[2] || matrix_column >= i_columns + i_padding[2]) {
                            sum = sum + 0;
                        } else {
                            matrix_row = matrix_row - i_padding[0];
                            matrix_column = matrix_column - i_padding[2];
                            sum = sum + (*(i_matrix_A + matrix_row * i_columns + matrix_column))
                                    * (*(i_kernel_B + rindex * i_kernel_size + pindex));
                        }
                    }
                }
            }
            *(o_matrix_C + index * (*o_columns) + jindex) = sum;
        }
    }
}

void ReLU_3D(element_t *i_matrix_A, size_t i_rows, size_t i_columns, size_t i_depth,
             element_t *o_matrix_B) {
    size_t index, jindex, kindex;
    for(index = 0; index < i_rows; index++) {
        for(jindex = 0; jindex < i_columns; jindex++) {
            for(kindex = 0; kindex < i_depth; kindex++) {
                if(*(i_matrix_A + index * i_columns + jindex * i_depth + kindex) < 0) {
                    *(o_matrix_B + index * i_columns + jindex * i_depth + kindex) = 0;
                } else {
                    *(o_matrix_B + index * i_columns + jindex * i_depth + kindex)
                    = *(i_matrix_A + index * i_columns + jindex * i_depth + kindex);
                }
            }
        }
    }
}



void ReLU_2D(element_t *i_matrix_A, size_t i_rows, size_t i_columns,
             element_t *o_matrix_B) {
    size_t index, jindex;
    for(index = 0; index < i_rows; index++) {
        for(jindex = 0; jindex < i_columns; jindex++) {
            if(*(i_matrix_A+ index * i_columns + jindex) < 0) {
                *(o_matrix_B + index * i_columns + jindex) = 0;
            } else {
                *(o_matrix_B + index * i_columns + jindex)
                = *(i_matrix_A+ index * i_columns + jindex);
            }
        }
    }
}

void max_pooling_3D(element_t *i_matrix_A, size_t i_rows, size_t i_columns, size_t i_depth,
                size_t i_filter_row, size_t i_filter_column,
                size_t i_stride_row, size_t i_stride_column,
                size_t i_padding_type,
                element_t *o_matrix_B, size_t *o_rows, size_t *o_columns) {
    size_t index, jindex, kindex, rindex, pindex;
    size_t matrix_row, matrix_column;
    element_t max;
    size_t padding;
    if(i_padding_type == 1) {
        padding = ( - 1) / 2;
    } else {
        padding = 0;
    }
    *o_rows = (i_rows + 2 * padding - i_filter_row) / i_stride_row + 1;
    *o_columns = (i_columns + 2 * padding - i_filter_column) / i_stride_column + 1;
    for(index = 0; index < *o_rows; index++) {
        for(jindex = 0; jindex < *o_columns; jindex++) {
            for(kindex = 0; kindex < i_depth; kindex++) {
                max = -1; // because it does'n have negative numbers
                for(rindex = 0; rindex < i_filter_row; rindex++) {
                    for(pindex = 0; pindex < i_columns; pindex++) {
                        matrix_row = index * i_stride_row + rindex;
                        matrix_column = jindex * i_stride_column + pindex;
                        if( matrix_row < padding || matrix_row >= i_rows + padding) {
                            max = max;
                        } else {
                            if( matrix_column < padding || matrix_column >= i_columns + padding) {
                                max = max;
                            } else {
                                matrix_row = matrix_row - padding;
                                matrix_column = matrix_column - padding;
                                if( (*(i_matrix_A + matrix_row * i_columns + matrix_column * i_depth + kindex)) >  max) {
                                    max = (*(i_matrix_A + matrix_row * i_columns + matrix_column * i_depth + kindex));
                                }
                            }
                        }
                    }
                }
                *(o_matrix_B + index * (*o_columns) + jindex * i_depth + kindex) = max;
            }
        }
    }

}

void max_pooling_2D(element_t *i_matrix_A, size_t i_rows, size_t i_columns,
                    size_t i_filter_row, size_t i_filter_column,
                    size_t i_stride_row, size_t i_stride_column,
                    size_t i_padding_type,
                    element_t *o_matrix_B, size_t *o_rows, size_t *o_columns) {
    size_t index, jindex, rindex, pindex;
    size_t matrix_row, matrix_column;
    size_t padding;
    if(i_padding_type == 1) {
        padding = (max(i_filter_row, i_filter_column) - 1) / 2;
    } else {
        padding = 0;
    }
    *o_rows = (i_rows + 2 * padding - i_filter_row) / i_stride_row + 1;
    *o_columns = (i_columns + 2 * padding - i_filter_column) / i_stride_column + 1;
    for(index = 0; index < *o_rows; index++) {
        for(jindex = 0; jindex < *o_columns; jindex++) {
            element_t max = -1; // because it does'n have negative numbers
            for(rindex = 0; rindex < i_filter_row; rindex++) {
                for(pindex = 0; pindex < i_filter_column; pindex++) {
                    matrix_row = index * i_stride_row + rindex;
                    matrix_column = jindex * i_stride_column + pindex;
                    if( matrix_row < padding || matrix_row >= i_rows + padding) {
                        max = max;
                    } else {
                        if( matrix_column < padding || matrix_column >= i_columns + padding) {
                            max = max;
                        } else {
                            matrix_row = matrix_row - padding;
                            matrix_column = matrix_column - padding;
                            if( (*(i_matrix_A + matrix_row * i_columns + matrix_column)) >  max) {
                                max = (*(i_matrix_A + matrix_row * i_columns + matrix_column));
                            }
                        }
                    }
                }
            }
            *(o_matrix_B + index * (*o_columns) + jindex) = max;
        }
    }

}

void full_conected(element_t i_vector_A[], size_t i_length,
                   element_t i_kernel_B[], element_t *o_element) {
    vector_T_multiply_vector(i_vector_A, i_kernel_B,
                             i_length, o_element);
}

void softmax(element_t i_vector_A[], size_t i_length,
             element_t o_vector_B[]) {
    size_t index;
    element_t sum, e_x;
    sum = 0;
    for(index = 0; index < i_length; index++) {
        if(i_vector_A[index] >= 0) {
            e_x = number_exponential(EXPONENTIAL_PRECISION, i_vector_A[index]);
            sum = sum + e_x;
        } else {
            e_x = number_exponential(EXPONENTIAL_PRECISION, -1*i_vector_A[index]);
            sum = sum + 1/e_x;
        }
    }

    for(index = 0; index < i_length; index++) {
        if(i_vector_A[index] >= 0) {
            e_x = number_exponential(EXPONENTIAL_PRECISION, i_vector_A[index]);
            o_vector_B[index] = e_x / sum;
        } else {
            e_x = number_exponential(EXPONENTIAL_PRECISION, -1*i_vector_A[index]);
            o_vector_B[index] = 1 / (e_x * sum);
        }
    }
}

void avg_pooling_3D(element_t *i_matrix_A, size_t i_rows, size_t i_columns, size_t i_depth,
                size_t i_filter_row, size_t i_filter_column,
                size_t i_stride_row, size_t i_stride_column,
                size_t i_padding_type,
                element_t *o_matrix_B, size_t *o_rows, size_t *o_columns) {
    size_t index, jindex, kindex, rindex, pindex;
    size_t matrix_row, matrix_column;
    element_t avg;
    size_t padding;
    if(i_padding_type == 1) {
        padding = (max(i_filter_row, i_filter_column) - 1) / 2;
    } else {
        padding = 0;
    }
    *o_rows = (i_rows + 2 * padding - i_filter_row) / i_stride_row + 1;
    *o_columns = (i_columns + 2 * padding - i_filter_column) / i_stride_column + 1;
    for(index = 0; index < *o_rows; index++) {
        for(jindex = 0; jindex < *o_columns; jindex++) {
            for(kindex = 0; kindex < i_depth; kindex++) {
                avg = 0; // because it does'n have negative numbers
                for(rindex = 0; rindex < i_filter_row; rindex++) {
                    for(pindex = 0; pindex < i_filter_column; pindex++) {
                        matrix_row = index * i_stride_row + rindex;
                        matrix_column = jindex * i_stride_column + pindex;
                        if( matrix_row < padding || matrix_row >= i_rows + padding) {
                            avg = avg + 0;
                        } else {
                            if( matrix_column < padding || matrix_column >= i_columns + padding) {
                                avg = avg + 0;
                            } else {
                                matrix_row = matrix_row - padding;
                                matrix_column = matrix_column - padding;
                                avg = avg + (*(i_matrix_A + matrix_row * i_columns + matrix_column * i_depth + kindex));
                            }
                        }
                    }
                }
                *(o_matrix_B + index * (*o_columns) + jindex * i_depth + kindex) = avg / (i_filter_row * i_filter_column);
            }
        }
    }

}

/* Alexnet implementation simulation for classifing
* http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf
* https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637
* https://engmrk.com/alexnet-implementation-using-keras/
* https://www.ritchieng.com/machine-learning/deep-learning/convs/
*/
size_t alexnet_simulation_classify(element_t *i_input_data, size_t i_rows,
                                   size_t i_columns, element_t *i_input_layer,
                                   element_t *i_conv_output, element_t *i_kernel,
                                   element_t *o_output_layer) {
    size_t index, jindex, kindex;
    size_t stride_row, stride_column, filter_row, filter_column;
    size_t start_depth, end_depth;
    size_t rows, columns;
    size_t aux_rows, aux_columns;
    size_t kernel_size;

    //element_t i_input_layer[290400] // 55x55x96
    //element_t i_conv_output[3025] // 55x55
    //element_t o_output_layer[290400] // 55x55x96
    //element_t i_kernel[9216] //6x6x256
    size_t current_depth;
    size_t total_depth;
    size_t padding_type;
    size_t padding[4] = {0,0,0,0};
    element_t full_conected_element;

    //prepare data for first layer
    stride_row = i_rows / 227;
    stride_column = i_columns / 227;
    current_depth = 3;
    filter_row = stride_row;
    filter_column = stride_column;
    padding_type = 1;
    avg_pooling_3D(i_input_data, i_rows, i_columns, current_depth,
                   filter_row, filter_column,
                   stride_row, stride_column,
                   padding_type,
                   i_input_layer, &aux_rows, &aux_columns);
    


    /* first layer
    * conv 96 of 11x11x3 no padding with stride 4 pad 0
    * relu
    * max pool 3x3 filter with stride 2
    */
    total_depth = current_depth;
    current_depth = 96;
    stride_row = 4;
    stride_column = 4;
    start_depth = 0;
    end_depth = 3;
    kernel_size = 11;
    rows = aux_rows;
    columns = aux_columns;
    padding_type = 0;
    for(kindex = 0; kindex < current_depth; kindex++) {
        convolution_3D(i_input_layer, rows, columns,
                       start_depth, end_depth, total_depth,
                       i_kernel, kernel_size,
                       padding_type, padding,
                       stride_row, stride_column,
                       i_conv_output, &aux_rows, &aux_columns);
        for(index = 0; index < aux_rows; index++) {
            for(jindex = 0; jindex < aux_columns; jindex++) {
                *(o_output_layer + index * aux_columns + jindex * current_depth + kindex)
                = *(i_conv_output + index * aux_columns + jindex);
            }
        }
    }

    rows = aux_rows;
    columns = aux_columns;
    ReLU_3D(o_output_layer, rows, columns, current_depth,
            o_output_layer);
    stride_row = 2;
    stride_column = 2;
    filter_row = 3;
    filter_column = 3;
    padding_type = 0;
    max_pooling_3D(o_output_layer, rows, columns, current_depth,
                   filter_row,  filter_column,
                   stride_row, stride_column,
                   padding_type,
                   i_input_layer, &aux_rows, &aux_columns);
    /* second layer
    * conv 256 of 5x5x96 with stride 1 pad 2
    * relu
    * max pool 3x3 filter with stride 2
    */
    total_depth = current_depth;
    current_depth = 256;
    stride_row = 1;
    stride_column = 1;
    start_depth = 0;
    end_depth = 96;
    kernel_size = 5;
    rows = aux_rows;
    columns = aux_columns;
    padding_type = 2;
    padding[0] = 2;
    padding[1] = 2;
    padding[2] = 2;
    padding[3] = 2;
    for(kindex = 0; kindex < current_depth; kindex++) {
        convolution_3D(i_input_layer, rows, columns,
                       start_depth, end_depth, total_depth,
                       i_kernel, kernel_size,
                       padding_type, padding,
                       stride_row, stride_column,
                       i_conv_output, &aux_rows, &aux_columns);
        for(index = 0; index < aux_rows; index++) {
            for(jindex = 0; jindex < aux_columns; jindex++) {
                *(o_output_layer + index * aux_columns + jindex * current_depth + kindex)
                = *(i_conv_output + index * aux_columns + jindex);
            }
        }
    }

    rows = aux_rows;
    columns = aux_columns;
    ReLU_3D(o_output_layer, rows, columns, current_depth,
            o_output_layer);
    stride_row = 2;
    stride_column = 2;
    filter_row = 3;
    filter_column = 3;
    padding_type = 0;
    max_pooling_3D(o_output_layer, rows, columns, current_depth,
                   filter_row,  filter_column,
                   stride_row, stride_column,
                   padding_type,
                   i_input_layer, &aux_rows, &aux_columns);
    /* third layer
    * conv 384 of 3x3x256 with stride 1 pad 1
    * relu
    */
    total_depth = current_depth;
    current_depth = 384;
    stride_row = 1;
    stride_column = 1;
    start_depth = 0;
    end_depth = 256;
    kernel_size = 3;
    rows = aux_rows;
    columns = aux_columns;
    padding_type = 2;
    padding[0] = 1;
    padding[1] = 1;
    padding[2] = 1;
    padding[3] = 1;
    for(kindex = 0; kindex < current_depth; kindex++) {
        convolution_3D(i_input_layer, rows, columns,
                       start_depth, end_depth, total_depth,
                       i_kernel, kernel_size,
                       padding_type, padding,
                       stride_row, stride_column,
                       i_conv_output, &aux_rows, &aux_columns);
        for(index = 0; index < aux_rows; index++) {
            for(jindex = 0; jindex < aux_columns; jindex++) {
                *(o_output_layer + index * aux_columns + jindex * current_depth + kindex)
                = *(i_conv_output + index * aux_columns + jindex);
            }
        }
    }

    rows = aux_rows;
    columns = aux_columns;
    ReLU_3D(o_output_layer, rows, columns, current_depth,
            i_input_layer);
    /* forth layer
    * conv 384 of 3x3x384 with stride 1 pad 1
    * relu
    */
    total_depth = current_depth;
    current_depth = 384;
    stride_row = 1;
    stride_column = 1;
    start_depth = 0;
    end_depth = 384;
    kernel_size = 3;
    rows = aux_rows;
    columns = aux_columns;
    padding_type = 2;
    padding[0] = 1;
    padding[1] = 1;
    padding[2] = 1;
    padding[3] = 1;
    for(kindex = 0; kindex < current_depth; kindex++) {
        convolution_3D(i_input_layer, rows, columns,
                       start_depth, end_depth, total_depth,
                       i_kernel, kernel_size,
                       padding_type, padding,
                       stride_row, stride_column,
                       i_conv_output, &aux_rows, &aux_columns);
        for(index = 0; index < aux_rows; index++) {
            for(jindex = 0; jindex < aux_columns; jindex++) {
                *(o_output_layer + index * aux_columns + jindex * current_depth + kindex)
                = *(i_conv_output + index * aux_columns + jindex);
            }
        }
    }
    rows = aux_rows;
    columns = aux_columns;
    ReLU_3D(o_output_layer, rows, columns, current_depth,
            i_input_layer);
    /* fifth layer
    * conv 256 of 3x3x384 with stride 1 pad 1
    * relu
    * max pool 3x3 filter with stride 2
    */
    total_depth = current_depth;
    current_depth = 256;
    stride_row = 1;
    stride_column = 1;
    start_depth = 0;
    end_depth = 384;
    kernel_size = 3;
    rows = aux_rows;
    columns = aux_columns;
    padding_type = 2;
    padding[0] = 1;
    padding[1] = 1;
    padding[2] = 1;
    padding[3] = 1;
    for(kindex = 0; kindex < current_depth; kindex++) {
        convolution_3D(i_input_layer, rows, columns,
                       start_depth, end_depth, total_depth,
                       i_kernel, kernel_size,
                       padding_type, padding,
                       stride_row, stride_column,
                       i_conv_output, &aux_rows, &aux_columns);
        for(index = 0; index < aux_rows; index++) {
            for(jindex = 0; jindex < aux_columns; jindex++) {
                *(o_output_layer + index * aux_columns + jindex * current_depth + kindex)
                = *(i_conv_output + index * aux_columns + jindex);
            }
        }
    }
    rows = aux_rows;
    columns = aux_columns;
    ReLU_3D(o_output_layer, rows, columns, current_depth,
            o_output_layer);
    stride_row = 2;
    stride_column = 2;
    filter_row = 3;
    filter_column = 3;
    padding_type = 0;
    max_pooling_3D(o_output_layer, rows, columns, current_depth,
                   filter_row,  filter_column,
                   stride_row, stride_column,
                   padding_type,
                   i_input_layer, &aux_rows, &aux_columns);
    /* sixth layer
    * conv 4096 of 6x6x256 with stride 1 pad 0
    */
    total_depth = current_depth;
    current_depth = 4096;
    stride_row = 1;
    stride_column = 1;
    start_depth = 0;
    end_depth = 256;
    kernel_size = 6;
    rows = aux_rows;
    columns = aux_columns;
    padding_type = 0;
    for(kindex = 0; kindex < current_depth; kindex++) {
        convolution_3D(i_input_layer, rows, columns,
                       start_depth, end_depth, total_depth,
                       i_kernel, kernel_size,
                       padding_type, padding,
                       stride_row, stride_column,
                       i_conv_output, &aux_rows, &aux_columns);
        for(index = 0; index < aux_rows; index++) {
            for(jindex = 0; jindex < aux_columns; jindex++) {
                *(o_output_layer + index * aux_columns + jindex * current_depth + kindex)
                = *(i_conv_output + index * aux_columns + jindex);
            }
        }
    }
    for(kindex = 0; kindex < current_depth; kindex++) {
        *(i_input_layer + kindex) = *(o_output_layer + kindex);
    }
    /* seventh layer
    * fully connected 4096->4096
    */
    total_depth = current_depth;
    current_depth = 4096;
    rows = aux_rows;
    columns = aux_columns;
    for(kindex = 0; kindex < current_depth; kindex++) {
        full_conected(i_input_layer, total_depth,
                      i_kernel, &full_conected_element);
        *(o_output_layer + kindex) = full_conected_element;
    }
    for(kindex = 0; kindex < current_depth; kindex++) {
        *(i_input_layer + kindex) = *(o_output_layer + kindex);
    }
    /* eight layer
    * fully connected 4096->1000
    */
    total_depth = current_depth;
    current_depth = 1000;
    rows = aux_rows;
    columns = aux_columns;
    for(kindex = 0; kindex < current_depth; kindex++) {
        full_conected(i_input_layer, total_depth,
                      i_kernel, &full_conected_element);
        *(o_output_layer + kindex) = full_conected_element;
    }
    for(kindex = 0; kindex < current_depth; kindex++) {
        *(i_input_layer + kindex) = *(o_output_layer + kindex);
    }

    /*
    * softmax layer
    */
    total_depth = current_depth;
    softmax(i_input_layer, total_depth, o_output_layer);

    element_t max = -1;
    size_t return_value = -1;
    for(kindex = 0; kindex < total_depth; kindex++) {
        if(*(o_output_layer + kindex) > max) {
            max = *(o_output_layer + kindex);
            return_value = kindex;
        }
    }

    return return_value;

}

/*
element_t g_input_layer[200704] // 56x56x64
element_t g_conv_output[12544] // 112x112
element_t g_inner_layer[200704] // 56x56x64
element_t g_output_layer[802816] // 112x112x64
//element_t g_kernel[4608] //3x3x512
element_t g_kernel[25008] //7x7x512
*/
/* Rosnet
* https://arxiv.org/pdf/1512.03385.pdf
* https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
*/
size_t rosnet_simulation_classifier(element_t *i_input_data, size_t i_rows,
                                   size_t i_columns, element_t *i_input_layer,
                                   element_t *i_conv_output, element_t *i_kernel,
                                   element_t *i_inner_layer,
                                   element_t *o_output_layer) {
    size_t index, jindex, kindex, rindex;
    size_t stride_row, stride_column, filter_row, filter_column;
    size_t start_depth, end_depth;
    size_t rows, columns;
    size_t aux_rows, aux_columns;
    size_t kernel_size;
    size_t current_depth;
    size_t total_depth;
    size_t padding_type;
    size_t padding[4] = {0,0,0,0};
    element_t full_conected_element;

    //prepare data for first layer
    stride_row = i_rows / 224;
    stride_column = i_columns / 224;
    current_depth = 3;
    filter_row = stride_row;
    filter_column = stride_column;
    padding_type = 1;
    avg_pooling_3D(i_input_data, i_rows, i_columns, current_depth,
                   filter_row, filter_column,
                   stride_row, stride_column,
                   padding_type,
                   i_input_layer, &aux_rows, &aux_columns);

    /* first layer
    * conv 64 of 7x7x3 no padding with stride 2 pad 1
    * relu
    * max pool 3x3 filter with stride 2
    */
    total_depth = current_depth;
    current_depth = 64;
    stride_row = 1;
    stride_column = 1;
    start_depth = 0;
    end_depth = 3;
    kernel_size = 7;
    rows = aux_rows;
    columns = aux_columns;
    padding_type = 1;
    for(kindex = 0; kindex < current_depth; kindex++) {
        convolution_3D(i_input_layer, rows, columns,
                       start_depth, end_depth, total_depth,
                       i_kernel, kernel_size,
                       padding_type, padding,
                       stride_row, stride_column,
                       i_conv_output, &aux_rows, &aux_columns);
        for(index = 0; index < aux_rows; index++) {
            for(jindex = 0; jindex < aux_columns; jindex++) {
                *(o_output_layer + index * aux_columns + jindex * current_depth + kindex)
                = *(i_conv_output + index * aux_columns + jindex);
            }
        }
    }

    rows = aux_rows;
    columns = aux_columns;
    ReLU_3D(o_output_layer, rows, columns, current_depth,
            o_output_layer);
    stride_row = 2;
    stride_column = 2;
    filter_row = 3;
    filter_column = 3;
    padding_type = 1;
    max_pooling_3D(o_output_layer, rows, columns, current_depth,
                   filter_row,  filter_column,
                   stride_row, stride_column,
                   padding_type,
                   i_input_layer, &aux_rows, &aux_columns);
    
    for(rindex = 0; rindex < 2; rindex++) {
        /* conv_1_1/1_2
        * conv 64 of 3x3x64 no padding with stride 1 pad 1
        * relu
        * conv 64 of 3x3x64 no padding with stride 1 pad 1
        * adition
        * relu
        */
        total_depth = current_depth;
        current_depth = 64;
        stride_row = 1;
        stride_column = 1;
        start_depth = 0;
        end_depth = current_depth;
        kernel_size = 3;
        rows = aux_rows;
        columns = aux_columns;
        padding_type = 1;
        for(kindex = 0; kindex < current_depth; kindex++) {
            convolution_3D(i_input_layer, rows, columns,
                        start_depth, end_depth, total_depth,
                        i_kernel, kernel_size,
                        padding_type, padding,
                        stride_row, stride_column,
                        i_conv_output, &aux_rows, &aux_columns);
            for(index = 0; index < aux_rows; index++) {
                for(jindex = 0; jindex < aux_columns; jindex++) {
                    *(i_inner_layer + index * aux_columns + jindex * current_depth + kindex)
                    = *(i_conv_output + index * aux_columns + jindex);
                }
            }
        }

        rows = aux_rows;
        columns = aux_columns;
        ReLU_3D(i_inner_layer, rows, columns, current_depth,
                i_inner_layer);
        

        for(kindex = 0; kindex < current_depth; kindex++) {
            convolution_3D(i_inner_layer, rows, columns,
                        start_depth, end_depth, total_depth,
                        i_kernel, kernel_size,
                        padding_type, padding,
                        stride_row, stride_column,
                        i_conv_output, &aux_rows, &aux_columns);
            for(index = 0; index < aux_rows; index++) {
                for(jindex = 0; jindex < aux_columns; jindex++) {
                    *(o_output_layer + index * aux_columns + jindex * current_depth + kindex)
                    = *(i_conv_output + index * aux_columns + jindex);
                }
            }
        }

        rows = aux_rows;
        columns = aux_columns;
        //addition
        for(index = 0; index < rows; index++) {
            for(jindex = 0; jindex < columns; jindex++) {
                for(kindex = 0; kindex < current_depth; kindex++) {
                    *(o_output_layer + index * columns + jindex * current_depth + kindex)
                    = *(o_output_layer + index * columns + jindex * current_depth + kindex) +
                    *(i_input_layer + index * columns + jindex * current_depth + kindex);
                }
            }
        }
        ReLU_3D(o_output_layer, rows, columns, current_depth,
                i_input_layer);
    }


    for(rindex = 0; rindex < 6; rindex++) {
        /* conv_1_1/1_2
        * conv 64 of 3x3x64 no padding with stride 1 pad 1
        * relu
        * conv 64 of 3x3x64 no padding with stride 1 pad 1
        * adition
        * relu
        */
        if(rindex%2 == 0) {
            total_depth = current_depth;
            current_depth = current_depth * 2;
            stride_row = 2;
            stride_column = 2;
        } else {
            total_depth = current_depth;
            current_depth = current_depth;
            stride_row = 1;
            stride_column = 1;
        }
        start_depth = 0;
        end_depth = current_depth;
        kernel_size = 3;
        rows = aux_rows;
        columns = aux_columns;
        padding_type = 1;
        for(kindex = 0; kindex < current_depth; kindex++) {
            convolution_3D(i_input_layer, rows, columns,
                        start_depth, end_depth, total_depth,
                        i_kernel, kernel_size,
                        padding_type, padding,
                        stride_row, stride_column,
                        i_conv_output, &aux_rows, &aux_columns);
            for(index = 0; index < aux_rows; index++) {
                for(jindex = 0; jindex < aux_columns; jindex++) {
                    *(i_inner_layer + index * aux_columns + jindex * current_depth + kindex)
                    = *(i_conv_output + index * aux_columns + jindex);
                }
            }
        }

        rows = aux_rows;
        columns = aux_columns;
        ReLU_3D(i_inner_layer, rows, columns, current_depth,
                i_inner_layer);
        

        for(kindex = 0; kindex < current_depth; kindex++) {
            convolution_3D(i_inner_layer, rows, columns,
                        start_depth, end_depth, total_depth,
                        i_kernel, kernel_size,
                        padding_type, padding,
                        stride_row, stride_column,
                        i_conv_output, &aux_rows, &aux_columns);
            for(index = 0; index < aux_rows; index++) {
                for(jindex = 0; jindex < aux_columns; jindex++) {
                    *(o_output_layer + index * aux_columns + jindex * current_depth + kindex)
                    = *(i_conv_output + index * aux_columns + jindex);
                }
            }
        }

        rows = aux_rows;
        columns = aux_columns;
        //addition
        for(index = 0; index < rows; index++) {
            for(jindex = 0; jindex < columns; jindex++) {
                for(kindex = 0; kindex < current_depth; kindex++) {
                    *(o_output_layer + index * columns + jindex * current_depth + kindex)
                    = *(o_output_layer + index * columns + jindex * current_depth + kindex) +
                    *(i_input_layer + index * columns + jindex * current_depth + kindex);
                }
            }
        }
        ReLU_3D(o_output_layer, rows, columns, current_depth,
                i_input_layer);
    }


    stride_row = 1;
    stride_column = 1;
    filter_row = 3;
    filter_column = 3;
    avg_pooling_3D(i_input_layer, rows, columns, current_depth,
                   filter_row,  filter_column,
                   stride_row, stride_column, padding_type,
                   o_output_layer, &aux_rows, &aux_columns);
    for(index = 0; index < rows; index++) {
        for(jindex = 0; jindex < columns; jindex++) {
            for(kindex = 0; kindex < current_depth; kindex++) {
                *(i_input_layer + index * columns + jindex * current_depth + kindex)
                = *(o_output_layer + index * columns + jindex * current_depth + kindex);
            }
        }
    }

    /* fully connected
    * conv 1000 of 7x7x512 with stride 1 pad 1
    */
    total_depth = current_depth;
    current_depth = 1000;
    stride_row = 1;
    stride_column = 1;
    start_depth = 0;
    end_depth = current_depth;
    kernel_size = 7;
    rows = aux_rows;
    columns = aux_columns;
    padding_type = 1;
    for(kindex = 0; kindex < current_depth; kindex++) {
        convolution_3D(i_input_layer, rows, columns,
                       start_depth, end_depth, total_depth,
                       i_kernel, kernel_size,
                       padding_type, padding,
                       stride_row, stride_column,
                       i_conv_output, &aux_rows, &aux_columns);
        for(index = 0; index < aux_rows; index++) {
            for(jindex = 0; jindex < aux_columns; jindex++) {
                *(o_output_layer + index * aux_columns + jindex * current_depth + kindex)
                = *(i_conv_output + index * aux_columns + jindex);
            }
        }
    }
    for(kindex = 0; kindex < current_depth; kindex++) {
        *(i_input_layer + kindex) = *(o_output_layer + kindex);
    }


    /*
    * softmax layer
    */
    total_depth = current_depth;
    softmax(i_input_layer, total_depth, o_output_layer);

    element_t max = -1;
    size_t return_value = -1;
    for(kindex = 0; kindex < total_depth; kindex++) {
        if(*(o_output_layer + kindex) > max) {
            max = *(o_output_layer + kindex);
            return_value = kindex;
        }
    }

    return return_value;
}
 
int main(void) {
    const size_t image_rows = 256;
    const size_t image_cols = 256;
    const size_t image_depth = 3;
    element_t test_image[image_rows * image_cols * image_depth];
    
    for(size_t i = 0; i < image_rows; i++) {
        for(size_t j = 0; j < image_cols; j++) {
            test_image[i*image_cols*image_depth + j*image_depth + 0] = (element_t)j / image_cols;
            test_image[i*image_cols*image_depth + j*image_depth + 1] = (element_t)i / image_rows;
            test_image[i*image_cols*image_depth + j*image_depth + 2] = 0;
        }
    }
    
    size_t answer = alexnet_simulation_classify(test_image, image_rows, 
                                              image_cols, g_input_layer, 
                                              g_conv_output, g_kernel, 
                                              g_output_layer);

    #ifdef DEBUG
        printf("Result: %f\n", answer);
    #endif
    return 0;
}