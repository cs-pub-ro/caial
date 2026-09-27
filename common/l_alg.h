#ifndef _L_ALG_H_
#define _L_ALG_H_
#include "common.h"

void *memset(void *ptr, int value, long unsigned int num);

#define EXPONENTIAL_PRECISION 8
/**
 * Returns approximate value of e^x
 * using sum of first n terms of Taylor Series
 */
element_t number_exponential(size_t i_precision, element_t i_x) 
{ 
	element_t sum = 1; // initialize sum of series 
    size_t index;
	for (index = i_precision - 1; index > 0; index--) {
		sum = 1 + i_x * sum / index; 
    }
	return sum; 
}


/**
 * Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
 * Input: i_x, a element.
 * Output: 1/(1 + e^-x).
 * 1/(1+e^-x) = e^x/(1+e^x)
 * o_y = f(i_x)
 */
void number_sigmoid (element_t i_x, element_t *o_y) {
    element_t e_x;
    if(i_x >= 0) {
        e_x = number_exponential(EXPONENTIAL_PRECISION, i_x);
        *o_y = e_x / (e_x + 1);
    } else {
        e_x = number_exponential(EXPONENTIAL_PRECISION, -1*i_x);
        *o_y = 1 / (1 + e_x);
    }
}

/**
 * Returns the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
 * where f(x) is sigmoid function.
 * Input: i_x .
 * Output: x(1 - x) for every element of the input vector input.
 */
void number_sigmoid_derivate (element_t i_x, element_t *o_y) {
    *o_y = i_x * (1 - i_x);
}


/**
 * The elementwise sum of two vectors.
 * Inputs:
 * 	i_vector_A: a vector
 * 	i_vector_B: a vector
 *  i_length : size of the vectors
 * Output: 
 *  o_vector_C: a vector C=A+B
 */
void vector_sum (element_t i_vector_A[], element_t i_vector_B[],
                 size_t i_length, element_t o_vector_C[]) {
    size_t index;
    for (index = 0; index < i_length; index++) {
        o_vector_C[index] = i_vector_A[index] + i_vector_B[index];
    }
}
/**
 * The elementwise differnece of two vectors.
 * Inputs:
 * 	i_vector_A: a vector
 * 	i_vector_B: a vector
 *  i_length : size of the vectors
 * Output: 
 *  o_vector_C: a vector C=A-B
 */
void vector_difference (element_t i_vector_A[], element_t i_vector_B[],
                        size_t i_length, element_t o_vector_C[]){
    size_t index;
    for (index = 0; index < i_length; index++) {
        o_vector_C[index] = i_vector_A[index] - i_vector_B[index];
    }
}

/**
 * The elementwise multiplication of two vectors.
 * Inputs:
 * 	i_vector_A: a vector
 * 	i_vector_B: a vector
 *  i_length : size of the vectors
 * Output: 
 *  o_vector_C: a vector C=A.*B
 */
void vector_multiply_elementwise (element_t i_vector_A[], element_t i_vector_B[],
                                  size_t i_length, element_t o_vector_C[]) {
    size_t index;
    for (index = 0; index < i_length; index++) {
        o_vector_C[index] = i_vector_A[index] * i_vector_B[index];
    }
}


/**
 * Product of a vector with a scalar
 * Inputs:
 * 	i_vector_A: a matrix i_length
 * 	i_length: length of i_vector_A 
 * 	i_scalar: scalar
 * Output: 
 *  o_vector_B: a vector B=A*scalar size i_length
 */
void vector_multiply_scalar(element_t i_vector_A[], size_t i_length,
                            element_t i_scalar, element_t o_vector_B[]) {
    size_t index;
    for (index = 0; index < i_length; index++) {
        o_vector_B[index] = i_vector_A[index] * i_scalar;
    }
}

/**
 * Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
 * Inputs:
 * 	i_vector_A: a matrix i_length
 * 	i_length: length of i_vector_A 
 * Output: 
 *  o_vector_B: a vector B=f(A) size i_length
 * 1/(1+e^-x) = e^x/(1+e^x)
 */
void vector_sigmoid (element_t i_vector_A[], size_t i_length,
                     element_t o_vector_B[]) {
    size_t index;
    element_t e_x;
    for (index = 0; index < i_length; index++) {
        if(i_vector_A[index] >= 0) {
            e_x = number_exponential(EXPONENTIAL_PRECISION, i_vector_A[index]);
            o_vector_B[index] = e_x / (e_x + 1);
        } else {
            e_x = number_exponential(EXPONENTIAL_PRECISION, -1*i_vector_A[index]);
            o_vector_B[index] = 1 / (1 + e_x);
        }
    }
}

/**
 * Returns the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
 * where f(x) is sigmoid function. g(x) = x(1-x)
 * 	i_vector_A: a matrix i_length
 * 	i_length: length of i_vector_A 
 * Output: 
 *  o_vector_B: a vector B=g(A) size i_length
 */
void vector_sigmoid_derivate (element_t i_vector_A[], size_t i_length,
                              element_t o_vector_B[])  {
    size_t index;
    for (index = 0; index < i_length; index++) {
        o_vector_B[index] = i_vector_A[index] * (1 - i_vector_A[index]);
    }
}

/**
 * Calculate the transpose matrix of input matrix.
 * Inputs:
 * 	i_matrix_A: a matrix
 * 	i_rows: number of rows for A
 *  i_columns : number of columns for A
 * Output: 
 *  o_matrix_B: a matrix B=A.T
 */
void matrix_transpose(element_t **i_matrix_A, size_t i_rows,
                      size_t i_columns, element_t **o_matrix_B) {
    size_t index, jindex;
    for(index = 0; index < i_rows; index++) {
        for(jindex = 0; jindex < i_columns; jindex++) {
            o_matrix_B[jindex][index] = i_matrix_A[index][jindex];
        }
    }
}

/**
 * Calculate the product of a matrix and a vector
 * Inputs:
 * 	i_matrix_A: a matrix i_rows x i_columns
 * 	i_rows: number of rows for A
 *  i_columns : number of columns for A
 *  i_vector_B: a vector of size i_columns x 1
 * Output: 
 *  o_vector_C: a vector C=A*B size i_rows x 1
 */
void matrix_multiply_vector(element_t *i_matrix_A, size_t i_rows, size_t i_columns,
                            element_t i_vector_B[], element_t o_vector_C[]) {
    size_t index, jindex;
    for(index = 0; index < i_rows; index++) {
        o_vector_C[index] = 0;
        for(jindex = 0; jindex < i_columns; jindex++) {
            o_vector_C[index] = o_vector_C[index]
                                + i_matrix_A[index * i_rows + jindex] * i_vector_B[jindex];
        }
    }
}

/**
 * Calculate the product of the transpose of a matrix and a vector
 * Inputs:
 * 	i_matrix_A: a matrix i_rows x i_columns
 * 	i_rows: number of rows for A
 *  i_columns : number of columns for A
 *  i_vector_B: a vector of size i_rows x 1
 * Output: 
 *  o_vector_C: a vector C=A*B size i_columns x 1
 */
void matrix_T_multiply_vector(element_t *i_matrix_A, size_t i_rows, size_t i_columns,
                            element_t i_vector_B[], element_t o_vector_C[]) {
    size_t index, jindex;
    for(index = 0; index < i_columns; index++) {
        o_vector_C[index] = 0;
        for(jindex = 0; jindex < i_rows; jindex++) {
            o_vector_C[index] = o_vector_C[index]
                                + i_matrix_A[jindex * i_rows + index] * i_vector_B[jindex];
        }
    }
}

/**
 * Product of the transpose of a matrix and a matrix
 * Inputs:
 * 	i_matrix_A: a matrix i_middle x i_rows
 * 	i_matrix_B: a matrix i_middle x i_columns
 * 	i_rows: number of rows for o_matrix_C and columns for i_matrix_A
 *  i_columns : number of columns for i_matrix_B and o_matrix_C
 *  i_middle: number of rows for i_matrix_A and i_matrix_B
 * Output: 
 *  o_matrix_C: a matrix C=A.T*B size i_rows x i_columns
 */
void matrix_T_multiply_matrix(element_t *i_matrix_A, element_t *i_matrix_B,
                            size_t i_rows, size_t i_middle, size_t i_columns,
                            element_t *o_matrix_C) {
	size_t index, jindex, kindex;
	for (index = 0; index < i_rows; index++) {
		for (jindex = 0; jindex < i_columns; jindex++) {
			o_matrix_C[index * i_rows + jindex] = 0;
			for (kindex = 0; kindex < i_middle; kindex++) {
				o_matrix_C[index * i_rows + jindex] = o_matrix_C[index * i_rows + jindex]
                                            + i_matrix_A[kindex * i_rows + index]
                                            * i_matrix_B[kindex * i_rows + jindex];
			}
		}
    }
}


/**
 * Product of two matrix
 * Inputs:
 * 	i_matrix_A: a matrix i_rows x i_middle
 * 	i_matrix_B: a matrix i_middle x i_columns
 * 	i_rows: number of rows for i_matrix_A and o_matrix_C
 *  i_columns : number of columns for i_matrix_B and o_matrix_C
 *  i_middle: number of cololumns for i_matrix_A and rows for i_matrix_B
 * Output: 
 *  o_matrix_C: a matrix C=A*B size i_rows x i_columns
 */
void matrix_multiply_matrix(element_t *i_matrix_A, element_t *i_matrix_B,
                            size_t i_rows, size_t i_middle, size_t i_columns,
                            element_t *o_matrix_C) {
	size_t index, jindex, kindex;
	for (index = 0; index < i_rows; index++) {
		for (jindex = 0; jindex < i_columns; jindex++) {
			o_matrix_C[index * i_columns + jindex] = 0;
			for (kindex = 0; kindex < i_middle; kindex++) {
				o_matrix_C[index * i_columns + jindex] = o_matrix_C[index * i_columns + jindex]
                                            + i_matrix_A[index * i_middle + kindex]
                                            * i_matrix_B[kindex * i_columns + jindex];
			}
		}
    }
}

/**
 * Product of a vector with transpose of a second vector resulting a matrix
 * Inputs:
 * 	i_vector_A: a vector of  i_rows x 1
 * 	i_vector_B: a vector of  i_columns x 1
 * 	i_rows: number of rows for o_matrix_C
 *  i_columns : number of columns for o_matrix_C
 * Output: 
 *  o_matrix_C: a matrix C=A*B.T size i_rows x i_columns
 */
void vector_multiply_vector_T(element_t i_vector_A[], element_t i_vector_B[],
                              size_t i_rows, size_t i_columns,
                              element_t *o_matrix_C) {
	size_t index, jindex;
	for (index = 0; index < i_rows; index++) {
		for (jindex = 0; jindex < i_columns; jindex++) {
			o_matrix_C[index * i_rows + jindex] = i_vector_A[index] * i_vector_B[jindex];
		}
    }
}

/**
 * Product of transpose of a vector with a second vector resulting an element
 * Inputs:
 * 	i_vector_A: a vector of  i_length
 * 	i_vector_B: a vector of  i_length
 *  i_length : size of the vectors
 * Output: 
 *  o_C: a matrix C=A.T*B size 1
 */
void vector_T_multiply_vector(element_t i_vector_A[], element_t i_vector_B[],
                              size_t i_length, element_t *o_C) {
    size_t index;
    *o_C = 0;
    for(index = 0; index < i_length; index++) {
        *o_C = *o_C + i_vector_A[index] * i_vector_B[index];
    }
}

/**
 * Product of two matrix elementwise
 * Inputs:
 * 	i_matrix_A: a matrix i_rows x i_columns
 * 	i_matrix_B: a matrix i_rows x i_columns
 * 	i_rows: number of rows
 *  i_columns : number of columns
 * Output: 
 *  o_matrix_C: a matrix C=A.*B size i_rows x i_columns
 */
void matrix_multiply_elementwise(element_t *i_matrix_A, element_t *i_matrix_B,
                                 size_t i_rows, size_t i_columns,
                                 element_t *o_matrix_C) {
	size_t index, jindex;
	for (index = 0; index < i_rows; index++) {
		for (jindex = 0; jindex < i_columns; jindex++) {
            o_matrix_C[index * i_rows + jindex] = i_matrix_A[index * i_rows + jindex]
                                        * i_matrix_B[index * i_rows + jindex];
		}
    }
}

/**
 * Sum of two matrix
 * Inputs:
 * 	i_matrix_A: a matrix i_rows x i_columns
 * 	i_matrix_B: a matrix i_rows x i_columns
 * 	i_rows: number of rows
 *  i_columns : number of columns
 * Output: 
 *  o_matrix_C: a matrix C=A+B size i_rows x i_columns
 */
void matrix_sum(element_t *i_matrix_A, element_t *i_matrix_B,
                size_t i_rows, size_t i_columns,
                element_t *o_matrix_C) {
	size_t index, jindex;
	for (index = 0; index < i_rows; index++) {
		for (jindex = 0; jindex < i_columns; jindex++) {
            o_matrix_C[index * i_rows + jindex] = i_matrix_A[index * i_rows + jindex]
                                        + i_matrix_B[index * i_rows + jindex];
		}
    }
}

/**
 *  Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
 * Inputs:
 * 	i_matrix_A: a matrix i_rows x i_columns
 * 	i_rows: number of rows
 *  i_columns : number of columns
 * Output: 
 *  o_matrix_B: a matrix B=f(A) size i_rows x i_columns
 */
void matrix_sigmoid(element_t *i_matrix_A,
                    size_t i_rows, size_t i_columns,
                    element_t *o_matrix_B) {
	size_t index, jindex;
    element_t e_x;
	for (index = 0; index < i_rows; index++) {
		for (jindex = 0; jindex < i_columns; jindex++) {
            if(i_matrix_A[index * i_rows + jindex] >= 0) {
                e_x = number_exponential(EXPONENTIAL_PRECISION,
                                         i_matrix_A[index * i_rows]);
                o_matrix_B[index * i_rows + jindex] = e_x / (e_x + 1);
            } else {
                e_x = number_exponential(EXPONENTIAL_PRECISION,
                                         -1*i_matrix_A[index * i_rows + jindex]);
                o_matrix_B[index * i_rows + jindex] = 1 / (1 + e_x);
            }
		}
    }
}
/**
 * Returns the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
 * where f(x) is sigmoid function. g(x) = x(1-x)
 * Inputs:
 * 	i_matrix_A: a matrix i_rows x i_columns
 * 	i_rows: number of rows
 *  i_columns : number of columns
 * Output: 
 *  o_matrix_B: a matrix B=g(A) size i_rows x i_columns
 */
void matrix_sigmoid_derivate(element_t **i_matrix_A,
                    size_t i_rows, size_t i_columns,
                    element_t **o_matrix_B) {
	size_t index, jindex;
	for (index = 0; index < i_rows; index++) {
		for (jindex = 0; jindex < i_columns; jindex++) {
            o_matrix_B[index][jindex] = i_matrix_A[index][jindex]
                                        * (1 - i_matrix_A[index][jindex]);
		}
    }
}

/**
 * Product of a vector tranpose and a matrix as a vector
 * Inputs:
 * 	i_vector_A: a matrix i_rows
 * 	i_matrix_B: a matrix i_rows x i_columns
 * 	i_rows: number of rows for i_matrix_A
 *  i_columns : number of columns for i_matrix_A
 * Output: 
 *  o_matrix_C: a matrix C=A.T*B size 1 x i_columns
 */
void vector_T_multiply_matrix(element_t i_vector_A[], element_t *i_matrix_B,
                              size_t i_rows, size_t i_columns,
                              element_t o_vector_C[]) {
    size_t index, jindex;
    for(index = 0; index < i_columns; index++) {
        o_vector_C[index] = 0;
        for(jindex = 0; jindex < i_rows; jindex++) {
            o_vector_C[index] = o_vector_C[index]
                                + i_vector_A[jindex] * (*(i_matrix_B + jindex * i_columns + index));
        }
    }
}


/**
* Functions related to inverse_matrix
* https://www.geeksforgeeks.org/adjoint-inverse-matrix/
* https://en.wikipedia.org/wiki/Crout_matrix_decomposition
*/

/*
* Crout matrix decomposition
* https://en.wikipedia.org/wiki/Crout_matrix_decomposition
* Inputs:
* 	i_matrix_A: a matrix i_size x i_size
* 	i_size: the size of the matrix
* Output: 
*  o_matrix_L: a matrix i_size x i_size
*  o_matrix_U: a matrix i_size x i_size
* A=L*U
*/
size_t matrix_crout_decomposition(element_t *i_matrix_A, size_t i_size,
                                  real_t *o_matrix_L, real_t *o_matrix_U) {

    size_t index, jindex, kindex;
    real_t sum;

    //initialise matrix
	for (index = 0; index < i_size; index++) {
	    for (jindex = 0; jindex < i_size; jindex++) {
		    o_matrix_U[index * i_size + jindex] = 0;
		    o_matrix_L[index * i_size + jindex] = 0;
        }
	}

    //initialise diagnogal for U
	for (index = 0; index < i_size; index++) {
		o_matrix_U[index * i_size + index] = 1;
	}

	for (jindex = 0; jindex < i_size; jindex++) {
		for (index = jindex; index < i_size; index++) {
			sum = 0;
			for (kindex = 0; kindex < jindex; kindex++) {
				sum = sum + o_matrix_L[index * i_size + kindex] * o_matrix_U[kindex * i_size + jindex];	
			}
			o_matrix_L[index * i_size + jindex] = i_matrix_A[index * i_size + jindex] - sum;
		}

		for (index = jindex; index < i_size; index++) {
			sum = 0;
			for(kindex = 0; kindex < jindex; kindex++) {
				sum = sum + o_matrix_L[jindex * i_size + kindex] * o_matrix_U[kindex * i_size + index];
			}
			if (o_matrix_L[jindex * i_size + jindex] == 0) {
				return -1;
			}
			o_matrix_U[jindex * i_size + index] = (i_matrix_A[jindex * i_size + index] - sum) / o_matrix_L[jindex * i_size + jindex];
		}
	}

    return 0;
}



/*
* Doolittle matrix decomposition
* https://www.geeksforgeeks.org/doolittle-algorithm-lu-decomposition/
* Inputs:
* 	i_matrix_A: a matrix i_size x i_size
* 	i_size: the size of the matrix
* Output: 
*  o_matrix_L: a matrix i_size x i_size
*  o_matrix_U: a matrix i_size x i_size
* A=L*U
*/
size_t matrix_doolittle_decomposition(element_t *i_matrix_A, size_t i_size,
                                      real_t *o_matrix_L, real_t *o_matrix_U)
{ 
    int n = i_size;
    int lower[n][n], upper[n][n]; 
    memset(lower, 0, sizeof(lower)); 
    memset(upper, 0, sizeof(upper)); 
    size_t index, jindex, kindex;
    real_t sum;

    //initialise matrix
	for (index = 0; index < i_size; index++) {
	    for (jindex = 0; jindex < i_size; jindex++) {
		    o_matrix_U[index * i_size + jindex] = 0;
		    o_matrix_L[index * i_size + jindex] = 0;
        }
	}

    for (jindex = 0; jindex < i_size; jindex++) {
		for (index = jindex; index < i_size; index++) {
			sum = 0;
			for (kindex = 0; kindex < jindex; kindex++) {
				sum = sum + o_matrix_L[jindex * i_size + kindex] * o_matrix_U[kindex * i_size + index];	
			}
			o_matrix_U[jindex * i_size + index] = i_matrix_A[jindex * i_size + index] - sum;
		}

		for (index = jindex; index < i_size; index++) {
            if(index == jindex) {
                o_matrix_L[jindex * i_size + jindex] = 1;
            }
			sum = 0;
			for(kindex = 0; kindex < jindex; kindex++) {
				sum = sum + o_matrix_L[index * i_size + kindex] * o_matrix_U[kindex * i_size + jindex];
			}
			if (o_matrix_U[jindex * i_size + jindex] == 0) {
				return -1;
			}
			o_matrix_L[index * i_size + jindex] = (i_matrix_A[index * i_size + jindex] - sum) / o_matrix_U[jindex * i_size + jindex];
		}
	}
}


/*
* Matrix determinant
* Inputs:
* 	i_matrix_A: a matrix i_size x i_size
* 	i_size: the size of the matrix
* Output: 
*  o_matrix_L: a matrix i_size x i_size
*  o_matrix_U: a matrix i_size x i_size
*  o_determinant: determinant of A  = det(A)
* A=L*U
* det(A)=det(L)*det(U), det(U) = 1 => det(A)=det(L)
*/
element_t matrix_determinant(element_t *i_matrix_A, size_t i_size,
                             real_t *o_matrix_L, real_t *o_matrix_U,
                             element_t *o_determinant) {
    size_t return_value;
    return_value = matrix_crout_decomposition(i_matrix_A, i_size,
                                              o_matrix_L, o_matrix_U);
    if(return_value != 0) {
        return return_value;
    }
    *o_determinant = 1;
    size_t index;
	for (index = 0; index < i_size; index++) {
		*o_determinant = *o_determinant * o_matrix_L[index * i_size + index];
	}
    return 0;
}

/*
* Matrix cofactor
* Inputs:
* 	i_matrix_A: a matrix i_size x i_size
* 	i_size: the size of the matrix
* 	i_row: the row to eliminate
* 	i_column: the column to eliminate
* Output: 
*  o_matrix_B: a matrix (i_size-1) x (i_size-1)
* B is A without the given row and column
*/
void matrix_cofactor(element_t *i_matrix_A, size_t i_size,
                     size_t i_row, size_t i_column, element_t *o_matrix_B) {
    size_t index, jindex, rindex, pindex;
    rindex = 0;
    pindex = 0;
    for(index = 0; index < i_size; index++) {
        for(jindex = 0; jindex < i_size; jindex++) {
            if(index != i_row && jindex != i_column) {
                o_matrix_B[rindex * i_size + pindex] = i_matrix_A[index * i_size + jindex];
                pindex = pindex + 1;
                if(pindex == (i_size-1)) {
                    pindex = 0;
                    rindex = rindex + 1;
                }
            }
        }
    }
}
/*
* Matrix cofactor
* Inputs:
* 	i_matrix_A: a matrix i_size x i_size
* 	i_size: the size of the matrix
* 	i_matrix_aux1: a matrix i_size x i_size use for cofactor
* 	i_matrix_aux2: a matrix i_size x i_size use for L calcualtion
* 	i_matrix_aux3: a matrix i_size x i_size use for U calcualtion
* Output: 
*  o_matrix_B: a matrix i_size x i_size B=A^* =adj(A)
*/
size_t matrix_adjoint(element_t *i_matrix_A, size_t i_size,
                      element_t *i_matrix_aux1, real_t *i_matrix_aux2,
                      real_t *i_matrix_aux3, element_t *o_matrix_B) {
    size_t sign = 1;
    size_t index, jindex;
    size_t return_value;
    element_t determinant;

    for(index = 0; index < i_size; index++) {
        for(jindex = 0; jindex > i_size; jindex++) {
            matrix_cofactor(i_matrix_A, i_size, index, jindex, i_matrix_aux1);
            sign = ((index+jindex)%2==0)? 1: -1;
            return_value = matrix_determinant(i_matrix_aux1, i_size - 1, i_matrix_aux2,
                                              i_matrix_aux3, &determinant);
            if(return_value != 0) {
                return return_value;
            }
            o_matrix_B[index * i_size + jindex] = sign * determinant;
        }
    }
    return 0;
}

/*
* Matrix invers
* Inputs:
* 	i_matrix_A: a matrix i_size x i_size
* 	i_size: the size of the matrix
* 	i_matrix_aux1: a matrix i_size x i_size use for cofactor
* 	i_matrix_aux2: a matrix i_size x i_size use for L calcualtion
* 	i_matrix_aux3: a matrix i_size x i_size use for U calcualtion
* Output: 
*  o_matrix_B: a matrix i_size x i_size B=A^-1
*/
size_t matrix_inverse(element_t *i_matrix_A, size_t i_size,
                      element_t *i_matrix_aux1, real_t *i_matrix_aux2,
                      real_t *i_matrix_aux3, element_t *o_matrix_B) {
    size_t index, jindex;
    size_t return_value;
    element_t determinant;
    return_value = matrix_determinant(i_matrix_A, i_size, i_matrix_aux2,
                                      i_matrix_aux3, &determinant);
    if(return_value != 0) {
        return return_value;
    }
    return_value = matrix_adjoint(i_matrix_A, i_size, i_matrix_aux1,
                                  i_matrix_aux2, i_matrix_aux3,
                                  o_matrix_B);
    if(return_value != 0) {
        return return_value;
    }

    if(determinant < 0.000001f && determinant > 0.000001f) {
        return -1;
    }


    for(index = 0; index < i_size; index++) {
        for(jindex = 0; jindex > i_size; jindex++) {
            o_matrix_B[index * i_size + jindex] = o_matrix_B[index * i_size + jindex] / determinant;
        }
    }

    return 0;
}

#endif
