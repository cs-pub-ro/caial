// Part of this code is taken from: https://www.codeproject.com/Articles/813185/Calculating-the-Number-PI-Through-Infinite-Sequenc
// Approximation of the number PI
// Language: C
// Author: Jose Cintra (jose.cintra@html-apps.info)
#include <stdlib.h>
#include <stdio.h>
#include "../common/common.h"
#define WITH_POSIT
// #define WITH_SQRT
#ifdef PFDEBUG
#include <stdio.h>
#endif
#ifdef WITH_SQRT
#include <math.h>
#endif
#include<stdint.h>
// variables
int sign_aux = 1;
element_t pi = 0.0;
element_t sign = 1.0;
element_t i = 1.0;
// constants
element_t one = 1.0;
element_t two = 2.0;
element_t three = 3.0;
element_t four = 4.0;
element_t minus_one = 4.0;
#ifdef WITH_POSIT
/*
// posit(32,2)
uint32_t posit_zero = 0x00000000;
uint32_t posit_one = 0x40000000;
uint32_t posit_two = 0x48000000;
uint32_t posit_three = 0x4c000000;
uint32_t posit_four = 0x50000000;
*/
// posit(32,3)
uint32_t posit_zero =  0x00000000;
uint32_t posit_one =   0x40000000;
uint32_t posit_two =   0x44000000;
uint32_t posit_three = 0x46000000;
uint32_t posit_four =  0x48000000;
uint32_t posit_minus_one = 0xc0000000;
#else
uint32_t fp32_zero = 0x00000000;
uint32_t fp32_one = 0x3f800000;
uint32_t fp32_two = 0x40000000;
uint32_t fp32_three = 0x40400000;
uint32_t fp32_four = 0x40800000;
#endif /* WITH_POSIT */
void init() {
#ifdef WITH_POSIT
  *((uint32_t*)&pi) = posit_zero;
  *((uint32_t*)&one) = posit_one;
  *((uint32_t*)&two) = posit_two;
  *((uint32_t*)&three) = posit_three;
  *((uint32_t*)&four) = posit_four;
  *((uint32_t*)&sign) = posit_one;
  *((uint32_t*)&minus_one) = posit_minus_one;
#else
  *((uint32_t*)&pi) = fp32_zero;
  *((uint32_t*)&one) = fp32_one;
  *((uint32_t*)&two) = fp32_two;
  *((uint32_t*)&three) = fp32_three;
  *((uint32_t*)&four) = fp32_four;
  *((uint32_t*)&sign) = fp32_one;
#endif /* WITH_POSIT */
}
#ifdef WITH_SQRT
void viete(int n) {
int k, j;
   pi = one;
for(k = n; k > 1; k--) {
      i = two;
for(j = 1; j < k; j++){
         i = two + sqrt(i);
      }
      i = sqrt(i);
      pi = pi * i / two;
   }
   pi = pi * sqrt(two) / two;
   pi = two / pi;
#ifdef PFDEBUG   
printf("\nAproximated value of PI = %1.16lf\n", pi);  
#endif
}
#endif /* WITH_SQRT */
void leibniz(int n) {   
for(i = 0; i < n; i++){
     pi = pi + sign * four / (two * i + one);
     sign = -sign;
   }
#ifdef PFDEBUG
printf("\nAproximated value of PI = %1.16lf\n", pi);  
#endif
}
void nilakantha(int n) {
// variables
element_t aux1,aux2,aux3, aux4, aux5;
#ifdef WITH_POSIT
  *((uint32_t*)&pi) = posit_three;
  *((uint32_t*)&i) = posit_two;
#else
  *((uint32_t*)&pi) = fp32_three;
  *((uint32_t*)&i) = fp32_two;
#endif
int j;
sign_aux = 1;
for(j = 2; j <= n*2; j += 2){
	aux1 = i + one;
	aux2 = i + two;
	aux3 = i * aux1;
	aux3 = aux3 * aux2;
	aux4 = four / aux3;
	aux5 = sign * aux4;
  //pi = pi + sign * (four / (i * (i + one) * (i + two)));
  pi = pi + aux5;
  sign = sign * minus_one;
  i = i + two;
  }
#ifdef PFDEBUG
printf("\nAproximated value of PI = %1.16lf\n", pi);  
#endif
}
int main() {
init();
// leibniz(100000000);
// viete(100);
nilakantha(200);
return 0;
}