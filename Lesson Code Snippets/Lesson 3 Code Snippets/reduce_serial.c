/* reduce_serial.c
 * Ernest Yeung
 * ernestyalumni@gmail.com
 * Demonstrates reduce with a serial implementation
 * cf. https://classroom.udacity.com/courses/cs344/lessons/86719951/concepts/876789040923#
 * Serial Implementation of Reduce
 * */
#include <stdio.h>  // printf

/* reduce_serial_int: serial implementation of reduce for integers */
int reduce_serial_int(int len, int *input_array) 
{
	int sum = 0;
	for (int i=0; i < len; i++) {
		sum += input_array[i];
	}
	return sum;
}


int main() {
	const int len_eg = 100; // len_eg: length example

	/* "boilerplate" to make interesting arrays for examples */
	int elts[len_eg];
	for (int i=0; i< len_eg;i++) {
		elts[i] = i;
	}


// instructor's original implementation	
	int sum = 0;
	for (int i=0; i < len_eg; i++) {
		sum += elts[i] ; 
	}
//	return sum;

	printf(" This is the result of summing 99 numbers : %d \n", sum );

	int sum2 = 0;
	sum2 = reduce_serial_int( len_eg, elts );

	printf(" This is the result, using a defined function called reduce_serial_int, \
			of summing 99 numbers : %d \n", sum2 );


}
