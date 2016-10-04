/* reduce_serial_vectors.cpp
 * Ernest Yeung
 * ernestyalumni@gmail.com
 * Demonstrates reduce with a serial implementation
 * with C++11/14 vector(s) (library)
 * cf. https://classroom.udacity.com/courses/cs344/lessons/86719951/concepts/876789040923#
 * Serial Implementation of Reduce
 * */

#include <iostream>
#include <vector>

using std::vector; // std::vector -> vector 

/* reduce_serial_int: serial implementation of reduce for integers */
int reduce_serial_int(const vector<int> input_vector) 
{
	int sum = 0;
	for (auto iter : input_vector) {
		sum += iter;
	}
	return sum;
} 
 
int main() {
	constexpr int len_eg { 100 } ;
	
	/* "boilerplate" to make interesting arrays for examples */
	vector<int> elts(len_eg,0);
	for (auto i=0; i< elts.size();i++) {
		elts[i] = i;
	}

	// instructor's original implementation, modified	
	int sum = 0;
	for (auto i=0; i < len_eg; i++) {
		sum += elts[i] ; 
	}
	
	std::cout << " This is the result of summing 99 numbers : " << 
		sum << std::endl;

	int sum2 { 0 };
	sum2 = reduce_serial_int( elts ); 

	std::cout << " This is the result, using a defined function called reduce_serial_int, \
		of summing 99 numbers : " << sum2 << std::endl;


}
 
