/* bitwiserightshift.cpp
 * Ernest Yeung
 * ernestyalumni@gmail.com
 * Demonstrates bitwise right shift and bitwise right shift assignment operators
 * Used in reduce.cu of 
 * cf. https://classroom.udacity.com/courses/cs344/lessons/86719951/concepts/876789040923#
 * Serial Implementation of Reduce
 * */
#include <iostream>
#include <vector>  // for std::vector
#include <algorithm> // for std::reverse

// cf. http://stackoverflow.com/questions/2686542/converting-integer-to-a-bit-representation

// convert - converts an int into a binary representation (i.e. 0's and 1's) in a vector Container
std::vector<int> convert(int x) {
	std::vector<int> ret;
	while(x) {
		if (x & 1) // & binary AND operator, copies a bit to result if it exists in both operands
			ret.push_back(1);
		else
			ret.push_back(0);
		x >>= 1;
	}
	std::reverse(ret.begin(),ret.end());
	return ret;
}

int main() {
	// This prints out 1024 and what 1024 in binary
	int ieg { 1024 };
	auto ieg_converted = convert( ieg );
	std::cout << ieg << std::endl; 
	for (auto digit : ieg_converted) 
		std::cout << digit << " " ; 
	std::cout << std::endl;

	// This prints out 1025 and what 1025 in binary
	ieg += 1; 
	ieg_converted = convert( ieg) ;	
	std::cout << ieg << std::endl; 
	for (auto digit : ieg_converted) 
		std::cout << digit << " " ; 
	std::cout << std::endl;

// Explanation of bitwise shift
// cf. http://stackoverflow.com/questions/6385792/what-does-a-bitwise-shift-left-or-right-do-and-what-is-it-used-for
	ieg = 1024; 
// This ends up dividing 1024 by 2, resulting in 512
	ieg >>= 1;
	ieg_converted = convert( ieg) ;	
	std::cout << ieg << std::endl; 
	for (auto digit : ieg_converted) 
		std::cout << digit << " " ; 
	std::cout << std::endl;

	std::cout << " I will demonstrate what the bitshift operators do, to remind us. " << std::endl;
	
	std::vector<int> leftshift_example;
	for (auto i=0; i<11; ++i) {
		leftshift_example.push_back(i) ; }
	std::cout <<   " Originally,             : " ;
	for (auto elem : leftshift_example ) {
		std::cout << " " << elem ; } 
	std::cout << "\n After application of << : " ;
	for (auto &elem : leftshift_example) {
		elem <<= 1; 
		std::cout << " " << elem ; 		}
	std::cout << "\n So effectively << 1 is the multiplication by 2 " << std::endl;
	
	
	std::vector<int> rightshift_example;
	for (auto i=0; i<11; ++i) {
		rightshift_example.push_back(i) ; }
	std::cout <<   " Originally,             : " ;
	for (auto elem : rightshift_example ) {
		std::cout << " " << elem ; } 
	std::cout << "\n After application of >> : " ;
	for (auto &elem : rightshift_example) {
		elem >>= 1; 
		std::cout << " " << elem ; 		}
	std::cout << "\n So effectively >> 1 is division by 2 " << std::endl;

	std::cout << " \n Further example of >> 1; we'll loop through a number of times. " << std::endl;
	for (auto i=11; i<22; ++i) {
		rightshift_example.push_back(i); 
		rightshift_example[i-11] = (i-11);
		}
	std::cout <<       " Originally,                             : " ;
	for (auto elem : rightshift_example ) {
		std::cout << " " << elem ; } 
	for (auto iter = 0 ; iter < 4 ; ++iter) {
		std::cout << "\n After application of >> for iteration " << iter << " : " ;
		for (auto &elem : rightshift_example) {
			elem >>= 1; 
			std::cout << " " << elem ; 		}
	}
	std::cout << "\n Indeed, consider the application of j = 1,2,3,... for >> j operator on elements - " << std::endl; 
	for (auto i=0; i<22; ++i) {
		rightshift_example[i] = i ; }
	std::cout <<   " Originally,               : " ;
	for (auto elem : rightshift_example ) {
		std::cout << " " << elem ; } 
	for (auto iter = 0 ; iter < 4 ; ++iter) {
		std::cout << "\n After application of << " << iter +1 << " : " ;
		for (auto elem : rightshift_example) {
			std::cout << " " << (elem >> (iter + 1)); 		}
	}

	std::cout << "\n Also check out modulus arithmetic" << std::endl;
	std::cout <<   " Originally,                 : " ;
	for (auto elem : rightshift_example ) {
		std::cout << " " << elem ; } 
	for (auto iter = 0 ; iter < 4 ; ++iter) {
		std::cout << "\n After application of % 2**" << iter+1 << " : " ;
		for (auto elem : rightshift_example) {
			std::cout << " " << (elem % (2 << iter) ); 		}
		std::cout << " with (2 << (iter + 1 )) being : " << (2 << iter  ) << std::endl;
	}
	
	std::cout << std::endl;

}
