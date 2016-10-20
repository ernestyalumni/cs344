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



}
