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

std::vector<int> convert(int x) {
	std::vector<int> ret;
	while(x) {
		if (x&1)
			ret.push_back(1);
		else
			ret.push_back(0);
		x>>=1;
	}
	std::reverse(ret.begin(),ret.end());
	return ret;
}

int main() {
	int ieg { 1024 };
	auto ieg_converted = convert( ieg );
	std::cout << ieg << std::endl; 
	for (auto digit : ieg_converted) 
		std::cout << digit << " " ; 
	std::cout << std::endl;

	ieg += 1; 
	ieg_converted = convert( ieg) ;	
	std::cout << ieg << std::endl; 
	for (auto digit : ieg_converted) 
		std::cout << digit << " " ; 
	std::cout << std::endl;

// Explanation of bitwise shift
// cf. http://stackoverflow.com/questions/6385792/what-does-a-bitwise-shift-left-or-right-do-and-what-is-it-used-for

	ieg = 1024; 
	ieg >>= 1;
	ieg_converted = convert( ieg) ;	
	std::cout << ieg << std::endl; 
	for (auto digit : ieg_converted) 
		std::cout << digit << " " ; 
	std::cout << std::endl;

}
