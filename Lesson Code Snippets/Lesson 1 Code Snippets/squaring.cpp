/* squaring.cpp
 * Ernest Yeung 
 * ernestyalumni@gmail.com
 * demonstrate squaring numbers in C++11/C++14
 * */
// Compiling tip: I compiled this with
// g++ -std=c++14 squaring.cpp
// or with flag -std=c++11, because constexpr needs it

#include <iostream> // std::cout


// Via C++11/C++14 container vector

#include <vector>
using std::vector;

int main(int argc, char ** argv) {
	constexpr unsigned NOMPTS { 64 } ; 
	vector<int> xvalints( NOMPTS, 0 ); // xvalints has NOMPTS elements with values 0
	vector<float> xvalfloats( NOMPTS, 0.f ) ; // xvalfloats has NOMPTS elements with values 0.f
	
	std::cout << " This is the size of xvalints   : " << xvalints.size() << std::endl ;
	std::cout << " This is the size of xvalfloats : " << xvalfloats.size() << std::endl ;

	// initialize the arrays with test values

	int initval_int { 0 };
	float initval_float { 0.f };
	
	for (auto iter = xvalints.begin(); iter != xvalints.end() ; ++iter ) {
		*iter += initval_int ;
		initval_int += 1;
	} 

	for (auto iter = xvalfloats.begin(); iter != xvalfloats.end() ; ++iter ) {
		*iter += initval_float ;
		initval_float += 1.f;
	} 

	
	for (auto &i : xvalints ) // for each element in xvalints (note: i is a reference)
		i *= i ; // square the element value
		
	for (auto &i : xvalfloats) // for each element in xvalfloats (note: i is a reference)
		i *= i ; // square the element value
	
	for (auto i = 0; i<10; ++i) {
		std::cout << " For the ith entry, which is i : " << i << " This is xvalints : " << 
			xvalints[i] << std::endl ;
		std::cout << " For the ith entry, which is i : " << i << " This is xvalfloats : " << 
			xvalfloats[i] << std::endl ;
			
	}
	
}
