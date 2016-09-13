/* squaring_02.cpp
 * Ernest Yeung 
 * ernestyalumni@gmail.com
 * demonstrate squaring numbers in C++11/C++14
 * but with an input and output vector
 * */
// Compiling tip: I compiled this with
// g++ -std=c++14 squaring.cpp
// or with flag -std=c++11, because constexpr needs it

#include <iostream> // std::cout
#include <chrono>

// Via C++11/C++14 container vector

#include <vector>
using std::vector;
using namespace std;

int main(int argc, char ** argv) {

	constexpr unsigned NOMPTS { 64 } ; 

	vector<int> xvalints_in( NOMPTS, 0 ); // xvalints has NOMPTS elements with values 0
	vector<float> xvalfloats_in( NOMPTS, 0.f ) ; // xvalfloats has NOMPTS elements with values 0.f

	vector<int> xvalints_out( NOMPTS, 0 ); // xvalints has NOMPTS elements with values 0
	vector<float> xvalfloats_out( NOMPTS, 0.f ) ; // xvalfloats has NOMPTS elements with values 0.f

	// initialize the vectors with test values

	int initval_int { 0 };
	float initval_float { 0.f };
	
	for (auto iter = xvalints_in.begin(); iter != xvalints_in.end() ; ++iter ) {
		*iter += initval_int ;
		initval_int += 1;
	} 

	for (auto iter = xvalfloats_in.begin(); iter != xvalfloats_in.end() ; ++iter ) {
		*iter += initval_float ;
		initval_float += 1.f;
	} 


	auto starttime = chrono::steady_clock::now();

	
	for (auto &i : xvalints_in ) // for each element in xvalints (note: i is a reference)
		i *= i ; // square the element value
		
	for (auto &i : xvalfloats_in) // for each element in xvalfloats (note: i is a reference)
		i *= i ; // square the element value


	auto outiter_int   { xvalints_out.begin() } ;
	auto outiter_float { xvalfloats_out.begin() } ;

	for (auto iter = xvalints_in.begin(); iter != xvalints_in.end() ; ++iter) {
		*outiter_int = *iter ; 
		++outiter_int; 
	}

	for (auto iter = xvalfloats_in.begin(); iter != xvalfloats_in.end() ; ++iter) {
		*outiter_float = *iter ; 
		++outiter_float; 
	}

	auto endtime = chrono::steady_clock::now() ;
	
	auto difftime = endtime - starttime; 
	
	std::cout << "The serial for loops (2) took : " << 
		chrono::duration <double, milli> (difftime).count() << " ms" << std::endl;

	
	for (auto i = 0; i<10; ++i) {
		std::cout << " For the ith entry, which is i : " << i << " This is xvalints_out   : " << 
			xvalints_out[i] << std::endl ;
		std::cout << " For the ith entry, which is i : " << i << " This is xvalfloats_out : " << 
			xvalfloats_out[i] << std::endl ;
			
	}
	
}
