#define _USE_MATH_DEFINES

#include <iostream>
#include <math.h>
#include <cmath>
#include <Bits.h>
#include <numeric>

#include <cuda_runtime.h>
#include <nvfunctional>

#include <chrono>
#include <iomanip>

#include "gausse_legendre.cuh"
#include "rectangle.cuh"
#include "romberg.cuh"
#include "simpson38.cuh"
#include "simpson.cuh"
#include "Parameters.cu"

using namespace std;
using Clock = chrono::steady_clock;

__device__ __host__  state f1(state x) {
	//return sin(x);
	return pow(exp(1.0),x);
}

void calculate_integral(state low_end, state high_end, state(*func)(state, state, int), string name, int i = 16) {
	
	state result = 0;
	state result_new = 0;

	for (int k = 0; i < INT_MAX; i *= 2, k++) {
		auto start = Clock::now();
		result_new = func(low_end, high_end, i);
		auto end = Clock::now();
		if (abs(result_new - result) < 1e-5 && k!=0) { 
			cout << setprecision(7) << fixed << name << ":\t" << result << "\t" << static_cast<chrono::duration<state>>(end - start).count() << endl << endl;
			break; 
		}
		else result = result_new;
	}	
}


int main() {

	state low_end = 0;
	state high_end =  M_PI;
	int steps = 1000000;
	int N = 8;


	//first time to run CUDA, time not measured
	rectangle_method_CUDA(low_end, high_end, 1);

	calculate_integral(low_end, high_end, &rectangle_method, "rectangle", steps);
	calculate_integral(low_end, high_end, &rectangle_method_CUDA, "rectangleCUDA", steps);
	calculate_integral(low_end, high_end, &rectangle_method_CUDA_red, "rectangleCUDAr", steps);
	calculate_integral(low_end, high_end, &simpson_method,  "simpson", steps);
	calculate_integral(low_end, high_end, &simpson_method_CUDA, "simpsonCUDA", steps);
	calculate_integral(low_end, high_end, &simpson_method_CUDA_red, "simpsonCUDAr", steps);
	calculate_integral(low_end, high_end, &simpson38_method, "simpson38", steps);
	calculate_integral(low_end, high_end, &simpson38_method_CUDA, "simpson38CUDA", steps);
	calculate_integral(low_end, high_end, &simpson38_method_CUDA_red, "simpson38CUDAr", steps);

	//Liczenie maksymalnie dla N = 16
	calculate_integral(low_end, high_end, &romberg_method, "romberg", N);
	calculate_integral(low_end, high_end, &romberg_method_CUDA_1, "rombergCUDA1", 64);

	//Liczenie maksymalnie dla N = 16
	calculate_integral(low_end, high_end, &romberg_method_CUDA_2, "rombergCUDA2", N);
	calculate_integral(low_end, high_end, &gauss_legendre_method, "gauss_legendre", 64);
	calculate_integral(low_end, high_end, &gauss_legendre_method_CUDA, "gausslegendreCUDA", 64);
	calculate_integral(low_end, high_end, &gauss_legendre_method_CUDA_red, "gausslegendreCUDAr", 64);


	cout<< endl<< rectangle_method_CUDA_red2(low_end, high_end, 15)<<endl;
	//cout<< gauss_legendre_method_CUDA(low_end, high_end, 15)<<endl;
	for (int i = 2; i < 5; i++) {
		//cout << endl<<pow(2, i) << " ";
		//romberg_method_CUDA_2(low_end, high_end, pow(2, i));
		//simpson38_method_CUDA_red(low_end, high_end, pow(10,i));
	}
	
	return 0;
}