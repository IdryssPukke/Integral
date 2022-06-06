#define _USE_MATH_DEFINES

#include <iostream>
#include <math.h>
#include <cmath>
#include <Bits.h>
#include <numeric>

#include <cuda_runtime.h>
#include <nvfunctional>

#include <chrono>
#include <map>

#define TPB 256


using namespace std;

/* Pozwala na szybk¹ zmianê pomiêdzy double oraz float
* double - wiêksza dok³adnoœæ, wolniejsze dzia³anie
* float  - mniejsza dok³adnoœæ, szybsze dzia³anie
*/
typedef double state;

__device__ __host__ state funkcja(state x);