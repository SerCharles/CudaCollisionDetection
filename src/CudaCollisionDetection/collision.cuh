#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Ball.hpp"

void UpdateBallsNaiveGPU(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, int N);