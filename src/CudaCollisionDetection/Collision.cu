#include"Collision.cuh"
#include "Ball.hpp"


/*
	������������߽���ײ
	������X��Χ��-X, X), Z��Χ(-Z, Z), Y��Χ(0, Y)
	���أ���
*/
__device__ void HandleCollisionBoard(Ball& ball, float XRange, float ZRange, float Height)
{
	if (ball.CurrentPlace.x - ball.Radius < -XRange)
	{
		ball.CurrentPlace.x = -XRange + ball.Radius;
		ball.CurrentSpeed.x = -ball.CurrentSpeed.x;
	}
	else if (ball.CurrentPlace.x + ball.Radius > XRange)
	{
		ball.CurrentPlace.x = XRange - ball.Radius;
		ball.CurrentSpeed.x = -ball.CurrentSpeed.x;
	}
	if (ball.CurrentPlace.z - ball.Radius < -ZRange)
	{
		ball.CurrentPlace.z = -ZRange + ball.Radius;
		ball.CurrentSpeed.z = -ball.CurrentSpeed.z;
	}
	else if (ball.CurrentPlace.z + ball.Radius > ZRange)
	{
		ball.CurrentPlace.z = ZRange - ball.Radius;
		ball.CurrentSpeed.z = -ball.CurrentSpeed.z;
	}
	if (ball.CurrentPlace.y - ball.Radius < 0)
	{
		ball.CurrentPlace.y = ball.Radius;
		ball.CurrentSpeed.y = -ball.CurrentSpeed.y;
	}
	else if (ball.CurrentPlace.y + ball.Radius > Height)
	{
		ball.CurrentPlace.y = Height - ball.Radius;
		ball.CurrentSpeed.y = -ball.CurrentSpeed.y;
	}
}



/*
	����������С�������˶�����߽���ײ
	�����������˶�ʱ�䣬X��Χ��-X, X), Z��Χ(-Z, Z), Y��Χ(0, Y)
	���أ���
*/
__device__ void BallMove(Ball& ball, float time, float XRange, float ZRange, float Height)
{
	ball.CurrentPlace.x = ball.CurrentPlace.x + ball.CurrentSpeed.x * time;
	ball.CurrentPlace.y = ball.CurrentPlace.y + ball.CurrentSpeed.y * time;
	ball.CurrentPlace.z = ball.CurrentPlace.z + ball.CurrentSpeed.z * time;
	HandleCollisionBoard(ball, XRange, ZRange, Height);
}

__global__ void UpdateBallsMoveNaive(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, int N)
{
	// ��ȡȫ������
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// ����
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < N; i += stride)
	{
		BallMove(balls[i], TimeOnce, XRange, ZRange, Height);
	}

}

void UpdateBallsNaiveGPU(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, int N)
{
	// �����й��ڴ�
	int nBytes = N * sizeof(Ball);
	Ball* balls_gpu;
	cudaMallocManaged((void**)&balls_gpu, nBytes);

	// ��ʼ������
	cudaMemcpy((void*)balls_gpu, (void*)balls, nBytes, cudaMemcpyHostToDevice);

	// ����kernel��ִ������
	dim3 blockSize(256);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
	// ִ��kernel
	UpdateBallsMoveNaive <<< gridSize, blockSize >>> (balls_gpu, TimeOnce, XRange, ZRange, Height, N);

	// ͬ��device ��֤�������ȷ����
	cudaDeviceSynchronize();

	// ��¼���
	cudaMemcpy((void*)balls, (void*)balls_gpu, nBytes, cudaMemcpyHostToDevice);

	// �ͷ��ڴ�
	cudaFree(balls);
}