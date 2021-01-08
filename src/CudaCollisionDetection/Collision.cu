#include"Collision.cuh"
#include "Ball.hpp"


//ͨ�ú���

__device__ float Dist(float x, float y, float z)
{
	return sqrt(x * x + y * y + z * z);
}

__device__ float Dist(Point& p)
{
	return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

__device__ float Multiply(Point& a, Point& b)
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

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

/*
	�������ж��������Ƿ���ײ
	��������a����b
	���أ���1����0
*/
__device__ bool JudgeCollision(Ball& a, Ball& b)
{
	float dist = 0;
	float dist_x = a.CurrentPlace.x - b.CurrentPlace.x;
	float dist_y = a.CurrentPlace.y - b.CurrentPlace.y;
	float dist_z = a.CurrentPlace.z - b.CurrentPlace.z;
	dist = Dist(dist_x, dist_y, dist_z);
	if (dist < a.Radius + b.Radius)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/*
	������������ײ������ٶ�
	��������a����b
	���أ���
*/
__device__ void ChangeSpeed(Ball& a, Ball& b)
{
	//�����ٶȰ����������任�������ٶȲ���
	float dist = 0;
	float diff_x = b.CurrentPlace.x - a.CurrentPlace.x;
	float diff_y = b.CurrentPlace.y - a.CurrentPlace.y;
	float diff_z = b.CurrentPlace.z - a.CurrentPlace.z;
	dist = Dist(diff_x, diff_y, diff_z);

	//���򣬷����ٶ�
	float rate_collide_a = (a.CurrentSpeed.x * diff_x + a.CurrentSpeed.y * diff_y + a.CurrentSpeed.z * diff_z) / dist / dist;
	float speed_collide_a_x = diff_x * rate_collide_a;
	float speed_collide_a_y = diff_y * rate_collide_a;
	float speed_collide_a_z = diff_z * rate_collide_a;

	float rate_collide_b = (b.CurrentSpeed.x * diff_x + b.CurrentSpeed.y * diff_y + b.CurrentSpeed.z * diff_z) / dist / dist;
	float speed_collide_b_x = diff_x * rate_collide_b;
	float speed_collide_b_y = diff_y * rate_collide_b;
	float speed_collide_b_z = diff_z * rate_collide_b;

	float unchanged_a_x = a.CurrentSpeed.x - speed_collide_a_x;
	float unchanged_a_y = a.CurrentSpeed.y - speed_collide_a_y;
	float unchanged_a_z = a.CurrentSpeed.z - speed_collide_a_z;

	float unchanged_b_x = b.CurrentSpeed.x - speed_collide_b_x;
	float unchanged_b_y = b.CurrentSpeed.y - speed_collide_b_y;
	float unchanged_b_z = b.CurrentSpeed.z - speed_collide_b_z;


	//����b������aײb���������߾����ٶ�
	float speed_collide_new_a_x = (speed_collide_a_x * (a.Weight - b.Weight) + speed_collide_b_x * (2 * b.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_a_y = (speed_collide_a_y * (a.Weight - b.Weight) + speed_collide_b_y * (2 * b.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_a_z = (speed_collide_a_z * (a.Weight - b.Weight) + speed_collide_b_z * (2 * b.Weight)) / (a.Weight + b.Weight);

	float speed_collide_new_b_x = (speed_collide_a_x * (2 * a.Weight) + speed_collide_b_x * (b.Weight - a.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_b_y = (speed_collide_a_y * (2 * a.Weight) + speed_collide_b_y * (b.Weight - a.Weight)) / (a.Weight + b.Weight);
	float speed_collide_new_b_z = (speed_collide_a_z * (2 * a.Weight) + speed_collide_b_z * (b.Weight - a.Weight)) / (a.Weight + b.Weight);

	a.CurrentSpeed.x = speed_collide_new_a_x + unchanged_a_x;
	a.CurrentSpeed.y = speed_collide_new_a_y + unchanged_a_y;
	a.CurrentSpeed.z = speed_collide_new_a_z + unchanged_a_z;

	b.CurrentSpeed.x = speed_collide_new_b_x + unchanged_b_x;
	b.CurrentSpeed.y = speed_collide_new_b_y + unchanged_b_y;
	b.CurrentSpeed.z = speed_collide_new_b_z + unchanged_b_z;
}

/*
����������֮�����ײ�����ɺ󣬴�������˶��Լ��ͱ߽����ײ�����У�
���������б�һ�ε�ʱ�䣬X��Χ(-X,X),Z��Χ(-Z,Z),Y��Χ(0,Y)�������
���أ��ޣ����Ǹ������б�
*/
__global__ void UpdateBallsMove(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, int N)
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


//�����㷨��غ���
/*
�����������㷨������ײ�����ٶȸ���
���������б�N����
���أ��ޣ����Ǹ������б�
*/
__global__ void HandleCollisionNaive(Ball* balls, int N)
{
	// ��ȡȫ������
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// ����
	int stride = blockDim.x * gridDim.x;
	for (int k = index; k < N * N; k += stride)
	{
		int i = k / N;
		int j = k % N;
		if(i < j)
		{
			if (JudgeCollision(balls[i], balls[j]))
			{
				ChangeSpeed(balls[i], balls[j]);
			}
		}
	}
}



/*
������GPU��ײ���+�˶������������������㷨��
���������б�һ�ε�ʱ�䣬X��Χ(-X,X),Z��Χ(-Z,Z),Y��Χ(0,Y)�������
���أ��ޣ����Ǹ������б�
*/
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
	HandleCollisionNaive << < gridSize, blockSize >> > (balls_gpu, N);
	// ͬ��device ��֤�������ȷ����
	cudaDeviceSynchronize();

	// ִ��kernel
	UpdateBallsMove <<< gridSize, blockSize >>> (balls_gpu, TimeOnce, XRange, ZRange, Height, N);
	// ͬ��device ��֤�������ȷ����
	cudaDeviceSynchronize();

	// ��¼���
	cudaMemcpy((void*)balls, (void*)balls_gpu, nBytes, cudaMemcpyDeviceToHost);

	// �ͷ��ڴ�
	cudaFree(balls_gpu);
}


//�ռ仮���õĹ��ߺ���
/*
����������ǰi�͵��㷨
������ԭʼ���飬����n
���أ�ԭʼ������ǰi��������
*/
__device__ void PrefixSum(uint32_t *values, unsigned int n) {
	int offset = 1;
	int a;
	uint32_t temp;

	// upsweep
	for (int d = n / 2; d; d /= 2) {
		__syncthreads();

		if (threadIdx.x < d) {
			a = (threadIdx.x * 2 + 1) * offset - 1;
			values[a + offset] += values[a];
		}

		offset *= 2;
	}

	if (!threadIdx.x) {
		values[n - 1] = 0;
	}

	// downsweep
	for (int d = 1; d < n; d *= 2) {
		__syncthreads();
		offset /= 2;

		if (threadIdx.x < d) {
			a = (threadIdx.x * 2 + 1) * offset - 1;
			temp = values[a];
			values[a] = values[a + offset];
			values[a + offset] += temp;
		}
	}
}

/*
����������㷨
������ԭʼ���飬���
���أ������Ϊ��
*/
__device__ void dSumReduce(unsigned int *values, unsigned int *out) {
	// wait for the whole array to be populated
	__syncthreads();

	// sum by reduction, using half the threads in each subsequent iteration
	unsigned int threads = blockDim.x;
	unsigned int half = threads / 2;

	while (half) {
		if (threadIdx.x < half) {
			// only keep going if the thread is in the first half threads
			for (int k = threadIdx.x + half; k < threads; k += half) {
				values[threadIdx.x] += values[k];
			}

			threads = half;
		}

		half /= 2;

		// make sure all the threads are on the same iteration
		__syncthreads();
	}

	// only let one thread update the current sum
	if (!threadIdx.x) {
		atomicAdd(out, values[0]);
	}
}


//�ռ仮���㷨��غ���
/*
��������ʼ��cells��objects���飬ǰ�߼�¼�������ڵĸ�����Ϣ������x��y��z��id��home����phantom�������߼�¼����id��home/phantom
�������յ�cell��phantom�����б�͸��������и��ָ�����Ϣ
���أ�����cells��objects�����cell_num
*/
__global__ void InitCellKernel(uint32_t *cells, uint32_t *objects, Ball* balls, int N,
	float XRange, float ZRange, float Height, float GridSize, int GridX, int GridY, int GridZ, unsigned int* cell_num) 
{
	extern __shared__ unsigned int t[];
	unsigned int count = 0;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
	{
		int current_cell_id = i * 8; //ÿ���������8��������
		int cell_info = 0;
		int object_info = 0;
		int current_count = 0;
		float x = balls[i].CurrentPlace.x;
		float y = balls[i].CurrentPlace.y;
		float z = balls[i].CurrentPlace.z;
		float radius = balls[i].Radius;

		//�ҵ�home cell
		int hash_x = (x + XRange) / GridSize;
		int hash_y = (y) / GridSize;
		int hash_z = (z + ZRange) / GridSize;
		cell_info = hash_x << 17 | hash_y << 9 | hash_z << 1 | HOME_CELL;
		object_info = i << 1 | HOME_OBJECT;
		cells[current_cell_id] = cell_info;
		objects[current_cell_id] = object_info;
		current_cell_id++;
		count++;
		current_count++;

		//��phantom
		for (int dx = -1; dx <= 1; dx++)
		{
			for (int dy = -1; dy <= 1; dy++)
			{
				for (int dz = -1; dz <= 1; dz++)
				{
					int new_hash_x = hash_x + dx;
					int new_hash_y = hash_y + dy;
					int new_hash_z = hash_z + dz;

					//�Լ�������
					if (dx == 0 && dy == 0 && dz == 0)
					{
						continue;
					}

					//Խ�粻����
					if (new_hash_x < 0 || new_hash_x >= GridX ||
						new_hash_y < 0 || new_hash_y >= GridY ||
						new_hash_z < 0 || new_hash_z >= GridZ)
					{
						continue;
					}

					float relative_x = 0;
					float relative_y = 0;
					float relative_z = 0;
					if (dx == 0)
					{
						relative_x = x;
					}
					else if (dx == -1)
					{
						relative_x = hash_x * GridSize - XRange;
					}
					else
					{
						relative_x = (hash_x + 1) * GridSize - XRange;
					}

					if (dz == 0)
					{
						relative_z = z;
					}
					else if (dz == -1)
					{
						relative_z = hash_z * GridSize - ZRange;
					}
					else
					{
						relative_z = (hash_z + 1) * GridSize - ZRange;
					}

					if (dy == 0)
					{
						relative_y = y;
					}
					else if (dy == -1)
					{
						relative_y = hash_y * GridSize;
					}
					else
					{
						relative_y = (hash_y + 1) * GridSize;
					}

					relative_x -= x;
					relative_y -= y;
					relative_z -= z;

					float dist = Dist(relative_x, relative_y, relative_z);
					if (dist < radius)
					{
						int cell_info = new_hash_x << 17 | new_hash_y << 9 | new_hash_z << 1 | PHANTOM_CELL;
						int object_info = i << 1 | PHANTOM_OBJECT;
						cells[current_cell_id] = cell_info;
						objects[current_cell_id] = object_info;
						current_cell_id++;
						count++;
						current_count++;
					}
				}
			}
		}

		//����
		while (current_count < 8)
		{

			cells[current_cell_id] = UINT32_MAX;
			objects[current_cell_id] = i << 2;
			current_cell_id++;
			current_count++;
		}

	}

	//����������ĸ��Ӹ���
	t[threadIdx.x] = count;
	dSumReduce(t, cell_num);
}

/*
��������ʼ��cells�� objects�����������
�������յ�cell��phantom�����б�͸��������и��ָ�����Ϣ���߳���Ϣ
���أ�����cells��objects�����cell_num
 */
 unsigned int InitCells(uint32_t *cells, uint32_t *objects, Ball* balls, int N,
	 float XRange, float ZRange, float Height, float GridSize, int GridX, int GridY, int GridZ, 
	 unsigned int* cell_count_temp, unsigned int num_blocks, unsigned int threads_per_block) {
	 unsigned int cell_count;
	 cudaMemset(&cell_count, 0, sizeof(unsigned int));
	 InitCellKernel << <num_blocks, threads_per_block,
		 threads_per_block * sizeof(unsigned int) >> > (
			 cells, objects, balls, N, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ, cell_count_temp);
	 cudaMemcpy(&cell_count, cell_count_temp, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	 return cell_count;
 }




 __global__ void RadixOrderKernel(uint32_t *keys_in, uint32_t *values_in,
	 uint32_t *keys_out, uint32_t *values_out,
	 uint32_t *radices, uint32_t *radix_sums,
	 unsigned int n, unsigned int cells_per_group,
	 int shift) 
 {
	 int num_radices = 1 << RADIX_LENGTH;
	 extern __shared__ uint32_t s[];
	 uint32_t *t = s + num_radices;
	 int group = threadIdx.x / THREADS_PER_GROUP;
	 int group_start = (blockIdx.x * GROUPS_PER_BLOCK + group) * cells_per_group;
	 int group_end = group_start + cells_per_group;
	 uint32_t k;

	 // initialize shared memory
	 for (int i = threadIdx.x; i < num_radices; i += blockDim.x) 
	 {
		 s[i] = radix_sums[i];

		 // copy the last element in each prefix-sum to a separate array
		 if (!((i + 1) % (num_radices / NUM_BLOCKS))) 
		 {
			 t[i / (num_radices / NUM_BLOCKS)] = s[i];
		 }
	 }

	 __syncthreads();

	 // add padding to array for prefix-sum
	 for (int i = threadIdx.x + NUM_BLOCKS; i < PADDED_BLOCKS; i += blockDim.x) {
		 t[i] = 0;
	 }

	 __syncthreads();

	 // calculate prefix-sum on radix counters
	 PrefixSum(t, PADDED_BLOCKS);
	 __syncthreads();

	 // add offsets to prefix-sum values
	 for (int i = threadIdx.x; i < num_radices; i += blockDim.x)
	 {
		 s[i] += t[i / (num_radices / NUM_BLOCKS)];
	 }

	 __syncthreads();

	 // add offsets to radix counters
	 for (int i = threadIdx.x; i < GROUPS_PER_BLOCK * num_radices; i += blockDim.x)
	 {
		 t[i] = radices[(i / GROUPS_PER_BLOCK * NUM_BLOCKS + blockIdx.x) *
			 GROUPS_PER_BLOCK + i % GROUPS_PER_BLOCK] + s[i / GROUPS_PER_BLOCK];
	 }

	 __syncthreads();

	 // rearrange key-value pairs
	 for (int i = group_start + threadIdx.x % THREADS_PER_GROUP; i < group_end &&
		 i < n; i += THREADS_PER_GROUP) {
		 // need only avoid bank conflicts by group
		 k = (keys_in[i] >> shift & num_radices - 1) * GROUPS_PER_BLOCK + group;

		 // write key-value pairs sequentially by thread in the thread group
		 for (int j = 0; j < THREADS_PER_GROUP; j++) {
			 if (threadIdx.x % THREADS_PER_GROUP == j) {
				 keys_out[t[k]] = keys_in[i];
				 values_out[t[k]] = values_in[i];
				 t[k]++;
			 }
		 }
	 }
 }

 /*
   ��������ǰ׺�ͣ�����radix_sums
   ������֮ǰ���������radices��Ҫ���µĽ��
   ���أ�radix_sums����
*/
 __global__ void RadixSumKernel(uint32_t *radices, uint32_t *radix_sums) 
 {
	 //����Ϊ256����ͽ��
	 extern __shared__ uint32_t shared_radix[];
	 uint32_t total;
	 uint32_t left = 0;
	 int num_radices = 1 << RADIX_LENGTH;
	 uint32_t *radix = radices + blockIdx.x * num_radices * GROUPS_PER_BLOCK;

	 for (int j = 0; j < num_radices / NUM_BLOCKS; j++)
	 {
		 //�ѵ�ǰblock��radixд�빲���ڴ�
		 for (int i = threadIdx.x; i < NUM_BLOCKS * GROUPS_PER_BLOCK; i += blockDim.x) 
		 {
			 shared_radix[i] = radix[i];
		 }

		 __syncthreads();

		 //����padding��256����֤Ч��
		 for (int i = threadIdx.x + NUM_BLOCKS * GROUPS_PER_BLOCK; i < PADDED_GROUPS; i += blockDim.x) 
		 {
			 shared_radix[i] = 0;
		 }

		 __syncthreads();

		 //total���ó���һ�ֵĽ��
		 //ֻ��һ���̸߳������
		 if (!threadIdx.x) 
		 {
			 total = shared_radix[PADDED_GROUPS - 1];
		 }

		 //��s��ǰ׺�ͣ�s[i]����0-i��Ԫ�صĺ�
		 PrefixSum(shared_radix, PADDED_GROUPS);
		 __syncthreads();

		 //������radix���飬Ŀǰradix��ֵ���ǰ׺����
		 for (int i = threadIdx.x; i < NUM_BLOCKS * GROUPS_PER_BLOCK; i += blockDim.x) 
		 {
			 radix[i] = shared_radix[i];
		 }

		 __syncthreads();

		 //1.total�����ۼӣ�total��˵���ÿһ��ǰ׺��֮��
		 //ֻ��һ���̸߳������
		 if (!threadIdx.x) 
		 {
			 total += shared_radix[PADDED_GROUPS - 1];

			 //�ӵ�����ǰ׺����ȥ������ǰ׺�ͳ���ֻ��256
			 radix_sums[blockIdx.x * num_radices / NUM_BLOCKS + j] = left;
			 total += left;
			 left = total;
		 }

		 //����radix���鵽��һλ
		 radix += NUM_BLOCKS * GROUPS_PER_BLOCK;
	 }
 }

 /*
	��������ʼ��radix����
	������cells���飬ȫ��radices��cells������8N����ÿ��cells������ƫ����
	���أ�����radices
 */
 __global__ void RadixTabulateKernel(uint32_t *cells, uint32_t *radices, int num_cells, unsigned int cells_per_group, int shift) 
 {
	 //ÿ��block����һ���ڴ棬��С��radix������256�� * block��group��С
	 //һ��group��������cells
	 extern __shared__ uint32_t shared_radix_by_blocks[];

	 //����ź�ÿ�鿪ʼ��������cell���
	 int group = threadIdx.x / THREADS_PER_GROUP;
	 int group_start = (blockIdx.x * GROUPS_PER_BLOCK + group) * cells_per_group;
	 int group_end = group_start + cells_per_group;


	 int radices_choice_num = 1 << RADIX_LENGTH;
	 //��ʼ�������ڴ�
	 for (int i = threadIdx.x; i < GROUPS_PER_BLOCK * radices_choice_num; i += blockDim.x) 
	 {
		 shared_radix_by_blocks[i] = 0;
	 }

	 __syncthreads();

	 for (int i = group_start + threadIdx.x % THREADS_PER_GROUP; i < group_end && i < num_cells; i += THREADS_PER_GROUP) 
	 {
		 //��ǰԪ�ص�radix
		 uint32_t current_item_radix = (cells[i] >> shift) & (radices_choice_num - 1);

		 //ӳ�䵽��Ӧ��group
		 uint32_t current_item_group_radix = current_item_radix * GROUPS_PER_BLOCK + group;

		 //��֤radix�����Ǵ��еģ���ֹ��ͻ
		 for (int j = 0; j < THREADS_PER_GROUP; j++) 
		 {
			 if (threadIdx.x % THREADS_PER_GROUP == j) {
				 shared_radix_by_blocks[current_item_group_radix]++;
			 }
		 }
	 }

	 __syncthreads();

	 //ȫ���ڴ�ҲҪ����
	 for (int i = threadIdx.x; i < GROUPS_PER_BLOCK * radices_choice_num; i += blockDim.x) 
	 {
		 
		 //i�ǰ�����洢��radix��Ϣ---ÿ�鹲��һ���ڴ�
		 //radices��ȫ�ֵģ������鶼��,��С��256*ÿ��block�����*block����
		 int block_num = i / GROUPS_PER_BLOCK * NUM_BLOCKS + blockIdx.x;
		 int radix_by_block = block_num * GROUPS_PER_BLOCK + i % GROUPS_PER_BLOCK;
		 radices[radix_by_block] = shared_radix_by_blocks[i];
	 }
 }

	 /**
 * @brief Radix sorts an array of objects using occupations as keys.
 * @param cells the input array of cells.
 * @param objects the input array of objects.
 * @param cells_temp sorted array of cells.
 * @param objects_temp array of objects sorted by corresponding cells.
 * @param radices working array to hold radix data.
 * @param radix_sums working array to hold radix prefix sums.
 * @param num_objects the number of objects included.
 */
void cudaSortCells(uint32_t *cells, uint32_t *objects, uint32_t *cells_temp, uint32_t *objects_temp, 
	uint32_t *radices, uint32_t *radix_sums, int N) 
{
	//һ���߳��鴦����ٸ�cell
	unsigned int cells_per_group = (N * 8 - 1) / NUM_BLOCKS / GROUPS_PER_BLOCK + 1;


	uint32_t *cells_swap;
	uint32_t *objects_swap;
	int num_radices = 1 << RADIX_LENGTH;
	//ÿ������8��bit
	for (int i = 0; i < 32; i += RADIX_LENGTH)
	{
		RadixTabulateKernel << <NUM_BLOCKS, GROUPS_PER_BLOCK * THREADS_PER_GROUP, 
			GROUPS_PER_BLOCK * num_radices * sizeof(uint32_t) >> > (
				cells, radices, N * 8, cells_per_group, i);
		RadixSumKernel << <NUM_BLOCKS, GROUPS_PER_BLOCK * THREADS_PER_GROUP,
			PADDED_GROUPS * sizeof(uint32_t) >> > (
				radices, radix_sums);
		RadixOrderKernel << <NUM_BLOCKS, GROUPS_PER_BLOCK * THREADS_PER_GROUP,
			num_radices * sizeof(uint32_t) + GROUPS_PER_BLOCK *
			num_radices * sizeof(uint32_t) >> > (
				cells, objects, cells_temp, objects_temp, radices, radix_sums,
				N * 8, cells_per_group, i);

		// cells sorted up to this bit are in cells_temp; swap for the next pass
		cells_swap = cells;
		cells = cells_temp;
		cells_temp = cells_swap;
		objects_swap = objects;
		objects = objects_temp;
		objects_temp = objects_swap;
	}
}

/*
�������ռ仮���㷨������ײ�����ٶȸ��£���������
���������б�X��Χ(-X,X),Z��Χ(-Z,Z),Y��Χ(0,Y)�����Ӵ�С��X���Ӹ�����Y���Ӹ�����Z���Ӹ�����N����
���أ��ޣ����Ǹ������б�
*/
void HandleCollisionGrid(Ball* balls, float XRange, float ZRange, float Height, 
	float GridSize, int GridX, int GridY, int GridZ, int N,
	unsigned int num_blocks, unsigned int threads_per_block)
{

	//�����ڴ�
	unsigned int cell_size = N * 8 * sizeof(uint32_t);
	unsigned int num_cells_occupied;
	/*unsigned int num_collisions;
	unsigned int num_collisions_cpu;
	unsigned int num_tests;
	unsigned int num_tests_cpu;*/

	uint32_t *cells_gpu;
	uint32_t *cells_gpu_temp;
	uint32_t *objects_gpu;
	uint32_t *objects_gpu_temp;
	unsigned int *temp_gpu;
	uint32_t *radices_gpu;
	uint32_t *radix_sums_gpu;

	int num_radices = 1 << RADIX_LENGTH;

	cudaMalloc((void **)&cells_gpu, cell_size);
	cudaMalloc((void **)&cells_gpu_temp, cell_size);
	cudaMalloc((void **)&objects_gpu, cell_size);
	cudaMalloc((void **)&objects_gpu_temp, cell_size);
	cudaMalloc((void **)&temp_gpu, 2 * sizeof(unsigned int));
	cudaMalloc((void **)&radices_gpu, NUM_BLOCKS * GROUPS_PER_BLOCK * num_radices * sizeof(uint32_t));
	cudaMalloc((void **)&radix_sums_gpu, num_radices * sizeof(uint32_t));

	//������
	num_cells_occupied = InitCells(cells_gpu, objects_gpu, balls, N,
		XRange, ZRange, Height, GridSize, GridX, GridY, GridZ,
		temp_gpu, num_blocks, threads_per_block);

	cudaDeviceSynchronize();
	uint32_t *kebab1 = (uint32_t*)malloc(cell_size);
	uint32_t *kebab2 = (uint32_t*)malloc(cell_size);

	cudaMemcpy((void*)kebab1, (void*)cells_gpu, cell_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)kebab2, (void*)objects_gpu, cell_size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N * 8; i++)
	{
		int type1 = kebab1[i] & 1;
		int z = (kebab1[i] >> 1) & 31;
		int y = (kebab1[i] >> 9) & 31;
		int x = (kebab1[i] >> 17) & 31;
		int type2 = 1 - (kebab2[i] & 1);
		int id = (kebab2[i] >> 1) & 7;
		printf(("x = %d, y = %d, z = %d, type1 = %d, type2 = %d, id = %d\n"), x, y, z, type1, type2, id);
	}
	free(kebab1);
	free(kebab2);

	cudaSortCells(cells_gpu, objects_gpu, cells_gpu_temp, objects_gpu_temp, radices_gpu,
		radix_sums_gpu, cell_size);
	/*num_collisions = cudaCellCollide(d_cells, d_objects, d_positions,
		d_velocities, d_dims, num_objects,
		num_cells, d_temp, &num_tests, num_blocks,
		threads_per_block);*/




	cudaFree(temp_gpu);
	cudaFree(cells_gpu);
	cudaFree(cells_gpu_temp);
	cudaFree(objects_gpu);
	cudaFree(objects_gpu_temp);
	cudaFree(radices_gpu);
	cudaFree(radix_sums_gpu);
}


/*
������GPU��ײ���+�˶��������������ռ仮���㷨��
���������б�һ�ε�ʱ�䣬X��Χ(-X,X),Z��Χ(-Z,Z),Y��Χ(0,Y)��һ�����Ӵ�С��X,Y,Z�ĸ��Ӹ����������
���أ��ޣ����Ǹ������б�
*/
void UpdateBallsGridGPU(Ball* balls, float TimeOnce, float XRange, float ZRange, float Height, 
	float GridSize, int GridX, int GridY, int GridZ, int N)
{
	//���ã�������Ҫ����block��thread
	unsigned int num_blocks = 100;
	unsigned int threads_per_block = 512;
	unsigned int object_size = (N - 1) / threads_per_block + 1;
	if (object_size < num_blocks) {
		num_blocks = object_size;
	}

	Ball* balls_gpu;
	unsigned int nBytes = N * sizeof(Ball);
	cudaMalloc((void**)&balls_gpu, nBytes);


	// ��ʼ������
	cudaMemcpy((void*)balls_gpu, (void*)balls, nBytes, cudaMemcpyHostToDevice);

	// ִ��kernel
	HandleCollisionGrid(balls_gpu, XRange, ZRange, Height, GridSize, GridX, GridY, GridZ, N, num_blocks, threads_per_block);
	// ͬ��device ��֤�������ȷ����
	cudaDeviceSynchronize();

	// ִ��kernel
	UpdateBallsMove << < num_blocks, threads_per_block>> > (balls_gpu, TimeOnce, XRange, ZRange, Height, N);
	// ͬ��device ��֤�������ȷ����
	cudaDeviceSynchronize();

	// ��¼���
	cudaMemcpy((void*)balls, (void*)balls_gpu, nBytes, cudaMemcpyDeviceToHost);

	// �ͷ��ڴ�
	cudaFree(balls_gpu);
}