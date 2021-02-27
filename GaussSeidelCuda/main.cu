#include <iostream>
#include <math.h>
#include <set>
#include <vector>
#include <time.h>
#include "EigenLibrary/Eigen/Dense"
#include "EigenLibrary/Eigen/Sparse"

#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 128

using namespace Eigen;

__global__ void PartitionSolver(float *A_value, int *A_index, float *b, float *x, int *paritions, int paritionStart, int partitionSize, int NNZRow)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (tid < partitionSize)
	{
		float sum = 0.0f;
		int ui = paritions[paritionStart + tid];
		float diagValue = 0.0f;
		int indexStarter = ui * (NNZRow + 1), valueStarter = ui * NNZRow;
		int matRowSize = A_index[indexStarter];
		for (int i = 0; i < matRowSize; i++)
		{
			int j = A_index[indexStarter + 1 + i];
			if (j != ui) sum += A_value[valueStarter + i] * x[j];
			else diagValue = A_value[valueStarter + i];
		}
		x[ui] = (b[ui] - sum) / diagValue;
	}
}

int ParallelGaussSeidel(SparseMatrix<float>& A, VectorXf& o_x, VectorXf& b, int *partitions, int numPartitions, std::vector<int> &partitionStarts)
{
	//store matrix in device in row major from a colmn major eigen sparse matrix
	int NNZ = (int)A.nonZeros();
	int N = (int)A.cols();//matrix size
	
	float *A_value;
	int *A_index;
	int NNZRow = 0;
	{
		std::vector<int> rowNNZ(N, 0);
		clock_t t1 = clock();
		for (int k = 0; k < A.outerSize(); ++k)
		{
			for (SparseMatrix<float>::InnerIterator it(A, k); it; ++it)
			{
				int row = (int)it.row();
				rowNNZ[row]++;
			}
		}
		
		for (int i = 0; i < N; i++)
		{
			if (rowNNZ[i] > NNZRow) NNZRow = rowNNZ[i];
		}

		float *values = (float *)malloc(NNZRow * N * sizeof(float));
		cudaMalloc((void**)&A_value, NNZRow * N * sizeof(float));
		int *indices = (int *)malloc((NNZRow + 1) * N * sizeof(int));
		cudaMalloc((void**)&A_index, (NNZRow + 1) * N * sizeof(int));

		for (int i = 0; i < N; i++) indices[i * (NNZRow + 1)] = 0;

		for (int k = 0; k < A.outerSize(); ++k)
		{
			for (SparseMatrix<float>::InnerIterator it(A, k); it; ++it)
			{
				int row = (int)it.row();
				int rowSize = indices[row * (NNZRow + 1)];
				values[row * NNZRow + rowSize] = it.value();
				indices[row * (NNZRow + 1) + 1 + rowSize] = (int)it.col();
				indices[row * (NNZRow + 1)]++;
			}
		}
		cudaMemcpy(A_value, values, NNZRow * N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(A_index, indices, (NNZRow + 1) * N * sizeof(int), cudaMemcpyHostToDevice);
		free(values);
		free(indices);
		clock_t elapsedTime = clock() - t1;
		std::cout << "Time to copy data to GPU (1): " << elapsedTime << std::endl;
	}

	//allocate GPU memeory for b
	float *d_b;
	cudaMalloc((void**)&d_b, N * sizeof(float));
	{
		clock_t t1 = clock();
		cudaMemcpy(d_b, b.data(), N * sizeof(float), cudaMemcpyHostToDevice);
		clock_t elapsedTime = clock() - t1;
		std::cout << "Time to copy data to GPU (2): " << elapsedTime << std::endl;
	}

	//allocate GPU memeory for x
	float *d_x;
	cudaMalloc((void**)&d_x, N * sizeof(float));
	{
		clock_t t1 = clock();
		cudaMemcpy(d_x, o_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
		clock_t elapsedTime = clock() - t1;
		std::cout << "Time to copy data to GPU (3): " << elapsedTime << std::endl;
	}

	int *d_partitions;//store partions in device 
	cudaMalloc((void**)&d_partitions, N * sizeof(int));
	cudaMemcpy(d_partitions, partitions, N * sizeof(int), cudaMemcpyHostToDevice);
	free(partitions);

	int iter;
	for (iter = 0; iter < 100; iter++)
	{
		for (size_t i = 0; i < numPartitions; i++)
		{
			int partitionSize = partitionStarts[i + 1] - partitionStarts[i];
			PartitionSolver<<<(partitionSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(A_value, A_index, d_b, d_x, d_partitions, partitionStarts[i], partitionSize, NNZRow);
			cudaDeviceSynchronize();
		}
		cudaMemcpy(o_x.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
		//std::cout << o_x << std::endl;
		
		//compute residue
		VectorXf r;
		r = b - A * o_x;
		float rNorm = r.norm();
		float tol = 0.001f;

		if (rNorm < tol) break;
	}
	//free GPU memory
	cudaFree(A_value);
	cudaFree(A_index);
	cudaFree(d_b);
	cudaFree(d_x);
	cudaFree(d_partitions);
	
	return iter;
}

__global__ void ColorInit(int *nodePalettes, int *nextColor, int *neighbours, int shrinkFactor, int stride, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n)
	{
		int maxColor = (int)(neighbours[stride * i] / shrinkFactor);
		if (maxColor == 0) maxColor = 1;
		nextColor[i] = maxColor;
		int starter = i * stride;
		nodePalettes[starter] = 0;
		for (int c = 0; c < maxColor; c++)
		{
			nodePalettes[starter]++;
			nodePalettes[starter + 1 + c] = c;
		}
	}
}

__global__ void RandomInit(unsigned int seed, curandState_t* states, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n)
	{
		curand_init(0, /* the seed can be the same for each core, here we pass the time in from the CPU */
			tid, /* the sequence number should be different for each core (unless you want all
						   cores to get the same sequence of numbers for some reason - use thread id! */
			0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
			&states[tid]);
	}
}

__global__ void TentativeColoring(int *nodeColor, int *nodePalettes, int *U, curandState_t* states, int stride, int USize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < USize)
	{
		int sid = U[tid];
		int starter = stride * sid;
		int paletteSize = nodePalettes[starter];
		
		int offset = curand(&states[sid]) % paletteSize;
		int offsetTemp = offset;
		
		while (offset < paletteSize)
		{
			if (nodePalettes[starter + 1 + offset] != -1)
			{
				break;
			}
			offset++;
		}
		
		if (offset == paletteSize)
		{	
			while (nodePalettes[starter + 1 + offsetTemp] == -1)
			{
				offsetTemp--;
			}
			offset = offsetTemp;
		}

		nodeColor[sid] = nodePalettes[starter + 1 + offset];
	}
}

__global__ void ResolveConflict(int *neighbours, int *nodeColor, int *nodePalettes, int *U, int stride, int USize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < USize)
	{
		bool legalNode = true;
		bool localMax = true;

		int sid = U[tid];
		int starter = stride * sid;
		int numNeighbours = neighbours[starter];

		int myColor = nodeColor[sid];
		for (int i = 0; i < numNeighbours; i++)
		{
			int nid = neighbours[starter + 1 + i];
			int neighbourColor = nodeColor[nid];
			if (myColor == neighbourColor)
			{
				legalNode = false;
			}
			if (sid < nid)
			{
				localMax = false;
			}
			if (!legalNode && !localMax) break;
		}

		if (legalNode || localMax)
		{
			for (int i = 0; i < numNeighbours; i++)
			{
				int nid = neighbours[starter + 1 + i];
				int n_starter = stride * nid;
				int n_paletteSize = nodePalettes[n_starter];
				for (int j = 0; j < n_paletteSize; j++)
				{
					if (nodePalettes[n_starter + 1 + j] == myColor)
					{
						nodePalettes[n_starter + 1 + j] = -1;//remove the color of current node from its neighbor's palette
						break;
					}
				}
			}
			U[tid] = -1;
		}
	}
}

__global__ void FeedHungry(int *nodePalettes, int *nextColor, int USize, int *U, int stride, bool noProgress)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < USize)
	{
		int sid = U[tid];
		int starter = stride * sid;
		
		bool isEmpty = true;
		int paletteSize = nodePalettes[starter];
		for (int i = 0; i < paletteSize; i++)
		{
			if (nodePalettes[starter + 1 + i] != -1)
			{
				isEmpty = false;
				break;
			}
		}
		
		if (isEmpty || noProgress)
		{
			nodePalettes[starter + 1 + paletteSize] = nextColor[sid];
			nextColor[sid] = nextColor[sid] + 1;
			nodePalettes[starter] = nodePalettes[starter] + 1;
		}
	}
}

int* GraphColoring(SparseMatrix<float> &A, int &o_numParitions, std::vector<int> &o_partitionStarts)
{
	int n = (int)A.rows();
	//construct graph
	std::vector<int> neighbourCounter(n, 0);
	for (int k = 0; k < A.outerSize(); ++k)
	{
		for (SparseMatrix<float>::InnerIterator it(A, k); it; ++it)
		{
			int col = (int)it.col();
			int row = (int)it.row();
			if (col != row) neighbourCounter[row]++;
		}
	}

	int maxDegree = 0;
	int minDegree = INT_MAX;
	for (int i = 0; i < n; i++)
	{
		int currentDegree = neighbourCounter[i];
		if (currentDegree > maxDegree)
		{
			maxDegree = currentDegree;
		}
		
		if (currentDegree < minDegree)
		{
			minDegree = currentDegree;
		}
	}
	//generate neigbour array
	int *neighbours;
	int stride = maxDegree + 1;
	{	
		neighbours = (int *)malloc(n * stride * sizeof(int));
		for (int i = 0; i < n; i++) neighbours[i * stride] = 0;
		for (int k = 0; k < A.outerSize(); ++k)
		{
			for (SparseMatrix<float>::InnerIterator it(A, k); it; ++it)
			{
				int col = (int)it.col();
				int row = (int)it.row();
				if (col != row)
				{
					int starter = row * stride;
					int i = neighbours[starter];
					neighbours[starter + 1 + i] = col;
					neighbours[starter]++;
				}
			}
		}
	}
	int *d_neighbours;
	cudaMalloc((void**)&d_neighbours, n * stride * sizeof(int));
	cudaMemcpy(d_neighbours, neighbours, n * stride * sizeof(int), cudaMemcpyHostToDevice);
	free(neighbours);

	//Color
	int *nodeColor = (int *)malloc(n * sizeof(int));//this will be used to store the final result in the end
	int *d_nodeColor;
	cudaMalloc((void**)&d_nodeColor, n * sizeof(int));
	int *d_nodePalettes;
	cudaMalloc((void**)&d_nodePalettes, n * stride * sizeof(int));
	int *d_nextColor;//next color that will be used for each node once all existing ones are consumed
	cudaMalloc((void**)&d_nextColor, n * sizeof(int));
	int shrinkFactor = minDegree + 1;
	ColorInit <<< (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(d_nodePalettes, d_nextColor, d_neighbours, shrinkFactor, stride, n);

	//U
	int USize = n;
	int *U = (int *)malloc(n * sizeof(int));
	int *tempU = (int *)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) U[i] = i;
	int *d_U;
	cudaMalloc((void**)&d_U, n * sizeof(int));
	cudaMemcpy(d_U, U, USize * sizeof(int), cudaMemcpyHostToDevice);

	//setup random function for each kernel
	curandState_t* states;
	cudaMalloc((void**)&states, n * sizeof(curandState_t));
	RandomInit <<< (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>((unsigned int)time(0), states, n);
	cudaDeviceSynchronize();
	
	std::cout << "Start to color graph..." << std::endl;
	int counter = 0;
	clock_t max1 = 0;
	clock_t max2 = 0;
	clock_t max3 = 0;
	clock_t t = clock();
	while (USize)
	{
		//std::cout << USize << std::endl;
		//tentative coloring
		{
			clock_t t = clock();
			TentativeColoring << < (USize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_nodeColor, d_nodePalettes, d_U, states, stride, USize);
			cudaDeviceSynchronize();
			clock_t elapsedTime = clock() - t;
			if (elapsedTime > max1) max1 = elapsedTime;
		}

		//conflict resolution
		bool noProgress = false;
		{
			clock_t t = clock();
			ResolveConflict << < (USize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_neighbours, d_nodeColor, d_nodePalettes, d_U, stride, USize);
			cudaDeviceSynchronize();
			cudaMemcpy(U, d_U, USize * sizeof(int), cudaMemcpyDeviceToHost);
			
			int newUSize = 0;
			std::vector<int> removed;
			for (int i = 0; i < USize; i++)
			{
				if (U[i] != -1)
				{
					tempU[newUSize] = U[i];
					newUSize++;
				}
				else 
				{
					removed.push_back(i);
				}
			}
			//swap U and tempU
			int *temp = U;
			U = tempU;
			tempU = temp;
			if (USize == newUSize) noProgress = true;
			USize = newUSize;
			cudaMemcpy(d_U, U, USize * sizeof(int), cudaMemcpyHostToDevice);
			clock_t elapsedTime = clock() - t;
			if (elapsedTime > max2) max2 = elapsedTime;
		}
		
		//feed the hungry
		{	
			clock_t t = clock();
			FeedHungry << < (USize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_nodePalettes, d_nextColor, USize, d_U, stride, noProgress);
			cudaDeviceSynchronize();
			clock_t elapsedTime = clock() - t;
			if (elapsedTime > max3) max3 = elapsedTime;
		}
		counter++;
	}
	std::cout << "tentative coloring: " << max1 << std::endl;
	std::cout << "conflict resolution: " << max2 << std::endl;
	std::cout << "feed the hungry: " << max3 << std::endl;

	clock_t elapsedTime = clock() - t;
	std::cout << "iterations to generate graph: " << counter << std::endl;
	std::cout << "time to finish all iterations: " << elapsedTime << std::endl;

	std::set<int> usedColors;
	cudaMemcpy(nodeColor, d_nodeColor, n * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++)
	{
		usedColors.insert(nodeColor[i]);
	}
	
	//generate partions
	o_numParitions = (int)usedColors.size();
	int *o_partitions = (int *)malloc(n * sizeof(int));
	{
		int partIt = 0;
		for (int ic = 0; ic < o_numParitions; ic++)
		{
			o_partitionStarts.push_back(partIt);
			auto setIt = usedColors.begin();
			advance(setIt, ic);
			int usedColor = *setIt;
			for (int i = 0; i < n; i++)
			{
				if (nodeColor[i] == usedColor)
				{
					o_partitions[partIt] = i;
					partIt++;
				}
			}
		}
		o_partitionStarts.push_back(partIt);
	}

	//free memory
	cudaFree(d_neighbours);
	cudaFree(d_nodeColor);
	cudaFree(d_nodePalettes);
	cudaFree(d_nextColor);
	cudaFree(d_U);
	cudaFree(states);

	free(nodeColor);
	free(U);
	free(tempU);

	return o_partitions;
}

void MatrixGenerator(MatrixXf& o_Mat)
{
	int n = (int)o_Mat.rows();
	for (int i = 0; i < n; ++i)
	{
		for (int j = i; j < n; ++j)
		{
			if (rand() % 9 == 0)
			{
				o_Mat(i, j) = float(rand() % 10);
				o_Mat(j, i) = o_Mat(i, j);
			}
		}
	}

	/*
	Now we iterate over every row, and modify the matrix to make sure it's diagonally dominant.
	*/
	for (int i = 0; i < n; ++i)
	{
		float diag = fabs(o_Mat(i, i));
		float row_sum = 0.0f;

		for (int j = 0; j < n; ++j)
		{
			if (i != j)
			{
				row_sum += fabs(o_Mat(i, j));
			}
		}

		/*
		Not diagonally dominant. So increase the diagonal value to fix that.
		*/
		if (!(diag >= row_sum))
		{
			o_Mat(i, i) += (row_sum - diag);
		}

		if (fabs(o_Mat(i, i)) < 0.00001f)
		{
			o_Mat(i, i) += 1.0f;
		}
	}

}
int GaussSeidel(SparseMatrix<float, RowMajor>& A, VectorXf& x, VectorXf& b)
{
	int iter;
	int n = (int)A.rows();
	for (iter = 0; iter < 100; iter++)
	{
		for (int k = 0; k < A.outerSize(); ++k)
		{
			float sum = 0.0f;
			float diagValue = 0.0f;
			for (SparseMatrix<float, RowMajor>::InnerIterator it(A, k); it; ++it)
			{
				int row = (int)it.row();
				int col = (int)it.col();
				if (row != col)
				{
					sum += it.value() * x(col);
				}
				else
				{
					diagValue = it.value();
				}
			}
			x(k) = (b(k) - sum) / diagValue;
		}
		//compute residue
		VectorXf r;
		r = b - A * x;
		float rNorm = r.norm();
		float tol = 0.001f;

		if (rNorm < tol) break;
	}

	return iter;
}

int main()
{
	//srand((unsigned int)time(NULL));
	srand(13000);
	int n = 10000;
	MatrixXf M(n, n);
	M.setZero();
	MatrixGenerator(M);
	SparseMatrix<float> M_sparse = M.sparseView();
	std::cout << "Matrix generated..." << std::endl << std::endl;
	//std::cout << M << std::endl;

	VectorXf expectedSolution(n);
	for (int i = 0; i < n; i++)
	{
		expectedSolution(i) = (float)i;
	}
	VectorXf b(n);
	b = M * expectedSolution;
	std::cout << expectedSolution(0) << ", " << expectedSolution(1) << ", " << "... " << expectedSolution(n - 2) << ", " << expectedSolution(n - 1) << std::endl;
	
	VectorXf x(n);
	
	//regular GS
	x.setZero();
	clock_t t = clock();
	SparseMatrix<float, RowMajor> Mr_sparse(M_sparse);
	int iter1 = GaussSeidel(Mr_sparse, x, b);
	//Eigen::SimplicialCholesky<SparseMatrix<float>> chol(M_sparse);
	//x = chol.solve(b);
	t = clock() - t;
	std::cout << "GS iterations: " << iter1 << std::endl;
	std::cout << x(0) << ", " << x(1) << ", " << "... " << x(n - 2) << ", " << x(n - 1) << std::endl;
	std::cout << "Elapsed time: " << t << std::endl << std::endl;
	
	
	//parallel GS
	x.setZero();
	int numParitions;
	std::vector<int> partitionStarts;
	t = clock();
	int *partitions = GraphColoring(M_sparse, numParitions, partitionStarts);
	clock_t graphTime = clock() - t;
	std::cout << "Time to generate graph: " << graphTime << std::endl;
	std::cout << "# of partitions: " << numParitions << std::endl;
	int iter2 = ParallelGaussSeidel(M_sparse, x, b, partitions, numParitions, partitionStarts);
	t = clock() - t;
	
	std::cout << x(0) << ", " << x(1) << ", " << "... " << x(n - 2) << ", " << x(n - 1) << std::endl;
	std::cout << "GS iterations: " << iter2 << std::endl;
	std::cout << "Elapsed time: " << t << std::endl;
	
	return 0;
}