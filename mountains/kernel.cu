#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess){ \
	printf("Error at %s:%d\n",__FILE__,__LINE__); \
	return EXIT_FAILURE;}} while(0)

const float POINT_DENSITY = 0.1;
const float RANDOM_RADIUS = 100;
const float SAMPLES = 50.0;

struct Point {
	float2 position;
	float2 direction;
	float height;
};

__global__ void setupKernel(curandState *state) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(id, 0, 0, &state[id]);
}


__device__ float2 operator + (const float2 a, const float2 b) {
	return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float2 operator * (const float2 a, const float s) {
	return make_float2(a.x * s, a.y * s);
}

__device__ float2 operator - (const float2 end, const float2 start) {
	return make_float2(end.x - start.x, end.y - start.y);
}

__device__ float2 normalize(const float2 vec) {
	float im = __frsqrt_rn(__fmul_rd(vec.x, vec.x) + __fmul_rd(vec.y, vec.y) );
	return vec * im;
}
__device__ float2 operator - (const float2 start) {
	return make_float2(-start.x, -start.y);
}

__device__ void writePoint(const Point point, float* heightmap, const int width, const int height) {
	if(point.position.x <= width && point.position.y <= height)
		heightmap[int(point.position.x + int(point.position.y) * width)] = point.height;
}

__device__ void writeVector(const float2 point, float* heightmap, const int width, const int height, const float value) {
	if (point.x <= width && point.y <= height)
		heightmap[int(point.x + int(point.y) * width)] = value;
}

__device__ float magnitude(float2 vec) {
	return __fsqrt_rn(__fmul_rd(vec.x, vec.x) + __fmul_rd(vec.y, vec.y));
}

__device__ float interpolate(float2 point, float2 target, float height) {
	return -magnitude(point - target) + height;
}

__device__ float interpolateAvoid(float2 point, float3 target) {
	return fminf(0.0, magnitude(make_float2(point.x - target.x, point.y - target.y)) - target.z);
}


__device__ float getHeight(float2 point, float2 target, float3* avoid, int avoidLength) {
	float height = interpolate(point, target, 1.0);
	for (int i = 0; i < avoidLength; i++) {
		height += interpolateAvoid(point, avoid[i]);
	}
	return height;
}

__device__ void printVector(const float2 vec) {
	printf("{%f, %f, %f}\n", vec.x, vec.y);
}

__global__ void test(const Point astart, const Point end, float3* results, float3 * blockResults, bool* reachedGoal, const int width, const int height, curandState* randomState, float randomRadius, float length, int points, int samples, float3 * avoidPoints, int avoidPointsLength) {
	Point start = astart;
	float2 endGoal = end.position;
	float2 randomVector;
	float2 bestPoint;
	float2 currentPoint;
	Point newPoint;
	float currentScore;
	float maxScore;
	bool hasReachedGoal = false;
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState localState = randomState[id];
	float threshold = 100.0;

	for (int i = 0; i < points; i++) {
		float2 center = start.position + start.direction * length;
		for (int j = 0; j < samples; j++) {
			randomVector = make_float2(curand_uniform(&localState) * 2.0 - 1.0, curand_uniform(&localState) * 2.0 - 1.0);
			currentPoint = center + randomVector * randomRadius;
			currentScore = getHeight(currentPoint, endGoal, avoidPoints, avoidPointsLength);
			if (maxScore < currentScore || j == 0) {
				maxScore = currentScore;
				bestPoint = currentPoint;
			}
		}
		results[id * points + i] = make_float3(bestPoint.x, bestPoint.y, 1.0);
		start.direction = normalize(bestPoint - start.position);
		start.position = bestPoint;	
		if (hasReachedGoal) {
			results[id * points + i] = make_float3(0.0, 0.0, 1.0);
		} else if (magnitude(endGoal - bestPoint) <= threshold) {
			hasReachedGoal = true;
		}
	}
	reachedGoal[id] = hasReachedGoal;
	randomState[id] = localState;
	
	__syncthreads();

	if (threadIdx.x != 0)
		return;
	int bestIndex = -1;
	for (int i = 0; i < blockDim.x; i++) {
		if (reachedGoal[i]) {
			bestIndex = i;
		}
	}
	if (bestIndex == -1) {
		bestIndex = 0;
	}
	for (int i = 0; i < points; i++) {
		blockResults[blockIdx.x * points + i] = results[points*bestIndex + i];
	}
}

__global__ void generateHeightmap(unsigned char* image, const int width, const int height, const int channels, const float3* points, const int pointsLength, const float slope, const Point start, const Point end) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float heightValue = 0.0;
	float2 coordinates = make_float2(id % width, id / width);
	float2 targetCoordinates;
	for (int i = 0; i < pointsLength; i++) {
		targetCoordinates = make_float2(points[i].x, points[i].y);
		if(targetCoordinates.x > 0.0 && targetCoordinates.y > 0.0)
			heightValue = fmaxf(heightValue, -slope * magnitude(coordinates - targetCoordinates) + points[i].z);
	}
	heightValue = fmaxf(heightValue, -slope * magnitude(coordinates - start.position) + start.height);
	heightValue = fmaxf(heightValue, -slope * magnitude(coordinates - end.position) + end.height);
	unsigned char white = (int)(heightValue * 255.0);
	image[id * 3] = white;
	image[id * 3 + 1] = white;
	image[id * 3 + 2] = white;
}

int generateTerrain(unsigned char* output, const int width, const int height, const int channels, const Point* Vecs, const float3 * avoidPoints, const int avoidPointsLength) {
	unsigned char value;
	float* floatOutput = (float*)malloc(sizeof(float) * width * height);
	curandState* devStates;
	int threadsPerBlock = 200;
	int blocks = 1;
	int totalThreads = blocks * threadsPerBlock;
	CUDA_CALL(cudaMalloc((void**)&devStates, totalThreads * sizeof(curandState)));

	setupKernel << <1, threadsPerBlock >> > (devStates);

	
	const float distance = sqrt(pow(Vecs[1].position.x - Vecs[0].position.x, 2) + pow(Vecs[1].position.y - Vecs[0].position.y, 2)) * 1.33;
	const int points = distance * POINT_DENSITY;
	const float randomRadius = RANDOM_RADIUS;
	const float length = distance/(float)points;
	
	//printf("distance: %f, points: %d, randomRadius: %f, length: %f\n", distance, points, randomRadius, length);
	const int samples = SAMPLES;

	float3* d_avoidPoints;

	CUDA_CALL(cudaMalloc((void**)&d_avoidPoints, avoidPointsLength * sizeof(float3)));
	CUDA_CALL(cudaMemcpy(d_avoidPoints, avoidPoints, avoidPointsLength * sizeof(float3), cudaMemcpyHostToDevice));

	float3* threadResults;
	float3* blockResults;
	float3* host_blockResults;
	bool * hasReachedGoal;
	host_blockResults = (float3*)malloc(blocks * points * sizeof(float3));
	CUDA_CALL(cudaMalloc((void**)&threadResults, totalThreads * points * sizeof(float3)));
	CUDA_CALL(cudaMalloc((void**)&hasReachedGoal, totalThreads * sizeof(bool)));
	CUDA_CALL(cudaMemset(threadResults, 0, totalThreads * points * sizeof(float3)));
	CUDA_CALL(cudaMalloc((void**)&blockResults, blocks * points * sizeof(float3))); 

	unsigned char* d_heightmap;
	CUDA_CALL(cudaMalloc((void**)&d_heightmap, width * height * channels * sizeof(unsigned char)));

	test<<<blocks, threadsPerBlock>>>(Vecs[0], Vecs[1], threadResults, blockResults, hasReachedGoal, width, height, devStates, randomRadius, length, points, samples, d_avoidPoints, avoidPointsLength);
	
	generateHeightmap<<<2000, 500>>>(d_heightmap, width, height, channels, blockResults, points, 0.01, Vecs[0], Vecs[1]);
	CUDA_CALL(cudaMemcpy(output, d_heightmap, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	free(floatOutput);
	free(host_blockResults);
	CUDA_CALL(cudaFree(hasReachedGoal));
	CUDA_CALL(cudaFree(threadResults));
	CUDA_CALL(cudaFree(blockResults));
	CUDA_CALL(cudaFree(devStates));
	CUDA_CALL(cudaFree(d_avoidPoints));
	CUDA_CALL(cudaFree(d_heightmap));
	return 0;
}

int main()
{
	int width, height, channels;
	width = 1000;
	height = 1000;
	channels = 3;
	unsigned char* output = (unsigned char*)malloc(sizeof(unsigned char) * width * height * channels);
	Point controlPoints[2] = {
		{100.0, 100.0, 1.0, 0.0, 1.0},
		{900, 900, -1.0, 0.0, 1.0}
	};
	const int avoidPointsLength = 2;
	float3* avoidPoints = (float3*)malloc(avoidPointsLength * sizeof(float3));
	if (avoidPointsLength > 0) {
		avoidPoints[0] = make_float3(200, 300, 200);
	}
	if (avoidPointsLength > 1) {
		avoidPoints[1] = make_float3(700, 600, 200);
	}
	generateTerrain(output, width, height, channels, controlPoints, avoidPoints, avoidPointsLength);
	stbi_write_png("output.png", width, height, channels, output, width * channels);
	free(output);
	free(avoidPoints);

	return 0;
}

