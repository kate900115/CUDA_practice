#include<stdio.h>

__global__ void stencil(float* in, float* out, int N, int M, int n, int BLOCKSIZE){
	__shared__ float shared_in[22][22];//BLOCKSIZE+2*n;
	int globalIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int globalIdx_y = blockIdx.y * blockDim.y + threadIdx.y;
	shared_in[threadIdx.y+n][threadIdx.x+n] = in[globalIdx_y*M + globalIdx_x];
	if (blockIdx.x>0){	
		if (threadIdx.x<n){
			shared_in[threadIdx.y+n][threadIdx.x] = in[(globalIdx_y)*M+globalIdx_x-n];
		}
	}
	if (blockIdx.y>0){	
		if (threadIdx.y<n){
			shared_in[threadIdx.y][threadIdx.x+n] = in[(globalIdx_y-n)*M+globalIdx_x];
		}
	}
	if (blockIdx.x<(int(M/BLOCKSIZE)-1)){
		if (threadIdx.x>BLOCKSIZE-n-1){	
			shared_in[threadIdx.y+n][threadIdx.x+2*n] = in[(globalIdx_y)*M+globalIdx_x+n];
		}
	}
	if (blockIdx.y<(int(N/BLOCKSIZE)-1)){
		if (threadIdx.y>BLOCKSIZE-n-1){
			shared_in[threadIdx.y+2*n][threadIdx.y+n] = in[(globalIdx_y+n)*M+globalIdx_x];
		}
	}
	__syncthreads();
	int value = 0;
	if ((globalIdx_x>=n)&&(globalIdx_x<M-n)&&(globalIdx_y>=n)&&(globalIdx_y<N-n))
	{
		for (int i=0; i<2*n+1; i++){
			value += shared_in[threadIdx.y+i][threadIdx.x+n];
			value += shared_in[threadIdx.y+n][threadIdx.x+i];
		}
		value = value - shared_in[threadIdx.y+n][threadIdx.x+n];	
	}
out[globalIdx_y*M + globalIdx_x] = value;
//	printf("out[%d][%d]=%f\n",globalIdx_y, globalIdx_x, out[globalIdx_y*M+globalIdx_x]);
}


int main(){
	float* h_a = NULL;
	float* h_b = NULL;
	float* d_a = NULL;
	float* d_b = NULL;

	int BLOCKSIZE = 16;
	int N = 512;
	int M = 256;
	int n = 3;

	h_a = (float*)malloc(M*N*sizeof(float));
	h_b = (float*)malloc(M*N*sizeof(float));
	cudaMalloc((void**)&d_a, M*N*sizeof(float));
	cudaMalloc((void**)&d_b, M*N*sizeof(float));

	if ((h_a==NULL)||(d_a==NULL)||(h_b==NULL)&&(d_b==NULL)){
		printf("Cannot allocate memory.\n");
	}

	memset(h_b, 0, M*N*sizeof(float));
	for (int i=0; i<N; i++){
		for (int j=0; j<M; j++){
			h_a[i*M+j]=i+j;
		}
	}

	cudaMemcpy(d_a, h_a, M*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, M*N*sizeof(float), cudaMemcpyHostToDevice);
	dim3 grid((M+BLOCKSIZE-1)/BLOCKSIZE, (N+BLOCKSIZE-1)/BLOCKSIZE, 1);
	dim3 block(BLOCKSIZE,BLOCKSIZE,1);	
	stencil<<<grid, block>>> (d_a, d_b, N, M, n, BLOCKSIZE);

	cudaMemcpy(h_b, d_b, M*N*sizeof(float), cudaMemcpyDeviceToHost);	
	for (int i=0; i<N; i++){
		for (int j=0; j<M; j++){
			printf("A[%d][%d]=%f\n",i,j,h_b[i*M+j]);
		}
	}
	return 0;

}
