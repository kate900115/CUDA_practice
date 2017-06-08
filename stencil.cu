#include<stdio.h>

__global__ void stencil(float* in, float* out, int N, int n, int BLOCKSIZE){
	if (blockIdx.x==0){
		__shared__ float shared_in[22];//BLOCKSIZE+2*n];
		int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
		shared_in[threadIdx.x+n] = in[globalIdx];
		if (threadIdx.x>(BLOCKSIZE-n-1)){
			shared_in[threadIdx.x+2*n] = in[globalIdx+n];
		}
		__syncthreads();
		int value = 0;
		if (threadIdx.x>n-1){
			for (int i=0; i<2*n+1; i++){
				value += shared_in[threadIdx.x+i];
			}
		}
		out[globalIdx] = value;
		
	}
	else if (blockIdx.x==(int(N/BLOCKSIZE)-1)){
		__shared__ float shared_in[22];
		int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
		shared_in[threadIdx.x+n] = in[globalIdx];
		if (threadIdx.x<n){
			shared_in[threadIdx.x] = in[globalIdx-n];
		}
		__syncthreads();
		int value = 0;
		if (threadIdx.x<BLOCKSIZE-n){
			for (int i=0; i<2*n+1; i++){
				value += shared_in[threadIdx.x+i];
			}
		}
		out[globalIdx] = value;
	}
	else{
		__shared__ float shared_in[22];
		int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
		shared_in[threadIdx.x+n] = in[globalIdx];
		if (threadIdx.x<n){
			shared_in[threadIdx.x] = in[globalIdx-n];
		}
		if (threadIdx.x>BLOCKSIZE-n-1){
			shared_in[threadIdx.x+2*n] = in[globalIdx+n];
		}
		__syncthreads();
		int value = 0;
		for (int i=0; i<2*n+1; i++){
			value += shared_in[threadIdx.x+i];
		}
		out[globalIdx] = value;
	}
}


int main(){
	float* h_a = NULL;
	float* h_b = NULL;
	float* d_a = NULL;
	float* d_b = NULL;

	int BLOCKSIZE = 16;
	int N = 512;
	int n = 3;

	h_a = (float*)malloc(N*sizeof(float));
	h_b = (float*)malloc(N*sizeof(float));
	cudaMalloc((void**)&d_a, N*sizeof(float));
	cudaMalloc((void**)&d_b, N*sizeof(float));

	if ((h_a==NULL)||(d_a==NULL)||(h_b==NULL)&&(d_b==NULL)){
		printf("Cannot allocate memory.\n");
	}

	memset(h_b, 0, N*sizeof(float));
	for (int i=0; i<N; i++){
		h_a[i]=i;
	}

	cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);
	
	stencil<<<(N+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>> (d_a, d_b, N, n, BLOCKSIZE);

	cudaMemcpy(h_b, d_b, N*sizeof(float), cudaMemcpyDeviceToHost);	
	for (int i=0; i<N; i++){
		printf("A[%d]=%f\n",i,h_b[i]);
	}
	return 0;

}
