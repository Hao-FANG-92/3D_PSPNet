// input: in_data (b,g,c), in_grid (b,n)
// output: out_data (b,n,c)
__global__ void grid_upsampling_gpu(int b,int n,int c,int g,const float * in_data,const int * in_grid,float * out_data){
    
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < c*n; index += stride)
    {

        int index_channel = index / n;
        int index_point = index % n;

        for (int index_batch = 0; index_batch < b; index_batch++)
        {
            int index_grid = in_grid[index_batch*n + index_point]; // in_grid[b, n]
            int index_in_data = index_batch*g*c + index_grid*c + index_channel; // in_data[b,g,c]
            int index_out_data = index_batch*n*c + index_point*c + index_channel; // out_data[b,n,c]

            out_data[index_out_data] = in_data[index_in_data];
        }

    }
}

__global__ void grid_upsampling_grad_gpu(int b,int n,int c,int g,const int * in_grid,const float * grad_out,float * grad_in){
    
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < b*c; index += stride)
    {
        int index_batch = index / c;
        int index_channel = index % c;

        /*for (int index_grid = 0; index_grid < g; index_grid++)
        {
            int index_grad_in = index_batch*g*c + index_grid*c + index_channel;
            grad_in[index_grad_in] = .0;
        }*/

        for (int index_point = 0; index_point < n; index_point++)
        {
            int index_grid = in_grid[index_batch*n + index_point]; // in_grid[b, n]
            int index_grad_out = index_batch*n*c + index_point*c + index_channel; // grad_out[b,n,c]
            int index_grad_in = index_batch*g*c + index_grid*c + index_channel; // grad_in[b,g,c]

            grad_in[index_grad_in] += 1 * grad_out[index_grad_out];
        }
    }
}

void gridupsamplingLauncher(int b,int n,int c,int g,const float * in_data,const int * in_grid,float * out_data){
    const int threads_per_block = 128;
    //const int number_of_blocks = (c*n + threads_per_block - 1)/threads_per_block;
    const int number_of_blocks = 8;

    cudaMemset(out_data,0,b*n*c*4);

    grid_upsampling_gpu<<<number_of_blocks, threads_per_block>>>(b,n,c,g,in_data,in_grid,out_data);
    //grid_upsampling_gpu<<<1, 1>>>(b,n,c,g,inp_data,grid_index,out);
    //cudaDeviceSynchronize();
}

void gridupsamplinggradLauncher(int b,int n,int c,int g,const int * in_grid,const float * grad_out,float * grad_in){
    const int threads_per_block = 128;
    //const int number_of_blocks = (b*c + threads_per_block - 1)/threads_per_block;
    const int number_of_blocks = 8;

    cudaMemset(grad_in,0,b*g*c*4);

    grid_upsampling_grad_gpu<<<number_of_blocks, threads_per_block>>>(b,n,c,g,in_grid,grad_out,grad_in);
    //grid_upsampling_grad_gpu<<<1, 1>>>(b,n,c,g,in_grid,grad_out,grad_in);
    //cudaDeviceSynchronize();
}
