// input: in_data (b,n,c), in_grid (b,n)
// output: out_data (b,g,c), out_pooling_mask (b,g,c)
__global__ void grid_pooling_gpu(int b,int n,int c,int g,const float * in_data,const int * in_grid,float * out_data,int * out_pooling_mask){
    
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < b*c; index += stride)
    {

        int index_batch = index / c;
        int index_channel = index % c;

        // initialization : really IMPORTANT
        /*for (int index_grid = 0; index_grid < g; index_grid++)
        {
            int index_max_pooling_mask = index_batch*g*c + index_grid*c + index_channel;
            out_pooling_mask[index_max_pooling_mask] = -1;
            out_data[index_max_pooling_mask] = .0;
        }*/

        for (int index_point = 0; index_point < n; index_point++)
        {
            int index_grid = in_grid[index_batch * n + index_point]; // in_grid[b,n]
        
            int index_input = index_batch*n*c + index_point*c + index_channel; // in_data[b,n,c]
        
            int index_output = index_batch*g*c + index_grid*c + index_channel; // out_data[b,g,c]
        
            if (out_data[index_output] < in_data[index_input])
            {
                out_data[index_output] = in_data[index_input];
                out_pooling_mask[index_output] = index_input;
            }
        }
    }
}


__global__ void grid_pooling_grad_gpu(int b,int n,int c,int g,const int * pooling_mask,const float * grad_out,float * out){
    
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < b*c; index += stride)
    {
        int index_batch = index / c;
        int index_channel = index % c;

        /*for (int index_point = 0; index_point < n; index_point++)
        {
            int index_grad_output = index_batch*n*c + index_point*c + index_channel;
            out[index_grad_output] = .0;
        }*/

        for (int index_grid = 0; index_grid < g; index_grid++)
        {
            int index_grad_output = index_batch*g*c + index_grid*c + index_channel;
            int index_max_point = pooling_mask[index_grad_output];

            if (index_max_point == -1) continue;
            //if (index_max_point >= b*g*c) continue;

            out[index_max_point] += 1 * grad_out[index_grad_output];
        }
    }
}


void gridpoolingLauncher(int b,int n,int c,int g,const float * in_data,const int * in_grid,float * out_data,int * out_pooling_mask){
    const int threads_per_block = 128;
    //const int number_of_blocks = (b*c + threads_per_block - 1)/threads_per_block;
    const int number_of_blocks = 8;

    //Fills the first count bytes of the memory area pointed to by devPtr with the constant byte value value.
    cudaMemset(out_data,0,b*g*c*4);
    cudaMemset(out_pooling_mask,-1,b*g*c*4);

    grid_pooling_gpu<<<number_of_blocks, threads_per_block>>>(b,n,c,g,in_data,in_grid,out_data,out_pooling_mask);
    //grid_pooling_gpu<<<1, 1>>>(b,n,c,g,inp_data,grid_index,out,grid_pooling_mask);
    //cudaDeviceSynchronize();
}


void gridpoolinggradLauncher(int b,int n,int c,int g,const int * pooling_mask,const float * grad_out,float * out){
    const int threads_per_block = 128;
    //const int number_of_blocks = (b*c + threads_per_block - 1)/threads_per_block;
    const int number_of_blocks = 8;

    cudaMemset(out,0,b*n*c*4);

    grid_pooling_grad_gpu<<<number_of_blocks, threads_per_block>>>(b,n,c,g,pooling_mask,grad_out,out);
    //grid_pooling_grad_gpu<<<number_of_blocks, threads_per_block>>>(b,n,c,g,pooling_mask,grad_out,out);
    //cudaDeviceSynchronize();
}
