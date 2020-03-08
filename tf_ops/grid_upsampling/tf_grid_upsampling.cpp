/* Grid pooling of points
 * Original author: Hao Fang
 * All Rights Reserved. 2018. 
 */
#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
#include <limits>

using namespace tensorflow;

REGISTER_OP("GridUpsampling")
  .Input("in_data: float32")
  .Input("in_grid: int32")
  .Output("out_data: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size, num_grids, channels
    c->WithRank(c->input(0), 3, &dims1);
    ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * num_points
    c->WithRank(c->input(1), 2, &dims2);
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims2, 1), c->Dim(dims1, 2)}); // batch_size * num_points * channels
    c->set_output(0, output);
    return Status::OK();
  });


REGISTER_OP("GridUpsamplingGrad")
  .Input("in_data: float32")
  .Input("in_grid: int32")
  .Input("grad_out: float32")
  .Output("grad_in: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });


void gridupsamplingLauncher(int b,int n,int c,int g,const float * inp_data,const int * grid_index,float * out);
class GridUpsamplingGpuOp: public OpKernel{
  public:
    explicit GridUpsamplingGpuOp(OpKernelConstruction* context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& in_data_tensor=context->input(0);
      auto in_data_flat=in_data_tensor.flat<float>();
      const float * in_data=&(in_data_flat(0));
      OP_REQUIRES(context,in_data_tensor.dims()==3,errors::InvalidArgument("Grid pooling layer expects (batch_size,num_points,channels) in_data shape"));

      int b=in_data_tensor.shape().dim_size(0);
      int g=in_data_tensor.shape().dim_size(1);
      int c=in_data_tensor.shape().dim_size(2);

      const Tensor& in_grid_tensor=context->input(1);
      auto in_grid_flat=in_grid_tensor.flat<int>();
      const int * in_grid=&(in_grid_flat(0));
      int n=in_grid_tensor.shape().dim_size(1);
      OP_REQUIRES(context,in_grid_tensor.dims()==2 && in_grid_tensor.shape().dim_size(0)==b && in_grid_tensor.shape().dim_size(1)==n,errors::InvalidArgument("ProbSample expects (batch_size,num_points) in_grid shape"));

      //std::cout << b << " " << g << " " << n << " " << c << std::endl;

      Tensor * out_data_tensor=nullptr;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,c},&out_data_tensor)); // std::numeric_limits<float>::min()
      auto out_data_flat=out_data_tensor->flat<float>();
      float * out_data=&(out_data_flat(0));

      gridupsamplingLauncher(b,n,c,g,in_data,in_grid,out_data);
    }
};
REGISTER_KERNEL_BUILDER(Name("GridUpsampling").Device(DEVICE_GPU), GridUpsamplingGpuOp);

void gridupsamplinggradLauncher(int b,int n,int c,int g,const int * in_grid,const float * grad_out,float * grad_in);
class GridUpsamplingGradGpuOp: public OpKernel{
  public:
    explicit GridUpsamplingGradGpuOp(OpKernelConstruction* context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& in_data_tensor=context->input(0);
      auto in_data_flat=in_data_tensor.flat<float>();
      const float * in_data=&(in_data_flat(0));
      OP_REQUIRES(context,in_data_tensor.dims()==3,errors::InvalidArgument("Grid pooling layer expects (batch_size,num_points,channels) in_data shape"));

      int b=in_data_tensor.shape().dim_size(0);
      int g=in_data_tensor.shape().dim_size(1);
      int c=in_data_tensor.shape().dim_size(2);

      const Tensor& in_grid_tensor=context->input(1);
      auto in_grid_flat=in_grid_tensor.flat<int>();
      const int * in_grid=&(in_grid_flat(0));
      int n=in_grid_tensor.shape().dim_size(1);
      OP_REQUIRES(context,in_grid_tensor.dims()==2 && in_grid_tensor.shape().dim_size(0)==b && in_grid_tensor.shape().dim_size(1)==n,errors::InvalidArgument("ProbSample expects (batch_size,num_points) in_grid shape"));

      const Tensor& grad_out_tensor=context->input(2);
      auto grad_out_flat=grad_out_tensor.flat<float>();
      const float * grad_out=&(grad_out_flat(0));
      OP_REQUIRES(context,grad_out_tensor.dims()==3,errors::InvalidArgument("Grid pooling gradient layer expects (batch_size,num_grids,channels) in_data shape"));

      Tensor * out_data_tensor=nullptr;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,g,c},&out_data_tensor));
      auto out_data_flat=out_data_tensor->flat<float>();
      float * out_data=&(out_data_flat(0));


      //for (int i = 0; i < b*n*c; i++)
        //std::cout << grad_out[0] << " ";
      //{std::cout << std::endl;}

      gridupsamplinggradLauncher(b,n,c,g,in_grid,grad_out,out_data);
    }
};
REGISTER_KERNEL_BUILDER(Name("GridUpsamplingGrad").Device(DEVICE_GPU), GridUpsamplingGradGpuOp);
