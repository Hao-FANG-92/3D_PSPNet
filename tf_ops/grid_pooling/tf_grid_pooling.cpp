/* Grid pooling of points
 * Original author: Hao Fang
 * All Rights Reserved. 2018. 
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <iostream>
#include <cuda_runtime.h>
#include <limits>

using namespace tensorflow;

REGISTER_OP("GridPooling")
  .Attr("num_grids: int")
  .Input("in_data: float32")
  .Input("in_grid: int32")
  .Output("out_data: float32")
  .Output("out_pooling_mask: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    int num_grids;
    TF_RETURN_IF_ERROR(c->GetAttr("num_grids", &num_grids));
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * num_points * channels
    c->WithRank(c->input(0), 3, &dims1);
    ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * num_points
    c->WithRank(c->input(1), 2, &dims2);
    
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims2, 0), num_grids, c->Dim(dims1, 2)}); // batch_size * num_grids * channels
    c->set_output(0, output);
    c->set_output(1, output);

    return Status::OK();
  });


REGISTER_OP("GridPoolingGrad")
  .Input("in_data: float32")
  .Input("pooling_mask: int32")
  .Input("grad_data: float32")
  .Output("grad_points: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });


void gridpoolingLauncher(int b,int n,int c,int g,const float * inp_data,const int * grid_index,float * out,int * grid_pooling_mask);
class GridPoolingGpuOp: public OpKernel{
  public:
    explicit GridPoolingGpuOp(OpKernelConstruction* context):OpKernel(context){
        OP_REQUIRES_OK(context, context->GetAttr("num_grids", &g));
        OP_REQUIRES(context, g > 0, errors::InvalidArgument("Grid pooling layer expects positive number of grids"));
    }
    void Compute(OpKernelContext * context)override{
      const Tensor& in_data_tensor=context->input(0);
      const Tensor& in_grid_tensor=context->input(1);
      auto in_data_flat=in_data_tensor.flat<float>();
      auto in_grid_flat=in_grid_tensor.flat<int>();
      const float * in_data=&(in_data_flat(0));
      const int * in_grid=&(in_grid_flat(0));

      int b=in_data_tensor.shape().dim_size(0);
      int n=in_data_tensor.shape().dim_size(1);
      int c=in_data_tensor.shape().dim_size(2);


      OP_REQUIRES(context,in_data_tensor.dims()==3,errors::InvalidArgument("Grid pooling layer expects (batch_size,num_points,channels) in_data shape"));
      

      OP_REQUIRES(context,in_grid_tensor.dims()==2 && in_grid_tensor.shape().dim_size(0)==b && in_grid_tensor.shape().dim_size(1)==n,errors::InvalidArgument("ProbSample expects (batch_size,num_points) in_grid shape"));


      Tensor * out_data_tensor=nullptr;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,g,c},&out_data_tensor)); // std::numeric_limits<float>::min()
      

      Tensor * out_pooling_mask_tensor=nullptr;

      OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,g,c},&out_pooling_mask_tensor));


      auto out_data_flat=out_data_tensor->flat<float>();
      float * out_data=&(out_data_flat(0));

      auto out_pooling_mask_flat=out_pooling_mask_tensor->flat<int>();
      int * out_pooling_mask=&(out_pooling_mask_flat(0));

      gridpoolingLauncher(b,n,c,g,in_data,in_grid,out_data,out_pooling_mask);
    }
   private:
    int g;
};
REGISTER_KERNEL_BUILDER(Name("GridPooling").Device(DEVICE_GPU), GridPoolingGpuOp);

void gridpoolinggradLauncher(int b,int n,int c,int g,const int * pooling_mask,const float * grad_out,float * out);
class GridPoolingGradGpuOp: public OpKernel{
  public:
    explicit GridPoolingGradGpuOp(OpKernelConstruction* context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& in_data_tensor=context->input(0);
      const Tensor& pooling_mask_tensor=context->input(1);
      const Tensor& grad_out_tensor=context->input(2);
      auto pooling_mask_flat=pooling_mask_tensor.flat<int>();
      auto grad_out_flat=grad_out_tensor.flat<float>();
      const int * pooling_mask=&(pooling_mask_flat(0)); 
      const float * grad_out=&(grad_out_flat(0));

      OP_REQUIRES(context,grad_out_tensor.dims()==3,errors::InvalidArgument("Grid pooling gradient layer expects (batch_size,num_grids,channels) in_data shape"));
      int b=pooling_mask_tensor.shape().dim_size(0);
      int g=pooling_mask_tensor.shape().dim_size(1);
      int c=pooling_mask_tensor.shape().dim_size(2);
      int n=in_data_tensor.shape().dim_size(1);

      Tensor * out_data_tensor=nullptr;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,c},&out_data_tensor));
      auto out_data_flat=out_data_tensor->flat<float>();
      float * out_data=&(out_data_flat(0));

      gridpoolinggradLauncher(b,n,c,g,pooling_mask,grad_out,out_data);
    }
};
REGISTER_KERNEL_BUILDER(Name("GridPoolingGrad").Device(DEVICE_GPU), GridPoolingGradGpuOp);
