#if GPU
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "yolov3_loss.h"
#include "yolov3.h"
#include <stdio.h>
#include <cfloat>
#include <iostream>
//#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "common_gpu.h"

using namespace std;

namespace dzhang{

__device__ float logistic_gradient_kernel(float x)
{
	return (1-x)*x;
}

__global__ void activate_array_kernel(const float *x, float *y, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) y[i] = logistic_gradient_kernel(x[i]);
}

extern "C" void activate_array_gpu(const float *x, float *y, int n)
{
    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, y, n);
    check_error(cudaPeekAtLastError());
}


float YoloLossLauncher(layer& l, network& net, const Eigen::GpuDevice& d)
{

    
    cuda_pull_array((float*)l.input, l.output, l.batch*l.inputs);
    
    cuda_pull_array(net.truth_gpu, (float*)net.truth, l.batch*l.max_boxes*(4+1));
    
    float loss = 0.0;
    yolov3_loss_pre_action(l,net);
   
    loss = yolov3_loss(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    
    return loss;

}


} // namespace dzhang

#endif
