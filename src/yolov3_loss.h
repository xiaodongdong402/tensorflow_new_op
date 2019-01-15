//#if !GOOGLE_CUDA
//#error This file must only be included when building with Cuda support
//#endif
#ifndef TENSORFLOW_USER_OPS_ROI_ALIGN_OP_GPU_H_
#define TENSORFLOW_USER_OPS_ROI_ALIGN_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

#include "yolov3.h"


namespace dzhang{

//namespace tensorflow {

float YoloLossLauncher(layer& l, network& net, const Eigen::GpuDevice& d);

//bool PSAlignPoolBackwardLauncher(const float* top_diff,
//    const int* mapping_channel, const int* argmax_position,
//    const int num_rois, const float spatial_scale, const int channels,
//    const int height, const int width,
//    const int pooled_height, const int pooled_width,
//    const int sample_height, const int sample_width,
//    const int output_dim,
//    float* bottom_diff, const float* bottom_rois, const Eigen::GpuDevice& d);

//}  // namespace tensorflow


} // namespace dzhang

#endif  // TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_GPU_H_
