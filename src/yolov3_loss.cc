#include <iostream>
#include <string.h>
#include <map>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/util/tensor_format.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "yolov3_loss.h"
#include "yolov3.h"

using namespace std;
using namespace tensorflow;

namespace dzhang{

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

using EVecXf = Eigen::VectorXf;
template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;

REGISTER_OP("YoloLoss")
    .Input("feature: float")
	.Input("ground_truth: float")
//	.Output("loss: float")
    .Output("delta: float")
	.Attr("mask: list(int)")
	.Attr("anchors_num: int = 3")
	.Attr("anchors: list(int)")
	.Attr("classes: int = 2")
	.Attr("img_height: int")
	.Attr("img_width: int")
	.Attr("ignore_thresh: float = 0.5")
	.Attr("is_training: bool = true")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle dims;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &dims));
      ::tensorflow::shape_inference::DimensionHandle channels;
      channels = c->Dim(dims, 3);

      ::tensorflow::shape_inference::ShapeHandle dims_rois;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &dims_rois));
      ::tensorflow::shape_inference::DimensionHandle num_rois;
      num_rois = c->Dim(dims_rois, 0);
//      c->set_output(0, output_shape);
//      c->set_output(1, output_shape);
      return ::tensorflow::Status::OK();
    })
	.Doc(R"doc()doc");

// Functor used by FusedBatchNormOp to do the computations.
template <typename Device, typename T>
struct YoloLoss;




float YoloLossForward(
		int batch,
		int w,
		int h,
		int n,
		int total,
		std::vector<int32>& anchors,
		std::vector<int32>& mask,
		int classes,
		int img_height,
		int img_width,
		const float* feature_ptr,
		const float* ground_truth,
		float ignore_thresh_,
		float* l_output_tensor_ptr,
		float* delta_ptr,
		int num_element,
		bool is_training_) {


	layer l(false, batch, w, h, n, total, mask, classes, feature_ptr, ignore_thresh_, l_output_tensor_ptr, delta_ptr, anchors);
	network net(false, is_training_, img_height, img_width, ground_truth, num_element);
	cout<<is_training_<<endl;
	yolov3_loss_pre_action(l,net);
	float loss = yolov3_loss(l,net);
	cout<<"####"<<endl;
	cout<<loss<<endl;
	return loss;
}


template <typename T>
struct YoloLoss<CPUDevice, T> {
  void operator()(OpKernelContext* context,
		  	  	  int anchors_num_,
		 		  std::vector<int32> anchors_,
				  std::vector<int32> mask_,
		 		  int classes_,
		 		  int img_height_,
		 		  int img_width_,
		 		  float ignore_thresh_,
				  bool is_training_) {

	  cout<<"#run on CPU"<<endl;
	    // Grab the input tensor
	    const Tensor& feature_tensor = context->input(0);
	    const Tensor& ground_truth_tensor = context->input(1);

	    int total_ = anchors_.size() / 2;


	    //NCHW
	    int batch = static_cast<int>(feature_tensor.dim_size(0));
	    int input_chnannel = static_cast<int>(feature_tensor.dim_size(1));
	    int h = static_cast<int>(feature_tensor.dim_size(2));
	    int w = static_cast<int>(feature_tensor.dim_size(3));

	    auto feature_ptr = feature_tensor.template flat<float>().data();
	    auto ground_truth_ptr = ground_truth_tensor.template flat<float>().data();

		Tensor* delta_tensor = nullptr;
		OP_REQUIRES_OK(context,
					context->allocate_output(0, feature_tensor.shape(), &delta_tensor));

		auto delta_ptr = delta_tensor->template flat<float>().data();

		Tensor l_output_tensor = Tensor(tensorflow::DataType::DT_FLOAT,feature_tensor.shape());

		auto l_output_tensor_ptr = l_output_tensor.template flat<float>().data();

	    // Call the cpu kernel launcher
		float loss = YoloLossForward(batch, w, h, anchors_num_, total_, anchors_, mask_,
					classes_, img_height_, img_width_, feature_ptr, ground_truth_ptr, ignore_thresh_,
					l_output_tensor_ptr, delta_ptr, ground_truth_tensor.NumElements(), is_training_);

		cout<<"&&&&"<<endl;

  }

};


#if GPU

float YoloLossForwardGPU(
  		int batch,
  		int w,
  		int h,
  		int n,
  		int total,
  		std::vector<int32>& anchors,
		std::vector<int32> mask,
  		int classes,
  		int img_height,
  		int img_width,
  		const float* feature_ptr,
  		const float* ground_truth,
  		float ignore_thresh_,
		float* l_output_tensor_ptr,
  		float* delta_ptr,
		int num_element,
		bool is_training_,
  		const Eigen::GpuDevice& d) {
  	layer l(true, batch, w, h, n, total, mask, classes, feature_ptr, ignore_thresh_, l_output_tensor_ptr, delta_ptr, anchors);
  	network net(true, is_training_, img_height, img_width, ground_truth,num_element);
  	float loss = YoloLossLauncher(l,net,d);
  	return loss;
}

template <typename T>
struct YoloLoss<GPUDevice, T> {
  void operator()(OpKernelContext* context,
		  int anchors_num_,
		  std::vector<int32> anchors_,
		  std::vector<int32> mask_,
		  int classes_,
		  int img_height_,
		  int img_width_,
		  float ignore_thresh_,
		  bool is_training_) {
	    // Grab the input tensor
	    const Tensor& feature_tensor = context->input(0);
	    const Tensor& ground_truth_tensor = context->input(1);

	    int total_ = anchors_.size() / 2;

	    //NCHW
	    int batch = static_cast<int>(feature_tensor.dim_size(0));
	    int input_chnannel = static_cast<int>(feature_tensor.dim_size(1));
	    int h = static_cast<int>(feature_tensor.dim_size(2));
	    int w = static_cast<int>(feature_tensor.dim_size(3));

	    auto feature_ptr = feature_tensor.template flat<float>().data();
	    auto ground_truth_ptr = ground_truth_tensor.template flat<float>().data();

		Tensor* delta_tensor = nullptr;
		OP_REQUIRES_OK(context,
					context->allocate_output(0, feature_tensor.shape(), &delta_tensor));

		auto delta_ptr = delta_tensor->template flat<float>().data();


		Tensor l_output_tensor;

	     OP_REQUIRES_OK(context, context->allocate_temp(
	    		 	 	 	 	 	 	 tensorflow::DataType::DT_FLOAT,
										 feature_tensor.shape(),
										 &l_output_tensor));

		auto l_output_tensor_ptr = l_output_tensor.template flat<float>().data();

	    // Call the cpu kernel launcher
		float loss = YoloLossForwardGPU(batch, w, h, anchors_num_, total_, anchors_, mask_,
					classes_, img_height_, img_width_, feature_ptr, ground_truth_ptr, ignore_thresh_, l_output_tensor_ptr,
					delta_ptr, ground_truth_tensor.NumElements(), is_training_, context->eigen_device<Eigen::GpuDevice>());

	//
	//	Tensor* output = nullptr;
	//	OP_REQUIRES_OK(context,
	//				context->allocate_output(0, TensorShape({1,1}), &output));
	//
	//	auto output_ptr = output->template flat<float>().data();
	//	output_ptr[0] = loss;



	  }

};
#endif



template <typename Device, typename T>
class YoloLossOp : public OpKernel {
 public:
  explicit YoloLossOp(OpKernelConstruction* context) : OpKernel(context) {

      OP_REQUIRES_OK(context, context->GetAttr("anchors_num", &anchors_num_));
      OP_REQUIRES_OK(context, context->GetAttr("anchors", &anchors_));
      OP_REQUIRES_OK(context, context->GetAttr("mask", &mask_));
      OP_REQUIRES_OK(context, context->GetAttr("classes", &classes_));
      OP_REQUIRES_OK(context, context->GetAttr("img_height", &img_height_));
      OP_REQUIRES_OK(context, context->GetAttr("img_width", &img_width_));
      OP_REQUIRES_OK(context, context->GetAttr("ignore_thresh", &ignore_thresh_));
      OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));


  }
   void Compute(OpKernelContext* context) override {
//
	   YoloLoss<Device,T>()(context,
				  anchors_num_,
				  anchors_,
				  mask_,
				  classes_,
				  img_height_,
				  img_width_,
				  ignore_thresh_,
				  is_training_);

   }

 private:
  int anchors_num_;
  std::vector<int32> anchors_;
  std::vector<int32> mask_;
  int classes_;
  int img_height_;
  int img_width_;
  float ignore_thresh_;
  bool is_training_;


};




REGISTER_KERNEL_BUILDER(Name("YoloLoss").Device(DEVICE_CPU), YoloLossOp<CPUDevice,float>);

#if GPU
REGISTER_KERNEL_BUILDER(Name("YoloLoss").Device(DEVICE_GPU), YoloLossOp<GPUDevice,float>);
#endif

} // namespace dzhang


