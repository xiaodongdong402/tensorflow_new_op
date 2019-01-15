#ifndef __YOLOV3_H__
#define __YOLOV3_H__
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "math.h"
#include <vector>

//namespace dzhang
//{

class box {
public:
	float x, y, w, h;
};

class layer {
public:
	layer();
	layer(bool is_gpu_, int batch_, int w_, int h_, int n_, int total_, std::vector<int>& mask_, int classes_, const float* feature_ptr, float ignore_thresh_, float* l_output_tensor_ptr_, float* delta_ptr, std::vector<int>& anchor);
	~layer();


public:
	bool is_gpu;
	int batch = 0; // batch size
	int h = 0; // feature map height
	int w = 0; // feature map width
	int n = 0; //anchors num
	int c = 0; //output channel num #default n*(classes + 4 + 1)
	int outputs; //output feature map size #default  h*w*n*(classes + 4 + 1)
	int inputs;
	int classes; //class num
	int max_boxes; // max box num #default 90
	float ignore_thresh; //ignore predict box thresh #default 0.5
	float truth_thresh; //predict truth thresh #default 1.0
	int total; //all scales anchors num
	int truths; //default 90*(4 + 1)
	float object_scale;
	float noobject_scale;

	float* output;
	const float * input;
	int *mask;
	float * biases;
	float * bias_updates;
//	float * truth;
	float * delta;
	int * map; //default 0

	#if GPU
		float* output_gpu;
		float* delta_gpu;
	#endif

};

class network {
public:
	network();
	network(bool is_gpu_, bool is_training, int h, int w,const float* truth, int num_element);
	~network();
public:
	int h; // image height
	int w; // image weight
	const float *truth; //ground truth
	bool is_gpu;
	bool is_training;
#if GPU
	float* truth_gpu;
#endif
};


class detection{
//public:
//	detection(){}
//	~detection(){}
public:
    box bbox;
    int classes;
    float *prob;
//    float *mask;
    float objectness;
    int sort_class;
};

box get_yolo_box(const float *x, float *biases, int n, int index, int i, int j,
		int lw, int lh, int w, int h, int stride);

int entry_index(layer&  l, int batch, int location, int entry);

box float_to_box(float *f, int stride);

float overlap(float x1, float w1, float x2, float w2);

float box_intersection(box a, box b);

float box_union(box a, box b);

float box_iou(box a, box b);

float delta_yolo_box(box truth, const float *x, float *biases, int n, int index,
		int i, int j, int lw, int lh, int w, int h, float *delta, float scale,
		int stride);

int int_index(int *a, int val, int n);

float mag_array(float *a, int n);

void delta_yolo_class(const float *output, float *delta, int index, int classs,
		int classes, int stride, float *avg_cat);

void make_yolo_layer(layer& l, int batch, int w, int h, int n, int total, int *mask, int classes, const float* feature_ptr, float ignore_thresh);

void deleta_layer_memory(layer& l);

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);

void yolov3_loss_pre_action(layer& l, network& net);

float yolov3_loss(layer& l, network& net);

//} // namespace dzhang

#endif //__YOLOV3_H__

