#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "yolov3.h"
#include "math.h"
#include "common_gpu.h"
#include <iostream>

using namespace std;

//namespace dzhang
//{

box get_yolo_box_(const float *x, float *biases, int n, int index, int i, int j,
		int lw, int lh, int w, int h, int stride) {
	box b;
	b.x = (i + x[index + 0 * stride]) / lw;
	b.y = (j + x[index + 1 * stride]) / lh;
	b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
	b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
//	if(b.w > 1.0) b.w = 1.0;
//	if(b.h > 1.0) b.h = 1.0;
	return b;
}


box get_yolo_box(const float *x, float *biases, int n, int index, int i, int j,
		int lw, int lh, int w, int h, int stride) {
	box b;
	b.x = (i + (x[index + 0 * stride])) / lw;
	b.y = (j + (x[index + 1 * stride])) / lh;
	b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
	b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
//	if(b.w > 1.0) b.w = 1.0;
//	if(b.h > 1.0) b.h = 1.0;
	return b;
}

int entry_index(layer& l, int batch, int location, int entry) {
	int n = location / (l.w * l.h);
	int loc = location % (l.w * l.h);
	int res =  batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1)
					+ entry * l.w * l.h + loc;
	return res;
}

box float_to_box(const float *f, int stride) {
	box b = { 0 };
	b.x = f[0];
	b.y = f[1 * stride];
	b.w = f[2 * stride];
	b.h = f[3 * stride];
	return b;
}

float overlap(float x1, float w1, float x2, float w2) {
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float box_intersection(box a, box b) {
	float w = overlap(a.x, a.w, b.x, b.w);
	float h = overlap(a.y, a.h, b.y, b.h);
	if (w < 0 || h < 0)
		return 0;
	float area = w * h;
	return area;
}

float box_union(box a, box b) {
	float i = box_intersection(a, b);
	float u = a.w * a.h + b.w * b.h - i;
	return u;
}

float box_iou(box a, box b) {
	return box_intersection(a, b) / box_union(a, b);
}

float delta_yolo_box(box truth, const float *x, float *biases, int n, int index,
		int i, int j, int lw, int lh, int w, int h, float *delta, float scale,
		int stride) {
	box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
	float iou = box_iou(pred, truth);

	float tx = (truth.x * lw - i);
	float ty = (truth.y * lh - j);
	float tw = log(truth.w * w / biases[2 * n]);
	float th = log(truth.h * h / biases[2 * n + 1]);

	if(tw < -100.0)
	{
		tw = -1.0;
	}

	if(th < -100.0)
	{
		th = -1.0;
	}
//	delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
//	delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
//	delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
//	delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);

	delta[index + 0 * stride] = scale * (x[index + 0 * stride] - tx);
	delta[index + 1 * stride] = scale * (x[index + 1 * stride] - ty);
	delta[index + 2 * stride] = scale * (x[index + 2 * stride] - tw);
	delta[index + 3 * stride] = scale * (x[index + 3 * stride] - th);

//	delta[index + 0 * stride] = (1.0 - delta[index + 0 * stride]) * delta[index + 0 * stride];
//	delta[index + 1 * stride] = (1.0 - delta[index + 1 * stride]) * delta[index + 1 * stride];



	return iou;
}

int int_index(int *a, int val, int n) {
	int i;
	for (i = 0; i < n; ++i) {
		if (a[i] == val)
			return i;
	}
	return -1;
}

float mag_array(float *a, int n) {

	int i;
	float sum = 0;
	for (i = 0; i < n; ++i) {
		sum += a[i] * a[i];
	}
	return sqrt(sum);
}

void delta_yolo_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat, int focal_loss)
{
    int n;
    if (delta[index + stride*class_id]){
        delta[index + stride*class_id] = output[index + stride*class_id] - 1;
        if(avg_cat) *avg_cat += output[index + stride*class_id];
        return;
    }
    // Focal loss
    if (focal_loss) {
        // Focal Loss
        float alpha = 0.5;    // 0.25 or 0.5
        //float gamma = 2;    // hardcoded in many places of the grad-formula

        int ti = index + stride*class_id;
        float pt = output[ti] + 0.000000000000001F;
        // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
        float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
        //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = (((n == class_id) ? 1 : 0) - output[index + stride*n]);

            delta[index + stride*n] *= alpha*grad;

            if (n == class_id) *avg_cat += output[index + stride*n];
        }
    }
    else {
        // default
        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = output[index + stride*n] - ((n == class_id) ? 1 : 0);
            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
}

void delta_yolo_class(const float *output, float *delta, int index, int classs,
		int classes, int stride, float *avg_cat) {

	int n;
	if (delta[index]) {
		delta[index + stride * classs] = 1 - output[index + stride * classs];
		if (avg_cat)
			*avg_cat += output[index + stride * classs];
		return;
	}
	for (n = 0; n < classes; ++n) {
		delta[index + stride * n] = ((n == classs) ? 1 : 0)
				- output[index + stride * n];
		if (n == classs && avg_cat)
			*avg_cat += output[index + stride * n];
	}
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}


layer::layer(bool is_gpu_, int batch_, int w_, int h_, int n_, int total_, std::vector<int>& mask_, int classes_, const float* feature_ptr, float ignore_thresh_,float* l_output_tensor_ptr_, float* delta_ptr, std::vector<int>& anchor_)
{
	    int i;
	    n = n_;
	    total = total_;
	    batch = batch_;
	    is_gpu = is_gpu_;
	    h = h_;
	    w = w_;
	    c = n*(classes_ + 4 + 1);
	    classes = classes_;

		max_boxes = 90;
		ignore_thresh = ignore_thresh_;
		ignore_thresh = 0.7;
		truth_thresh = 1.0;

	    outputs = h*w*n*(classes + 4 + 1);
	    inputs = outputs;
	    truths = max_boxes*(4 + 1);


		map = (int*) calloc(classes, sizeof(int));
		biases = (float*) calloc(total * 2, sizeof(float));
		bias_updates = (float*) calloc(n * 2, sizeof(float));
//		output = (float*) calloc(batch * outputs, sizeof(float));
	    //		output = new float[batch * outputs];
	    //		map = new int[classes];
	    //		biases = new float[total * 2];
	    //		bias_updates = new float[n * 2];

		if (is_gpu)
		{
			delta = (float*) calloc(batch * outputs, sizeof(float));
			output = (float*) calloc(batch * outputs, sizeof(float));
#if GPU
			output_gpu = l_output_tensor_ptr_;
			delta_gpu = delta_ptr;
#endif
		}
		else
		{
			delta = delta_ptr;
			output = l_output_tensor_ptr_;
		}



	    for(int j = 0; j < classes; j++)
	    {
	    	map[j] = j;
	    }

	    mask = &mask_[0];


//	    float anchor_[] = {10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326};
	    for(i = 0; i < total*2; ++i){
	        biases[i] = anchor_[i];
	    }
	    input = feature_ptr;


//		#if GPU
//			output_gpu = dzhang::cuda_make_array(output, batch * outputs);
//			delta_gpu = dzhang::cuda_make_array(delta, batch * outputs);
//		#endif


}

layer::~layer()
{

	free(biases);
	free(bias_updates);
	free(map);
	if (is_gpu)
	{
#if GPU
		free(output);
		free(delta);
#endif
	}

//	free(output);
//	free(delta);

//	delete [] biases;
//	delete [] bias_updates;
//	delete [] map;
//	delete [] output;

}

network::network(bool is_gpu_, bool is_training_, int img_h, int img_w,const float* truth_, int num_element)
{
	h = img_h;
	w =img_w;

	is_gpu = is_gpu_;
	is_training = is_training_;

	if (is_gpu)
	{
		truth = (float*) calloc(num_element, sizeof(float));
#if GPU
		truth_gpu = (float*)truth_;
#endif
	}
	else
	{
		truth = truth_;
	}

}

network::~network()
{

	if (is_gpu)
	{
#if GPU
		free((float*)truth);
#endif
	}

}



float logistic_activate(float x) {
	return 1. / (1. + exp(-x));
}


void activate_array(const float *x, float* y, const int n)
{
    int i;
    for(i = 0; i < n; ++i){
        y[i] = logistic_activate(x[i]);
    }
}


void yolov3_loss_pre_action(layer& l, network& net)
{
	int b,n;

	if(l.is_gpu)
	{
		for (b = 0; b < l.batch; ++b){
			for(n = 0; n < l.n; ++n){
				int index = entry_index(l, b, n*l.w*l.h, 0);
				activate_array(l.output + index, l.output + index, 2*l.w*l.h);
				index = entry_index(l, b, n*l.w*l.h, 4);
				activate_array(l.output + index, l.output + index, (1+l.classes)*l.w*l.h);
			}
		}
		return;

	}


	memcpy(l.output, l.input, l.outputs*l.batch*sizeof(float));

	for (b = 0; b < l.batch; ++b){
		for(n = 0; n < l.n; ++n){
			int index = entry_index(l, b, n*l.w*l.h, 0);
			activate_array(l.input + index, l.output + index, 2*l.w*l.h);
			index = entry_index(l, b, n*l.w*l.h, 4);
			activate_array(l.input + index, l.output + index, (1+l.classes)*l.w*l.h);
		}
	}

}


float logistic_activate_gradient(float x) {
	return x * ( 1.0 - x );
}


void activate_array_gradient(const float *x, float* y, const int n)
{
    int i;
    for(i = 0; i < n; ++i){
        y[i] = logistic_activate_gradient(x[i]);
    }
}



void yolov3_loss_pre_action_gradient(layer& l, network& net)
{
	int b,n;
	if(l.is_gpu)
	{
		for (b = 0; b < l.batch; ++b){
			for(n = 0; n < l.n; ++n){
//				int index = entry_index(l, b, n*l.w*l.h, 0);
//				activate_array_gradient(l.delta + index, l.delta + index, 2*l.w*l.h);
				int index = entry_index(l, b, n*l.w*l.h, 4);
				activate_array_gradient(l.delta + index, l.delta + index, (1+l.classes)*l.w*l.h);
			}
		}
		return;

	}


	memcpy(l.output, l.input, l.outputs*l.batch*sizeof(float));

	for (b = 0; b < l.batch; ++b){
		for(n = 0; n < l.n; ++n){
			int index = entry_index(l, b, n*l.w*l.h, 0);
			activate_array(l.input + index, l.output + index, 2*l.w*l.h);
			index = entry_index(l, b, n*l.w*l.h, 4);
			activate_array(l.input + index, l.output + index, (1+l.classes)*l.w*l.h);
		}
	}

}


//############################################################################################################

int yolo_num_detections(layer& l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    cout<<"count: "<<count<<endl;
    return count;
}

int nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

void avg_flipped_yolo(layer& l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer& l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}



int num_detections(layer &l, float thresh)
{
    int i;
    int s = yolo_num_detections(l, thresh);
    return s;
}


detection *make_network_boxes(layer& l, float thresh, int *num)
{

	int i;
    int nboxes = num_detections(l, thresh);
    if(num) *num = nboxes;
    cout<<nboxes<<endl;
    if (nboxes == 0)
    {
    	return nullptr;
    }
//    detection *dets = new detection[nboxes];
    detection *dets = (detection*)calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = (float*)calloc(l.classes, sizeof(float));
    }
    return dets;
}

void fill_network_boxes(layer& l, network& net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
	int count = get_yolo_detections(l, w, h, net.w, net.h, thresh, map, relative, dets);
//    dets += count;
}

detection *get_network_boxes(layer& l, network& net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes(l, thresh, num);
    if(!dets)
    {
    	return nullptr;
    }
    fill_network_boxes(l, net, w, h, thresh, hier, map, relative, dets);
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
    }
    free(dets);
}

void get_dect_boxes(layer &l, network& net)
{
	int i,j;
	float thresh = 0.5;
	float hier_thresh = 0.5;
	int nboxes = 0;

	int img_w = net.w;
	int img_h = net.h;

	detection *dets = get_network_boxes(l, net, img_w, img_h, thresh, hier_thresh, 0, 1, &nboxes);
	if (!dets)
	{
		return;
	}
	float nms = 0.5;
	if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

	int tmp_num = 0;
	for(i = 0; i < nboxes; ++i){
		box b = dets[i].bbox;
		l.delta[tmp_num++] = (b.x-b.w/2.);
		l.delta[tmp_num++] = (b.x+b.w/2.);
		l.delta[tmp_num++] = (b.y-b.h/2.);
		l.delta[tmp_num++] = (b.y+b.h/2.);
		for(j = 0; j < l.classes; j++ )
		{
			l.delta[tmp_num++] = dets[i].prob[j];
		}

    }

	free_detections(dets,nboxes);

}

//############################################################################################################


float sum_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}


// x,y have use logistic function
float yolov3_loss(layer& l, network& net) {


	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));


	if( !net.is_training)
	{
		get_dect_boxes(l,net);
		return -1.0;
	}

	int i, j, b, t, n;

	float avg_iou = 0;
	float recall = 0;
	float recall75 = 0;
	float avg_cat = 0;
	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	int class_count = 0;

	for (b = 0; b < l.batch; ++b) {
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w; ++i) {
				for (n = 0; n < l.n; ++n) {

					int box_index = entry_index(l, b,
							n * l.w * l.h + j * l.w + i, 0);
					box pred = get_yolo_box(l.output, l.biases, l.mask[n],
							box_index, i, j, l.w, l.h, net.w, net.h, l.w * l.h);
					float best_iou = 0;
					int best_t = 0;
					// dzhang debug: find the predict box that have the max iou whith ground truth box
					for (t = 0; t < l.max_boxes; ++t) {
						box truth = float_to_box(
								net.truth + t * (4 + 1) + b * l.truths, 1);
						if (!truth.x)
							break;
						float iou = box_iou(pred, truth);
						if (iou > best_iou) {
							best_iou = iou;
							best_t = t;
						}
					}
					int obj_index = entry_index(l, b,
							n * l.w * l.h + j * l.w + i, 4);
					avg_anyobj += l.output[obj_index];
					//box condidence gradient
					float condidence_scale = 0.5;
//					l.delta[obj_index] = ( 0 - l.output[obj_index] ) * condidence_scale;
					l.delta[obj_index] =  ( l.output[obj_index] - 0) * condidence_scale;
					if (best_iou > l.ignore_thresh) {
						l.delta[obj_index] = 0;
					}
					if (best_iou > l.truth_thresh) {
						l.delta[obj_index] = 1 - l.output[obj_index];

						int classs = net.truth[best_t * (4 + 1) + b * l.truths
								+ 4];
						if (l.map)
							classs = l.map[classs];
						int class_index = entry_index(l, b,
								n * l.w * l.h + j * l.w + i, 4 + 1);
						delta_yolo_class(l.output, l.delta, class_index, classs,
								l.classes, l.w * l.h, 0);
						delta_yolo_class(l.output, l.delta, class_index, classs, l.classes, l.w*l.h, &avg_cat, false);
						box truth = float_to_box(
								net.truth + best_t * (4 + 1) + b * l.truths, 1);
						delta_yolo_box(truth, l.output, l.biases, l.mask[n],
								box_index, i, j, l.w, l.h, net.w, net.h,
								l.delta, (2 - truth.w * truth.h), l.w * l.h);
					}

				}
			}
		}



		for (t = 0; t < l.max_boxes; ++t) {
			box truth = float_to_box(net.truth + t * (4 + 1) + b * l.truths, 1);

			if (!truth.x)
				break;
			float best_iou = 0;
			int best_n = 0;
			i = (truth.x * l.w);
			j = (truth.y * l.h);
			box truth_shift = truth;
			truth_shift.x = truth_shift.y = 0;
			for (n = 0; n < l.total; ++n) {
				box pred = { 0 };
				pred.w = l.biases[2 * n] / net.w;
				pred.h = l.biases[2 * n + 1] / net.h;
				float iou = box_iou(pred, truth_shift);
				if (iou > best_iou) {
					best_iou = iou;
					best_n = n;
				}
			}

			int mask_n = int_index(l.mask, best_n, l.n);
			if (mask_n >= 0) {
				int box_index = entry_index(l, b,
						mask_n * l.w * l.h + j * l.w + i, 0);

				float iou = 0.0;
				float scale = (2 - truth.w * truth.h);
				iou = delta_yolo_box(truth, l.output, l.biases, best_n,
						box_index, i, j, l.w, l.h, net.w, net.h, l.delta,
						scale, l.w * l.h);

				int obj_index = entry_index(l, b,
						mask_n * l.w * l.h + j * l.w + i, 4);
				avg_obj += l.output[obj_index];
				//box condidence gradient
//				l.delta[obj_index] = 1 - l.output[obj_index];
				l.delta[obj_index] = ( l.output[obj_index] - 1.0 );

				int classs = net.truth[t * (4 + 1) + b * l.truths + 4];
				if (l.map)
					classs = l.map[classs];
				int class_index = entry_index(l, b,
						mask_n * l.w * l.h + j * l.w + i, 4 + 1);
//				delta_yolo_class(l.output, l.delta, class_index, classs,
//						l.classes, l.w * l.h, &avg_cat);
				delta_yolo_class(l.output, l.delta, class_index, classs, l.classes, l.w*l.h, &avg_cat, false);


				++count;
				++class_count;
				if (iou > .5)
					recall += 1;
				if (iou > .75)
					recall75 += 1;
				avg_iou += iou;
			}
		}

	}


	float loss = pow(mag_array(l.delta, l.outputs * l.batch), 2.0);

	printf("Region Loss: %f, Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n",
			loss, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);


//	yolov3_loss_pre_action_gradient(l,net);

	return loss;

}

//void backward_yolo_layer(const layer l, network net)
//{
//   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
//}


//} // namespace dzhang
