#include "CornerPool.h"

void TopCornerPoolLayer::reshape(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* out = top[0];
	Blob* data = bottom[0];
	out->reshapeLike(data);
}

void TopCornerPoolLayer::setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* out = top[0];
	Blob* data = bottom[0];

	int batch_size = data->num();
	int channels = data->channel();
	int height = data->height();
	int width = data->width();

	max_ids_ = newBlobByShape(batch_size, channels, height, width);
	float* mask = max_ids_.get()->mutable_cpu_data();

	//memset(mask, -1, sizeof(float) * max_ids_.count());
	for (int n = 0; n < batch_size; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int w = 0; w < width; ++w) {
				mask[(height - 2) * width + w] = (height - 2) * width + w;
			}
			mask += height * width;
		}
	}
}

void TopCornerPoolLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* bottom_data = bottom[0];
	Blob* top_data = top[0];

	int batch_size = bottom_data->num();
	float* bottom_data_ptr = bottom_data->mutable_cpu_data();
	float* top_data_ptr = top_data->mutable_cpu_data();

	for (int n = 0; n < batch_size; ++n){
		for (int c = 0; c < bottom_data->channel(); ++c){
			Mat input(bottom_data->height(), bottom_data->width(), CV_32FC1, bottom_data_ptr + bottom_data->offset(n, c));
			Mat tmp(top_data->height(), top_data->width(), CV_32FC1, top_data_ptr + top_data->offset(n, c));
			input.copyTo(tmp);
			for (int i = 0; i < tmp.cols; i++){
				float max = 0.0f;
				for (int j = tmp.rows - 1; j >= 0; j--){
					if (j == (tmp.rows - 1)){
						max = tmp.at<float>(j, i);
					}
					if (tmp.at<float>(j, i) > max){
						max = tmp.at<float>(j, i);
					}
					else{
						tmp.at<float>(j, i) = max;
					}
				}
			}
		}
	}
}

void TopCornerPoolLayer::backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down) {
	Blob* out = top[0];
	int batch_size = out->count();
	int channels = out->channel();
	int height = out->height();
	int width = out->width();

	const float* top_diff = top[0]->cpu_diff();
	float* bottom_diff = bottom[0]->mutable_cpu_diff();

	//caffe_set(bottom[0]->count(), 0.0f, bottom_diff);

	memset(bottom_diff, 0, sizeof(float) * bottom[0]->count());

	float* max_ids_ptr = max_ids_.get()->mutable_cpu_data();

	for (int n = 0; n < batch_size; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
					int index = h * width + w;
					int bottom_index = max_ids_ptr[index];
					if (bottom_index == -1)
						continue;
					bottom_diff[bottom_index] += top_diff[index];
				}
			}
			bottom_diff += width * height;
			top_diff += width * height;
			max_ids_ptr += width * height;
		}
	}
}

TopCornerPoolLayer::~TopCornerPoolLayer() {
	max_ids_.reset();
}

void LeftCornerPoolLayer::reshape(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* out = top[0];
	Blob* data = bottom[0];
	out->reshapeLike(data);
}

void LeftCornerPoolLayer::setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* out = top[0];
	Blob* data = bottom[0];

	int batch_size = data->num();
	int channels = data->channel();
	int height = data->height();
	int width = data->width();

	max_ids_ = newBlobByShape(batch_size, channels, height, width);
	float* mask = max_ids_.get()->mutable_cpu_data();

	//memset(mask, -1, sizeof(float) * max_ids_.count());
	for (int n = 0; n < batch_size; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < height; ++h) {
				mask[(h + 1) * width - 1] = (h + 1) * width - 1;
			}
			mask += height * width;
		}
	}
}

void LeftCornerPoolLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* bottom_data = bottom[0];
	Blob* top_data = top[0];

	int batch_size = bottom_data->num();
	float* bottom_data_ptr = bottom_data->mutable_cpu_data();
	float* top_data_ptr = top_data->mutable_cpu_data();

	for (int n = 0; n < batch_size; ++n){
		for (int c = 0; c < bottom_data->channel(); ++c){
			Mat input(bottom_data->height(), bottom_data->width(), CV_32FC1, bottom_data_ptr + bottom_data->offset(n, c));
			Mat tmp(top_data->height(), top_data->width(), CV_32FC1, top_data_ptr + top_data->offset(n, c));
			input.copyTo(tmp);
			for (int i = 0; i < tmp.rows; i++){
				float max = 0.0f;
				for (int j = tmp.cols - 1; j >= 0; j--){
					//printf("%f\n", tmp.at<float>(i, j));
					if (j == (tmp.cols - 1)){
						max = tmp.at<float>(i, j);
					}
					if (tmp.at<float>(i, j) >= max){
						max = tmp.at<float>(i, j);
					}
					else{
						tmp.at<float>(i, j) = max;
					}
				}
			}
		}
	}
}

void LeftCornerPoolLayer::backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down) {
	Blob* out = top[0];
	int batch_size = out->count();
	int channels = out->channel();
	int height = out->height();
	int width = out->width();

	const float* top_diff = top[0]->cpu_diff();
	float* bottom_diff = bottom[0]->mutable_cpu_diff();

	//caffe_set(bottom[0]->count(), 0.0f, bottom_diff);

	memset(bottom_diff, 0, sizeof(float) * bottom[0]->count());

	float* max_ids_ptr = max_ids_.get()->mutable_cpu_data();

	for (int n = 0; n < batch_size; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
					int index = h * width + w;
					int bottom_index = max_ids_ptr[index];
					if (bottom_index == -1)
						continue;
					bottom_diff[bottom_index] += top_diff[index];
				}
			}
			bottom_diff += width * height;
			top_diff += width * height;
			max_ids_ptr += width * height;
		}
	}
}

LeftCornerPoolLayer::~LeftCornerPoolLayer() {
	max_ids_.reset();
}

void BottomCornerPoolLayer::reshape(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* out = top[0];
	Blob* data = bottom[0];
	out->reshapeLike(data);
}

void BottomCornerPoolLayer::setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* out = top[0];
	Blob* data = bottom[0];

	int batch_size = data->num();
	int channels = data->channel();
	int height = data->height();
	int width = data->width();

	max_ids_ = newBlobByShape(batch_size, channels, height, width);
	float* mask = max_ids_.get()->mutable_cpu_data();

	//memset(mask, -1, sizeof(float) * max_ids_.count());
	for (int n = 0; n < batch_size; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int w = 0; w < width; ++w) {
				mask[w] = w;
			}
			mask += height * width;
		}
	}
}

void BottomCornerPoolLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	float* max_ids_ptr = max_ids_.get()->mutable_cpu_data();

	Blob* bottom_data = bottom[0];
	Blob* top_data = top[0];

	int batch_size = bottom_data->num();
	float* bottom_data_ptr = bottom_data->mutable_cpu_data();
	float* top_data_ptr = top_data->mutable_cpu_data();

	for (int n = 0; n < batch_size; ++n){
		for (int c = 0; c < bottom_data->channel(); ++c){
			Mat input(bottom_data->height(), bottom_data->width(), CV_32FC1, bottom_data_ptr + bottom_data->offset(n, c));
			Mat tmp(top_data->height(), top_data->width(), CV_32FC1, top_data_ptr + top_data->offset(n, c));
			Mat mask(bottom_data->height(), bottom_data->width(), CV_32FC1, max_ids_->offset(n, c));
			input.copyTo(tmp);
			for (int i = 0; i < tmp.cols; i++){
				float max = 0.0f;
				for (int j = 0; j <tmp.rows; j++){
					if (j == 0)
						max = tmp.at<float>(j, i);
					if (tmp.at<float>(j, i) > max)
						max = tmp.at<float>(j, i);
					else
						tmp.at<float>(j, i) = max;
				}
			}
		}
	}
}

void BottomCornerPoolLayer::backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down) {
	Blob* out = top[0];
	int batch_size = out->count();
	int channels = out->channel();
	int height = out->height();
	int width = out->width();

	const float* top_diff = top[0]->cpu_diff();
	float* bottom_diff = bottom[0]->mutable_cpu_diff();

	//caffe_set(bottom[0]->count(), 0.0f, bottom_diff);

	memset(bottom_diff, 0, sizeof(float) * bottom[0]->count());

	float* max_ids_ptr = max_ids_.get()->mutable_cpu_data();

	for (int n = 0; n < batch_size; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
					int index = h * width + w;
					int bottom_index = max_ids_ptr[index];
					if (bottom_index == -1)
						continue;
					bottom_diff[bottom_index] += top_diff[index];
				}
			}
			bottom_diff += width * height;
			top_diff += width * height;
			max_ids_ptr += width * height;
		}
	}
}

BottomCornerPoolLayer::~BottomCornerPoolLayer() {
	max_ids_.reset();
}

void RightCornerPoolLayer::reshape(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* out = top[0];
	Blob* data = bottom[0];
	out->reshapeLike(data);
}

void RightCornerPoolLayer::setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* out = top[0];
	Blob* data = bottom[0];

	int batch_size = data->num();
	int channels = data->channel();
	int height = data->height();
	int width = data->width();

	max_ids_ = newBlobByShape(batch_size, channels, height, width);
	float* mask = max_ids_.get()->mutable_cpu_data();

	//memset(mask, -1, sizeof(float) * max_ids_.count());
	for (int n = 0; n < batch_size; n++) {
		for (int c = 0; c < channels; c++) {
			for (int h = 0; h < height; h++) {
				mask[h * width] = h * width;
			}
			mask += height * width;
		}
	}
}

void RightCornerPoolLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* bottom_data = bottom[0];
	Blob* top_data = top[0];
	int batch_size = bottom_data->num();

	float* bottom_data_ptr = bottom_data->mutable_cpu_data();
	float* top_data_ptr = top_data->mutable_cpu_data();
	for (int n = 0; n < batch_size; ++n){
		for (int c = 0; c < bottom_data->channel(); ++c){
			Mat input(bottom_data->height(), bottom_data->width(), CV_32FC1, bottom_data_ptr + bottom_data->offset(n, c));
			Mat tmp(top_data->height(), top_data->width(), CV_32FC1, top_data_ptr + top_data->offset(n, c));
			input.copyTo(tmp);
			for (int i = 0; i < tmp.rows; i++){
				float max = 0.0f;
				for (int j = 0; j < tmp.cols; j++){
					//printf("%f\n", tmp.at<float>(i, j));
					if (j == 0){
						max = tmp.at<float>(i, j);
					}
					if (tmp.at<float>(i, j) >= max){
						max = tmp.at<float>(i, j);
					}
					else{
						tmp.at<float>(i, j) = max;
					}
				}
			}
		}
	}
}

void RightCornerPoolLayer::backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down) {
	Blob* out = top[0];
	int batch_size = out->count();
	int channels = out->channel();
	int height = out->height();
	int width = out->width();

	const float* top_diff = top[0]->cpu_diff();
	float* bottom_diff = bottom[0]->mutable_cpu_diff();

	//caffe_set(bottom[0]->count(), 0.0f, bottom_diff);

	memset(bottom_diff, 0, sizeof(float) * bottom[0]->count());

	float* max_ids_ptr = max_ids_.get()->mutable_cpu_data();

	for (int n = 0; n < batch_size; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
					int index = h * width + w;
					int bottom_index = max_ids_ptr[index];
					if (bottom_index == -1)
						continue;
					bottom_diff[bottom_index] += top_diff[index];
				}
			}
			bottom_diff += width * height;
			top_diff += width * height;
			max_ids_ptr += width * height;
		}
	}
}

RightCornerPoolLayer::~RightCornerPoolLayer() {
	max_ids_.reset();
}