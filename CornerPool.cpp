#include "CornerPool.h"

void TopCornerPoolLayer::reshape(Blob** bottom, int numBottom, Blob** top, int numTop) {

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


	out->reshapeLike(data);
}

void TopCornerPoolLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* data = bottom[0];
	Blob* out = top[0];
	
	int batch_size = data->num();
	int channel = data->channel();
	int height = data->height();
	int width = data->width();

	float* input = data->mutable_cpu_data();
	float* ptr = out->mutable_cpu_data();

	float* max_ids_ptr = max_ids_.get()->mutable_cpu_data();

	memcpy(ptr, input, data->count());

	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < channel; j++) {
			//int size = height * width;
			Mat temp = Mat(height, width, CV_32F, ptr);
			for (int h = height - 2; h >= 0; h--) {
				for (int w = 0; w < width; w++) {
					int index = h * width + w;
					int compare_index = (h + 1) * width + w;
					ptr[index] = ptr[index] < ptr[compare_index] ? ptr[compare_index] : ptr[index];
					max_ids_ptr[index] = ptr[index] < ptr[compare_index] ? max_ids_ptr[compare_index] : index;
				}
					
			}
			ptr += height * width;
			max_ids_ptr += height * width;
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


	out->reshapeLike(data);
}

void LeftCornerPoolLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* data = bottom[0];
	Blob* out = top[0];

	int batch_size = data->num();
	int channel = data->channel();
	int height = data->height();
	int width = data->width();

	float* input = data->mutable_cpu_data();
	float* ptr = out->mutable_cpu_data();

	float* max_ids_ptr = max_ids_.get()->mutable_cpu_data();

	memcpy(ptr, input, data->count());

	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < channel; j++) {
			//int size = height * width;
			for (int h = 0; h < height; ++h) {
				for (int w = width - 2; w >= 0; --w) {
					int index = h * width + w;
					int compare_index = h * width + w + 1;
					ptr[index] = ptr[index] < ptr[compare_index] ? ptr[compare_index] : ptr[index];
					max_ids_ptr[index] = ptr[index] < ptr[compare_index] ? max_ids_ptr[compare_index] : max_ids_ptr[index];
				}

			}
			ptr += height * width;
			max_ids_ptr += height * width;
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

	out->reshapeLike(data);
}

void BottomCornerPoolLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* data = bottom[0];
	Blob* out = top[0];

	int batch_size = data->num();
	int channel = data->channel();
	int height = data->height();
	int width = data->width();

	float* input = data->mutable_cpu_data();
	float* ptr = out->mutable_cpu_data();

	float* max_ids_ptr = max_ids_.get()->mutable_cpu_data();

	memcpy(ptr, input, data->count());

	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < channel; j++) {
			//int size = height * width;
			for (int h = 1; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
					int index = h * width + w;
					int compare_index = (h - 1) * width + w;
					ptr[index] = ptr[index] < ptr[compare_index] ? ptr[compare_index] : ptr[index];
					max_ids_ptr[index] = ptr[index] < ptr[compare_index] ? max_ids_ptr[compare_index] : max_ids_ptr[index];
				}

			}
			ptr += height * width;
			max_ids_ptr += height * width;
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


	out->reshapeLike(data);
}

void RightCornerPoolLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* data = bottom[0];
	Blob* out = top[0];

	int batch_size = data->num();
	int channel = data->channel();
	int height = data->height();
	int width = data->width();

	float* input = data->mutable_cpu_data();
	float* ptr = out->mutable_cpu_data();

	float* max_ids_ptr = max_ids_.get()->mutable_cpu_data();

	memcpy(ptr, input, data->count());

	for (int i = 0; i < batch_size; i++) {
		for (int j = 0; j < channel; j++) {
			//int size = height * width;
			for (int h = 0; h < height; ++h) {
				for (int w = 1; w < width; ++w) {
					int index = h * width + w - 1;
					int compare_index = h * width + w - 1;
					ptr[index] = ptr[index] < ptr[compare_index] ? ptr[compare_index] : ptr[index];
					max_ids_ptr[index] = ptr[index] < ptr[compare_index] ? max_ids_ptr[compare_index] : max_ids_ptr[index];
				}

			}
			ptr += height * width;
			max_ids_ptr += height * width;
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