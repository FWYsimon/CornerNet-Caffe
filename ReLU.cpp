#include "ReLU.h"

void ReLU::setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop) {

}

void ReLU::reshape(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* bottom_data = bottom[0];
	Blob* top_data = top[0];
	top_data->reshapeLike(bottom_data);
}

void ReLU::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* bottom_data = bottom[0];
	Blob* top_data = top[0];

	const float* bottom_data_ptr = bottom_data->cpu_data();
	float* top_data_ptr = top_data->mutable_cpu_data();
	for (int i = 0; i < bottom_data->count(); ++i) {
		if (_isnan(bottom_data_ptr[i])) {
			top_data_ptr[i] = 0;
			continue;
		}
		top_data_ptr[i] = bottom_data_ptr[i] > 0 ? bottom_data_ptr[i] : 0;
	}
}

void ReLU::backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down) {

}