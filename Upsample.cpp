#include "UpSample.h"

map<string, string> parseParamStr(const char* str){
	map<string, string> o;
	if (str){
		char* prev = 0;
		char* p = (char*)str;
		int stage = 0;
		string name, value;

		while (*p){
			while (*p){ if (*p != ' ') break; p++; }
			prev = p;

			while (*p){ if (*p == ' ' || *p == ':') break; p++; }
			if (*p) name = string(prev, p);

			while (*p){ if (*p != ' ' && *p != ':' || *p == '\'') break; p++; }
			bool has_yh = *p == '\'';
			if (has_yh) p++;
			prev = p;

			while (*p){ if (has_yh && *p == '\'' || !has_yh && (*p == ' ' || *p == ';')) break; p++; }
			if (p != prev){
				value = string(prev, p);
				o[name] = value;

				p++;
				while (*p){ if (*p != ' ' && *p != ';' && *p != '\'') break; p++; }
			}
		}
	}
	return o;
}

int getParamInt(map<string, string>& p, const string& key, int default_ = 0){
	if (p.find(key) == p.end())
		return default_;
	return atoi(p[key].c_str());
}

void Upsample::setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop) {
	map<string, string> param = parseParamStr(param_str);
	const int scale = getParamInt(param, "scale");
	const int upsample_w = getParamInt(param, "upsample_w");
	const int upsample_h = getParamInt(param, "upsample_h");

	scale_ = scale;

	Blob* top_data = top[0];
	Blob* bottom_data = bottom[0];
	top_data->reshape(bottom_data->num(), bottom_data->channel(), upsample_h, upsample_w);
}

void Upsample::reshape(Blob** bottom, int numBottom, Blob** top, int numTop) {

}

void Upsample::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* top_data = top[0];
	float* top_data_ptr = top_data->mutable_cpu_data();

	memset(top_data_ptr, 0, top_data->count() * sizeof(float));

	int batch_size = top_data->num();
	int channels = top_data->channel();
	int height = top_data->height();
	int width = top_data->width();

	Blob* bottom_data = bottom[0];
	float* bottom_data_ptr = bottom_data->mutable_cpu_data();

	int bottom_height = bottom_data->height();
	int bottom_width = bottom_data->width();

	for (int n = 0; n < batch_size; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int i = 0; i < height * width; ++i) {
				int row = i / width;
				int col = i % height;
				int index = row / scale_ * bottom_width + col / scale_;
				top_data_ptr[i] = bottom_data_ptr[index];
			}
			bottom_data_ptr += bottom_height * bottom_width;
			top_data_ptr += height * width;
		}
	}
}

void Upsample::backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down) {

}