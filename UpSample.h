#pragma once

#include "common.h"

class Upsample :public BaseLayer {
public:
	SETUP_LAYERFUNC(Upsample);
	virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual void backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down);
	virtual void reshape(Blob** bottom, int numBottom, Blob** top, int numTop);

private:
	int scale_;
};