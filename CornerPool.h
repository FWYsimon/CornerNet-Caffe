#pragma once

#include "common.h"

class TopCornerPoolLayer :public BaseLayer {
public:
	SETUP_LAYERFUNC(TopCornerPoolLayer);
	virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual void backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down);
	virtual void reshape(Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual ~TopCornerPoolLayer();

private:
	shared_ptr<Blob> max_ids_;
};

class LeftCornerPoolLayer :public BaseLayer {
public:
	SETUP_LAYERFUNC(LeftCornerPoolLayer);
	virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual void backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down);
	virtual void reshape(Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual ~LeftCornerPoolLayer();

private:
	shared_ptr<Blob> max_ids_;
};

class BottomCornerPoolLayer :public BaseLayer {
public:
	SETUP_LAYERFUNC(BottomCornerPoolLayer);
	virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual void backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down);
	virtual void reshape(Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual ~BottomCornerPoolLayer();

private:
	shared_ptr<Blob> max_ids_;
};

class RightCornerPoolLayer :public BaseLayer {
public:
	SETUP_LAYERFUNC(RightCornerPoolLayer);
	virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual void backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down);
	virtual void reshape(Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual ~RightCornerPoolLayer();

private:
	shared_ptr<Blob> max_ids_;
};