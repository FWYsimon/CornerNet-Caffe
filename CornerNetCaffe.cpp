#include "common.h"

#include "CornerPool.h"
#include "UpSample.h"
#include "ReLU.h"

const float ae_threshold = 0.5;
const int nms_kernel = 3;
const int top_k = 100;
const float nms_threshold = 0.5;
const int max_per_image = 1000;
const bool merge_bbox = false;

vector<string> labelmap = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "baer", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
"broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dinning table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

struct CornerBBox {
	float tl_x;
	float tl_y;
	float br_x;
	float br_y;
	int cls;
	float tl_score;
	float br_score;
	float score;

	CornerBBox(float tl_x, float tl_y, float br_x, float br_y, float tl_score, float br_score, int cls, float score) :
		tl_x(tl_x), tl_y(tl_y), br_x(br_x), br_y(br_y), tl_score(tl_score), br_score(br_score), cls(cls), score(score){

	}
};

float sigmoid(float x) {
	return 1. / (1. + exp(-x));
}

void max_pool2d(Blob* heat, int kernel, int stride, int pad) {
	int batch_size = heat->num();
	int cat = heat->channel();
	int height = heat->height();
	int width = heat->width();

	int pooled_height_ = static_cast<int>(ceil(static_cast<float>(
		height + 2 * pad - kernel) / stride)) + 1;

	int pooled_width_ = static_cast<int>(ceil(static_cast<float>(
		width + 2 * pad - kernel) / stride)) + 1;

	if (pad) {
		if ((pooled_height_ - 1) * stride >= height + pad) {
			--pooled_height_;
		}
		if ((pooled_width_ - 1) * stride >= width + pad) {
			--pooled_width_;
		}
	}

	float* bottom_data = heat->mutable_cpu_data();

	shared_ptr<Blob> top_data_blob = newBlobByShape(batch_size, cat, height, width);
	//top_data_blob->reshapeLike(heat);
	float* top_data = top_data_blob.get()->mutable_cpu_data();

	for (int i = 0; i < top_data_blob->count(); ++i)
		top_data[i] = -FLT_MAX;

	for (int n = 0; n < batch_size; ++n) {
		for (int c = 0; c < cat; ++c) {
			for (int ph = 0; ph < pooled_height_; ++ph) {
				for (int pw = 0; pw < pooled_width_; ++pw) {
					int hstart = ph * stride - pad;
					int wstart = pw * stride - pad;
					int hend = min(hstart + kernel, height);
					int wend = min(wstart + kernel, width);
					hstart = max(hstart, 0);
					wstart = max(wstart, 0);
					const int pool_index = ph * pooled_width_ + pw;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							const int index = h * width + w;
							if (bottom_data[index] > top_data[pool_index]) {
								top_data[pool_index] = bottom_data[index];
							}
						}
					}
				}
			}
			// compute offset
			bottom_data += height * width;
			top_data += pooled_width_ * pooled_height_;
		}
	}

	top_data = top_data_blob.get()->mutable_cpu_data();
	bottom_data = heat->mutable_cpu_data();

	for (int i = 0; i < top_data_blob->count(); ++i) {
		if (top_data[i] != bottom_data[i]) {
			top_data[i] = 0;
		}
		bottom_data[i] = top_data[i];
	}

	top_data_blob.reset();
}

void nms(Blob* heat, int kernel) {
	int pad = (kernel - 1) / 2;

	max_pool2d(heat, kernel, 1, pad);
}

void heat_topk(vector<float> scores, int K, int height, int width, vector<float>& topk_scores, vector<int>& topk_inds, vector<int>& topk_clses, vector<float>& topk_ys, vector<float>& topk_xs) {
	topk_scores = scores;

	topk_inds.resize(scores.size());
	iota(topk_inds.begin(), topk_inds.end(), 0);
	//std::sort(temp_topk_scores.rbegin(), temp_topk_scores.rend());
	std::sort(topk_inds.begin(), topk_inds.end(), [&scores](int a, int b) {
		return scores[a] > scores[b];
	});

	std::sort(topk_scores.rbegin(), topk_scores.rend());
	topk_scores.resize(K);
	topk_inds.resize(K);

	topk_clses.resize(K);
	topk_ys.resize(K);
	topk_xs.resize(K);

	for (int i = 0; i < K; ++i) {
		topk_clses[i] = (int)(topk_inds[i] / (height * width));

		topk_inds[i] = topk_inds[i] % (height * width);
		topk_ys[i] = topk_inds[i] / width;
		topk_xs[i] = topk_inds[i] % width;
	}

}

void topk(vector<float> scores, int num_dets, vector<int>& inds, vector<float>& topk_scores) {
	inds.resize(scores.size());
	topk_scores = scores;
	iota(inds.begin(), inds.end(), 0);
	//std::sort(temp_topk_scores.rbegin(), temp_topk_scores.rend());
	std::sort(inds.begin(), inds.end(), [&scores](int a, int b) {
		return scores[a] > scores[b];
	});

	std::sort(topk_scores.rbegin(), topk_scores.rend());
	topk_scores.resize(num_dets);
	inds.resize(num_dets);
}

void gather_offset_feat(vector<float> offset, vector<int> inds, vector<float>& topk_offset) {
	int feature_area = offset.size() / 2;
	int K = inds.size();
	topk_offset.resize(K * 2);

	for (int i = 0; i < K; ++i) {
		int index = inds[i];

		float x_offset = offset[index];
		float y_offset = offset[index + feature_area];

		topk_offset[i] = x_offset;
		topk_offset[i + K] = y_offset;
	}
}

void gather_embed_feat(vector<float> embed, vector<int> inds, vector<float>& topk_embed) {
	int K = inds.size();
	topk_embed.resize(K);
	for (int i = 0; i < K; ++i) {
		int index = inds[i];
		topk_embed[i] = embed[index];
	}
}

void gather_feat(Blob* feat, float* inds, int K) {
	//shared_ptr<Blob> trans_feat = feat->transpose(0, 2, 3, 1);
	//gather_feat(feat, inds, K);
}

vector<vector<CornerBBox>> decode(Blob* tl_heat, Blob* tl_embed, Blob* tl_offset, Blob* br_heat, Blob* br_embed, Blob* br_offset, int K = 100, int kernel = 1, float ae_threshold = 1.0f, int num_dets = 1000) {
	int batch_size = tl_heat->num();
	int channels = tl_heat->channel();
	int height = tl_heat->height();
	int width = tl_heat->width();

	float* tl_heat_ptr = tl_heat->mutable_cpu_data();
	float* br_heat_ptr = br_heat->mutable_cpu_data();

	vector<float> vec_tl_heat(tl_heat_ptr, tl_heat_ptr + tl_heat->channel() * tl_heat->height() * tl_heat->width());
	vector<float> vec_br_heat(br_heat_ptr, br_heat_ptr + tl_heat->channel() * tl_heat->height() * tl_heat->width());

	CV_Assert(tl_heat->count() == br_heat->count());

	for (int n = 0; n < batch_size; ++n) {
		for (int c = 0; c < channels; ++c) {
			Mat tl_heat_mat(tl_heat->height(), tl_heat->width(), CV_32FC1, tl_heat_ptr + tl_heat->offset(n, c));
			Mat br_heat_mat(br_heat->height(), br_heat->width(), CV_32FC1, br_heat_ptr + br_heat->offset(n, c));
			for (int i = 0; i < tl_heat_mat.rows; ++i) {
				for (int j = 0; j < tl_heat_mat.cols; ++j) {
					tl_heat_mat.at<float>(i, j) = sigmoid(tl_heat_mat.at<float>(i, j));
					br_heat_mat.at<float>(i, j) = sigmoid(br_heat_mat.at<float>(i, j));
				}
			}
		}
	}

	// perform nms on heatmaps
	nms(tl_heat, kernel);
	nms(br_heat, kernel);

	float* tl_offset_ptr = tl_offset->mutable_cpu_data();
	float* br_offset_ptr = br_offset->mutable_cpu_data();

	float* tl_embed_ptr = tl_embed->mutable_cpu_data();
	float* br_embed_ptr = br_embed->mutable_cpu_data();

	int heat_step = tl_heat->channel() * tl_heat->height() * tl_heat->width();
	int offset_step = tl_offset->channel() * tl_offset->height() * tl_offset->width();
	int embed_step = tl_embed->channel() * tl_embed->height() * tl_embed->width();

	vector<vector<CornerBBox>> result;

	for (int n = 0; n < batch_size; ++n) {
		vector<float> vec_tl_heat(tl_heat_ptr, tl_heat_ptr + heat_step);
		vector<float> vec_br_heat(br_heat_ptr, br_heat_ptr + heat_step);
		vector<float> tl_scores, tl_ys, tl_xs, br_scores, br_ys, br_xs;
		vector<int> tl_inds, tl_clses, br_inds, br_clses;

		heat_topk(vec_tl_heat, K, height, width, tl_scores, tl_inds, tl_clses, tl_ys, tl_xs);
		heat_topk(vec_br_heat, K, height, width, br_scores, br_inds, br_clses, br_ys, br_xs);

		vector<float> vec_tl_offset(tl_offset_ptr, tl_offset_ptr + offset_step);
		vector<float> vec_br_offset(br_offset_ptr, br_offset_ptr + offset_step);
		vector<float> tl_offsets, br_offsets;

		gather_offset_feat(vec_tl_offset, tl_inds, tl_offsets);
		gather_offset_feat(vec_br_offset, br_inds, br_offsets);

		for (int i = 0; i < K; ++i) {
			tl_xs[i] += tl_offsets[i];
			tl_ys[i] += tl_offsets[i + K];
			br_xs[i] += br_offsets[i];
			br_ys[i] += br_offsets[i + K];
		}

		vector<float> vec_tl_embed(tl_embed_ptr, tl_embed_ptr + embed_step);
		vector<float> vec_br_embed(br_embed_ptr, br_embed_ptr + embed_step);
		vector<float> tl_embeds, br_embeds;
		gather_embed_feat(vec_tl_embed, tl_inds, tl_embeds);
		gather_embed_feat(vec_br_embed, br_inds, br_embeds);

		vector<float> scores(K * K);
		for (int i = 0; i < K; ++i) {
			for (int j = 0; j < K; ++j) {
				int index = i * K + j;
				scores[index] = (tl_scores[i] + br_scores[j]) / 2;

				// reject boxes based on classes
				if (tl_clses[i] != br_clses[j])
					scores[index] = -1;

				// reject boxes based on distances
				float dist = fabs(tl_embeds[i] - br_embeds[j]);
				if (dist > ae_threshold)
					scores[index] = -1;

				// reject boxes based on widths and heights
				if (br_xs[i] < tl_xs[j])
					scores[index] = -1;
				if (br_ys[i] < tl_ys[j])
					scores[index] = -1;
			}
		}

		vector<float> top_scores;
		vector<int> top_inds;
		topk(scores, num_dets, top_inds, top_scores);

		vector<CornerBBox> one_batch_result;
		for (int i = 0; i < num_dets; ++i) {
			int index = top_inds[i];
			float score = top_scores[i];

			int x_index = index / K;
			int y_index = index % K;

			float tl_x = tl_xs[x_index];
			float tl_y = tl_ys[y_index];
			float br_x = br_xs[x_index];
			float br_y = br_ys[y_index];
			float tl_score = tl_scores[x_index];
			float br_score = br_scores[y_index];
			int cls = tl_clses[x_index];

			CornerBBox cbbox(tl_x, tl_y, br_x, br_y, tl_score, br_score, cls, score);
			one_batch_result.push_back(cbbox);
		}
		result.push_back(one_batch_result);

		tl_heat_ptr += heat_step;
		br_heat_ptr += heat_step;
		tl_offset_ptr += offset_step;
		br_offset_ptr += offset_step;
		tl_embed_ptr += embed_step;
		br_embed_ptr += embed_step;
	}
	return result;
}

Mat crop_image(Mat image, int ctx, int cty, int height, int width, vector<float>& border) {
	int im_height = image.rows;
	int im_width = image.cols;

	Mat cropped_image = Mat::zeros(height, width, CV_8UC3);

	int x0 = max(0, ctx - width / 2);
	int x1 = min(ctx + width / 2, im_width);
	int y0 = max(0, cty - height / 2);
	int y1 = min(cty + height / 2, im_height);

	int left = ctx - x0;
	int right = x1 - ctx;
	int top = cty - y0;
	int bottom = y1 - cty;

	int cropped_cty = height / 2;
	int cropped_ctx = width / 2;

	int y_slice_start = cropped_cty - top;
	int y_slice_end = cropped_cty + bottom;
	int x_slice_start = cropped_ctx - left;
	int x_slice_end = cropped_ctx + right;

	int offset_y = y_slice_start - y0;
	int offset_x = x_slice_start - x0;
	for (int h = y_slice_start; h < y_slice_end; ++h) {
		for (int w = x_slice_start; w < x_slice_end; ++w) {
			//cropped_image.at<int>(h, w) = image.at<int>(h - offset_y, w - offset_x);
			cropped_image.at<Vec3b>(h, w)[0] = image.at<Vec3b>(h - offset_y, w - offset_x)[0];
			cropped_image.at<Vec3b>(h, w)[1] = image.at<Vec3b>(h - offset_y, w - offset_x)[1];
			cropped_image.at<Vec3b>(h, w)[2] = image.at<Vec3b>(h - offset_y, w - offset_x)[2];
		}
	}

	border.push_back(cropped_cty - top);
	border.push_back(cropped_cty + bottom);
	border.push_back(cropped_ctx - left);
	border.push_back(cropped_ctx + right);

	return cropped_image;
}

vector<CornerBBox> soft_nms(vector<CornerBBox> boxes, float sigma = 0.5, float Nt = 0.3, float threshold = 0.001, int method = 0) {
	int N = boxes.size();

	float max_score = 0;
	int max_pos = 0;
	int pos = 0;

	float weight;

	for (int i = 0; i < N; ++i) {
		max_score = boxes[i].score;
		max_pos = i;

		float tx1 = boxes[i].tl_x;
		float ty1 = boxes[i].tl_y;
		float tx2 = boxes[i].br_x;
		float ty2 = boxes[i].br_y;

		float ts = boxes[i].score;

		pos = i + 1;
		while (pos < N) {
			if (max_score < boxes[pos].score) {
				max_score = boxes[pos].score;
				max_pos = pos;
			}
			pos += 1;
		}
		// add max box as a detection 
		boxes[i].tl_x = boxes[max_pos].tl_x;
		boxes[i].tl_y = boxes[max_pos].tl_y;
		boxes[i].br_x = boxes[max_pos].br_x;
		boxes[i].br_y = boxes[max_pos].br_y;

		// swap ith box with position of max box
		boxes[max_pos].tl_x = tx1;
		boxes[max_pos].tl_y = ty1;
		boxes[max_pos].br_x = tx2;
		boxes[max_pos].br_y = ty2;
		boxes[max_pos].score = ts;

		tx1 = boxes[i].tl_x;
		ty1 = boxes[i].tl_y;
		tx2 = boxes[i].br_x;
		ty2 = boxes[i].br_y;

		ts = boxes[i].score;

		pos = i + 1;
		// NMS iterations, note that N changes if detection boxes fall below threshold
		while (pos < N) {
			float x1 = boxes[pos].tl_x;
			float y1 = boxes[pos].tl_y;
			float x2 = boxes[pos].br_x;
			float y2 = boxes[pos].br_y;
			float s = boxes[pos].score;

			float area = (x2 - x1 + 1) * (y2 - y1 + 1);
			float iw = (min(tx2, x2) - max(tx1, x1) + 1);
			if (iw > 0) {
				float ih = (min(ty2, y2) - max(ty1, y1) + 1);
				if (ih > 0) {
					float ua = (tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih;
					// #iou between max box and detection box
					float ov = iw * ih / ua;

					// linear
					if (method == 1) {
						if (ov > Nt)
							weight = 1 - ov;
						else
							weight = 1;
					}
					else if (method == 2) { // gaussian 
						weight = exp(-(ov * ov) / sigma);
					}
					else { // original nms
						if (ov > Nt)
							weight = 0;
						else
							weight = 1;
					}

					boxes[pos].score = weight * boxes[pos].score;

					// if box score falls below threshold, discard the box by swapping with last box
					// update N
					if (boxes[pos].score < threshold) {
						boxes[pos].tl_x = boxes[N - 1].tl_x;
						boxes[pos].tl_y = boxes[N - 1].tl_y;
						boxes[pos].br_x = boxes[N - 1].br_x;
						boxes[pos].br_y = boxes[N - 1].br_y;
						boxes[pos].score = boxes[N - 1].score;
						N -= 1;
						pos -= 1;
					}
				}
			}
			pos += 1;
		}
	}
	vector<CornerBBox> out = vector<CornerBBox>(boxes.begin(), boxes.begin() + N);
	return out;
}

int main() {
	setGPU(0);
	cc::installRegister();

	INSTALL_LAYER(TopCornerPoolLayer);
	INSTALL_LAYER(LeftCornerPoolLayer);
	INSTALL_LAYER(BottomCornerPoolLayer);
	INSTALL_LAYER(RightCornerPoolLayer);
	INSTALL_LAYER(Upsample);
	INSTALL_LAYER(ReLU);

	string deploy = "cornernet.prototxt";
	string model = "cornernet.caffemodel";
	shared_ptr<Net> net = loadNetFromPrototxt(deploy.c_str());
	net->weightsFromFile(model.c_str());

	Blob* tl_heat = net->blob("conv_blob172");

	Blob* tl_embed = net->blob("conv_blob176");
	Blob* tl_offset = net->blob("conv_blob180");

	Blob* br_heat = net->blob("conv_blob174");
	Blob* br_embed = net->blob("conv_blob178");
	Blob* br_offset = net->blob("conv_blob182");

	Mat im = imread("test.jpg");
	Mat show;
	im.copyTo(show);

	int height = im.rows;
	int width = im.cols;

	int scale = 1;

	int new_height = height * scale;
	int new_width = width * scale;

	int new_center_x = new_width / 2;
	int new_center_y = new_height / 2;

	int inp_height = new_height | 127;
	int inp_width = new_width | 127;

	int out_height = (inp_height + 1) / 4;
	int out_width = (inp_width + 1) / 4;
	float height_ratio = (float)out_height / inp_height;
	float width_ratio = (float)out_width / inp_width;

	resize(im, im, Size(new_width, new_height));

	//crop image
	vector<float> border;
	Mat cropped_image = crop_image(im, new_center_x, new_center_y, inp_height, inp_width, border);

	Scalar mean(0.40789654, 0.44719302, 0.47026115);
	Scalar std(0.28863828, 0.27408164, 0.27809835);

	cropped_image.convertTo(cropped_image, CV_32F);
	cropped_image /= 255.0;

	for (int i = 0; i < cropped_image.rows; ++i) {
		for (int j = 0; j < cropped_image.cols; ++j) {
			for (int n = 0; n < 3; n++) {
				cropped_image.at<cv::Vec3f>(i, j)[n] -= mean[n];
				cropped_image.at<cv::Vec3f>(i, j)[n] /= std[n];
			}
		}
	}

	Blob* input = net->input_blob(0);
	input->reshape(1, 3, cropped_image.rows, cropped_image.cols);

	net->reshape();
	input->setData(0, cropped_image);

	net->forward();

	vector<vector<CornerBBox>> result = decode(tl_heat, tl_embed, tl_offset, br_heat, br_embed, br_offset, top_k, nms_kernel, ae_threshold, max_per_image);


	int categories = labelmap.size();
	//rescale dets
	vector<CornerBBox> all_boxes = result[0];
	vector<vector<CornerBBox>> keep_boxes(categories);
	vector<float> scores;
	for (int i = 0; i < all_boxes.size(); ++i) {
		float tl_x = all_boxes[i].tl_x / height_ratio;
		float tl_y = all_boxes[i].tl_y / width_ratio;
		float br_x = all_boxes[i].br_x / height_ratio;
		float br_y = all_boxes[i].br_y / width_ratio;
		tl_x -= border[2];
		br_x -= border[2];
		tl_y -= border[0];
		br_y -= border[0];

		all_boxes[i].tl_x = max(0, tl_x);
		all_boxes[i].tl_y = max(0, tl_y);
		all_boxes[i].br_x = min(br_x, width);
		all_boxes[i].br_y = min(br_y, height);
		if (all_boxes[i].score > -1) {
			keep_boxes[all_boxes[i].cls].push_back(all_boxes[i]);
			scores.push_back(all_boxes[i].score);
		}
	}

	const int box_per_image = 100;
	int count_box = 0;
	vector<CornerBBox> top_bboxes;
	for (int i = 0; i < categories; ++i) {
		vector<CornerBBox> box_after_nms = soft_nms(keep_boxes[i], 0.5f, nms_threshold, 2);
		top_bboxes.insert(top_bboxes.end(), box_after_nms.begin(), box_after_nms.end());
		count_box += box_after_nms.size();
	}

	vector<int> remove_inds;
	if (count_box > box_per_image) {
		std::sort(scores.begin(), scores.end(), greater<float>());
		float thresh = scores[box_per_image];
		for (int i = 0; i < top_bboxes.size(); ++i) {
			if (top_bboxes[i].score < thresh)
				remove_inds.push_back(i);
		}
	}

	for (int i = 0; i < remove_inds.size(); ++i)
		top_bboxes.erase(top_bboxes.begin() + remove_inds[i]);

	for (int i = 0; i < top_bboxes.size(); ++i) {
		CornerBBox cbbox = top_bboxes[i];
		if (cbbox.score <= 0.5)
			continue;
		Rect box(cbbox.tl_x, cbbox.tl_y, cbbox.br_x - cbbox.tl_x, cbbox.br_y - cbbox.tl_y);
		putText(show, labelmap[cbbox.cls], Point(cbbox.tl_x, cbbox.tl_y), 1, 2, Scalar(255, 0, 0), 2);
		rectangle(show, box, Scalar(255, 255, 0), 2);
	}

	imshow("demo", show);
	waitKey(0);

	return 0;
}

