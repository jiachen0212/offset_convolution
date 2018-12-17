#include <vector>

#include "caffe/layers/conv_off_layer.hpp"

namespace caffe {

template <typename Dtype>
void DeforConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* offset = this->blobs_[1]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {  // number of bottom
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight, offset,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[2]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

// kernel function for get offset diff and weight diff
template <typename Dtype>
__global__ void get_diff_kernel(const int num, const Dtype* new_weight_diff, const Dtype* weight, const Dtype* offset,
  Dtype* weight_diff, Dtype* offset_tmp, const int in_channel, const int kernel_h, const int kernel_w){
  CUDA_KERNEL_LOOP(index, num) {
  int outc = index / (in_channel * kernel_h * kernel_w);
  int inc = (index / (kernel_h * kernel_w)) % in_channel;
  int kh = (index / kernel_w) % kernel_h;
  int kw = index % kernel_w;

  int weight_idx              = ((outc * in_channel + inc) * kernel_h + kh) * kernel_w + kw;
  int new_weight_idx          = weight_idx * 4;
  int offset_idx              = ((inc * kernel_h + kh) * kernel_w + kw) * 2;
  int offset_tmp_idx          = weight_idx * 2;

  float off_x                 = offset[offset_idx] - floor(offset[offset_idx]);
  float off_y                 = offset[offset_idx + 1] - floor(offset[offset_idx + 1]);
                              // cause diff, so need +=    each smaple's diff, add sum
  weight_diff[weight_idx]     += new_weight_diff[new_weight_idx] * (1 - off_x) * (1 - off_y) +
                              new_weight_diff[new_weight_idx + 1] * (1 - off_x) * off_y +
                              new_weight_diff[new_weight_idx + 2] * off_x * (1 - off_y) +
                              new_weight_diff[new_weight_idx + 3] * off_x * off_y;

  // offset_tmp_ [outc, inc, kh, kw, 2]  adding between independent threads
  // diff 太大的话,加个scale或者上下限它控制更新.
  offset_tmp[offset_tmp_idx]     += weight[weight_idx] * (new_weight_diff[new_weight_idx] * (off_y - 1)  // diff +=
                                                    - new_weight_diff[new_weight_idx + 1] * off_y
                                                    + new_weight_diff[new_weight_idx + 2] * (1 - off_y)
                                                    + new_weight_diff[new_weight_idx + 3] * off_y);

  offset_tmp[offset_tmp_idx + 1] += weight[weight_idx] * (new_weight_diff[new_weight_idx] * (off_x - 1)  // diff +=
                                                    + new_weight_diff[new_weight_idx + 1] * (1 - off_x)
                                                    - new_weight_diff[new_weight_idx + 2] * off_x
                                                    + new_weight_diff[new_weight_idx + 3] * off_x);
  }
}


template <typename Dtype>
void get_diff(const Dtype* new_weight_diff,const Dtype* weight, const Dtype* offset, Dtype* weight_diff, Dtype* offset_tmp,
  const int out_channel, const int in_channel, const int kernel_h, const int kernel_w) {
  int num = out_channel * in_channel * kernel_h * kernel_w;
  get_diff_kernel<Dtype><<<CAFFE_GET_BLOCKS(num),
                             CAFFE_CUDA_NUM_THREADS>>>(num, new_weight_diff, weight, offset, weight_diff, offset_tmp,
                             in_channel, kernel_h, kernel_w);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void get_diff<float>(const float* new_weight_diff,const float* weight, const float* offset, float* weight_diff, float* offset_tmp, const int out_channel, const int in_channel, const int kernel_h, const int kernel_w);
template void get_diff<double>(const double* new_weight_diff,const double* weight, const double* offset, double* weight_diff, double* offset_tmp, const int out_channel, const int in_channel, const int kernel_h, const int kernel_w);


template <typename Dtype>
void DeforConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* offset = this->blobs_[1]->gpu_data();
  Dtype* offset_diff = this->blobs_[1]->mutable_gpu_diff();

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    if (this->bias_term_ && this->param_propagate_down_[2]) { // Bias gradient, if necessary.
      Dtype* bias_diff = this->blobs_[2]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || this->param_propagate_down_[1] || propagate_down[i]) { // gradient w.r.t. weight. Note that we will accumulate diffs.
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        if (this->param_propagate_down_[0]) {
          Dtype* new_weight_diff = this->off_weight_->mutable_gpu_diff(); // gener new_weight_diff then dis to weight_diff and offset_diff.
          Dtype* offset_tmp = this->offset_diff_tmp_->mutable_gpu_diff();
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_, // bottom data 2 bottom_col(*4), bottom_col * top diff get new_weight_diff
              top_diff + n * this->top_dim_, new_weight_diff, offset);
          const int out_channel = this->blobs_[0]->shape(0); // new_weight_diff dis to offset diff and weight diff
          const int in_channel = this->blobs_[0]->shape(1);
          const int k_h = this->blobs_[0]->shape(2);
          const int k_w = this->blobs_[0]->shape(3);
          // cause offset_tmp will add in each sample time, so need to set zero each sample time, or loss paofei..
          // caffe_gpu_set(this->offset_diff_tmp_->count(), Dtype(0.0), offset_tmp);
          get_diff(new_weight_diff, weight, offset, weight_diff, offset_tmp, out_channel, in_channel, k_h, k_w);
          const Dtype* outc_tmp = this->outc_tmp_->gpu_data();
          // caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, in_channel * k_h * k_w * 2, // get offset_diff, offset_tmp_ * outc_tmp = offset_diff
          // out_channel, (Dtype)1., outc_tmp, offset_tmp, (Dtype)1., offset_diff);   // alpha beta: 1 1  c = alpha*ab + beta*c  each sample y sum up
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, in_channel * k_h * k_w * 2,
          out_channel, (Dtype)1., outc_tmp, offset_tmp, (Dtype)0., offset_diff);   // if off_tmp not set 0.0 eatch sample time, then bata set to 0.
        }
        if (propagate_down[i]) {  // gradient w.r.t. bottom data, if necessary.
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight, offset,
              bottom_diff + n * this->bottom_dim_);  // get bottom_diff: [inc, height, width]
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DeforConvolutionLayer);

}  // namespace caffe

