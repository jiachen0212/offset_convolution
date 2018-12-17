#include <vector>

#include "caffe/util/im2col_off.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, Dtype* data_col, const Dtype* offset) {
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int outputs = output_h * output_w;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
  for (int k_h = 0; kernel_h; ++k_h){
        for (int k_w = 0; kernel_w; ++k_w){
          int input_row = k_h - pad_h;
         for (int i = 0; output_h; ++i){
            int input_col = k_w - pad_w;
            for (int j = 0; output_w; ++j){
          // offset [in_c, k_h, k_w, 2]
          int new_row = input_row + floor(offset[((channel * kernel_h + k_h) * kernel_w + k_w) * 2 + 0]);
          int new_col = input_col + floor(offset[((channel * kernel_h + k_h) * kernel_w + k_w) * 2 + 1]);
          vector<int> ner1(2);
          ner1.push_back(floor(new_row));
          ner1.push_back(floor(new_col));
          vector<int> ner2(2);
          ner2.push_back(floor(new_row));
          ner2.push_back(floor(new_col) + 1);
          vector<int> ner3(2);
          ner3.push_back(floor(new_row) + 1);
          ner3.push_back(floor(new_col));
          vector<int> ner4(2);
          ner4.push_back(floor(new_row) + 1);
          ner4.push_back(floor(new_col) + 1);
          if (ner1[0] < 0 || ner1[0] >= height || ner1[1] < 0 || ner1[1] >= width){
            data_col[((k_h * kernel_h + k_w) * 4 + 0 ) * outputs + i * output_w + j] = 0;  // *data_col is pointer
          }
          else{
            data_col[((k_h * kernel_h + k_w) * 4 + 0 ) * outputs + i * output_w + j] = data_im[ner1[0] * width + ner1[1]];
            }
          if (ner2[0] < 0 || ner2[0] >= height || ner2[1] < 0 || ner2[1] >= width){
            data_col[((k_h * kernel_h + k_w) * 4 + 1 ) * outputs + i * output_w + j] = 0;
          }
          else{
            data_col[((k_h * kernel_h + k_w) * 4 + 1 ) * outputs + i * output_w + j] = data_im[ner2[0] * width + ner2[1]];
            }
          if (ner3[0] < 0 || ner3[0] >= height || ner3[1] < 0 || ner3[1] >= width){
            data_col[((k_h * kernel_h + k_w) * 4 + 2 ) * outputs + i * output_w + j] = 0;
          }
          else{
            data_col[((k_h * kernel_h + k_w) * 4 + 2 ) * outputs + i * output_w + j] = data_im[ner3[0] * width + ner3[1]];
            }
          if (ner4[0] < 0 || ner4[0] >= height || ner4[1] < 0 || ner4[1] >= width){
            data_col[((k_h * kernel_h + k_w) * 4 + 3 ) * outputs + i * output_w + j] = 0;
          }
          else{
            data_col[((k_h * kernel_h + k_w) * 4 + 3 ) * outputs + i * output_w + j] = data_im[ner4[0] * width + ner4[1]];
            }
          input_col += stride_w;
        }
        input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w, float* data_col, const float* offset);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w, double* data_col, const double* offset);

template <typename Dtype>
inline void im2col_nd_core_cpu(const Dtype* data_input, const bool im2col,
    const int num_spatial_axes, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_output) {
  if (!im2col) {
    int im_size = im_shape[0];
    for (int i = 0; i < num_spatial_axes; ++i) {
      im_size *= im_shape[1 + i];
    }
    caffe_set(im_size, Dtype(0), data_output);
  }
  int kernel_size = 1;
  for (int i = 0; i < num_spatial_axes; ++i) {
    kernel_size *= kernel_shape[i];
  }
  const int channels_col = col_shape[0];
  vector<int> d_offset(num_spatial_axes, 0);
  vector<int> d_iter(num_spatial_axes, 0);
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int offset = c_col;
    for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
      if (d_i < num_spatial_axes - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented; ) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      int index_col = c_col;
      int index_im = c_col / kernel_size;
      bool is_padding = false;
      for (int d_i = 0; d_i < num_spatial_axes; ++d_i) {
        const int d = d_iter[d_i];
        const int d_im = d * stride[d_i] - pad[d_i] +
            d_offset[d_i] * dilation[d_i];
        is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_im *= im_shape[d_i + 1];
        index_im += d_im;
      }
      if (im2col) {
        if (is_padding) {
          data_output[index_col] = 0;
        } else {
          data_output[index_col] = data_input[index_im];
        }
      } else if (!is_padding) {  // col2im
        data_output[index_im] += data_input[index_col];
      }
      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      incremented = false;
      for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
        const int d_max = col_shape[d_i + 1];
        DCHECK_LT(d_iter[d_i], d_max);
        if (d_iter[d_i] == d_max - 1) {
          d_iter[d_i] = 0;
        } else {  // d_iter[d_i] < d_max - 1
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    }  // while(incremented) {
  }  // for (int c = 0; c < channels_col; ++c) {
}

template <typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col) {
  const bool kIm2Col = true;
  im2col_nd_core_cpu(data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
                  kernel_shape, pad, stride, dilation, data_col);
}

// Explicit instantiation
template void im2col_nd_cpu<float>(const float* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_col);
template void im2col_nd_cpu<double>(const double* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, Dtype* data_im, const Dtype* offset) {
  // data_col: [in_c*k_h*k_w*4, out_h*out_w]
  // offset: [in_c, k_h, k_w, 2]
  caffe_set(height * width * channels, Dtype(0), data_im); // initializer the data_im ..
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int outs = output_h * output_w;
  const int channel_size = height * width;
  int maxind = 4 * channels * kernel_h * kernel_w * outs;
  for (int channel = channels; channel--; data_im += channel_size) {
    int  ind = 0;
    while (ind < maxind) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        // offset += not -=
        input_row += floor(offset[((channel * kernel_h + kernel_row) * kernel_w + kernel_col) * 2 + 0]);
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            // data_col += output_w;
            ind += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            input_col += floor(offset[((channel * kernel_h + kernel_row) * kernel_w + kernel_col) * 2 + 1]);
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += data_col[ind % outs + (ind / outs) * 4 * outs];
              }
              if (input_col < width - 1) {
                data_im[input_row * width + input_col + 1] += data_col[ind % outs + (ind / outs) * 4 * outs + outs];
              }
              if ((is_a_ge_zero_and_a_lt_b(input_col, width)) && (input_row < height - 1)) {
                data_im[(input_row  + 1) * width + input_col] += data_col[ind % outs + (ind / outs) * 4 * outs + 2 * outs];
              }
              if ((input_col < width - 1) && (input_row < height - 1)) {
                data_im[(input_row  + 1) * width + input_col + 1] += data_col[ind % outs + (ind / outs) * 4 * outs + 3 * outs];
              }
              // data_col++;
              ind ++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w, float* data_im,  const float* offset);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w, double* data_im, const double* offset);

template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im) {
  const bool kIm2Col = false;
  im2col_nd_core_cpu(data_col, kIm2Col, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, dilation, data_im);
}

// Explicit instantiation
template void col2im_nd_cpu<float>(const float* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_im);
template void col2im_nd_cpu<double>(const double* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_im);


}  // namespace caffe
