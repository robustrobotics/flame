/**
 * This file is part of FLaME.
 * Copyright (C) 2017 W. Nicholas Greene (wng@csail.mit.edu)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see <http://www.gnu.org/licenses/>.
 *
 * @file visualization.h
 * @author W. Nicholas Greene
 * @date 2017-08-18 18:58:49 (Fri)
 */

#pragma once

#include <limits>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "flame/types.h"
#include "flame/utils/assert.h"

namespace flame {

namespace utils {

/**
 * @brief Draw a rectangle on an image.
 */
template <typename T>
void rectangle(const Point2i& top_left, const Point2i& bot_right,
               const T& color, Image<T>* img, int thickness = 1) {
  int xstart = top_left.x;
  int xend = bot_right.x;
  int ystart = top_left.y;
  int yend = bot_right.y;

  // Draw horizontal lines.
  for (int ii = -thickness/2; ii <= thickness/2; ++ii) {
    for (int jj = xstart; jj <= xend; ++jj) {
      (*img)(ystart + ii, jj) = color;
      (*img)(yend + ii, jj) = color;
    }
  }

  // Draw vertical lines. (Switch drawing order to prefer walking over columns.)
  for (int ii = ystart; ii <= yend; ++ii) {
    for (int jj = -thickness/2; jj <= thickness/2; ++jj) {
      (*img)(ii, xstart + jj) = color;
      (*img)(ii, xend + jj) = color;
    }
  }

  return;
}

// Adapted from http://axonflux.com/handy-rgb-to-hsl-and-rgb-to-hsv-color-model-c
inline float hueToRGB(float p, float q, float t) {
  if (t < 0)
    t += 1;
  if (t > 1)
    t -= 1;
  if (t < 1.0/6)
    return p + (q - p) * 6 * t;
  if (t < 1.0/2)
    return q;
  if (t < 2.0/3)
    return p + (q - p) * (2.0/3 - t) * 6;

  return p;
}

/**
 * Converts an HSL color value to RGB. Conversion formula
 * adapted from http://en.wikipedia.org/wiki/HSL_color_space.
 * Assumes h, s, and l are contained in the set [0, 1] and
 * returns r, g, and b in the set [0, 255].
 *
 * @param   Number  h       The hue
 * @param   Number  s       The saturation
 * @param   Number  l       The lightness
 * @return  Array           The RGB representation
 */
inline cv::Vec3b hslToRGB(float h, float s, float l) {
    float r, g, b;

    if (s == 0) {
      r = g = b = l; // achromatic
    } else {

      float q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      float p = 2 * l - q;
      r = hueToRGB(p, q, h + 1.0/3);
      g = hueToRGB(p, q, h);
      b = hueToRGB(p, q, h - 1.0/3);
    }

    cv::Vec3b c(255, 255, 255);  // white
    c[0] = b * 255;
    c[1] = g * 255;
    c[2] = r * 255;

    return c;
  }

/**
 * @brief Maps an inward point normal to a color.
 */
inline cv::Vec3b normalMap(float nx, float ny, float nz) {
  cv::Vec3b color(0, 0, 0);

  uint8_t red = 255 * (nx + 1)/2;
  uint8_t green = 255 * (ny + 1)/2;
  uint8_t blue = 127 * nz + 127;
  color[0] = blue;
  color[1] = green;
  color[2] = red;

  return color;
}

/**
 * \brief Jet colormap.
 *
 * Matlab-like jet colormap.
 *
 * @param v[in] Value.
 * @param vmin[in] Minimum value limit.
 * @param vmax[in] Maximum value limit.
 * @return Color.
 */
inline cv::Vec3b jet(float v, float vmin, float vmax) {
  cv::Vec3b c(255, 255, 255);  // white
  float dv;

  if (v < vmin)
    v = vmin;
  if (v > vmax)
    v = vmax;
  dv = vmax - vmin;

  if (v < (vmin + 0.25 * dv)) {
    c[2] = 0;
    c[1] = static_cast<uint8_t>(255 * (4 * (v - vmin) / dv));
  } else if (v < (vmin + 0.5 * dv)) {
    c[2] = 0;
    c[0] = static_cast<uint8_t>(255 * (1 + 4 * (vmin + 0.25 * dv - v) / dv));;
  } else if (v < (vmin + 0.75 * dv)) {
    c[2] = static_cast<uint8_t>(255 * (4 * (v - vmin - 0.5 * dv) / dv));
    c[0] = 0;
  } else {
    c[1] = static_cast<uint8_t>(255 * (1 + 4 * (vmin + 0.75 * dv - v) / dv));
    c[0] = 0;
  }

  return c;
}

/**
 * \brief Maps a value linearly between two colors.
 */
inline cv::Vec3b blendColor(const cv::Vec3b& cmin, const cv::Vec3b& cmax,
                            float v, float vmin, float vmax) {
  // Clamp value.
  if (v < vmin) {
    v = vmin;
  }
  if (v > vmax) {
    v = vmax;
  }

  float alpha = (v - vmin) / (vmax - vmin);
  float invalpha = 1.0f - alpha;

  return cv::Vec3b(invalpha*cmin[0] + alpha*cmax[0],
                   invalpha*cmin[1] + alpha*cmax[1],
                   invalpha*cmin[2] + alpha*cmax[2]);
}

/**
 * \brief Inverse depth colormap.
 *
 * Colormap for displaying inverse depth.
 *
 * @param id[in] Inverse depth.
 * @return Color.
 */
inline cv::Vec3b idepthColor(float id) {
  // rainbow between 0 and 4
  float r = (0 - id) * 255 / 1.0;
  r = (r < 0) ? -r : r;

  float g = (1-id) * 255 / 1.0;
  g = (g < 0) ? -g : g;

  float b = (2-id) * 255 / 1.0;
  b = (b < 0) ? -b : b;

  uchar rc = r < 0 ? 0 : (r > 255 ? 255 : r);
  uchar gc = g < 0 ? 0 : (g > 255 ? 255 : g);
  uchar bc = b < 0 ? 0 : (b > 255 ? 255 : b);

  return cv::Vec3b(255 - rc, 255 - gc, 255 - bc);
}

/**
 * \brief Convert 32-bit Census value to 8-bit gray value.
 *
 * @param[in] val Input Census value.
 * @param[in] min Minimum Census value.
 * @param[in] max Maximum Census value.
 * @return Gray scale value.
 */
inline cv::Vec3b censusToGray(const uint32_t val, const uint32_t min,
                              const uint32_t max) {
  uint32_t range = max - min;
  range = (range == 0) ? 1 : range;
  uint8_t gray = 255 * (static_cast<float>(val) - min) / range;
  return cv::Vec3b(gray, gray, gray);
}

/**
 * \brief Apply a colormap to a line, interpolating along the values.
 */
template <typename ColorMap>
inline void applyColorMapLine(const cv::Point2f& A, const cv::Point2f& B,
                              float A_val, float B_val, ColorMap map,
                              float alpha, cv::Mat* img) {
  FLAME_ASSERT(img->channels() == 3);
  FLAME_ASSERT(img->isContinuous());
#if CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION >= 3
  FLAME_ASSERT(img->type() == cv::traits::Type<cv::Vec3b>::value);
#else
  FLAME_ASSERT(img->type() == cv::DataType<cv::Vec3b>::type);
#endif

  cv::LineIterator it(*img, A, B);
  float slope0 = (B_val - A_val) / (it.count);

  for (int ii = 0; ii < it.count; ++ii, ++it) {
    float val_ii = A_val + ii * slope0;

    cv::Vec3b color = map(val_ii);
    (*it)[0] = color[0] * alpha + (*it)[0] * (1.0f - alpha);
    (*it)[1] = color[1] * alpha + (*it)[1] * (1.0f - alpha);
    (*it)[2] = color[2] * alpha + (*it)[2] * (1.0f - alpha);
  }

  return;
}

/**
 * \brief Apply a colormap to an image.
 *
 * Long Description
 *
 * @param img[in] Input image.
 * @param map[in] Colormap function object/functor.
 * @param out[out] Colormapped image.
 */
template <typename ChannelType, typename ColorMap>
void applyColorMap(const cv::Mat& img, ColorMap map, cv::Mat3b* out) {
  FLAME_ASSERT(img.channels() == 1);
  FLAME_ASSERT(img.isContinuous());

  if (out->empty()) {
    out->create(img.rows, img.cols);
  }

  const ChannelType* in_ptr = reinterpret_cast<ChannelType*>(img.data);
  cv::Vec3b* out_ptr = reinterpret_cast<cv::Vec3b*>(out->data);
  for (int ii = 0; ii < img.rows * img.cols; ++ii) {
    out_ptr[ii] = map(in_ptr[ii], out_ptr[ii]);
  }

  return;
}

#if CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2
// cv::Mat_<uint32_t> is not supported in OpenCV 3.3.

/**
 * \brief Convert a 32-bit Census image to RGB image.
 *
 * Most of the convenient functions that make this easy in OpenCV don't work for
 * uint32_t.
 *
 * @param[in] img Input Census image.
 * @param[ou] out Output image.
 */
inline void censusToRGB(const cv::Mat_<uint32_t>& img, cv::Mat3b* out,
                        uint32_t min = 0, uint32_t max = 10000000) {
  FLAME_ASSERT(img.isContinuous());

  if (out->empty()) {
    out->create(img.rows, img.cols);
  }

  // Create RGB image.
  for (int ii = 0; ii < img.rows; ++ii) {
    for (int jj = 0; jj < img.cols; ++jj) {
      // (*out)(ii, jj) = censusToGray(img(ii, jj), min, max);
      (*out)(ii, jj) = jet(img(ii, jj), min, max);
    }
  }

  return;
}
#endif

// /**
//  * \brief Colormap for an inverse depth estimate.
//  *
//  * @param lastFrameID[in] Visualization type?
//  * @return RGB color.
//  */
// inline cv::Vec3b getVisualizationColor(const InverseFlameimate& est, int lastFrameID) {
//   if (params::debugDisplay == 0 ||
//       params::debugDisplay == 1) {
//     float id;
//     if (params::debugDisplay == 0) {
//       id = est.idepth_smoothed;
//     } else {// if(params::debugDisplay == 1)
//       id = est.idepth;
//     }

//     if (id < 0) {
//       return cv::Vec3b(255, 255, 255);
//     }

//     // rainbow between 0 and 4
//     float r = (0 - id) * 255 / 1.0;
//     r = (r < 0) ? -r : r;

//     float g = (1-id) * 255 / 1.0;
//     g = (g < 0) ? -g : g;

//     float b = (2-id) * 255 / 1.0;
//     b = (b < 0) ? -b : b;

//     uchar rc = r < 0 ? 0 : (r > 255 ? 255 : r);
//     uchar gc = g < 0 ? 0 : (g > 255 ? 255 : g);
//     uchar bc = b < 0 ? 0 : (b > 255 ? 255 : b);

//     return cv::Vec3b(255 - rc, 255 - gc, 255 - bc);
//   }

//   // plot validity counter
//   if (params::debugDisplay == 2) {
//     float f = est.validity_counter * (255.0 /
//       (params::VALIDITY_COUNTER_MAX_VARIABLE + params::VALIDITY_COUNTER_MAX));
//     uchar v = f < 0 ? 0 : (f > 255 ? 255 : f);

//     return cv::Vec3b(0, v, v);
//   }

//   // plot var
//   if (params::debugDisplay == 3 || params::debugDisplay == 4) {
//     float idv;
//     if (params::debugDisplay == 3) {
//       idv = est.idepth_var_smoothed;
//     } else {
//       idv = est.idepth_var;
//     }

//     float var = - 0.5 * log10(idv);

//     var = var * 255 * 0.333;
//     if (var > 255) {
//       var = 255;
//     }
//     if (var < 0) {
//       return cv::Vec3b(0, 0, 255);
//     }

//     return cv::Vec3b(255 - var, var, 0);// bw
//   }

//   // plot skip
//   if (params::debugDisplay == 5) {
//     float f = (est.next_stereo_min_frame_id - lastFrameID) * (255.0 / 100);
//     uchar v = f < 0 ? 0 : (f > 255 ? 255 : f);
//     return cv::Vec3b(v, 0, v);
//   }

//   return cv::Vec3b(255, 255, 255);
// }

}  // namespace utils

}  // namespace flame
