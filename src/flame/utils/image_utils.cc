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
 * @file image_utils.cc
 * @author W. Nicholas Greene
 * @date 2017-08-18 18:53:35 (Fri)
 */

#include "flame/utils/image_utils.h"
#include "flame/utils/rasterization.h"

namespace flame {

namespace utils {

#ifdef __SSE__

/**
 * @brief SSE optimized specialization for uint8_t images.
 */
template <>
void getCentralGradient(int width, int height, const uint8_t* img_ptr,
                        float* gradx_ptr, float* grady_ptr) {
  __m128 half = _mm_set_ps(0.5f, 0.5f, 0.5f, 0.5);

  // Compute gradx.
  for (uint32_t ii = 0; ii < height; ++ii) {
    uint32_t jj = 0;
    for (jj = 1; jj < width - 4 - 1; jj+=4) {
      int idx = ii * width + jj;

      __m128 left = _mm_set_ps(img_ptr[idx + 2], img_ptr[idx + 1],
                               img_ptr[idx], img_ptr[idx - 1]);
      __m128 right = _mm_set_ps(img_ptr[idx + 4], img_ptr[idx + 3],
                                img_ptr[idx + 2], img_ptr[idx + 1]);
      __m128 diffx = _mm_sub_ps(right, left);
      __m128 gx = _mm_mul_ps(half, diffx);
      _mm_storeu_ps(gradx_ptr + idx, gx);
    }

    // Finish remainder of row without SSE if not divisble by 4.
    for (; jj < width - 1; ++jj) {
      int idx = ii * width + jj;
      float left = img_ptr[idx - 1];
      float right = img_ptr[idx + 1];
      gradx_ptr[idx] = 0.5 * (right - left);
    }
  }

  // Compute gradient for first/last columns using forward/backward differences.
  for (uint32_t ii = 0; ii < height; ++ii) {
    // First column (forward difference).
    float left = img_ptr[ii * width];
    float right = img_ptr[ii * width + 1];
    gradx_ptr[ii * width] = right - left;

    // Last column (backward difference).
    left = img_ptr[ii * width + width - 2];
    right = img_ptr[ii * width + width - 1];
    gradx_ptr[ii * width + width - 1] = right - left;
  }

  // Compute grady.
  for (uint32_t ii = 1; ii < height - 1; ++ii) {
    uint32_t jj = 0;
    for (jj = 0; jj < width - 4; jj+=4) {
      int idx = ii * width + jj;

      __m128 up = _mm_set_ps(img_ptr[idx - width + 3], img_ptr[idx - width + 2],
                             img_ptr[idx - width + 1], img_ptr[idx - width]);
      __m128 down = _mm_set_ps(img_ptr[idx + width + 3], img_ptr[idx + width + 2],
                             img_ptr[idx + width + 1], img_ptr[idx + width]);
      __m128 diffy = _mm_sub_ps(down, up);
      __m128 gy = _mm_mul_ps(half, diffy);
      _mm_storeu_ps(grady_ptr + idx, gy);
    }

    // Finish remainder of row without SSE if not divisble by 4.
    for (; jj < width; ++jj) {
      int idx = ii * width + jj;
      float up = img_ptr[idx - width];
      float down = img_ptr[idx + width];
      grady_ptr[idx] = 0.5 * (down - up);
    }
  }

  // Compute gradient for first/last rows using forward/backward differences.
  for (uint32_t ii = 0; ii < width; ++ii) {
    // First row (forward difference).
    float up = img_ptr[ii];
    float down = img_ptr[ii + width];
    grady_ptr[ii] = down - up;

    // Last row (backward difference).
    up = img_ptr[(height - 2) * width + ii];
    down = img_ptr[(height - 1) * width + ii];
    grady_ptr[(height - 1) * width + ii] = down - up;
  }

  return;
}

/**
 * @brief SSE optimized specialization for uint8_t images.
 */
template<>
void getCentralGradient(const cv::Mat_<uint8_t>& img,
                        cv::Mat_<float>* gradx,
                        cv::Mat_<float>* grady) {
  int width = img.cols;
  int height = img.rows;

  // Allocate outputs if needed.
  if (gradx->empty()) {
    gradx->create(height, width);
  }
  if (grady->empty()) {
    grady->create(height, width);
  }

  FLAME_ASSERT(img.isContinuous());
  FLAME_ASSERT(gradx->isContinuous());
  FLAME_ASSERT(grady->isContinuous());

  // Grab raw pointers to data.
  const uint8_t* img_ptr = reinterpret_cast<uint8_t*>(img.data);
  float* gradx_ptr = reinterpret_cast<float*>(gradx->data);
  float* grady_ptr = reinterpret_cast<float*>(grady->data);

  getCentralGradient<uint8_t, float>(width, height, img_ptr, gradx_ptr, grady_ptr);

  return;
}

/**
 * @brief SSE optimized specialization for float images.
 */
template <>
void getCentralGradient(int width, int height, const float* img_ptr,
                        float* gradx_ptr, float* grady_ptr) {
  __m128 half = _mm_set_ps(0.5f, 0.5f, 0.5f, 0.5);

  // Compute gradx.
  for (uint32_t ii = 0; ii < height; ++ii) {
    uint32_t jj = 0;
    for (jj = 1; jj < width - 4 - 1; jj+=4) {
      int idx = ii * width + jj;

      __m128 left = _mm_loadu_ps(img_ptr + idx - 1);
      __m128 right = _mm_loadu_ps(img_ptr + idx + 1);
      __m128 diffx = _mm_sub_ps(right, left);
      __m128 gx = _mm_mul_ps(half, diffx);
      _mm_storeu_ps(gradx_ptr + idx, gx);
    }

    // Finish remainder of row without SSE if not divisble by 4.
    for (; jj < width - 1; ++jj) {
      int idx = ii * width + jj;
      float left = img_ptr[idx - 1];
      float right = img_ptr[idx + 1];
      gradx_ptr[idx] = 0.5 * (right - left);
    }
  }

  // Compute gradient for first/last columns using forward/backward differences.
  for (uint32_t ii = 0; ii < height; ++ii) {
    // First column (forward difference).
    float left = img_ptr[ii * width];
    float right = img_ptr[ii * width + 1];
    gradx_ptr[ii * width] = right - left;

    // Last column (backward difference).
    left = img_ptr[ii * width + width - 2];
    right = img_ptr[ii * width + width - 1];
    gradx_ptr[ii * width + width - 1] = right - left;
  }

  // Compute grady.
  for (uint32_t ii = 1; ii < height - 1; ++ii) {
    uint32_t jj = 0;
    for (jj = 0; jj < width - 4; jj+=4) {
      int idx = ii * width + jj;

      __m128 up = _mm_loadu_ps(img_ptr + idx - width);
      __m128 down = _mm_loadu_ps(img_ptr + idx + width);
      __m128 diffy = _mm_sub_ps(down, up);
      __m128 gy = _mm_mul_ps(half, diffy);
      _mm_storeu_ps(grady_ptr + idx, gy);
    }

    // Finish remainder of row without SSE if not divisble by 4.
    for (; jj < width; ++jj) {
      int idx = ii * width + jj;
      float up = img_ptr[idx - width];
      float down = img_ptr[idx + width];
      grady_ptr[idx] = 0.5 * (down - up);
    }
  }

  // Compute gradient for first/last rows using forward/backward differences.
  for (uint32_t ii = 0; ii < width; ++ii) {
    // First row (forward difference).
    float up = img_ptr[ii];
    float down = img_ptr[ii + width];
    grady_ptr[ii] = down - up;

    // Last row (backward difference).
    up = img_ptr[(height - 2) * width + ii];
    down = img_ptr[(height - 1) * width + ii];
    grady_ptr[(height - 1) * width + ii] = down - up;
  }

  return;
}

/**
 * @brief SSE optimized specialization for float images.
 */
template<>
void getCentralGradient(const cv::Mat_<float>& img,
                        cv::Mat_<float>* gradx,
                        cv::Mat_<float>* grady) {
  int width = img.cols;
  int height = img.rows;

  // Allocate outputs if needed.
  if (gradx->empty()) {
    gradx->create(height, width);
  }
  if (grady->empty()) {
    grady->create(height, width);
  }

  FLAME_ASSERT(img.isContinuous());
  FLAME_ASSERT(gradx->isContinuous());
  FLAME_ASSERT(grady->isContinuous());

  // Grab raw pointers to data.
  const float* img_ptr = reinterpret_cast<float*>(img.data);
  float* gradx_ptr = reinterpret_cast<float*>(gradx->data);
  float* grady_ptr = reinterpret_cast<float*>(grady->data);

  getCentralGradient<float, float>(width, height, img_ptr, gradx_ptr, grady_ptr);

  return;
}

#endif

// Liang-Barsky function by Daniel White @
// http://www.skytopia.com/project/articles/compsci/clipping.html This function
// inputs 8 numbers, and outputs 4 new numbers (plus a boolean value to say
// whether the clipped line is drawn at all).
//
bool clipLineLiangBarsky(float xmin, float xmax,
                         float ymin, float ymax,
                         float x0, float y0,
                         float x1, float y1,
                         float* x0_clip, float* y0_clip,
                         float* x1_clip, float* y1_clip) {
  FLAME_ASSERT(!std::isnan(x0));
  FLAME_ASSERT(!std::isnan(y0));
  FLAME_ASSERT(!std::isnan(x1));
  FLAME_ASSERT(!std::isnan(y1));

  FLAME_ASSERT(!std::isnan(xmin));
  FLAME_ASSERT(!std::isnan(xmax));
  FLAME_ASSERT(!std::isnan(ymin));
  FLAME_ASSERT(!std::isnan(ymax));

  float t0 = 0.0f;
  float t1 = 1.0f;
  float xdelta = x1 - x0;
  float ydelta = y1 - y0;

  for (int edge = 0; edge < 4; edge++) {
    float p = 1.0f;
    float q = 0.0f;
    float r = 0.0f;

    if (edge == 0) {
      p = -xdelta;
      q = -(xmin-x0);
    } else if (edge == 1) {
      p = xdelta;
      q = (xmax-x0);
    } else if (edge == 2) {
      p = -ydelta;
      q = -(ymin-y0);
    } else if (edge == 3) {
      p = ydelta;
      q = (ymax-y0);
    }

    r = q/p;
    if (p == 0 && q < 0)
      return false;   // Don't draw line at all. (parallel line outside)

    if (p < 0) {
      if (r > t1)
        return false;         // Don't draw line at all.
      else if (r > t0)
        t0 = r;            // Line is _clipped!
    } else if (p > 0) {
      if (r < t0)
        return false;      // Don't draw line at all.
      else if (r < t1)
        t1 = r;         // Line is _clipped!
    }
  }

  float x0_clip_int = x0 + t0 * xdelta;
  float y0_clip_int = y0 + t0 * ydelta;
  float x1_clip_int = x0 + t1 * xdelta;
  float y1_clip_int = y0 + t1 * ydelta;

  /* Final check for numerical errors. */
  if (x0_clip_int < xmin)
    x0_clip_int = xmin;
  if (x0_clip_int > xmax)
    x0_clip_int = xmax;
  if (y0_clip_int < ymin)
    y0_clip_int = ymin;
  if (y0_clip_int > ymax)
    y0_clip_int = ymax;

  if (x1_clip_int < xmin)
    x1_clip_int = xmin;
  if (x1_clip_int > xmax)
    x1_clip_int = xmax;
  if (y1_clip_int < ymin)
    y1_clip_int = ymin;
  if (y1_clip_int > ymax)
    y1_clip_int = ymax;

  if (!(x0_clip_int >= xmin)) {
    fprintf(stderr, "x0_clip_int(%f) !>= xmin(%f)\n",
            x0_clip_int, xmin);
  }

  FLAME_ASSERT(x0_clip_int >= xmin);
  FLAME_ASSERT(x0_clip_int <= xmax);
  FLAME_ASSERT(y0_clip_int >= ymin);
  FLAME_ASSERT(y0_clip_int <= ymax);
  FLAME_ASSERT(x1_clip_int >= xmin);
  FLAME_ASSERT(x1_clip_int <= xmax);
  FLAME_ASSERT(y1_clip_int >= ymin);
  FLAME_ASSERT(y1_clip_int <= ymax);

  /* Set output. */
  *x0_clip = x0_clip_int;
  *y0_clip = y0_clip_int;
  *x1_clip = x1_clip_int;
  *y1_clip = y1_clip_int;

  return true;        // (clipped) line is drawn
}

void interpolateMesh(const std::vector<Triangle>& triangles,
                     const std::vector<cv::Point2f>& vertices,
                     const std::vector<float>& values,
                     const std::vector<bool>& vtx_validity,
                     const std::vector<bool>& tri_validity,
                     cv::Mat* img) {
  for (int ii = 0; ii < triangles.size(); ++ii) {
    if (tri_validity[ii] &&
        vtx_validity[triangles[ii][0]] && vtx_validity[triangles[ii][1]] &&
        vtx_validity[triangles[ii][2]]) {
      // Triangle spits out points in clockwise order, but drawing function
      // expects CCW.
      utils::DrawShadedTriangleBarycentric(vertices[triangles[ii][2]],
                                           vertices[triangles[ii][1]],
                                           vertices[triangles[ii][0]],
                                           values[triangles[ii][2]],
                                           values[triangles[ii][1]],
                                           values[triangles[ii][0]],
                                           img);
    }
  }

  return;
}

void interpolateInverseMesh(const std::vector<Triangle>& triangles,
                            const std::vector<cv::Point2f>& vertices,
                            const std::vector<float>& ivalues,
                            const std::vector<bool>& validity,
                            cv::Mat* img) {
  for (int ii = 0; ii < triangles.size(); ++ii) {
    if (validity[triangles[ii][0]] && validity[triangles[ii][1]] &&
        validity[triangles[ii][2]]) {
      // Triangle spits out points in clockwise order, but drawing function
      // expects CCW.
      utils::DrawShadedTriangleBarycentric(vertices[triangles[ii][2]],
                                           vertices[triangles[ii][1]],
                                           vertices[triangles[ii][0]],
                                           1.0f/ivalues[triangles[ii][2]],
                                           1.0f/ivalues[triangles[ii][1]],
                                           1.0f/ivalues[triangles[ii][0]],
                                           img);
    }
  }

  return;
}

}  // namespace utils

}  // namespace flame
