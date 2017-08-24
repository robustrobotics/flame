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
 * @file image_utils.h
 * @author W. Nicholas Greene
 * @date 2017-08-18 18:50:27 (Fri)
 */

#pragma once

#include <vector>

#ifdef __SSE__
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "flame/types.h"
#include "flame/utils/assert.h"
#include "flame/utils/triangulator.h"
#include "flame/utils/visualization.h"

namespace flame {

namespace utils {

/**
 * @brief Returns smallest multiple of mult that is larger than val.
 */
template <typename T>
inline T least_multiple(T mult, T val) {
  T rem = val % mult;
  if (rem == 0) {
    return val;
  } else if (val <= mult) {
    return mult;
  } else {
    return val + mult - rem;
  }
}

/**
 * @brief Fast round to int.
 *
 * std::round() is notoriously slow (I think because it needs to handle all
 * edge cases). This is a faster implementation I found here:
 * https://stackoverflow.com/questions/485525/round-for-float-in-c
 */
inline int fast_roundf(float r) {
  return (r > 0.0f) ? (r + 0.5f) : (r - 0.5f);
}

/**
 * @brief Fast ceiling function.
 *
 * Taken from https://www.codeproject.com/Tips/700780/Fast-floor-ceiling-functions
 */
template <typename T>
inline int fast_ceil(T r) {
  int ret = static_cast<int>(r);
  return (ret < r) ? ++ret : ret;
}

/**
 * @brief Fast absolute value.
 *
 * std::fabs<T> appears to be slow on my system, while std::abs is fast. This
 * function ensures a faster abs is used.
 */
template <typename T>
inline T fast_abs(T r) {
  return (r > 0) ? r : -r;
}

/**
 * @brief Computes (unscaled) barycentric coordinates of p given triangle (v0,
 * v1, v2) in *clockwise* winding in image coordinates (x right, y down).
 *
 * Coordinates are unscaled. To get true coordinates in [0-1], divide by area2,
 * which is the area of the parallelogram defined by (v0, v1, v2).
 *
 * Adapted from:
 * https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
 */
template <typename T>
void barycentricCoords(const cv::Point_<T>& v0, const cv::Point_<T>& v1,
                       const cv::Point_<T>& v2, const cv::Point_<T>& p,
                       T* alpha, T* beta, T* gamma, T* area2) {
  *area2 = -v1.y * v2.x + v0.y * (-v1.x + v2.x) + v0.x * (v1.y - v2.y) +
      v1.x * v2.y;
  *alpha = v0.y * v2.x - v0.x * v2.y + (v2.y - v0.y) * p.x + (v0.x - v2.x) * p.y;
  *beta = v0.x * v1.y - v0.y * v1.x + (v0.y - v1.y) * p.x + (v1.x - v0.x) * p.y;
  *gamma = *area2 - *alpha - *beta;
  return;
}

/**
 * @brief Test if point lies in the interior of triangle given barycentric
 * coordinates computed using barycentricCoords.
 *
 * Will return false if point is on the edge of triangle.
 */
template <typename T>
bool insideTriangle(T alpha, T beta, T gamma) {
  return (alpha > 0) && (beta > 0) && (gamma > 0);
}

/**
 * @brief Test if point lies on the border of triangle given barycentric
 * coordinates computed using barycentricCoords.
 */
template <typename T>
bool onTriangleBorder(T alpha, T beta, T gamma, T area2) {
  if ((alpha < 0) || (beta < 0) || (gamma < 0) || (area2 < 0)) {
    // Point is outside triangle.
    return false;
  } else {
    // Test if on edge.
    return (alpha == 0) || (beta == 0) || (gamma == 0) ||
        (alpha == area2) || (beta == area2) || (gamma == area2);
  }
}

/**
 * @brief Returns true if D is inside circumcirlce of triangle (A, B, C) defined
 * in *clockwise* order in image coordinates (x right, y down).
 *
 * Uses determinant test outlined here:
 * https://en.wikipedia.org/wiki/Delaunay_triangulation
 */
template <typename T>
bool insideCircumCircle(const cv::Point_<T>& A, const cv::Point_<T>& B,
                        const cv::Point_<T>& C, const cv::Point_<T>& D) {
  Eigen::Matrix<T, 3, 3> M;
  M(0, 0) = A.x - D.x;
  M(0, 1) = A.y - D.y;
  M(0, 2) = (A.x*A.x - D.x*D.x) + (A.y*A.y - D.y*D.y);

  M(1, 0) = B.x - D.x;
  M(1, 1) = B.y - D.y;
  M(1, 2) = (B.x*B.x - D.x*D.x) + (B.y*B.y - D.y*D.y);

  M(2, 0) = C.x - D.x;
  M(2, 1) = C.y - D.y;
  M(2, 2) = (C.x*C.x - D.x*D.x) + (C.y*C.y - D.y*D.y);

  return (M.determinant() > 0);
}

/**
 * @brief Return true if number is a power of two.
 *
 * Only valid for unsigned integer types.
 * Taken from https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
 */
template <typename T>
inline bool isPowerOfTwo(T val) {
  FLAME_ASSERT(val > 0);
  return val && !(val & (val - 1));
}

/**
 * @brief Checks if forming power-of-two pyramid with base dimension dim and
 * num_levels is valid (dim is divisible by 2^num_levels).
 */
template <typename T>
inline bool isPyramidValid(T dim, T num_levels) {
  FLAME_ASSERT(dim > 0);
  FLAME_ASSERT(num_levels > 0);
  return (num_levels > 1) ? (dim % (1 << (num_levels - 1)) == 0) : true;
}

/**
 * @brief Compute bilinear interpolation weights.
 */
inline void bilinearWeights(float x, float y, float* w00, float* w01,
                            float* w10, float* w11) {
  int x_floor = static_cast<int>(x);
  int y_floor = static_cast<int>(y);

  float dx = x - x_floor;
  float dy = y - y_floor;

  /* Compute rectangles using only 1 multiply (taken from LSD-SLAM). */
  *w11 = dx * dy;
  *w01 = dx - *w11;
  *w10 = dy - *w11;
  *w00 = 1.0f - dx - dy + *w11;

  return;
}

/**
 * @brief Bilinear interpolation function.
 *
 * Returns the approximate pixel value at a point (x, y) by interpolating
 * among its neighbors.
 *
 * @param[in] rows Image rows.
 * @param[in] cols Image columns.
 * @param[in] step Row step in bytes.
 * @param[in] data Raw pixel data.
 * @param[in] x Row coordinate (horizontal).
 * @param[in] y Column coordinate (vertical).
 * @return Interpolated value.
 */
template <typename ChannelType, typename RetType>
inline RetType bilinearInterp(uint32_t rows, uint32_t cols, std::size_t step,
                              const void* data, float x, float y) {
  FLAME_ASSERT(x >= 0);
  FLAME_ASSERT(y >= 0);
  FLAME_ASSERT(x < cols - 1);
  FLAME_ASSERT(y < rows - 1);
  // FLAME_ASSERT(img.channels() == 1);
  // FLAME_ASSERT(img.isContinuous());
  FLAME_ASSERT(rows > 1);
  FLAME_ASSERT(cols > 1);

  int x_floor = static_cast<int>(x);
  int y_floor = static_cast<int>(y);

  float w00, w01, w10, w11;
  bilinearWeights(x, y, &w00, &w01, &w10, &w11);

  const uint8_t* datab =
      &(static_cast<const uint8_t*>(data))[y_floor * step + x_floor * sizeof(ChannelType)];

  return w00 * (*reinterpret_cast<const ChannelType*>(datab)) +
      w01 * (*reinterpret_cast<const ChannelType*>(datab + sizeof(ChannelType))) +
      w10 * (*reinterpret_cast<const ChannelType*>(datab + step)) +
      w11 * (*reinterpret_cast<const ChannelType*>(datab + sizeof(ChannelType) + step));
}

/**
 * @brief Bilinear interpolation function.
 *
 * Returns the approximate pixel value at a point (x, y) by interpolating
 * among its neighbors.
 *
 * @param img[in] Input image.
 * @param x[in] X-coordinate (horizontal).
 * @param Y[in] Y-coordinate (vertical).
 * @return Interpolated value.
 */
template <typename ChannelType, typename RetType>
inline RetType bilinearInterp(const cv::Mat& img, float x, float y) {
  return bilinearInterp<ChannelType, RetType>(img.rows, img.cols, img.step,
                                              img.data, x, y);
}

#ifdef __SSE4_1__

/**
 * @brief Template specifalization of bilinearInterp optimized using SSE.
 *
 * NOTE: I'm not seeing any performance boost above the non-SSE version, but
 * keeping this around for now.
 */
inline float bilinearInterpSSE(const cv::Mat& img, float x, float y) {
  // std::cout << "(" << x << ", " << y << ")" << std::endl;

  FLAME_ASSERT(x >= 0);
  FLAME_ASSERT(y >= 0);
  FLAME_ASSERT(x < img.cols - 1);
  FLAME_ASSERT(y < img.rows - 1);
  FLAME_ASSERT(img.channels() == 1);
  FLAME_ASSERT(img.isContinuous());
  FLAME_ASSERT(img.rows > 1);
  FLAME_ASSERT(img.cols > 1);

  int x_floor = static_cast<int>(x);
  int y_floor = static_cast<int>(y);

  // cppcheck-suppress assignBoolToPointer
  const uint8_t* data = img.ptr<uint8_t>(0) +
    y_floor*img.cols + x_floor;

  /* Compute rectangles using only 1 multiply (taken from LSD-SLAM). */
  float dx = x - x_floor;
  float dy = y - y_floor;
  float r_00 = dx * dy;
  float r_01 = dx - r_00;
  float r_10 = dy - r_00;
  float r_11 = 1 - dx - dy + r_00;

  __m128 weights = _mm_set_ps(r_11, r_10, r_01, r_00);
  __m128 vals = _mm_set_ps(data[0], data[img.cols], data[1], data[img.cols + 1]);

  const int mask = 0xFF;
  __m128 dot_prod = _mm_dp_ps(weights, vals, mask);

  float ret;
  _mm_store_ss(&ret, dot_prod);

  return ret;
}

#endif

/**
 * @brief Apply a max-filter with a 3x3-sample window to image in place.
 *
 * Modified from LSD-SLAM.
 *
 * Replaces every value with the maximum value in a 3x3 window. Useful for
 * minimum-suppression (e.g. computing the maximum gradient).
 *
 * @param[in/out] img Input image. Will be modified in place.
 */
template <typename ChannelType>
void applyMaxFilter3(cv::Mat_<ChannelType>* img) {
  int width = img->cols;
  int height = img->rows;

  // Only process points not along the first/last row/column.
  ChannelType* curr_ptr = reinterpret_cast<ChannelType*>(img->data) + width + 1;
  ChannelType* end_ptr =
    reinterpret_cast<ChannelType*>(img->data) + width * (height - 1) - 1;

  // Apply max-filter up/down into tmp buffer.
  cv::Mat_<ChannelType> tmp(height, width);
  ChannelType* tmp_ptr = reinterpret_cast<ChannelType*>(tmp.data) + width+1;
  for (; curr_ptr < end_ptr; curr_ptr++, tmp_ptr++) {
    ChannelType up = curr_ptr[-width];
    ChannelType center = curr_ptr[0];
    if (up < center) {
      up = center;
    }
    ChannelType down = curr_ptr[width];
    if (up < down) {
      *tmp_ptr = down;
    } else {
      *tmp_ptr = up;
    }
  }

  // Apply max-filter left/right into output.
  curr_ptr = reinterpret_cast<ChannelType*>(img->data) + width+1;
  end_ptr = reinterpret_cast<ChannelType*>(img->data) + width * (height - 1) - 1;
  tmp_ptr = reinterpret_cast<ChannelType*>(tmp.data) + width+1;
  for (; curr_ptr < end_ptr; curr_ptr++, tmp_ptr++) {
    ChannelType left = tmp_ptr[-1];
    ChannelType center = tmp_ptr[0];
    if (left < center) {
      left = center;
    }
    ChannelType right = tmp_ptr[1];
    if (left < right) {
      *curr_ptr = right;
    } else {
      *curr_ptr = left;
    }
  }

  return;
}

/**
 * @brief Compute image gradients using Sobel operators.
 *
 * The Sobel operator combines a Gaussian blur with a discrete derivative.
 *
 * @param img[in] Input image.
 * @param gradx[out] X gradient (horizontal).
 * @param grady[out] Y gradient (vertical).
 * @return Return Description
 */
template <typename GradientType>
void getSobelGradient(const cv::Mat& img, cv::Mat_<GradientType>* gradx,
                      cv::Mat_<GradientType>* grady) {
  if (gradx->empty()) {
    gradx->create(img.size());
  }
  if (grady->empty()) {
    grady->create(img.size());
  }

  cv::Sobel(img, *gradx, cv::DataType<float>::type, 1, 0);
  cv::Sobel(img, *grady, cv::DataType<float>::type, 0, 1);

  // Sobel derivatives are not normalized.
  *gradx = *gradx/8;
  *grady = *grady/8;

  return;
}

/**
 * @brief Compute central gradient.
 *
  * Modified from LSD-SLAM.
 *
 * The derivative kernel is simply 0.5 * [-1, 0, 1] for x and y. No
 * blurring/smoothing is performed.
 *
 * @param[in] width Image width.
 * @param[in] height Image height.
 * @param[in] img Image buffer (no padding bytes allowed).
 * @param[out] gradx Horizontal gradient (must be pre-allocated).
 * @param[out] grady Vertical gradient (must be pre-allocated).
 */
template <typename ChannelType, typename GradientType>
void getCentralGradient(int width, int height, const ChannelType* img_ptr,
                        GradientType* gradx_ptr, GradientType* grady_ptr) {
  // Compute gradx.
  for (uint32_t ii = 0; ii < height; ++ii) {
    for (uint32_t jj = 1; jj < width - 1; ++jj) {
      int idx = ii * width + jj;

      GradientType left = img_ptr[idx - 1];
      GradientType right = img_ptr[idx + 1];
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
    for (uint32_t jj = 0; jj < width; ++jj) {
      int idx = ii * width + jj;

      GradientType up = img_ptr[idx - width];
      GradientType down = img_ptr[idx + width];
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
 * @brief Compute image gradients using central differences.
 *
 * Modified from LSD-SLAM.
 *
 * The derivative kernel is simply 0.5 * [-1, 0, 1] for x and y. No
 * blurring/smoothing is performed.
 *
 * @param[in] img Input image.
 * @param[out] gradx Gradient in the horizontal direction.
 * @param[out] grady Gradient in the vertical direction.
 */
template <typename ChannelType, typename GradientType>
void getCentralGradient(const cv::Mat_<ChannelType>& img,
                        cv::Mat_<GradientType>* gradx,
                        cv::Mat_<GradientType>* grady) {
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
  const ChannelType* img_ptr = reinterpret_cast<ChannelType*>(img.data);
  GradientType* gradx_ptr = reinterpret_cast<GradientType*>(gradx->data);
  GradientType* grady_ptr = reinterpret_cast<GradientType*>(grady->data);

  getCentralGradient<ChannelType, GradientType>(width, height, img_ptr,
                                                gradx_ptr, grady_ptr);

  return;
}

#ifdef __SSE__

/**
 * @brief SSE optimized specialization for uint8_t images.
 */
template <>
void getCentralGradient(int width, int height, const uint8_t* img_ptr,
                        float* gradx_ptr, float* grady_ptr);

/**
 * @brief SSE optimized specialization for uint8_t images.
 */
template<>
void getCentralGradient(const cv::Mat_<uint8_t>& img,
                        cv::Mat_<float>* gradx,
                        cv::Mat_<float>* grady);

/**
 * @brief SSE optimized specialization for float images.
 */
template <>
void getCentralGradient(int width, int height, const float* img_ptr,
                        float* gradx_ptr, float* grady_ptr);

/**
 * @brief SSE optimized specialization for float images.
 */
template<>
void getCentralGradient(const cv::Mat_<float>& img,
                        cv::Mat_<float>* gradx,
                        cv::Mat_<float>* grady);

#endif

/**
 * @brief Return the magnitude of the gradients.
 *
 * @param[in] gradx Horizontal gradient.
 * @param[in] grady Vertical gradient.
 * @param[out] mag Magnitude.
 */
template <typename GradientType>
void getGradientMag(const cv::Mat_<GradientType>& gradx,
                    const cv::Mat_<GradientType>& grady,
                    cv::Mat_<GradientType>* mag) {
  FLAME_ASSERT(gradx.size() == grady.size());

  if (mag->empty()) {
    mag->create(gradx.size());
  }

  GradientType* mag_ptr = reinterpret_cast<GradientType*>((*mag).data);
  GradientType* gradx_ptr = reinterpret_cast<GradientType*>(gradx.data);
  GradientType* grady_ptr = reinterpret_cast<GradientType*>(grady.data);

  for (int ii = 0; ii < gradx.rows; ++ii) {
    for (int jj = 0; jj < gradx.cols; ++jj) {
      int idx = ii * gradx.cols + jj;
      GradientType dx = gradx_ptr[idx];
      GradientType dy = grady_ptr[idx];
      mag_ptr[idx] = sqrt(dx*dx + dy*dy);
    }
  }

  return;
}

/**
 * @brief Clip line with Liang-Barsky algorithm.
 *
 * Clip a line defined by endpoints (x0, y0) and (x1, y1)
 * to fit in window defined by (xmin, xmax) and (ymin, ymax).
 *
 * @param xmin[in] Left edge of window.
 * @param xmax[in] Right edge.of window.
 * @param ymin[in] Bottom edge of window.
 * @param ymax[in] Top edge of window.
 * @param x0[in] X-coordinate of endpoint 0.
 * @param y0[in] Y-coordinate of endpoint 0.
 * @param x1[in] X-coordinate of endpoint 1.
 * @param y1[in] Y-coordinate of endpoint 1.
 * @param x0_clip[out] X-coordinate of clipped endpoint 0.
 * @param y0_clip[out] Y-coordinate of clipped endpoint 0.
 * @param x1_clip[out] X-coordinate of clipped endpoint 1.
 * @param y1_clip[out] Y-coordinate of clipped endpoint 1.
 * @return True if line intersects window, False otherwise.
 */
bool clipLineLiangBarsky(float xmin, float xmax,
                         float ymin, float ymax,
                         float x0, float y0,
                         float x1, float y1,
                         float* x0_clip, float* y0_clip,
                         float* x1_clip, float* y1_clip);

inline bool clipLineLiangBarsky(const Point2f& top_left, const Point2f& bot_right,
                                const Point2f& x0, const Point2f& x1,
                                Point2f* x0_clip, Point2f* x1_clip) {
  return clipLineLiangBarsky(top_left.x, bot_right.x,
                             top_left.y, bot_right.y,
                             x0.x, x0.y, x1.x, x1.y,
                             &(x0_clip->x), &(x0_clip->y),
                             &(x1_clip->x), &(x1_clip->y));
}

inline bool clipLineLiangBarsky(const Point2f& top_left, const Point2f& bot_right,
                                Point2f* x0, Point2f* x1) {
  return clipLineLiangBarsky(top_left.x, bot_right.x,
                             top_left.y, bot_right.y,
                             x0->x, x0->y, x1->x, x1->y,
                             &(x0->x), &(x0->y),
                             &(x1->x), &(x1->y));
}

/**
 * @brief Interpolate a mesh of values.
 *
 * Vertices are expected in clockwise order.
 *
 * @param[in] triangles Triangle list.
 * @param[in] vertices Mesh vertices.
 * @param[in] values Values to interpolate.
 * @param[in] vtx_validity Validity of vertices.
 * @param[in] tri_validity Validity of triangles.
 * @param[out] img Output image. Must be initialized!
 */
void interpolateMesh(const std::vector<Triangle>& triangles,
                     const std::vector<cv::Point2f>& vertices,
                     const std::vector<float>& values,
                     const std::vector<bool>& vtx_validity,
                     const std::vector<bool>& tri_validity,
                     cv::Mat* img);
inline void interpolateMesh(const std::vector<Triangle>& triangles,
                            const std::vector<cv::Point2f>& vertices,
                            const std::vector<float>& values,
                            const std::vector<bool>& vtx_validity,
                            cv::Mat* img) {
  std::vector<bool> tri_validity(triangles.size(), true);
  interpolateMesh(triangles, vertices, values, vtx_validity, tri_validity, img);
  return;
}

/**
 * @brief Interpolate a mesh of inverse values.
 *
 * Interpolates the inverse of a mesh of values. Vertices are expected in
 * clockwise order.
 *
 * @param[in] triangles Triangle list.
 * @param[in] vertices Mesh vertices.
 * @param[in] ivalues Inverse values to interpolate.
 * @param[in] validity Validity of vertices.
 * @param[out] img Output image. Must be initialized!
 */
void interpolateInverseMesh(const std::vector<Triangle>& triangles,
                            const std::vector<cv::Point2f>& vertices,
                            const std::vector<float>& ivalues,
                            const std::vector<bool>& validity,
                            cv::Mat* img);

/**
 * @brief Draw a colormapped wireframe on top of an image.
 *
 * @tparam ColorMap cv::Vec3b ColorMap(float value)
 * @param[in] triangles Triangle list.
 * @param[in] vertices Mesh vertices.
 * @param[in] values Color of each vertex.
 * @param[in] vtx_validity Validity of vertices.
 * @param[in] tri_validity Validity of triangles.
 * @param[in] colormap ColorMap to apply.
 * @param[in] alpha Alpha blending ratio between source image and wireframe.
 * @param[out] img Output image. Must be initialized!
 */
template <typename ColorMap>
void drawColorMappedWireframe(const std::vector<Triangle>& triangles,
                              const std::vector<cv::Point2f>& vertices,
                              const std::vector<float>& values,
                              const std::vector<bool>& vtx_validity,
                              const std::vector<bool>& tri_validity,
                              ColorMap colormap, float alpha, cv::Mat3b* img) {
  for (int ii = 0; ii < triangles.size(); ++ii) {
    if (tri_validity[ii] &&
        vtx_validity[triangles[ii][0]] && vtx_validity[triangles[ii][1]] &&
        vtx_validity[triangles[ii][2]]) {
      applyColorMapLine(vertices[triangles[ii][0]], vertices[triangles[ii][1]],
                        values[triangles[ii][0]], values[triangles[ii][1]],
                        colormap, alpha, img);

      applyColorMapLine(vertices[triangles[ii][1]], vertices[triangles[ii][2]],
                        values[triangles[ii][1]], values[triangles[ii][2]],
                        colormap, alpha, img);

      applyColorMapLine(vertices[triangles[ii][0]], vertices[triangles[ii][2]],
                        values[triangles[ii][0]], values[triangles[ii][2]],
                        colormap, alpha, img);
    }
  }

  return;
}

}  // namespace utils

}  // namespace flame
