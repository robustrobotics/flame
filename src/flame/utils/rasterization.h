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
 * @file rasterization.h
 * @author W. Nicholas Greene
 * @date 2017-08-18 19:05:08 (Fri)
 */

#include <emmintrin.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace flame {

namespace utils {

/**
 * \brief Sort indices by increasing y-coord.
 */
inline void SortVertices(cv::Point* p1, cv::Point* p2, cv::Point* p3) {
  cv::Point tmp;
  if (p1->y > p2->y) {
    tmp = *p1;
    *p1 = *p2;
    *p2 = tmp;
  }

  if (p1->y > p3->y) {
    tmp = *p1;
    *p1 = *p3;
    *p3 = tmp;
  }

  if (p2->y > p3->y) {
    tmp = *p2;
    *p2 = *p3;
    *p3 = tmp;
  }

  return;
}

/**
 * \brief Draw an line of interpolated values.
 *
 * Draw a line between p1 and p2, interpolating values linearly from v1 to v2.
 * Assumes img is a 32-bit float image.
 */
void DrawLineInterpolated(cv::Point p1, cv::Point p2,
                          float v1, float v2, cv::Mat* img);

/**
 * \brief Draw a shaded triangle.
 *
 * Assumes the vertices are sorted in increasing y order.
 * Assumes the top of the triangle is flat such that p1.y = p2.y and that p1.x < p2.x.
 *
 * Based on:
 * http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
 */
void DrawTopFlatShadedTriangle(cv::Point p1, cv::Point p2, cv::Point p3,
                               float v1, float v2, float v3, cv::Mat* img);

/**
 * \brief Draw a shaded triangle.
 *
 * Assumes the vertices are sorted in increasing y order.
 * Assumes the bottom of the triangle is flat such that p2.y = p3.y and that p2.x < p3.x.
 *
 * Based on:
 * http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
 */
void DrawBottomFlatShadedTriangle(cv::Point p1, cv::Point p2, cv::Point p3,
                                  float v1, float v2, float v3, cv::Mat* img);

/**
 * \brief Draw a shaded triangle.
 */
void DrawShadedTriangle(cv::Point p1, cv::Point p2, cv::Point p3,
                        float v1, float v2, float v3, cv::Mat* img);

/**
 * \brief Draw a shaded triangle.
 *
 * Assumes points are in counter-clockwise order.
 *
 * Uses the barycentric approach described here:
 * https://fgiesen.wordpress.com/2013/02/06/the-barycentric-conspirac/
 */
void DrawShadedTriangleBarycentric(cv::Point p1, cv::Point p2, cv::Point p3,
                                   float v1, float v2, float v3, cv::Mat* img);

inline int orient2d(const cv::Point& a, const cv::Point& b, const cv::Point& c) {
  return (b.y - a.y)*c.x - (b.x - a.x)*c.y + (b.x*a.y - a.x*b.y);
}

inline int min3(int x, int y, int z) {
  return x < y ? (x < z ? x : z) : (y < z ? y : z);
}

inline int max3(int x, int y, int z) {
  return x > y ? (x > z ? x : z) : (y > z ? y : z);
}

struct Edge {
  static const int stepXSize = 4;
  static const int stepYSize = 1;

  // __m128 is the SSE 128-bit packed float type (4 floats).
  __m128 oneStepX;
  __m128 oneStepY;

  __m128 init(const cv::Point& v0, const cv::Point& v1,
              const cv::Point& origin) {
    // Edge setup
    float A = v1.y - v0.y;
    float B = v0.x - v1.x;
    float C = v1.x*v0.y - v0.x*v1.y;

    // Step deltas
    // __m128i y = _mm_set1_ps(x) sets y[0..3] = x.
    oneStepX = _mm_set1_ps(A*stepXSize);
    oneStepY = _mm_set1_ps(B*stepYSize);

    // x/y values for initial pixel block
    // NOTE: Set operations have arguments in reverse order!
    // __m128 y = _mm_set_epi32(x3, x2, x1, x0) sets y0 = x0, etc.
    __m128 x = _mm_set_ps(origin.x + 3, origin.x + 2, origin.x + 1, origin.x);
    __m128 y = _mm_set1_ps(origin.y);

    // Edge function values at origin
    // A*x + B*y + C.
    __m128 A4 = _mm_set1_ps(A);
    __m128 B4 = _mm_set1_ps(B);
    __m128 C4 = _mm_set1_ps(C);

    return _mm_add_ps(_mm_add_ps(_mm_mul_ps(A4, x), _mm_mul_ps(B4, y)), C4);
  }
};

}  // namespace utils

}  // namespace flame
