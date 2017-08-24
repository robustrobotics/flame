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
 * @file rasterization.cc
 * @author W. Nicholas Greene
 * @date 2017-08-18 19:05:24 (Fri)
 */

#include <stdio.h>
#include <iostream>

#include "flame/utils/rasterization.h"

namespace flame {

namespace utils {

void DrawLineInterpolated(cv::Point p1, cv::Point p2,
                          float v1, float v2, cv::Mat* img) {
  float length = sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y));
  float slope = (v2 - v1)/length;

  float val = v1;
  cv::LineIterator it(*img, p1, p2);
  int count = it.count;

  for (int ii = 0; ii < count; ++ii) {
    float* ptr = reinterpret_cast<float*>(*it);
    *ptr = val;
    val += slope;
    it++;
  }

  return;
}

void DrawTopFlatShadedTriangle(cv::Point p1, cv::Point p2, cv::Point p3,
                               float v1, float v2, float v3, cv::Mat*img) {
  // Check for special case (three vertices are adjacent, don't waste time
  // interpolating).
  if ((abs(p2.x - p1.x) == 1) && (abs(p3.y - p2.y) == 1)) {
    img->at<float>(p1.y, p1.x) = v1;
    img->at<float>(p2.y, p2.x) = v2;
    img->at<float>(p3.y, p3.x) = v3;
    return;
  }

  cv::LineIterator it1(*img, p1, p3);
  cv::LineIterator it2(*img, p2, p3);

  float slope1 = (v3 - v1)/(p3.y - p1.y);
  float slope2 = (v3 - v2)/(p3.y - p2.y);

  float val1 = v1;
  float val2 = v2;

  for (int y = p1.y; y <= p3.y; ++y) {
    // Walk along the line from p1-p2 until we get to y.
    while (it1.pos().y < y) {
      it1++;
    }

    // Walk along the line from p2-p3 until we get to y.
    while (it2.pos().y < y) {
      it2++;
    }

    // Fill this row of the triangle.
    DrawLineInterpolated(it1.pos(), it2.pos(), val1, val2, img);

    // Increment values.
    val1 += slope1;
    val2 += slope2;
  }

  return;
}

void DrawBottomFlatShadedTriangle(cv::Point p1, cv::Point p2, cv::Point p3,
                                  float v1, float v2, float v3, cv::Mat*img) {
  // Check for special case (three vertices are adjacent, don't waste time
  // interpolating).
  if ((abs(p2.y - p1.y) == 1) && (abs(p3.x - p2.x) == 1)) {
    img->at<float>(p1.y, p1.x) = v1;
    img->at<float>(p2.y, p2.x) = v2;
    img->at<float>(p3.y, p3.x) = v3;
    return;
  }

  cv::LineIterator it1(*img, p1, p2);
  cv::LineIterator it2(*img, p1, p3);

  float slope1 = (v2 - v1)/(p2.y - p1.y);
  float slope2 = (v3 - v1)/(p3.y - p1.y);

  float val1 = v1;
  float val2 = v1;

  for (int y = p1.y; y <= p3.y; ++y) {
    // Walk along the line from p1-p2 until we get to y.
    while (it1.pos().y < y) {
      it1++;
    }

    // Walk along the line from p2-p3 until we get to y.
    while (it2.pos().y < y) {
      it2++;
    }

    // Fill this row of the triangle.
    DrawLineInterpolated(it1.pos(), it2.pos(), val1, val2, img);

    // Increment values.
    val1 += slope1;
    val2 += slope2;
  }

  return;
}

void DrawShadedTriangle(cv::Point p1, cv::Point p2, cv::Point p3,
                        float v1, float v2, float v3, cv::Mat* img) {
  SortVertices(&p1, &p2, &p3);

  if (p1.y == p2.y) {
    DrawTopFlatShadedTriangle(p1, p2, p3, v1, v2, v3, img);
  } else if (p2.y == p3.y) {
    DrawBottomFlatShadedTriangle(p1, p2, p3, v1, v2, v3, img);
  } else {
    // Split triangle and draw the top and bottom halves.
    float point_slope = static_cast<float>(p3.x - p1.x) / (p3.y - p1.y);
    int x4 = static_cast<int>(p1.x + point_slope * (p2.y - p1.y));
    cv::Point p4(x4, p2.y);

    float length = sqrt((p3.x - p1.x)*(p3.x - p1.x) + (p3.y - p1.y)*(p3.y - p1.y));
    float val_slope = (v3 - v1) / length;
    float l = sqrt((p3.x - p4.x)*(p3.x - p4.x) + (p3.y - p4.y)*(p3.y - p4.y));
    float v4 = v1 + val_slope * l;

    // printf("v4 = %f\n", v4);
    // printf("val_slope = %f, l = %f, length = %f\n", val_slope, l, length);

    DrawBottomFlatShadedTriangle(p1, p2, p4, v1, v2, v4, img);
    DrawTopFlatShadedTriangle(p2, p4, p3, v2, v4, v3, img);
  }

  return;
}

void DrawShadedTriangleBarycentric(cv::Point p1, cv::Point p2, cv::Point p3,
                                   float v1, float v2, float v3, cv::Mat* img) {
  // Compute triangle bounding box
  int xmin = min3(p1.x, p2.x, p3.x);
  int ymin = min3(p1.y, p2.y, p3.y);
  int xmax = max3(p1.x, p2.x, p3.x);
  int ymax = max3(p1.y, p2.y, p3.y);

  cv::Point p(xmin, ymin);
  Edge e12, e23, e31;

  // __m128 is the SSE 128-bit packed float type (4 floats).
  __m128 w1_row = e23.init(p2, p3, p);
  __m128 w2_row = e31.init(p3, p1, p);
  __m128 w3_row = e12.init(p1, p2, p);

  // Values as 4 packed floats.
  __m128 v14 = _mm_set1_ps(v1);
  __m128 v24 = _mm_set1_ps(v2);
  __m128 v34 = _mm_set1_ps(v3);

  // Rasterize
  for (p.y = ymin; p.y <= ymax; p.y += Edge::stepYSize) {
    // Determine barycentric coordinates
    __m128 w1 = w1_row;
    __m128 w2 = w2_row;
    __m128 w3 = w3_row;

    for (p.x = xmin; p.x <= xmax; p.x += Edge::stepXSize) {
      // If p is on or inside all edges, render pixel.
      __m128 zero = _mm_set1_ps(0.0f);

      // (w1 >= 0) && (w2 >= 0) && (w3 >= 0)
      // mask tells whether we should set the pixel.
      __m128 mask = _mm_and_ps(_mm_cmpge_ps(w1, zero),
                               _mm_and_ps(_mm_cmpge_ps(w2, zero),
                                          _mm_cmpge_ps(w3, zero)));

      // w1 + w2 + w3
      __m128 norm = _mm_add_ps(w1, _mm_add_ps(w2, w3));

      // v1*w1 + v2*w2 + v3*w3 / norm
      __m128 vals = _mm_div_ps(_mm_add_ps(_mm_mul_ps(v14, w1),
                                          _mm_add_ps(_mm_mul_ps(v24, w2),
                                                     _mm_mul_ps(v34, w3))), norm);

      // Grab original data.  We need to use different store/load functions if
      // the address is not aligned to 16-bytes.
      uint32_t addr = sizeof(float)*(p.y*img->cols + p.x);
      if (addr % 16 == 0) {
        float* img_ptr = reinterpret_cast<float*>(&(img->data[addr]));
        __m128 data = _mm_load_ps(img_ptr);

        // Set values using mask.
        // If mask is true, use vals, otherwise use data.
        __m128 res = _mm_or_ps(_mm_and_ps(mask, vals), _mm_andnot_ps(mask, data));
        _mm_store_ps(img_ptr, res);
      } else {
        // Address is not 16-byte aligned. Need to use special functions to load/store.
        float* img_ptr = reinterpret_cast<float*>(&(img->data[addr]));
        __m128 data = _mm_loadu_ps(img_ptr);

        // Set values using mask.
        // If mask is true, use vals, otherwise use data.
        __m128 res = _mm_or_ps(_mm_and_ps(mask, vals), _mm_andnot_ps(mask, data));
        _mm_storeu_ps(img_ptr, res);
      }

      // One step to the right.
      w1 = _mm_add_ps(w1, e23.oneStepX);
      w2 = _mm_add_ps(w2, e31.oneStepX);
      w3 = _mm_add_ps(w3, e12.oneStepX);
    }

    // Row step.
    w1_row = _mm_add_ps(w1_row, e23.oneStepY);
    w2_row = _mm_add_ps(w2_row, e31.oneStepY);
    w3_row = _mm_add_ps(w3_row, e12.oneStepY);
  }

  return;
}

}  // namespace utils

}  // namespace flame
