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
 * @file image_utils_test.cc
 * @author W. Nicholas Greene
 * @date 2017-06-16 11:30:14 (Fri)
 */

#include <stdio.h>

#include <iostream>
#include <chrono>

#include <Eigen/Core>

#include <opencv2/highgui/highgui.hpp>

#include "gtest/gtest.h"

#include "flame/utils/image_utils.h"
#include "flame/utils/stats_tracker.h"

namespace flame {

namespace utils {

namespace  {

typedef std::chrono::high_resolution_clock clock;
typedef std::chrono::duration<double, std::milli> msec;

TEST(ImageUtilsTest, insideTriangleTest) {
  cv::Point2f v0(0, 0);
  cv::Point2f v1(1, 0);
  cv::Point2f v2(1, 1);

  float alpha, beta, gamma, area2;

  // Inside tests.
  barycentricCoords<float>(v0, v1, v2, cv::Point2f(0.75f, 0.25f),
                           &alpha, &beta, &gamma, &area2);
  EXPECT_TRUE(insideTriangle<float>(alpha, beta, gamma));
  EXPECT_FALSE(onTriangleBorder<float>(alpha, beta, gamma, area2));

  barycentricCoords<float>(v0, v1, v2, cv::Point2f(0.75f, 0.5f),
                           &alpha, &beta, &gamma, &area2);
  EXPECT_TRUE(insideTriangle<float>(alpha, beta, gamma));
  EXPECT_FALSE(onTriangleBorder<float>(alpha, beta, gamma, area2));

  // Outside tests.
  barycentricCoords<float>(v0, v1, v2, cv::Point2f(0.25f, 0.75f),
                           &alpha, &beta, &gamma, &area2);
  EXPECT_FALSE(insideTriangle<float>(alpha, beta, gamma));
  EXPECT_FALSE(onTriangleBorder<float>(alpha, beta, gamma, area2));

  barycentricCoords<float>(v0, v1, v2, cv::Point2f(0.0f, 0.5f),
                           &alpha, &beta, &gamma, &area2);
  EXPECT_FALSE(insideTriangle<float>(alpha, beta, gamma));
  EXPECT_FALSE(onTriangleBorder<float>(alpha, beta, gamma, area2));

  barycentricCoords<float>(v0, v1, v2, cv::Point2f(0.0f, 1.0f),
                           &alpha, &beta, &gamma, &area2);
  EXPECT_FALSE(insideTriangle<float>(alpha, beta, gamma));
  EXPECT_FALSE(onTriangleBorder<float>(alpha, beta, gamma, area2));

  // Corner tests.
  barycentricCoords<float>(v0, v1, v2, cv::Point2f(0.0f, 0.0f),
                           &alpha, &beta, &gamma, &area2);
  EXPECT_FALSE(insideTriangle<float>(alpha, beta, gamma));
  EXPECT_TRUE(onTriangleBorder<float>(alpha, beta, gamma, area2));

  barycentricCoords<float>(v0, v1, v2, cv::Point2f(1.0f, 1.0f),
                           &alpha, &beta, &gamma, &area2);
  EXPECT_FALSE(insideTriangle<float>(alpha, beta, gamma));
  EXPECT_TRUE(onTriangleBorder<float>(alpha, beta, gamma, area2));

  // Edge tests.
  barycentricCoords<float>(v0, v1, v2, cv::Point2f(0.5f, 0.0f),
                           &alpha, &beta, &gamma, &area2);
  EXPECT_FALSE(insideTriangle<float>(alpha, beta, gamma));
  EXPECT_TRUE(onTriangleBorder<float>(alpha, beta, gamma, area2));

  barycentricCoords<float>(v0, v1, v2, cv::Point2f(1.0f, 0.5f),
                           &alpha, &beta, &gamma, &area2);
  EXPECT_FALSE(insideTriangle<float>(alpha, beta, gamma));
  EXPECT_TRUE(onTriangleBorder<float>(alpha, beta, gamma, area2));

  barycentricCoords<float>(v0, v1, v2, cv::Point2f(0.5f, 0.5f),
                           &alpha, &beta, &gamma, &area2);
  EXPECT_FALSE(insideTriangle<float>(alpha, beta, gamma));
  EXPECT_TRUE(onTriangleBorder<float>(alpha, beta, gamma, area2));

  barycentricCoords<float>(v0, v1, v2, cv::Point2f(0.75f, 0.75f),
                           &alpha, &beta, &gamma, &area2);
  EXPECT_FALSE(insideTriangle<float>(alpha, beta, gamma));
  EXPECT_TRUE(onTriangleBorder<float>(alpha, beta, gamma, area2));

  return;
}

TEST(ImageUtilsTest, insideCircumCircleTest) {
  cv::Point2f A(0, 0);
  cv::Point2f B(1, 0);
  cv::Point2f C(1, 1);

  EXPECT_TRUE(insideCircumCircle<float>(A, B, C, (A + B + C) * 0.3333));

  float theta = 2*M_PI/3;
  A.x = cos(0);
  A.y = sin(0);
  B.x = cos(theta);
  B.y = sin(theta);
  C.x = cos(2*theta);
  C.y = sin(2*theta);
  EXPECT_TRUE(insideCircumCircle<float>(A, B, C, cv::Point2f(0.0f, 0.0f)));
  EXPECT_TRUE(insideCircumCircle<float>(A, B, C, cv::Point2f(0.95f, 0.0f)));
  EXPECT_TRUE(insideCircumCircle<float>(A, B, C, cv::Point2f(0.0f, 0.95f)));
  EXPECT_TRUE(insideCircumCircle<float>(A, B, C, cv::Point2f(-0.95f, 0.0f)));
  EXPECT_TRUE(insideCircumCircle<float>(A, B, C, cv::Point2f(0.0f, -0.95f)));

  EXPECT_FALSE(insideCircumCircle<float>(A, B, C, cv::Point2f(2, 0)));
  EXPECT_FALSE(insideCircumCircle<float>(A, B, C, cv::Point2f(-2, 0)));
  EXPECT_FALSE(insideCircumCircle<float>(A, B, C, cv::Point2f(0.95f, 0.95f)));
  EXPECT_FALSE(insideCircumCircle<float>(A, B, C, cv::Point2f(-0.95f, -0.95f)));
  EXPECT_FALSE(insideCircumCircle<float>(A, B, C, cv::Point2f(1.1f, 0.0f)));
  EXPECT_FALSE(insideCircumCircle<float>(A, B, C, cv::Point2f(-1.1f, 0.0f)));
  EXPECT_FALSE(insideCircumCircle<float>(A, B, C, cv::Point2f(0.0f, 1.1f)));
  EXPECT_FALSE(insideCircumCircle<float>(A, B, C, cv::Point2f(0.0f, -1.1f)));

  return;
}

/**
 * @brief Test basic bilinear interpolation using OpenCV container.
 */
TEST(ImageUtilsTest, bilinearInterpCV) {
  cv::Mat img = (cv::Mat_<uint8_t>(2, 2) << 91, 210, 162, 95);

  float ret = bilinearInterp<uint8_t, float>(img, 0.5, 0.2);
  EXPECT_NEAR(146.1, ret, 1e-5);

  ret = bilinearInterp<uint8_t, float>(img, 0.2, 0.5);
  EXPECT_NEAR(131.70001, ret, 1e-5);

  ret = bilinearInterp<uint8_t, float>(img, 0.5, 0.5);
  EXPECT_NEAR(139.5, ret, 1e-5);

  ret = bilinearInterp<uint8_t, float>(img, 0.2, 0.2);
  EXPECT_NEAR(121.5600052, ret, 1e-5);

  return;
}

/**
 * @brief Test central gradient with image that increases horizontally.
 */
TEST(ImageUtilsTest, getCentralGradientHorizontal) {
  int width = 640;
  int height = 480;
  cv::Mat1b img(height, width);
  cv::Mat1f gradx(height, width);
  cv::Mat1f grady(height, width);

  float slope = 255.0f / width;
  for (int ii = 0; ii < height; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      img.at<uint8_t>(ii, jj) = fast_roundf(slope * jj);
    }
  }

  // Compute central gradient.
  StatsTracker stats;
  stats.tick("getCentralGradient");

  getCentralGradient<uint8_t, float>(img, &gradx, &grady);

  stats.tock("getCentralGradient");

  printf("getCentralGradient = %f\n", stats.timings("getCentralGradient"));

  // Check output.
  for (uint32_t ii = 0; ii < height; ++ii) {
    for (uint32_t jj = 1; jj < width - 1; ++jj) {
      float left = img.at<uint8_t>(ii, jj - 1);
      float right = img.at<uint8_t>(ii, jj + 1);
      float expx = 0.5 * (right - left);
      EXPECT_NEAR(expx, gradx(ii, jj), 1e-6) << "(" << ii << ", " << jj << ")";
    }
  }
  for (uint32_t ii = 0; ii < height; ++ii) {
    // First column (forward difference).
    float left = img.at<uint8_t>(ii, 0);
    float right = img.at<uint8_t>(ii, 1);
    float expx = right - left;
    EXPECT_NEAR(expx, gradx(ii, 0), 1e-6);

    // Last column (backward difference).
    left = img.at<uint8_t>(ii, width - 2);
    right = img.at<uint8_t>(ii, width - 1);
    expx = right - left;
    EXPECT_NEAR(expx, gradx(ii, width - 1), 1e-6);
  }

  for (int ii = 1; ii < height - 1; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      float up = img.at<uint8_t>(ii - 1, jj);
      float down = img.at<uint8_t>(ii + 1, jj);
      float expy = 0.5 * (down - up);
      EXPECT_NEAR(expy, grady(ii, jj), 1e-6);
    }
  }
  for (uint32_t ii = 0; ii < width; ++ii) {
    // First row (forward difference).
    float up = img.at<uint8_t>(0, ii);
    float down = img.at<uint8_t>(1, ii);
    float expy = down - up;
    EXPECT_NEAR(expy, grady(0, ii), 1e-6);

    // Last row (backward difference).
    up = img.at<uint8_t>(height - 2, ii);
    down = img.at<uint8_t>(height - 1, ii);
    expy = down - up;
    EXPECT_NEAR(expy, grady(height - 1, ii), 1e-6);
  }

  // cv::imshow("getCentralGradientHorizontal", img);
  // cv::waitKey(0);

  return;
}

/**
 * @brief Test central gradient with image that increases vertically.
 */
TEST(ImageUtilsTest, getCentralGradientVertical) {
  int width = 640;
  int height = 480;
  cv::Mat1b img(height, width);
  cv::Mat1f gradx(height, width);
  cv::Mat1f grady(height, width);

  float slope = 255.0f / height;
  for (int ii = 0; ii < height; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      img.at<uint8_t>(ii, jj) = fast_roundf(slope * ii);
    }
  }

  // Compute central gradient.
  StatsTracker stats;
  stats.tick("getCentralGradient");

  getCentralGradient<uint8_t, float>(img, &gradx, &grady);

  stats.tock("getCentralGradient");

  printf("getCentralGradient = %f\n", stats.timings("getCentralGradient"));

  // Check output.
  for (uint32_t ii = 0; ii < height; ++ii) {
    for (uint32_t jj = 1; jj < width - 1; ++jj) {
      float left = img.at<uint8_t>(ii, jj - 1);
      float right = img.at<uint8_t>(ii, jj + 1);
      float expx = 0.5 * (right - left);
      EXPECT_NEAR(expx, gradx(ii, jj), 1e-6) << "(" << ii << ", " << jj << ")";
    }
  }
  for (uint32_t ii = 0; ii < height; ++ii) {
    // First column (forward difference).
    float left = img.at<uint8_t>(ii, 0);
    float right = img.at<uint8_t>(ii, 1);
    float expx = right - left;
    EXPECT_NEAR(expx, gradx(ii, 0), 1e-6);

    // Last column (backward difference).
    left = img.at<uint8_t>(ii, width - 2);
    right = img.at<uint8_t>(ii, width - 1);
    expx = right - left;
    EXPECT_NEAR(expx, gradx(ii, width - 1), 1e-6);
  }

  for (int ii = 1; ii < height - 1; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      float up = img.at<uint8_t>(ii - 1, jj);
      float down = img.at<uint8_t>(ii + 1, jj);
      float expy = 0.5 * (down - up);
      EXPECT_NEAR(expy, grady(ii, jj), 1e-6);
    }
  }
  for (uint32_t ii = 0; ii < width; ++ii) {
    // First row (forward difference).
    float up = img.at<uint8_t>(0, ii);
    float down = img.at<uint8_t>(1, ii);
    float expy = down - up;
    EXPECT_NEAR(expy, grady(0, ii), 1e-6);

    // Last row (backward difference).
    up = img.at<uint8_t>(height - 2, ii);
    down = img.at<uint8_t>(height - 1, ii);
    expy = down - up;
    EXPECT_NEAR(expy, grady(height - 1, ii), 1e-6);
  }

  // cv::imshow("getCentralGradientVertical", img);
  // cv::waitKey(0);

  return;
}

/**
 * @brief Test central gradient with uint8_t image that increases
 * horizontally. Uses SSE optimized code.
 */
TEST(ImageUtilsTest, getCentralGradientHorizontalSSEb) {
  int width = 640;
  int height = 480;

  cv::Mat1b img(height, width);
  cv::Mat1f gradx(height, width);
  cv::Mat1f grady(height, width);

  float slope = 255.0f / width;
  for (int ii = 0; ii < height; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      img(ii, jj) = fast_roundf(slope * jj);
    }
  }

  // Compute central gradient.
  StatsTracker stats;
  stats.tick("getCentralGradient");

  getCentralGradient<uint8_t, float>(img, &gradx, &grady);

  stats.tock("getCentralGradient");

  printf("getCentralGradient = %f\n", stats.timings("getCentralGradient"));

  // Check output.
  for (uint32_t ii = 0; ii < height; ++ii) {
    for (uint32_t jj = 1; jj < width - 1; ++jj) {
      float left = img.at<uint8_t>(ii, jj - 1);
      float right = img.at<uint8_t>(ii, jj + 1);
      float expx = 0.5 * (right - left);
      EXPECT_NEAR(expx, gradx(ii, jj), 1e-6) << "(" << ii << ", " << jj << ")";
    }
  }
  for (uint32_t ii = 0; ii < height; ++ii) {
    // First column (forward difference).
    float left = img.at<uint8_t>(ii, 0);
    float right = img.at<uint8_t>(ii, 1);
    float expx = right - left;
    EXPECT_NEAR(expx, gradx(ii, 0), 1e-6);

    // Last column (backward difference).
    left = img.at<uint8_t>(ii, width - 2);
    right = img.at<uint8_t>(ii, width - 1);
    expx = right - left;
    EXPECT_NEAR(expx, gradx(ii, width - 1), 1e-6);
  }

  for (int ii = 1; ii < height - 1; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      float up = img.at<uint8_t>(ii - 1, jj);
      float down = img.at<uint8_t>(ii + 1, jj);
      float expy = 0.5 * (down - up);
      EXPECT_NEAR(expy, grady(ii, jj), 1e-6);
    }
  }
  for (uint32_t ii = 0; ii < width; ++ii) {
    // First row (forward difference).
    float up = img.at<uint8_t>(0, ii);
    float down = img.at<uint8_t>(1, ii);
    float expy = down - up;
    EXPECT_NEAR(expy, grady(0, ii), 1e-6);

    // Last row (backward difference).
    up = img.at<uint8_t>(height - 2, ii);
    down = img.at<uint8_t>(height - 1, ii);
    expy = down - up;
    EXPECT_NEAR(expy, grady(height - 1, ii), 1e-6);
  }

  // cv::imshow("getCentralGradientHorizontal", img);
  // cv::waitKey(0);

  return;
}

/**
 * @brief Test central gradient with float image that increases
 * horizontally. Uses SSE optimized code.
 */
TEST(ImageUtilsTest, getCentralGradientHorizontalSSEf) {
  int width = 640;
  int height = 480;

  cv::Mat1f img(height, width);
  cv::Mat1f gradx(height, width);
  cv::Mat1f grady(height, width);

  float slope = 255.0f / width;
  for (int ii = 0; ii < height; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      img(ii, jj) = fast_roundf(slope * jj);
    }
  }

  // Compute central gradient.
  StatsTracker stats;
  stats.tick("getCentralGradient");

  getCentralGradient<float, float>(img, &gradx, &grady);

  stats.tock("getCentralGradient");

  printf("getCentralGradient = %f\n", stats.timings("getCentralGradient"));

  // Check output.
  for (uint32_t ii = 0; ii < height; ++ii) {
    for (uint32_t jj = 1; jj < width - 1; ++jj) {
      float left = img(ii, jj - 1);
      float right = img(ii, jj + 1);
      float expx = 0.5 * (right - left);
      EXPECT_NEAR(expx, gradx(ii, jj), 1e-6) << "(" << ii << ", " << jj << ")";
    }
  }
  for (uint32_t ii = 0; ii < height; ++ii) {
    // First column (forward difference).
    float left = img(ii, 0);
    float right = img(ii, 1);
    float expx = right - left;
    EXPECT_NEAR(expx, gradx(ii, 0), 1e-6);

    // Last column (backward difference).
    left = img(ii, width - 2);
    right = img(ii, width - 1);
    expx = right - left;
    EXPECT_NEAR(expx, gradx(ii, width - 1), 1e-6);
  }

  for (int ii = 1; ii < height - 1; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      float up = img(ii - 1, jj);
      float down = img(ii + 1, jj);
      float expy = 0.5 * (down - up);
      EXPECT_NEAR(expy, grady(ii, jj), 1e-6);
    }
  }
  for (uint32_t ii = 0; ii < width; ++ii) {
    // First row (forward difference).
    float up = img(0, ii);
    float down = img(1, ii);
    float expy = down - up;
    EXPECT_NEAR(expy, grady(0, ii), 1e-6);

    // Last row (backward difference).
    up = img(height - 2, ii);
    down = img(height - 1, ii);
    expy = down - up;
    EXPECT_NEAR(expy, grady(height - 1, ii), 1e-6);
  }

  // cv::imshow("getCentralGradientHorizontal", img);
  // cv::waitKey(0);

  return;
}

/**
 * @brief Test central gradient with uint8_t image that increases
 * vertically. Uses SSE optimized code.
 */
TEST(ImageUtilsTest, getCentralGradientVerticalSSEb) {
  int width = 640;
  int height = 480;

  cv::Mat1b img(height, width);
  cv::Mat1f gradx(height, width);
  cv::Mat1f grady(height, width);

  float slope = 255.0f / width;
  for (int ii = 0; ii < height; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      img(ii, jj) = fast_roundf(slope * jj);
    }
  }

  // Compute central gradient.
  StatsTracker stats;
  stats.tick("getCentralGradient");

  getCentralGradient<uint8_t, float>(img, &gradx, &grady);

  stats.tock("getCentralGradient");

  printf("getCentralGradient = %f\n", stats.timings("getCentralGradient"));

  // Check output.
  for (uint32_t ii = 0; ii < height; ++ii) {
    for (uint32_t jj = 1; jj < width - 1; ++jj) {
      float left = img.at<uint8_t>(ii, jj - 1);
      float right = img.at<uint8_t>(ii, jj + 1);
      float expx = 0.5 * (right - left);
      EXPECT_NEAR(expx, gradx(ii, jj), 1e-6) << "(" << ii << ", " << jj << ")";
    }
  }
  for (uint32_t ii = 0; ii < height; ++ii) {
    // First column (forward difference).
    float left = img.at<uint8_t>(ii, 0);
    float right = img.at<uint8_t>(ii, 1);
    float expx = right - left;
    EXPECT_NEAR(expx, gradx(ii, 0), 1e-6);

    // Last column (backward difference).
    left = img.at<uint8_t>(ii, width - 2);
    right = img.at<uint8_t>(ii, width - 1);
    expx = right - left;
    EXPECT_NEAR(expx, gradx(ii, width - 1), 1e-6);
  }

  for (int ii = 1; ii < height - 1; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      float up = img.at<uint8_t>(ii - 1, jj);
      float down = img.at<uint8_t>(ii + 1, jj);
      float expy = 0.5 * (down - up);
      EXPECT_NEAR(expy, grady(ii, jj), 1e-6);
    }
  }
  for (uint32_t ii = 0; ii < width; ++ii) {
    // First row (forward difference).
    float up = img.at<uint8_t>(0, ii);
    float down = img.at<uint8_t>(1, ii);
    float expy = down - up;
    EXPECT_NEAR(expy, grady(0, ii), 1e-6);

    // Last row (backward difference).
    up = img.at<uint8_t>(height - 2, ii);
    down = img.at<uint8_t>(height - 1, ii);
    expy = down - up;
    EXPECT_NEAR(expy, grady(height - 1, ii), 1e-6);
  }

  // cv::imshow("getCentralGradientVertical", img);
  // cv::waitKey(0);

  return;
}

/**
 * @brief Test central gradient with float image that increases vertically. Uses
 * SSE optimized code.
 */
TEST(ImageUtilsTest, getCentralGradientVerticalSSEf) {
  int width = 640;
  int height = 480;

  cv::Mat1f img(height, width);
  cv::Mat1f gradx(height, width);
  cv::Mat1f grady(height, width);

  float slope = 255.0f / width;
  for (int ii = 0; ii < height; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      img(ii, jj) = fast_roundf(slope * jj);
    }
  }

  // Compute central gradient.
  StatsTracker stats;
  stats.tick("getCentralGradient");

  getCentralGradient<float, float>(img, &gradx, &grady);

  stats.tock("getCentralGradient");

  printf("getCentralGradient = %f\n", stats.timings("getCentralGradient"));

  // Check output.
  for (uint32_t ii = 0; ii < height; ++ii) {
    for (uint32_t jj = 1; jj < width - 1; ++jj) {
      float left = img(ii, jj - 1);
      float right = img(ii, jj + 1);
      float expx = 0.5 * (right - left);
      EXPECT_NEAR(expx, gradx(ii, jj), 1e-6) << "(" << ii << ", " << jj << ")";
    }
  }
  for (uint32_t ii = 0; ii < height; ++ii) {
    // First column (forward difference).
    float left = img(ii, 0);
    float right = img(ii, 1);
    float expx = right - left;
    EXPECT_NEAR(expx, gradx(ii, 0), 1e-6);

    // Last column (backward difference).
    left = img(ii, width - 2);
    right = img(ii, width - 1);
    expx = right - left;
    EXPECT_NEAR(expx, gradx(ii, width - 1), 1e-6);
  }

  for (int ii = 1; ii < height - 1; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      float up = img(ii - 1, jj);
      float down = img(ii + 1, jj);
      float expy = 0.5 * (down - up);
      EXPECT_NEAR(expy, grady(ii, jj), 1e-6);
    }
  }
  for (uint32_t ii = 0; ii < width; ++ii) {
    // First row (forward difference).
    float up = img(0, ii);
    float down = img(1, ii);
    float expy = down - up;
    EXPECT_NEAR(expy, grady(0, ii), 1e-6);

    // Last row (backward difference).
    up = img(height - 2, ii);
    down = img(height - 1, ii);
    expy = down - up;
    EXPECT_NEAR(expy, grady(height - 1, ii), 1e-6);
  }

  // cv::imshow("getCentralGradientVertical", img);
  // cv::waitKey(0);

  return;
}

TEST(ImageUtilsTest, LiangBarkyInside) {

  float x0 = 1.0f;
  float y0 = 1.0f;
  float x1 = 2.0f;
  float y1 = 2.0f;

  float x0_clip, y0_clip;
  float x1_clip, y1_clip;

  EXPECT_TRUE(clipLineLiangBarsky(0, 3, 0, 3,
                                  x0, y0,
                                  x1, y1,
                                  &x0_clip, &y0_clip,
                                  &x1_clip, &y1_clip));

  float tol = 1e-6;
  EXPECT_NEAR(x0, x0_clip, tol);
  EXPECT_NEAR(y0, y0_clip, tol);
  EXPECT_NEAR(x1, x1_clip, tol);
  EXPECT_NEAR(y1, y1_clip, tol);
}

TEST(ImageUtilsTest, LiangBarskyXY0outside) {
  float x0 = 1.0f;
  float y0 = 1.0f;
  float x1 = 2.0f;
  float y1 = 2.0f;

  float x0_clip, y0_clip;
  float x1_clip, y1_clip;

  EXPECT_TRUE(clipLineLiangBarsky(1.1, 3, 1.2, 3,
                                  x0, y0,
                                  x1, y1,
                                  &x0_clip, &y0_clip,
                                  &x1_clip, &y1_clip));
  float tol = 1e-6;
  EXPECT_NEAR(1.2, x0_clip, tol);
  EXPECT_NEAR(1.2, y0_clip, tol);
  EXPECT_NEAR(x1, x1_clip, tol);
  EXPECT_NEAR(y1, y1_clip, tol);
}

TEST(ImageUtilsTest, LiangBarskyXY1outside) {
  float x0 = 1.0f;
  float y0 = 1.0f;
  float x1 = 2.0f;
  float y1 = 2.0f;

  float x0_clip, y0_clip;
  float x1_clip, y1_clip;

  EXPECT_TRUE(clipLineLiangBarsky(1, 1.5, 1, 1.25,
                                  x0, y0,
                                  x1, y1,
                                  &x0_clip, &y0_clip,
                                  &x1_clip, &y1_clip));
  float tol = 1e-6;
  EXPECT_NEAR(x0, x0_clip, tol);
  EXPECT_NEAR(y0, y0_clip, tol);
  EXPECT_NEAR(1.25, x1_clip, tol);
  EXPECT_NEAR(1.25, y1_clip, tol);
}

TEST(ImageUtilsTest, LiangBarskyBothoutside) {
  float x0 = 1.0f;
  float y0 = 1.0f;
  float x1 = 2.0f;
  float y1 = 2.0f;

  float x0_clip, y0_clip;
  float x1_clip, y1_clip;

  EXPECT_TRUE(clipLineLiangBarsky(1.1, 1.5, 1.15, 1.25,
                                  x0, y0,
                                  x1, y1,
                                  &x0_clip, &y0_clip,
                                  &x1_clip, &y1_clip));
  float tol = 1e-6;
  EXPECT_NEAR(1.15, x0_clip, tol);
  EXPECT_NEAR(1.15, y0_clip, tol);
  EXPECT_NEAR(1.25, x1_clip, tol);
  EXPECT_NEAR(1.25, y1_clip, tol);
}

TEST(ImageUtilsTest, LiangBarskyFalse) {
  float x0 = -3.40414f;
  float y0 = 1.95745f;
  float x1 = -3.26889f;
  float y1 = 2.17682f;

  float xmin = 2 + 1e-3 + 1;
  float xmax = 640 - 2 - 1e-3 - 2;
  float ymin = 2 + 1e-3 + 1;
  float ymax = 480 - 2 - 1e-3 - 2;

  float x0_clip, y0_clip;
  float x1_clip, y1_clip;
  EXPECT_FALSE(clipLineLiangBarsky(xmin, xmax,
                                   ymin, ymax,
                                   x0, y0,
                                   x1, y1,
                                   &x0_clip, &y0_clip,
                                   &x1_clip, &y1_clip));
}

/**
 * \brief Interpolate a simple mesh of values.
 */
TEST(ImageUtilsTest, interpolateMeshTest) {
  std::vector<Triangle> triangles = { Triangle(0, 1, 2) };
  std::vector<cv::Point2f> vertices(3);
  vertices[0] = cv::Point2f(10, 10);
  vertices[1] = cv::Point2f(20, 10);
  vertices[2] = cv::Point2f(10, 20);

  std::vector<float> values = { 1.0f, 2.0f, 3.0f };
  std::vector<bool> validity = { true, true, true };

  cv::Mat1f img(40, 40, std::numeric_limits<float>::quiet_NaN());

  interpolateMesh(triangles, vertices, values, validity, &img);

  float tol = 1e-3;
  EXPECT_NEAR(1.0f, img(10, 10), tol);
  EXPECT_NEAR(2.0f, img(10, 20), tol);
  EXPECT_NEAR(3.0f, img(20, 10), tol);

  // Test center of triangle.
  cv::Point2f center(15, 15);

  cv::Point2f gradient((values[1] - values[0]) / 10,
                       (values[2] - values[0]) / 10);
  float val = gradient.x * (center.x - vertices[0].x) +
      gradient.y * (center.y - vertices[0].y) + values[0];

  EXPECT_NEAR(val, img(center.y, center.x), tol);
}

/**
 * \brief Interpolate the inverse of a simple mesh of values.
 */
TEST(ImageUtilsTest, interpolateInverseMeshTest) {
  std::vector<Triangle> triangles = { Triangle(0, 1, 2) };
  std::vector<cv::Point2f> vertices(3);
  vertices[0] = cv::Point2f(10, 10);
  vertices[1] = cv::Point2f(20, 10);
  vertices[2] = cv::Point2f(10, 20);

  std::vector<float> values = { 1.0f, 2.0f, 3.0f };
  std::vector<float> ivalues = { 1.0f/1.0f, 1.0f/2.0f, 1.0f/3.0f };
  std::vector<bool> validity = { true, true, true };

  cv::Mat1f img(40, 40, std::numeric_limits<float>::quiet_NaN());

  interpolateInverseMesh(triangles, vertices, ivalues, validity, &img);

  float tol = 1e-3;
  EXPECT_NEAR(1.0f, img(10, 10), tol);
  EXPECT_NEAR(2.0f, img(10, 20), tol);
  EXPECT_NEAR(3.0f, img(20, 10), tol);

  // Test center of triangle.
  cv::Point2f center(15, 15);

  cv::Point2f gradient((values[1] - values[0]) / 10,
                       (values[2] - values[0]) / 10);
  float val = gradient.x * (center.x - vertices[0].x) +
      gradient.y * (center.y - vertices[0].y) + values[0];

  EXPECT_NEAR(val, img(center.y, center.x), tol);
}

}  // namespace

}  // namespace utils

}  // namespace flame
