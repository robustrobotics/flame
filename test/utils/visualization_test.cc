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
 * @file visualization_test.cc
 * @author W. Nicholas Greene
 * @date 2017-08-18 19:15:38 (Fri)
 */

#include "gtest/gtest.h"

#include "flame/utils/visualization.h"

namespace flame {

namespace utils {

TEST(VisualizationTest, blendColorTest1) {
  cv::Vec3b red(0, 0, 255);
  cv::Vec3b green(0, 255, 0);
  cv::Vec3b blend = blendColor(red, green, -1.0f, 0.0f, 1.0f);

  EXPECT_EQ(red[0], blend[0]);
  EXPECT_EQ(red[1], blend[1]);
  EXPECT_EQ(red[2], blend[2]);
}

TEST(VisualizationTest, blendColorTest2) {
  cv::Vec3b red(0, 0, 255);
  cv::Vec3b green(0, 255, 0);
  cv::Vec3b blend = blendColor(red, green, 2.0f, 0.0f, 1.0f);

  EXPECT_EQ(green[0], blend[0]);
  EXPECT_EQ(green[1], blend[1]);
  EXPECT_EQ(green[2], blend[2]);
}

TEST(VisualizationTest, blendColorTest3) {
  cv::Vec3b red(0, 0, 255);
  cv::Vec3b green(0, 255, 0);
  cv::Vec3b blend = blendColor(red, green, 0.5f, 0.0f, 1.0f);

  EXPECT_EQ(0, blend[0]);
  EXPECT_EQ(127, blend[1]);
  EXPECT_EQ(127, blend[2]);
}

TEST(VisualizationTest, censusToGrayTest1) {
  cv::Vec3b ret = censusToGray(0, 0, 100);
  EXPECT_EQ(0, ret[0]);
  EXPECT_EQ(0, ret[1]);
  EXPECT_EQ(0, ret[2]);
}

TEST(VisualizationTest, censusToGrayTest2) {
  cv::Vec3b ret = censusToGray(100, 0, 100);
  EXPECT_EQ(255, ret[0]);
  EXPECT_EQ(255, ret[1]);
  EXPECT_EQ(255, ret[2]);
}

TEST(VisualizationTest, censusToGrayTest3) {
  cv::Vec3b ret = censusToGray(50, 0, 100);
  EXPECT_EQ(127, ret[0]);
  EXPECT_EQ(127, ret[1]);
  EXPECT_EQ(127, ret[2]);
}

// TEST(VisualizationTest, censusToRGBTest1) {
//   cv::Mat_<uint32_t> img(3, 3, static_cast<uint32_t>(0));
//   // For some reason operator() is giving weird results. just work on raw data.
//   // img(1, 0) = 10000;
//   // img(0, 1) = 20000;

//   uint32_t* img_ptr = reinterpret_cast<uint32_t*>(img.data);
//   img_ptr[0 * img.cols + 1] = 10000;
//   img_ptr[1 * img.cols + 0] = 20000;

//   cv::Mat3b out(3, 3);
//   cv::Vec3b* out_ptr = reinterpret_cast<cv::Vec3b*>(out.data);

//   censusToRGB(img, &out);

//   EXPECT_EQ(img.rows, out.rows);
//   EXPECT_EQ(img.cols, out.cols);
//   for (int ii = 0; ii < img.rows; ++ii) {
//     for (int jj = 0; jj < img.cols; ++jj) {
//       if ((ii == 0) && (jj == 1)) {
//         EXPECT_EQ(127, out_ptr[ii * img.cols + jj][0]);
//         EXPECT_EQ(127, out_ptr[ii * img.cols + jj][1]);
//         EXPECT_EQ(127, out_ptr[ii * img.cols + jj][2]);
//       } else if ((ii == 1) && (jj == 0)) {
//         EXPECT_EQ(255, out_ptr[ii * img.cols + jj][0]);
//         EXPECT_EQ(255, out_ptr[ii * img.cols + jj][1]);
//         EXPECT_EQ(255, out_ptr[ii * img.cols + jj][2]);
//       } else {
//         EXPECT_EQ(0, out_ptr[ii * img.cols + jj][0]);
//         EXPECT_EQ(0, out_ptr[ii * img.cols + jj][1]);
//         EXPECT_EQ(0, out_ptr[ii * img.cols + jj][2]);
//       }
//     }
//   }
// }

}  // namespace utils

}  // namespace flame
