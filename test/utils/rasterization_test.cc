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
 * @file rasterization_test.cc
 * @author W. Nicholas Greene
 * @date 2017-08-23 15:13:53 (Wed)
 */

#include <unistd.h>

#include <chrono>

#include <boost/filesystem.hpp>

#include <opencv2/highgui/highgui.hpp>

#include "gtest/gtest.h"

#include "flame/utils/rasterization.h"

namespace fs = boost::filesystem;

typedef std::chrono::high_resolution_clock hclock;
typedef std::chrono::duration<float, std::milli> msec;

namespace  {

/**
 * \brief Compare bytes of two images.
 */
bool cvCompare(const cv::Mat& a, const cv::Mat& b) {
  const uint8_t* a_ptr = a.data;
  const uint8_t* b_ptr = b.data;

  int num_bytes = a.total() * a.elemSize();
  for (int ii = 0; ii < num_bytes; ++ii) {
    if (a_ptr[ii] != b_ptr[ii]) {
      return false;
    }
  }

  return true;
}

/**
 * \brief Compare gray scale values of two images.
 */
bool cvCompareNear(const cv::Mat& a, const cv::Mat& b, float tol) {
  const uint8_t* a_ptr = a.data;
  const uint8_t* b_ptr = b.data;

  if ((a.channels() != 1) || (b.channels() != 1)) {
    return false;
  }

  if (a.rows != b.rows) {
    return false;
  }

  if (a.cols != b.cols) {
    return false;
  }

  for (int ii = 0; ii < a.rows; ++ii) {
    for (int jj = 0; jj < a.cols; ++jj) {
      float diff = static_cast<float>(a.at<uint8_t>(ii, jj)) -
        static_cast<float>(b.at<uint8_t>(ii, jj));
      if (fabs(diff) > tol) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace

namespace flame {

TEST(RasterizationTest, SortVerticesTestInOrder) {
  cv::Point p1(100, 100);
  cv::Point p2(200, 100);
  cv::Point p3(150, 200);

  utils::SortVertices(&p1, &p2, &p3);

  EXPECT_LE(p1.y, p2.y);
  EXPECT_LE(p2.y, p3.y);
}

TEST(RasterizationTest, SortVerticesTestDescending) {
  cv::Point p1(100, 300);
  cv::Point p2(200, 200);
  cv::Point p3(150, 100);

  utils::SortVertices(&p1, &p2, &p3);

  EXPECT_LE(p1.y, p2.y);
  EXPECT_LE(p2.y, p3.y);
}

TEST(RasterizationTest, SortVerticesTestMixed) {
  cv::Point p1(100, 100);
  cv::Point p2(200, 300);
  cv::Point p3(150, 200);

  utils::SortVertices(&p1, &p2, &p3);

  EXPECT_LE(p1.y, p2.y);
  EXPECT_LE(p2.y, p3.y);
}

TEST(RasterizationTest, DrawLineInterpolatedTest) {
  cv::Mat img(480, 640, cv::DataType<float>::type, cv::Scalar(0));

  cv::Point p1(100, 100);
  cv::Point p2(200, 200);
  utils::DrawLineInterpolated(p1, p2, 0.0f, 1.0f, &img);

  // Convert to 8-bit for comparison.
  cv::Mat out_img;
  cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
  img.convertTo(out_img, cv::DataType<uint8_t>::type);

  // Load truth image and compare.
  char exe_str[200];
  readlink("/proc/self/exe", exe_str, 200);
  fs::path exe_path(exe_str);
  std::string base_dir = exe_path.parent_path().string();

  cv::Mat truth = cv::imread(base_dir +
    "/../data/RasterizationTest_DrawLineInterpolatedTest.png",
    cv::IMREAD_GRAYSCALE);

  EXPECT_TRUE(cvCompare(truth, out_img));

  // Save image.
  cv::imwrite(base_dir + "/RasterizationTest_DrawLineInterpolatedTest.png", out_img);

  // Display image.
  // cv::namedWindow("DrawLineInterpolatedTest", CV_WINDOW_AUTOSIZE);
  // cv::imshow("DrawLineInterpolatedTest", img);

  // while (true) {
  //   int c;
  //   c = cv::waitKey(10);

  //   if (static_cast<char>(c) == 27) {
  //     cv::destroyAllWindows();
  //     break;
  //   }
  // }
}

TEST(RasterizationTest, DrawTopFlatShadedTriangle) {
  cv::Mat img(480, 640, cv::DataType<float>::type, cv::Scalar(0));

  cv::Point p1(100, 100);
  cv::Point p2(200, 100);
  cv::Point p3(150, 200);
  utils::DrawTopFlatShadedTriangle(p1, p2, p3, 0.0f, 0.0f, 1.0f, &img);

  // Convert to 8-bit for comparison.
  cv::Mat out_img;
  cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
  img.convertTo(out_img, cv::DataType<uint8_t>::type);

  // Load truth image and compare.
  char exe_str[200];
  readlink("/proc/self/exe", exe_str, 200);
  fs::path exe_path(exe_str);
  std::string base_dir = exe_path.parent_path().string();

  cv::Mat truth = cv::imread(base_dir +
    "/../data/RasterizationTest_DrawTopFlatShadedTriangle.png",
    cv::IMREAD_GRAYSCALE);

  EXPECT_TRUE(cvCompare(truth, out_img));

  // Save image.
  cv::imwrite(base_dir + "/RasterizationTest_DrawTopFlatShadedTriangle.png", out_img);

  // Display image.
  // cv::namedWindow("DrawTopFlatShadedTriangleTest", CV_WINDOW_AUTOSIZE);
  // cv::imshow("DrawTopFlatShadedTriangleTest", out_img);

  // while (true) {
  //   int c;
  //   c = cv::waitKey(10);

  //   if (static_cast<char>(c) == 27) {
  //     cv::destroyAllWindows();
  //     break;
  //   }
  // }
}

TEST(RasterizationTest, DrawBottomFlatShadedTriangle) {
  cv::Mat img(480, 640, cv::DataType<float>::type, cv::Scalar(0));

  cv::Point p1(150, 100);
  cv::Point p2(100, 200);
  cv::Point p3(200, 200);
  utils::DrawBottomFlatShadedTriangle(p1, p2, p3, 0.0f, 1.0f, 1.0f, &img);

  // Convert to 8-bit for comparison.
  cv::Mat out_img;
  cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
  img.convertTo(out_img, cv::DataType<uint8_t>::type);

  // Load truth image and compare.
  char exe_str[200];
  readlink("/proc/self/exe", exe_str, 200);
  fs::path exe_path(exe_str);
  std::string base_dir = exe_path.parent_path().string();

  cv::Mat truth = cv::imread(base_dir +
    "/../data/RasterizationTest_DrawBottomFlatShadedTriangle.png",
    cv::IMREAD_GRAYSCALE);

  EXPECT_TRUE(cvCompare(truth, out_img));

  // Save image.
  cv::imwrite(base_dir + "/RasterizationTest_DrawBottomFlatShadedTriangle.png", out_img);

  // Display image.
  // cv::namedWindow("DrawBottomFlatShadedTriangleTest", CV_WINDOW_AUTOSIZE);
  // cv::imshow("DrawBottomFlatShadedTriangleTest", img);

  // while (true) {
  //   int c;
  //   c = cv::waitKey(10);

  //   if (static_cast<char>(c) == 27) {
  //     cv::destroyAllWindows();
  //     break;
  //   }
  // }
}

TEST(RasterizationTest, DrawShadedTriangleTop) {
  cv::Mat img(480, 640, cv::DataType<float>::type, cv::Scalar(0));

  cv::Point p1(100, 100);
  cv::Point p2(200, 100);
  cv::Point p3(150, 200);
  utils::DrawShadedTriangle(p1, p2, p3, 0.0f, 0.0f, 1.0f, &img);

  // Convert to 8-bit for comparison.
  cv::Mat out_img;
  cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
  img.convertTo(out_img, cv::DataType<uint8_t>::type);

  // Load truth image and compare.
  char exe_str[200];
  readlink("/proc/self/exe", exe_str, 200);
  fs::path exe_path(exe_str);
  std::string base_dir = exe_path.parent_path().string();

  cv::Mat truth = cv::imread(base_dir +
    "/../data/RasterizationTest_DrawShadedTriangleTop.png",
    cv::IMREAD_GRAYSCALE);

  EXPECT_TRUE(cvCompare(truth, out_img));

  // Save image.
  cv::imwrite(base_dir + "/RasterizationTest_DrawShadedTriangleTop.png", out_img);

  // Display image.
  // cv::namedWindow("DrawShadedTriangleTest", CV_WINDOW_AUTOSIZE);
  // cv::imshow("DrawShadedTriangleTest", img);

  // while (true) {
  //   int c;
  //   c = cv::waitKey(10);

  //   if (static_cast<char>(c) == 27) {
  //     cv::destroyAllWindows();
  //     break;
  //   }
  // }
}

TEST(RasterizationTest, DrawShadedTriangleBottom) {
  cv::Mat img(480, 640, cv::DataType<float>::type, cv::Scalar(0));

  cv::Point p1(150, 100);
  cv::Point p2(100, 200);
  cv::Point p3(200, 200);
  utils::DrawShadedTriangle(p1, p2, p3, 0.0f, 1.0f, 1.0f, &img);

  // Convert to 8-bit for comparison.
  cv::Mat out_img;
  cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
  img.convertTo(out_img, cv::DataType<uint8_t>::type);

  // Load truth image and compare.
  char exe_str[200];
  readlink("/proc/self/exe", exe_str, 200);
  fs::path exe_path(exe_str);
  std::string base_dir = exe_path.parent_path().string();

  cv::Mat truth = cv::imread(base_dir +
    "/../data/RasterizationTest_DrawShadedTriangleBottom.png",
    cv::IMREAD_GRAYSCALE);

  EXPECT_TRUE(cvCompare(truth, out_img));

  // Save image.
  cv::imwrite(base_dir + "/RasterizationTest_DrawShadedTriangleBottom.png", out_img);

  // Display image.
  // cv::namedWindow("DrawShadedTriangleTest", CV_WINDOW_AUTOSIZE);
  // cv::imshow("DrawShadedTriangleTest", img);

  // while (true) {
  //   int c;
  //   c = cv::waitKey(10);

  //   if (static_cast<char>(c) == 27) {
  //     cv::destroyAllWindows();
  //     break;
  //   }
  // }
}

TEST(RasterizationTest, DrawShadedTriangle1) {
  cv::Mat img(480, 640, cv::DataType<float>::type, cv::Scalar(0));

  cv::Point p1(150, 100);
  cv::Point p2(100, 200);
  cv::Point p3(200, 300);
  utils::DrawShadedTriangle(p1, p2, p3, 0.0f, 1.0f, 0.0f, &img);

  // Convert to 8-bit for comparison.
  cv::Mat out_img;
  cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
  img.convertTo(out_img, cv::DataType<uint8_t>::type);

  // Load truth truth image and compare.
  char exe_str[200];
  readlink("/proc/self/exe", exe_str, 200);
  fs::path exe_path(exe_str);
  std::string base_dir = exe_path.parent_path().string();

  cv::Mat truth = cv::imread(base_dir +
    "/../data/RasterizationTest_DrawShadedTriangle1.png",
    cv::IMREAD_GRAYSCALE);

  EXPECT_TRUE(cvCompareNear(truth, out_img, 1.0f));

  // Save image.
  cv::imwrite(base_dir + "/RasterizationTest_DrawShadedTriangle1.png", out_img);

  // Display image.
  // cv::namedWindow("DrawShadedTriangleTest", CV_WINDOW_AUTOSIZE);
  // cv::imshow("DrawShadedTriangleTest", img);

  // while (true) {
  //   int c;
  //   c = cv::waitKey(10);

  //   if (static_cast<char>(c) == 27) {
  //     cv::destroyAllWindows();
  //     break;
  //   }
  // }
}

TEST(RasterizationTest, DrawShadedTriangle2) {
  cv::Mat img(480, 640, cv::DataType<float>::type, cv::Scalar(0));

  cv::Point p1(150, 100);
  cv::Point p2(200, 200);
  cv::Point p3(100, 300);
  utils::DrawShadedTriangle(p1, p2, p3, 0.0f, 1.0f, 0.0f, &img);

  // Convert to 8-bit for comparison.
  cv::Mat out_img;
  cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
  img.convertTo(out_img, cv::DataType<uint8_t>::type);

  // Load truth image and compare.
  char exe_str[200];
  readlink("/proc/self/exe", exe_str, 200);
  fs::path exe_path(exe_str);
  std::string base_dir = exe_path.parent_path().string();

  cv::Mat truth = cv::imread(base_dir +
    "/../data/RasterizationTest_DrawShadedTriangle2.png",
    cv::IMREAD_GRAYSCALE);

  EXPECT_TRUE(cvCompareNear(truth, out_img, 1.0f));

  // Save image.
  cv::imwrite(base_dir + "/RasterizationTest_DrawShadedTriangle2.png", out_img);

  // Display image.
  // cv::namedWindow("DrawShadedTriangleTest", CV_WINDOW_AUTOSIZE);
  // cv::imshow("DrawShadedTriangleTest", img);

  // while (true) {
  //   int c;
  //   c = cv::waitKey(10);

  //   if (static_cast<char>(c) == 27) {
  //     cv::destroyAllWindows();
  //     break;
  //   }
  // }
}

TEST(RasterizationTest, DrawShadedTriangleOutOfOrder1) {
  cv::Mat img(480, 640, cv::DataType<float>::type, cv::Scalar(0));

  cv::Point p1(200, 300);
  cv::Point p2(150, 100);
  cv::Point p3(100, 200);
  utils::DrawShadedTriangle(p1, p2, p3, 0.0f, 1.0f, 0.0f, &img);

  // Convert to 8-bit for comparison.
  cv::Mat out_img;
  cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
  img.convertTo(out_img, cv::DataType<uint8_t>::type);

  // Load truth image and compare.
  char exe_str[200];
  readlink("/proc/self/exe", exe_str, 200);
  fs::path exe_path(exe_str);
  std::string base_dir = exe_path.parent_path().string();

  cv::Mat truth = cv::imread(base_dir +
    "/../data/RasterizationTest_DrawShadedTriangleOutOfOrder1.png",
    cv::IMREAD_GRAYSCALE);

  EXPECT_TRUE(cvCompareNear(truth, out_img, 1.0f));

  // Save image.
  cv::imwrite(base_dir + "/RasterizationTest_DrawShadedTriangleOutOfOrder1.png", out_img);

  // Display image.
  // cv::namedWindow("DrawShadedTriangleTest", CV_WINDOW_AUTOSIZE);
  // cv::imshow("DrawShadedTriangleTest", img);

  // while (true) {
  //   int c;
  //   c = cv::waitKey(10);

  //   if (static_cast<char>(c) == 27) {
  //     cv::destroyAllWindows();
  //     break;
  //   }
  // }
}

TEST(RasterizationTest, DrawShadedTriangleOutOfOrder2) {
  cv::Mat img(480, 640, cv::DataType<float>::type, cv::Scalar(0));

  cv::Point p1(100, 300);
  cv::Point p2(150, 100);
  cv::Point p3(200, 200);
  utils::DrawShadedTriangle(p1, p2, p3, 0.0f, 1.0f, 0.0f, &img);

  // Convert to 8-bit for comparison.
  cv::Mat out_img;
  cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
  img.convertTo(out_img, cv::DataType<uint8_t>::type);

  // Load truth image and compare.
  char exe_str[200];
  readlink("/proc/self/exe", exe_str, 200);
  fs::path exe_path(exe_str);
  std::string base_dir = exe_path.parent_path().string();

  cv::Mat truth = cv::imread(base_dir +
    "/../data/RasterizationTest_DrawShadedTriangleOutOfOrder2.png",
    cv::IMREAD_GRAYSCALE);

  EXPECT_TRUE(cvCompareNear(truth, out_img, 1.0f));

  // Save image.
  cv::imwrite(base_dir + "/RasterizationTest_DrawShadedTriangleOutOfOrder2.png", out_img);

  // Display image.
  // cv::namedWindow("DrawShadedTriangleTest", CV_WINDOW_AUTOSIZE);
  // cv::imshow("DrawShadedTriangleTest", img);

  // while (true) {
  //   int c;
  //   c = cv::waitKey(10);

  //   if (static_cast<char>(c) == 27) {
  //     cv::destroyAllWindows();
  //     break;
  //   }
  // }
}

TEST(RasterizationTest, DrawShadedTriangleBarycentric1) {
  cv::Mat img(480, 640, cv::DataType<float>::type, cv::Scalar(0));

  cv::Point p1(150, 100);
  cv::Point p2(100, 200);
  cv::Point p3(200, 300);
  utils::DrawShadedTriangleBarycentric(p1, p2, p3, 0.0f, 1.0f, 0.0f, &img);

  // Convert to 8-bit for comparison.
  cv::Mat out_img;
  cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
  img.convertTo(out_img, cv::DataType<uint8_t>::type);

  // Load image and compare.
  char exe_str[200];
  readlink("/proc/self/exe", exe_str, 200);
  fs::path exe_path(exe_str);
  std::string base_dir = exe_path.parent_path().string();

  cv::Mat truth = cv::imread(base_dir +
    "/../data/RasterizationTest_DrawShadedTriangleBarycentric1.png",
    cv::IMREAD_GRAYSCALE);

  EXPECT_TRUE(cvCompare(truth, out_img));

  // Save image.
  cv::imwrite(base_dir + "/RasterizationTest_DrawShadedTriangleBarycentric1.png", out_img);

  // Diplay image.
  // cv::namedWindow("DrawShadedTriangleBarycentricTest1", CV_WINDOW_AUTOSIZE);
  // cv::imshow("DrawShadedTriangleBarycentricTest1", img);

  // while (true) {
  //   int c;
  //   c = cv::waitKey(10);

  //   if (static_cast<char>(c) == 27) {
  //     cv::destroyAllWindows();
  //     break;
  //   }
  // }
}

TEST(RasterizationTest, DrawShadedTriangleBarycentric2) {
  cv::Mat img(480, 640, cv::DataType<float>::type, cv::Scalar(0));

  cv::Point p1(150, 100);
  cv::Point p2(100, 300);
  cv::Point p3(200, 200);
  utils::DrawShadedTriangleBarycentric(p1, p2, p3, 0.0f, 0.0f, 1.0f, &img);

  // Convert to 8-bit for comparison.
  cv::Mat out_img;
  cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
  img.convertTo(out_img, cv::DataType<uint8_t>::type);

  // Load truth image and compare.
  char exe_str[200];
  readlink("/proc/self/exe", exe_str, 200);
  fs::path exe_path(exe_str);
  std::string base_dir = exe_path.parent_path().string();

  cv::Mat truth = cv::imread(base_dir +
    "/../data/RasterizationTest_DrawShadedTriangleBarycentric2.png",
    cv::IMREAD_GRAYSCALE);

  EXPECT_TRUE(cvCompare(truth, out_img));

  // Save image.
  cv::imwrite(base_dir + "/RasterizationTest_DrawShadedTriangleBarycentric2.png", out_img);

  // Display image.
  // cv::namedWindow("DrawShadedTriangleBarycentricTest2", CV_WINDOW_AUTOSIZE);
  // cv::imshow("DrawShadedTriangleBarycentricTest2", img);

  // while (true) {
  //   int c;
  //   c = cv::waitKey(10);

  //   if (static_cast<char>(c) == 27) {
  //     cv::destroyAllWindows();
  //     break;
  //   }
  // }
}

// TEST(RasterizationTest, TimingTest2x2) {
//   int num_tests = 100;
//   int w = 640;
//   int h = 480;
//   int l = 2;

//   cv::Mat img(h, w, cv::DataType<float>::type);

//   // Hacky, but time it here.
//   std::chrono::time_point<hclock> start = hclock::now();

//   for (int ii = 0; ii < num_tests; ++ii) {
//     for (int y = 0; y < h - l; y+=l) {
//       for (int x = 0; x < w - l; x+=l) {
//         if ((x >= w) || (y >= h)) {
//           continue;
//         }

//         cv::Point p1(x, y);
//         cv::Point p2(x + l - 1, y);
//         cv::Point p3(x, y + l -1);
//         cv::Point p4(x + l - 1, y + l - 1);

//         utils::DrawShadedTriangle(p1, p2, p3, 0.0f, 1.0f, 1.0f, &img);
//         utils::DrawShadedTriangle(p2, p3, p4, 1.0f, 1.0f, 0.0f, &img);
//       }
//     }
//   }

//   msec ms = hclock::now() - start;
//   EXPECT_LT(ms.count()/num_tests, 2 * 3) << ms.count();

//   // Save image.
//   // char exe_str[200];
//   // readlink("/proc/self/exe", exe_str, 200);
//   // fs::path exe_path(exe_str);
//   // std::string base_dir = exe_path.parent_path().string();
//   // cv::imwrite(base_dir + "/RasterizationTest_TimingTest2x2.png", img);

//   // Display image.
//   // cv::namedWindow("TimingTest2x2", CV_WINDOW_AUTOSIZE);
//   // cv::imshow("TimingTest2x2", img);

//   // while (true) {
//   //   int c;
//   //   c = cv::waitKey(10);

//   //   if (static_cast<char>(c) == 27) {
//   //     cv::destroyAllWindows();
//   //     break;
//   //   }
//   // }

//   return;
// }

// TEST(RasterizationTest, TimingTest4x4) {
//   int num_tests = 100;
//   int w = 640;
//   int h = 480;
//   int l = 4;

//   cv::Mat img(h, w, cv::DataType<float>::type);

//   // Hacky, but time it here.
//   std::chrono::time_point<hclock> start = hclock::now();

//   for (int ii = 0; ii < num_tests; ++ii) {
//     for (int y = 0; y < h - l; y+=l) {
//       for (int x = 0; x < w - l; x+=l) {
//         if ((x >= w) || (y >= h)) {
//           continue;
//         }

//         cv::Point p1(x, y);
//         cv::Point p2(x + l - 1, y);
//         cv::Point p3(x, y + l -1);
//         cv::Point p4(x + l - 1, y + l - 1);

//         utils::DrawShadedTriangle(p1, p2, p3, 0.0f, 1.0f, 1.0f, &img);
//         utils::DrawShadedTriangle(p2, p3, p4, 1.0f, 1.0f, 0.0f, &img);
//       }
//     }
//   }

//   msec ms = hclock::now() - start;
//   EXPECT_LT(ms.count()/num_tests, 2.5 * 13) << ms.count();

//   // Save image.
//   // char exe_str[200];
//   // readlink("/proc/self/exe", exe_str, 200);
//   // fs::path exe_path(exe_str);
//   // std::string base_dir = exe_path.parent_path().string();
//   // cv::imwrite(base_dir + "/RasterizationTest_TimingTest4x4.png", img);

//   // Display image.
//   // cv::namedWindow("TimingTest4x4", CV_WINDOW_AUTOSIZE);
//   // cv::imshow("TimingTest4x4", img);

//   // while (true) {
//   //   int c;
//   //   c = cv::waitKey(10);

//   //   if (static_cast<char>(c) == 27) {
//   //     cv::destroyAllWindows();
//   //     break;
//   //   }
//   // }

//   return;
// }

// TEST(RasterizationTest, TimingTest2x2Barycentric) {
//   int num_tests = 100;
//   int w = 640;
//   int h = 480;
//   int l = 2;

//   cv::Mat img(h, w, cv::DataType<float>::type);

//   // Hacky, but time it here.
//   std::chrono::time_point<hclock> start = hclock::now();

//   for (int ii = 0; ii < num_tests; ++ii) {
//     for (int y = 0; y < h - l; y+=l) {
//       for (int x = 0; x < w - l; x+=l) {
//         if ((x >= w) || (y >= h)) {
//           continue;
//         }

//         cv::Point p1(x, y);
//         cv::Point p2(x + l - 1, y);
//         cv::Point p3(x, y + l -1);
//         cv::Point p4(x + l - 1, y + l - 1);

//         utils::DrawShadedTriangleBarycentric(p1, p3, p2, 0.0f, 1.0f, 1.0f, &img);
//         utils::DrawShadedTriangleBarycentric(p2, p3, p4, 1.0f, 1.0f, 0.0f, &img);
//       }
//     }
//   }

//   msec ms = hclock::now() - start;
//   EXPECT_LT(ms.count()/num_tests, 3 * 4) << ms.count();

//   // Save image.
//   // char exe_str[200];
//   // readlink("/proc/self/exe", exe_str, 200);
//   // fs::path exe_path(exe_str);
//   // std::string base_dir = exe_path.parent_path().string();
//   // cv::imwrite(base_dir + "/RasterizationTest_TimingTest2x2Barycentric.png", img);

//   // Display image.
//   // cv::namedWindow("TimingTest2x2Barycentric", CV_WINDOW_AUTOSIZE);
//   // cv::imshow("TimingTest2x2Barycentric", img);

//   // while (true) {
//   //   int c;
//   //   c = cv::waitKey(10);

//   //   if (static_cast<char>(c) == 27) {
//   //     cv::destroyAllWindows();
//   //     break;
//   //   }
//   // }

//   return;
// }

// TEST(RasterizationTest, TimingTest4x4Barycentric) {
//   int num_tests = 100;
//   int w = 640;
//   int h = 480;
//   int l = 4;

//   cv::Mat img(h, w, cv::DataType<float>::type);

//   // Hacky, but time it here.
//   std::chrono::time_point<hclock> start = hclock::now();

//   for (int ii = 0; ii < num_tests; ++ii) {
//     for (int y = 0; y < h - l; y+=l) {
//       for (int x = 0; x < w - l; x+=l) {
//         if ((x >= w) || (y >= h)) {
//           continue;
//         }

//         cv::Point p1(x, y);
//         cv::Point p2(x + l - 1, y);
//         cv::Point p3(x, y + l -1);
//         cv::Point p4(x + l - 1, y + l - 1);

//         utils::DrawShadedTriangleBarycentric(p1, p3, p2, 0.0f, 1.0f, 1.0f, &img);
//         utils::DrawShadedTriangleBarycentric(p2, p3, p4, 1.0f, 1.0f, 0.0f, &img);
//       }
//     }
//   }

//   msec ms = hclock::now() - start;
//   EXPECT_LT(ms.count()/num_tests, 2.5 * 1.5) << ms.count();

//   // Save image.
//   // char exe_str[200];
//   // readlink("/proc/self/exe", exe_str, 200);
//   // fs::path exe_path(exe_str);
//   // std::string base_dir = exe_path.parent_path().string();
//   // cv::imwrite(base_dir + "/RasterizationTest_TimingTest4x4Barycentric.png", img);

//   // Display image.
//   // cv::namedWindow("TimingTest4x4Barycentric", CV_WINDOW_AUTOSIZE);
//   // cv::imshow("TimingTest4x4Barycentric", img);

//   // while (true) {
//   //   int c;
//   //   c = cv::waitKey(10);

//   //   if (static_cast<char>(c) == 27) {
//   //     cv::destroyAllWindows();
//   //     break;
//   //   }
//   // }

//   return;
// }

// TEST(RasterizationTest, TimingTest8x8Barycentric) {
//   int num_tests = 100;
//   int w = 640;
//   int h = 480;
//   int l = 8;

//   cv::Mat img(h, w, cv::DataType<float>::type);

//   // Hacky, but time it here.
//   std::chrono::time_point<hclock> start = hclock::now();

//   for (int ii = 0; ii < num_tests; ++ii) {
//     for (int y = 0; y < h - l; y+=l) {
//       for (int x = 0; x < w - l; x+=l) {
//         if ((x >= w) || (y >= h)) {
//           continue;
//         }

//         cv::Point p1(x, y);
//         cv::Point p2(x + l - 1, y);
//         cv::Point p3(x, y + l -1);
//         cv::Point p4(x + l - 1, y + l - 1);

//         utils::DrawShadedTriangleBarycentric(p1, p3, p2, 0.0f, 1.0f, 1.0f, &img);
//         utils::DrawShadedTriangleBarycentric(p2, p3, p4, 1.0f, 1.0f, 0.0f, &img);
//       }
//     }
//   }

//   msec ms = hclock::now() - start;
//   EXPECT_LT(ms.count()/num_tests, 2 * 1) << ms.count();

//   // Save image.
//   // char exe_str[200];
//   // readlink("/proc/self/exe", exe_str, 200);
//   // fs::path exe_path(exe_str);
//   // std::string base_dir = exe_path.parent_path().string();
//   // cv::imwrite(base_dir + "/RasterizationTest_TimingTest8x8Barycentric.png", img);

//   // Display image.
//   // cv::namedWindow("TimingTest8x8Barycentric", CV_WINDOW_AUTOSIZE);
//   // cv::imshow("TimingTest8x8Barycentric", img);

//   // while (true) {
//   //   int c;
//   //   c = cv::waitKey(10);

//   //   if (static_cast<char>(c) == 27) {
//   //     cv::destroyAllWindows();
//   //     break;
//   //   }
//   // }

//   return;
// }

}  // namespace flame
