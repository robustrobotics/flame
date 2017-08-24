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
 * @file epipolar_geometry_test.cc
 * @author W. Nicholas Greene
 * @date 2017-08-18 19:18:41 (Fri)
 */

#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sophus/se3.hpp>

#include "gtest/gtest.h"

#include "flame/stereo/epipolar_geometry.h"

namespace flame {

namespace stereo {

/**
 * \brief Test minDepthProjection with translations in X direction.
 */
TEST(EpipolarGeometryTest, minDepthProjectionXTranslate1) {
  Eigen::Matrix3f K;
  K << 525, 0, 640.0/2,
    0, 525, 480.0/2,
    0, 0, 1;

  cv::Point2f u_ref(320, 240);
  EpipolarGeometry<float> epigeo(K, K.inverse());

  // Positive X translation
  epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(2, 0, 0));
  cv::Point2f u_min1;
  epigeo.minDepthProjection(u_ref, &u_min1);

  EXPECT_TRUE(u_min1.x > K(0, 2)*2);
  EXPECT_EQ(240, u_min1.y);

  // Negative X translation.
  epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(-2, 0, 0));
  cv::Point2f u_min2;
  epigeo.minDepthProjection(u_ref, &u_min2);

  EXPECT_TRUE(u_min2.x < 0);
  EXPECT_EQ(240, u_min2.y);
}

/**
 * \brief Test minDepthProjection with translations in X direction.
 */
TEST(EpipolarGeometryTest, minDepthProjectionXTranslate2) {
  Eigen::Matrix3f K;
  K << 525, 0, 640.0/2,
    0, 525, 480.0/2,
    0, 0, 1;

  cv::Point2f u_ref(320, 0);
  EpipolarGeometry<float> epigeo(K, K.inverse());

  // Positive X translation
  epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(2, 0, 0));
  cv::Point2f u_min1;
  epigeo.minDepthProjection(u_ref, &u_min1);

  EXPECT_TRUE(u_min1.x > K(0, 2)*2);
  EXPECT_NEAR(0, u_min1.y, 1e-3);

  // Negative X translation.
  epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(-2, 0, 0));
  cv::Point2f u_min2;
  epigeo.minDepthProjection(u_ref, &u_min2);

  EXPECT_TRUE(u_min2.x < 0);
  EXPECT_NEAR(0, u_min2.y, 1e-3);
}

/**
 * \brief Test minDepthProjection with translations in Y direction.
 */
TEST(EpipolarGeometryTest, minDepthProjectionYTranslate1) {
  Eigen::Matrix3f K;
  K << 525, 0, 640.0/2,
    0, 525, 480.0/2,
    0, 0, 1;

  cv::Point2f u_ref(320, 240);
  EpipolarGeometry<float> epigeo(K, K.inverse());

  // Positive Y translation
  epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0, 2, 0));
  cv::Point2f u_min1;
  epigeo.minDepthProjection(u_ref, &u_min1);

  EXPECT_TRUE(u_min1.y > K(1, 2)*2);
  EXPECT_NEAR(320, u_min1.x, 1e-3);

  // Negative Y translation.
  epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0, -2, 0));
  cv::Point2f u_min2;
  epigeo.minDepthProjection(u_ref, &u_min2);

  EXPECT_TRUE(u_min2.y < 0);
  EXPECT_NEAR(320, u_min2.x, 1e-3);
}

/**
 * \brief Test minDepthProjection with translations in Y direction.
 */
TEST(EpipolarGeometryTest, minDepthProjectionYTranslate2) {
  Eigen::Matrix3f K;
  K << 525, 0, 640.0/2,
    0, 525, 480.0/2,
    0, 0, 1;

  cv::Point2f u_ref(0, 240);
  EpipolarGeometry<float> epigeo(K, K.inverse());

  // Positive Y translation
  epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0, 2, 0));
  cv::Point2f u_min1;
  epigeo.minDepthProjection(u_ref, &u_min1);

  EXPECT_TRUE(u_min1.y > K(1, 2)*2);
  EXPECT_NEAR(0, u_min1.x, 1e-3);

  // Negative Y translation.
  epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0, -2, 0));
  cv::Point2f u_min2;
  epigeo.minDepthProjection(u_ref, &u_min2);

  EXPECT_TRUE(u_min2.y < 0);
  EXPECT_NEAR(0, u_min2.x, 1e-3);
}

/**
 * \brief Test minDepthProjection when yawed.
 */
TEST(EpipolarGeometryTest, minDepthProjection60Yaw) {
  Eigen::Matrix3f K;
  K << 525, 0, 640.0/2,
    0, 525, 480.0/2,
    0, 0, 1;

  Eigen::AngleAxisf aa21(-M_PI/3, Eigen::Vector3f::UnitY());
  Eigen::Quaternionf q21(aa21);
  Eigen::Vector3f t21(2, 0, 0);

  Eigen::Quaternionf q12(q21.inverse());
  Eigen::Vector3f t12(-(q12 * t21));

  cv::Point2f u_ref(320, 240);
  EpipolarGeometry<float> epigeo(K, K.inverse());
  double tol = 1e-4;

  // ref = 1, cmp = 2
  epigeo.loadGeometry(q12, t12);

  cv::Point2f u_min1;
  epigeo.minDepthProjection(u_ref, &u_min1);

  EXPECT_NEAR(16.8910904, u_min1.x, tol);
  EXPECT_NEAR(240, u_min1.y, tol);

  // ref = 2, cmp = 1
  epigeo.loadGeometry(q21, t21);

  cv::Point2f u_min2;
  epigeo.minDepthProjection(u_ref, &u_min2);

  EXPECT_NEAR(1049999424, u_min2.x, tol);
  EXPECT_NEAR(240, u_min2.y, tol);
}

/**
 * \brief Test using real data point where ref cam is in front of cmp cam.
 */
TEST(EpipolarGeometryTest, minDepthProjectionRefFontCmp) {
  Eigen::Matrix3f K;
  K << 535.43310546875f, 0.0f, 320.106652814575f,
    0.0f, 539.212524414062f, 247.632132204719f,
    0.0f, 0.0f, 1.0f;

  Eigen::Quaternionf q_ref_to_cmp(0.999138, -0.000878, 0.041493, 0.000386);
  Eigen::Vector3f t_ref_to_cmp(-0.221092, -0.036134, 0.084099);

  EpipolarGeometry<float> epigeo(K, K.inverse());
  epigeo.loadGeometry(q_ref_to_cmp, t_ref_to_cmp);

  cv::Point2f u_zero;
  epigeo.minDepthProjection(cv::Point2f(320, 240), &u_zero);

  double tol = 1e-2;
  EXPECT_NEAR(-1087.525391, u_zero.x, tol);
  EXPECT_NEAR(15.954912, u_zero.y, tol);
}

/**
 * \brief Test using real data point where ref cam is behind cmp cam.
 */
TEST(EpipolarGeometryTest, minDepthProjectionRefBehindCmp) {
  Eigen::Matrix3f K;
  K << 535.43310546875f, 0.0f, 320.106652814575f,
    0.0f, 539.212524414062f, 247.632132204719f,
    0.0f, 0.0f, 1.0f;

  Eigen::Quaternionf q_ref_to_cmp(-0.999853, 0.014856, -0.005249, -0.006822);
  Eigen::Vector3f t_ref_to_cmp(-0.258187, 0.040849, -0.054990);

  EpipolarGeometry<float> epigeo(K, K.inverse());
  epigeo.loadGeometry(q_ref_to_cmp, t_ref_to_cmp);

  cv::Point2f u_zero;
  epigeo.minDepthProjection(cv::Point2f(320, 240), &u_zero);

  double tol = 1e-1;
  EXPECT_NEAR(187.65597534179688, u_zero.x, tol);
  EXPECT_NEAR(278.55392456054688, u_zero.y, tol);
}

/**
 * \brief Test maxDepthProjection when ref and cmp camera have the same pose.
 */
TEST(EpipolarGeometryTest, maxDepthProjectionIdentity) {
  Eigen::Matrix3f K;
  K << 525, 0, 640/2,
    0, 525, 480/2,
    0, 0, 1;
  Eigen::Matrix3f Kinv(K.inverse());

  EpipolarGeometry<float> epigeo(K, Kinv);
  epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero());

  cv::Point2f u_ref(320, 240);
  cv::Point2f u_cmp;
  epigeo.maxDepthProjection(u_ref, &u_cmp);

  float tol = 1e-3;
  EXPECT_NEAR(u_ref.x, u_cmp.x, tol);
  EXPECT_NEAR(u_ref.y, u_cmp.y, tol);
}

/**
 * \brief Test maxDepthProjection when yawed.
 */
TEST(EpipolarGeometryTest, maxDepthProjection30Yaw) {
  Eigen::Matrix3f K;
  K << 525, 0, 640/2,
    0, 525, 480/2,
    0, 0, 1;
  Eigen::AngleAxisf aa_right(-M_PI/6, Eigen::Vector3f::UnitY());
  Eigen::Quaternionf q_right(aa_right);

  EpipolarGeometry<float> epigeo(K, K.inverse());
  epigeo.loadGeometry(q_right, Eigen::Vector3f::Zero());

  cv::Point2f u_ref(320, 240);
  cv::Point2f u_cmp;
  epigeo.maxDepthProjection(u_ref, &u_cmp);

  double tol = 1e-4;
  EXPECT_NEAR(16.891090393066406, u_cmp.x, tol);
  EXPECT_NEAR(240, u_cmp.y, tol);
}

/**
 * \brief Test maxDepthProjection when rolled.
 */
TEST(EpipolarGeometryTest, maxDepthProjection30Roll) {
  Eigen::Matrix3f K;
  K << 525, 0, 640/2,
    0, 525, 480/2,
    0, 0, 1;
  Eigen::AngleAxisf aa_right(-M_PI/6, Eigen::Vector3f::UnitX());
  Eigen::Quaternionf q_right(aa_right);

  EpipolarGeometry<float> epigeo(K, K.inverse());
  epigeo.loadGeometry(q_right, Eigen::Vector3f::Zero());

  cv::Point2f u_ref(320, 240);
  cv::Point2f u_cmp;
  epigeo.maxDepthProjection(u_ref, &u_cmp);

  double tol = 1e-4;
  EXPECT_NEAR(320, u_cmp.x, tol);
  EXPECT_NEAR(543.10888671875, u_cmp.y, tol);
}

TEST(EpipolarGeometryTest, epiline60Yaw) {
  Eigen::Matrix3f K;
  K << 525, 0, 640/2,
       0, 525, 480/2,
       0, 0, 1;

  Eigen::AngleAxisf aa_right_to_left(-M_PI/3, Eigen::Vector3f::UnitY());
  Eigen::Quaternionf q_right_to_left(aa_right_to_left);
  Eigen::Vector3f t_right_to_left(2, 0, 0);

  Eigen::Quaternionf q_left_to_right(aa_right_to_left.inverse());
  Eigen::Vector3f t_left_to_right(-(q_right_to_left * t_right_to_left));

  EpipolarGeometry<float> epigeo(K, K.inverse());
  epigeo.loadGeometry(q_left_to_right, t_left_to_right);

  cv::Point2f u_ref(320, 240);
  cv::Point2f u_inf, epi;
  epigeo.epiline(u_ref, &u_inf, &epi);

  double tol = 1e-4;
  EXPECT_NEAR(1, epi.x, tol);
  EXPECT_NEAR(0, epi.y, tol);
}

TEST(EpipolarGeometryTest, epiline60Roll) {
  Eigen::Matrix3f K;
  K << 525, 0, 640/2,
       0, 525, 480/2,
       0, 0, 1;

  Eigen::AngleAxisf aa_right_to_left(M_PI/3, Eigen::Vector3f::UnitX());
  Eigen::Quaternionf q_right_to_left(aa_right_to_left);
  Eigen::Vector3f t_right_to_left(0, 2, 0);

  Eigen::Quaternionf q_left_to_right(aa_right_to_left.inverse());
  Eigen::Vector3f t_left_to_right(-(q_right_to_left * t_right_to_left));

  EpipolarGeometry<float> epigeo(K, K.inverse());
  epigeo.loadGeometry(q_left_to_right, t_left_to_right);

  cv::Point2f u_ref(320, 240);
  cv::Point2f u_inf, epi;
  epigeo.epiline(u_ref, &u_inf, &epi);

  double tol = 1e-4;
  EXPECT_NEAR(0, epi.x, tol);
  EXPECT_NEAR(1, epi.y, tol);
}

/**
 * \brief Test point at (1, 0, 10) with cameras 1m apart and yawed -15 deg.
 *
 * T1 = -15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (1, 0, 10)
 */
TEST(EpipolarGeometryTest, disparityToDepthTest1) {
  Eigen::Matrix3f K;
  K << 525.0f, 0.0f, 320.0f,
    0.0f, 525.0f, 240.0f,
    0.0f, 0.0f, 1.0f;
  Eigen::Matrix3f Kinv(K.inverse());

  // Geometry of cameras and landmarks.
  Eigen::AngleAxisf aa(-M_PI/12, Eigen::Vector3f::UnitY());
  Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
  Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                  Eigen::Vector3f(1.0f, 0.0f, 0.0f));
  Eigen::Vector3f p_world(1.0f, 0.0f, 10.0f);

  // Project into cameras.
  Sophus::SE3f T12(T2.inverse() * T1);
  Sophus::SE3f T21(T1.inverse() * T2);

  cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                    T1.translation(),
                                                    p_world);
  cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                    T2.translation(),
                                                    p_world);

  cv::Point2f u_inf, epi;
  float tol = 1e-4;

  // Depth from camera 1.
  EpipolarGeometry<float> epigeo1(K, Kinv);
  epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
  float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
  float depth1 = epigeo1.disparityToDepth(u1, u_inf, epi, disp1);
  EXPECT_NEAR((T1.inverse() * p_world)(2), depth1, tol);

  // Depth from camera 2.
  EpipolarGeometry<float> epigeo2(K, Kinv);
  epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
  float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
  float depth2 = epigeo2.disparityToDepth(u2, u_inf, epi, disp2);
  EXPECT_NEAR((T2.inverse() * p_world)(2), depth2, tol);
}

/**
 * \brief Test point at (-1, 0, 10) with cameras 1m apart and yawed -15 deg.
 *
 * T1 = -15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (-1, 0, 10)
 */
TEST(EpipolarGeometryTest, disparityToDepthTest2) {
  Eigen::Matrix3f K;
  K << 525.0f, 0.0f, 320.0f,
    0.0f, 525.0f, 240.0f,
    0.0f, 0.0f, 1.0f;
  Eigen::Matrix3f Kinv(K.inverse());

  // Geometry of cameras and landmarks.
  Eigen::AngleAxisf aa(-M_PI/12, Eigen::Vector3f::UnitY());
  Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
  Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                  Eigen::Vector3f(1.0f, 0.0f, 0.0f));
  Eigen::Vector3f p_world(-1.0f, 0.0f, 10.0f);

  // Project into cameras.
  Sophus::SE3f T12(T2.inverse() * T1);
  Sophus::SE3f T21(T1.inverse() * T2);

  cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                    T1.translation(),
                                                    p_world);
  cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                    T2.translation(),
                                                    p_world);

  cv::Point2f u_inf, epi;
  float tol = 1e-4;

  // Depth from camera 1.
  EpipolarGeometry<float> epigeo1(K, Kinv);
  epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
  float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
  float depth1 = epigeo1.disparityToDepth(u1, u_inf, epi, disp1);
  EXPECT_NEAR((T1.inverse() * p_world)(2), depth1, tol);

  // Depth from camera 2.
  EpipolarGeometry<float> epigeo2(K, Kinv);
  epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
  float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
  float depth2 = epigeo2.disparityToDepth(u2, u_inf, epi, disp2);
  EXPECT_NEAR((T2.inverse() * p_world)(2), depth2, tol);
}

/**
 * \brief Test point at (0, 1, 10) with cameras 1m apart and yawed 15 deg.
 *
 * T1 = 15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (0, 1, 10)
 */
TEST(EpipolarGeometryTest, disparityToDepthTest3) {
  Eigen::Matrix3f K;
  K << 525.0f, 0.0f, 320.0f,
    0.0f, 525.0f, 240.0f,
    0.0f, 0.0f, 1.0f;
  Eigen::Matrix3f Kinv(K.inverse());

  // Geometry of cameras and landmarks.
  Eigen::AngleAxisf aa(M_PI/12, Eigen::Vector3f::UnitY());
  Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
  Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                  Eigen::Vector3f(1.0f, 0.0f, 0.0f));
  Eigen::Vector3f p_world(0.0f, 1.0f, 10.0f);

  // Project into cameras.
  Sophus::SE3f T12(T2.inverse() * T1);
  Sophus::SE3f T21(T1.inverse() * T2);

  cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                    T1.translation(),
                                                    p_world);
  cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                    T2.translation(),
                                                    p_world);

  cv::Point2f u_inf, epi;
  float tol = 1e-4;

  // Depth from camera 1.
  EpipolarGeometry<float> epigeo1(K, Kinv);
  epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
  float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
  float depth1 = epigeo1.disparityToDepth(u1, u_inf, epi, disp1);
  EXPECT_NEAR((T1.inverse() * p_world)(2), depth1, tol);

  // Depth from camera 2.
  EpipolarGeometry<float> epigeo2(K, Kinv);
  epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
  float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
  float depth2 = epigeo2.disparityToDepth(u2, u_inf, epi, disp2);
  EXPECT_NEAR((T2.inverse() * p_world)(2), depth2, tol);
}

/**
 * \brief Test point at (0, -1, 10) with cameras 1m apart and yawed 15 deg.
 *
 * T1 = 15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (0, -1, 10)
 */
TEST(EpipolarGeometryTest, disparityToDepthTest4) {
  Eigen::Matrix3f K;
  K << 525.0f, 0.0f, 320.0f,
    0.0f, 525.0f, 240.0f,
    0.0f, 0.0f, 1.0f;
  Eigen::Matrix3f Kinv(K.inverse());

  // Geometry of cameras and landmarks.
  Eigen::AngleAxisf aa(M_PI/12, Eigen::Vector3f::UnitY());
  Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
  Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                  Eigen::Vector3f(1.0f, 0.0f, 0.0f));
  Eigen::Vector3f p_world(0.0f, -1.0f, 10.0f);

  // Project into cameras.
  Sophus::SE3f T12(T2.inverse() * T1);
  Sophus::SE3f T21(T1.inverse() * T2);

  cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                    T1.translation(),
                                                    p_world);
  cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                    T2.translation(),
                                                    p_world);

  cv::Point2f u_inf, epi;
  float tol = 1e-4;

  // Depth from camera 1.
  EpipolarGeometry<float> epigeo1(K, Kinv);
  epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
  float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
  float depth1 = epigeo1.disparityToDepth(u1, u_inf, epi, disp1);
  EXPECT_NEAR((T1.inverse() * p_world)(2), depth1, tol);

  // Depth from camera 2.
  EpipolarGeometry<float> epigeo2(K, Kinv);
  epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
  float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
  float depth2 = epigeo2.disparityToDepth(u2, u_inf, epi, disp2);
  EXPECT_NEAR((T2.inverse() * p_world)(2), depth2, tol);
}

/**
 * \brief Test point at (1, 0, 10) with cameras 1m apart and yawed -15 deg.
 *
 * T1 = -15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (1, 0, 10)
 */
TEST(EpipolarGeometryTest, disparityToInverseDepthTest1) {
  Eigen::Matrix3f K;
  K << 525.0f, 0.0f, 320.0f,
    0.0f, 525.0f, 240.0f,
    0.0f, 0.0f, 1.0f;
  Eigen::Matrix3f Kinv(K.inverse());

  // Geometry of cameras and landmarks.
  Eigen::AngleAxisf aa(-M_PI/12, Eigen::Vector3f::UnitY());
  Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
  Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                  Eigen::Vector3f(1.0f, 0.0f, 0.0f));
  Eigen::Vector3f p_world(1.0f, 0.0f, 10.0f);

  // Project into cameras.
  Sophus::SE3f T12(T2.inverse() * T1);
  Sophus::SE3f T21(T1.inverse() * T2);

  cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                    T1.translation(),
                                                    p_world);
  cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                    T2.translation(),
                                                    p_world);

  cv::Point2f u_inf, epi;
  float tol = 1e-4;

  // Depth from camera 1.
  EpipolarGeometry<float> epigeo1(K, Kinv);
  epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
  float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
  float idepth1 = epigeo1.disparityToInverseDepth(u1, u_inf, epi, disp1);
  EXPECT_NEAR(1.0f/(T1.inverse() * p_world)(2), idepth1, tol);

  // Depth from camera 2.
  EpipolarGeometry<float> epigeo2(K, Kinv);
  epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
  float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
  float idepth2 = epigeo2.disparityToInverseDepth(u2, u_inf, epi, disp2);
  EXPECT_NEAR(1.0f/(T2.inverse() * p_world)(2), idepth2, tol);
}

/**
 * \brief Test point at (-1, 0, 10) with cameras 1m apart and yawed -15 deg.
 *
 * T1 = -15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (-1, 0, 10)
 */
TEST(EpipolarGeometryTest, disparityToInverseDepthTest2) {
  Eigen::Matrix3f K;
  K << 525.0f, 0.0f, 320.0f,
    0.0f, 525.0f, 240.0f,
    0.0f, 0.0f, 1.0f;
  Eigen::Matrix3f Kinv(K.inverse());

  // Geometry of cameras and landmarks.
  Eigen::AngleAxisf aa(-M_PI/12, Eigen::Vector3f::UnitY());
  Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
  Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                  Eigen::Vector3f(1.0f, 0.0f, 0.0f));
  Eigen::Vector3f p_world(-1.0f, 0.0f, 10.0f);

  // Project into cameras.
  Sophus::SE3f T12(T2.inverse() * T1);
  Sophus::SE3f T21(T1.inverse() * T2);

  cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                    T1.translation(),
                                                    p_world);
  cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                    T2.translation(),
                                                    p_world);

  cv::Point2f u_inf, epi;
  float tol = 1e-4;

  // Depth from camera 1.
  EpipolarGeometry<float> epigeo1(K, Kinv);
  epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
  float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
  float idepth1 = epigeo1.disparityToInverseDepth(u1, u_inf, epi, disp1);
  EXPECT_NEAR(1.0f/(T1.inverse() * p_world)(2), idepth1, tol);

  // Depth from camera 2.
  EpipolarGeometry<float> epigeo2(K, Kinv);
  epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
  float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
  float idepth2 = epigeo2.disparityToInverseDepth(u2, u_inf, epi, disp2);
  EXPECT_NEAR(1.0f/(T2.inverse() * p_world)(2), idepth2, tol);
}

/**
 * \brief Test point at (0, 1, 10) with cameras 1m apart and yawed 15 deg.
 *
 * T1 = 15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (0, 1, 10)
 */
TEST(EpipolarGeometryTest, disparityToInverseDepthTest3) {
  Eigen::Matrix3f K;
  K << 525.0f, 0.0f, 320.0f,
    0.0f, 525.0f, 240.0f,
    0.0f, 0.0f, 1.0f;
  Eigen::Matrix3f Kinv(K.inverse());

  // Geometry of cameras and landmarks.
  Eigen::AngleAxisf aa(M_PI/12, Eigen::Vector3f::UnitY());
  Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
  Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                  Eigen::Vector3f(1.0f, 0.0f, 0.0f));
  Eigen::Vector3f p_world(0.0f, 1.0f, 10.0f);

  // Project into cameras.
  Sophus::SE3f T12(T2.inverse() * T1);
  Sophus::SE3f T21(T1.inverse() * T2);

  cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                    T1.translation(),
                                                    p_world);
  cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                    T2.translation(),
                                                    p_world);

  cv::Point2f u_inf, epi;
  float tol = 1e-4;

  // Depth from camera 1.
  EpipolarGeometry<float> epigeo1(K, Kinv);
  epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
  float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
  float idepth1 = epigeo1.disparityToInverseDepth(u1, u_inf, epi, disp1);
  EXPECT_NEAR(1.0f/(T1.inverse() * p_world)(2), idepth1, 1e-2);

  // Depth from camera 2.
  EpipolarGeometry<float> epigeo2(K, Kinv);
  epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
  float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
  float idepth2 = epigeo2.disparityToInverseDepth(u2, u_inf, epi, disp2);
  EXPECT_NEAR(1.0f/(T2.inverse() * p_world)(2), idepth2, tol);
}

/**
 * \brief Test point at (0, -1, 10) with cameras 1m apart and yawed 15 deg.
 *
 * T1 = 15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (0, -1, 10)
 */
TEST(EpipolarGeometryTest, disparityToInverseDepthTest4) {
  Eigen::Matrix3f K;
  K << 525.0f, 0.0f, 320.0f,
    0.0f, 525.0f, 240.0f,
    0.0f, 0.0f, 1.0f;
  Eigen::Matrix3f Kinv(K.inverse());

  // Geometry of cameras and landmarks.
  Eigen::AngleAxisf aa(M_PI/12, Eigen::Vector3f::UnitY());
  Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
  Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                  Eigen::Vector3f(1.0f, 0.0f, 0.0f));
  Eigen::Vector3f p_world(0.0f, -1.0f, 10.0f);

  // Project into cameras.
  Sophus::SE3f T12(T2.inverse() * T1);
  Sophus::SE3f T21(T1.inverse() * T2);

  cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                    T1.translation(),
                                                    p_world);
  cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                    T2.translation(),
                                                    p_world);

  cv::Point2f u_inf, epi;
  float tol = 1e-4;

  // Depth from camera 1.
  EpipolarGeometry<float> epigeo1(K, Kinv);
  epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
  float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
  float idepth1 = epigeo1.disparityToInverseDepth(u1, u_inf, epi, disp1);
  EXPECT_NEAR(1.0f/(T1.inverse() * p_world)(2), idepth1, 1e-2);

  // Depth from camera 2.
  EpipolarGeometry<float> epigeo2(K, Kinv);
  epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
  float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
  float idepth2 = epigeo2.disparityToInverseDepth(u2, u_inf, epi, disp2);
  EXPECT_NEAR(1.0f/(T2.inverse() * p_world)(2), idepth2, tol);
}


/**
 * \brief Test point at (1, 0, 10) with cameras 1m apart and yawed -15 deg.
 *
 * Test projecting point between two cameras.
 *
 * T1 = -15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (1, 0, 10)
 */
TEST(EpipolarGeometryTest, projectTest1) {
  Eigen::Matrix3f K;
  K << 525.0f, 0.0f, 320.0f,
    0.0f, 525.0f, 240.0f,
    0.0f, 0.0f, 1.0f;
  Eigen::Matrix3f Kinv(K.inverse());

  // Geometry of cameras and landmarks.
  Eigen::AngleAxisf aa(-M_PI/12, Eigen::Vector3f::UnitY());
  Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
  Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                  Eigen::Vector3f(1.0f, 0.0f, 0.0f));
  Eigen::Vector3f p_world(1.0f, 0.0f, 10.0f);

  // Project into cameras.
  Sophus::SE3f T12(T2.inverse() * T1);
  Sophus::SE3f T21(T1.inverse() * T2);

  cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                    T1.translation(),
                                                    p_world);
  cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                    T2.translation(),
                                                    p_world);

  EpipolarGeometry<float> epigeo(K, Kinv);

  epigeo.loadGeometry(T21.unit_quaternion(), T21.translation());

  cv::Point2f u_cmp = epigeo.project(u2, 1.0f/p_world(2));
  EXPECT_NEAR(u1.x, u_cmp.x, 1e-4);
  EXPECT_NEAR(u1.y, u_cmp.y, 1e-4);
}

}  // namespace stereo

}  // namespace flame
