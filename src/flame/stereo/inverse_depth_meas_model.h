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
 * @file inverse_depth_noise_model.h
 * @author W. Nicholas Greene
 * @date 2016-07-19 20:19:52 (Tue)
 */

#pragma once

#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>

#include <sophus/se3.hpp>

#include "flame/stereo/epipolar_geometry.h"

namespace flame {

namespace stereo {

/**
 * @brief Class to compute noisy inverse depth measurements using the LSD-SLAM
 * noise model.
 */
class InverseDepthMeasModel final {
 public:
  // Noise parameters.
  struct Params {
    Params() {}

    int win_size = 5; // Window size for padding.
    float pixel_var = 16.0f;
    float epipolar_line_var = 1.0f;

    bool verbose = false; // Print verbose errors.
  };

  InverseDepthMeasModel(const Eigen::Matrix3f& K,
                        const Eigen::Matrix3f& Kinv,
                        const Params& params = Params());
  ~InverseDepthMeasModel() = default;

  InverseDepthMeasModel(const InverseDepthMeasModel& rhs) = default;
  InverseDepthMeasModel& operator=(const InverseDepthMeasModel& rhs) = default;

  InverseDepthMeasModel(InverseDepthMeasModel&& rhs) = default;
  InverseDepthMeasModel& operator=(InverseDepthMeasModel&& rhs) = default;

  /**
   * @brief Load geometry data.
   */
  void loadGeometry(const Sophus::SE3f& T_ref, const Sophus::SE3f& T_cmp) {
    T_ref_to_cmp_ = T_cmp.inverse() * T_ref;
    epigeo_.loadGeometry(T_ref_to_cmp_.unit_quaternion(),
                         T_ref_to_cmp_.translation());
    inited_geo_ = true;
    return;
  }

  /**
   * @brief Load non-padded image data.
   */
  void loadImages(const cv::Mat& img_ref, const cv::Mat& img_cmp,
                  const cv::Mat1f& gradx_ref, const cv::Mat1f& grady_ref,
                  const cv::Mat1f& gradx_cmp, const cv::Mat1f& grady_cmp) {
    FLAME_ASSERT(img_ref.isContinuous());
    FLAME_ASSERT(img_cmp.isContinuous());
    FLAME_ASSERT(gradx_ref.isContinuous());
    FLAME_ASSERT(grady_ref.isContinuous());
    FLAME_ASSERT(gradx_cmp.isContinuous());
    FLAME_ASSERT(grady_cmp.isContinuous());

    // Allocate memory.
    int border = params_.win_size / 2 + 1;
    img_ref_.create(img_ref.rows + 2 * border, img_ref.cols + 2 * border, 0);
    img_cmp_.create(img_ref.rows + 2 * border, img_ref.cols + 2 * border, 0);
    gradx_ref_.create(img_ref.rows + 2 * border, img_ref.cols + 2 * border);
    grady_ref_.create(img_ref.rows + 2 * border, img_ref.cols + 2 * border);
    gradx_cmp_.create(img_cmp.rows + 2 * border, img_cmp.cols + 2 * border);
    grady_cmp_.create(img_cmp.rows + 2 * border, img_cmp.cols + 2 * border);

    // Pad images.
    cv::copyMakeBorder(img_ref, img_ref_, border, border, border, border,
                       cv::BORDER_REFLECT_101);
    cv::copyMakeBorder(img_cmp, img_cmp_, border, border, border, border,
                       cv::BORDER_REFLECT_101);
    cv::copyMakeBorder(gradx_ref, gradx_ref_, border, border, border, border,
                       cv::BORDER_CONSTANT);
    cv::copyMakeBorder(grady_ref, grady_ref_, border, border, border, border,
                       cv::BORDER_CONSTANT);
    cv::copyMakeBorder(gradx_cmp, gradx_cmp_, border, border, border, border,
                       cv::BORDER_CONSTANT);
    cv::copyMakeBorder(grady_cmp, grady_cmp_, border, border, border, border,
                       cv::BORDER_CONSTANT);

    inited_imgs_ = true;
    return;
  }

  /**
   * @brief Load padded image data.
   */
  void loadPaddedImages(const cv::Mat& img_ref, const cv::Mat& img_cmp,
                        const cv::Mat1f& gradx_ref, const cv::Mat1f& grady_ref,
                        const cv::Mat1f& gradx_cmp, const cv::Mat1f& grady_cmp) {
    FLAME_ASSERT(img_ref.isContinuous());
    FLAME_ASSERT(img_cmp.isContinuous());
    FLAME_ASSERT(gradx_ref.isContinuous());
    FLAME_ASSERT(grady_ref.isContinuous());
    FLAME_ASSERT(gradx_cmp.isContinuous());
    FLAME_ASSERT(grady_cmp.isContinuous());

    img_ref_ = img_ref;
    img_cmp_ = img_cmp;
    gradx_ref_ = gradx_ref;
    grady_ref_ = grady_ref;
    gradx_cmp_ = gradx_cmp;
    grady_cmp_ = grady_cmp;

    inited_imgs_ = true;
    return;
  }

  /**
   * @brief Return inverse depth mean and variance.
   */
  bool idepth(const cv::Point2f& u_ref, const cv::Point2f& u_cmp,
              float* mu, float* var) const;

 private:
  bool inited_geo_;
  bool inited_imgs_;
  Params params_;

  cv::Mat img_ref_;
  cv::Mat img_cmp_;
  cv::Mat1f gradx_ref_;
  cv::Mat1f grady_ref_;
  cv::Mat1f gradx_cmp_;
  cv::Mat1f grady_cmp_;

  Sophus::SE3f T_ref_to_cmp_;
  stereo::EpipolarGeometry<float> epigeo_;
};

}  // namespace stereo

}  // namespace flame
