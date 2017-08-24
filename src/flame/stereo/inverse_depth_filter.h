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
 * @file inverse_depth_filter.h
 * @author W. Nicholas Greene
 * @date 2016-09-17 00:23:50 (Sat)
 */

#pragma once

#include "flame/stereo/line_stereo.h"
#include "flame/stereo/epipolar_geometry.h"

namespace flame {

namespace stereo {

/**
 * @brief Namespace for functions that perform LSD-SLAM style inverse depth
 * filtering.
 */
namespace inverse_depth_filter {

enum Status {
  SUCCESS = 0,
  FAIL_REF_PATCH_GRADIENT = 1,
  FAIL_AMBIGUOUS_MATCH = 2,
  FAIL_MAX_COST = 3,
};

/**
 * @brief Parameter struct for inverse_depth_filter.
 */
struct Params {
  Params() {}

  int win_size = 5; // Size of cost window.
  float search_sigma = 2.0f; // Size of search region (+/- search_sigma * sigma).

  float min_grad_mag = 5.0f;

  // Bounds for idepth search region.
  float idepth_min = 1e-3;
  float idepth_max = 2.0f;

  // Bounds for epipolar search region.
  float epilength_min = 3.0f;
  float epilength_max = 32.0f;

  float process_var_factor = 1.01f; // IDepth variance is multiplied by this each timestep.
  float process_fail_var_factor = 1.1f; // IDepth variance is multiplied by this if search fails.

  stereo::line_stereo::Params sparams;
};

/**
 * @brief Predict step.
 */
bool predict(const stereo::EpipolarGeometry<float>& epigeo,
             float process_var_factor,
             const cv::Point2f& u_ref, float mu, float var,
             cv::Point2f* u_cmp, float* mu_pred, float* var_pred);

/**
 * @brief Compute search region based on idepth moments.
 */
bool getSearchRegion(const Params& params,
                     const stereo::EpipolarGeometry<float>& epigeo,
                     int width, int height,
                     const cv::Point2f& u_ref,
                     float mu_prior, float var_prior,
                     cv::Point2f* start, cv::Point2f* end, cv::Point2f* epi);

/**
 * @brief Search for matching pixel in search region.
 */
Status search(const Params& params,
              const stereo::EpipolarGeometry<float>& epigeo,
              float rescale_factor,
              const cv::Mat1b& img_ref,
              const cv::Mat1b& img_cmp,
              const cv::Point2f& u_ref,
              const cv::Point2f& u_start,
              const cv::Point2f& u_end,
              cv::Point2f* u_cmp);

/**
 * @brief Fuse measurement with pred.
 */
bool update(float mu_pred, float var_pred,
            float mu_meas, float var_meas,
            float* mu_post, float* var_post,
            float outlier_sigma_thresh = 2.0f);

}  // namespace inverse_depth_filter

}  // namespace stereo

}  // namespace flame
