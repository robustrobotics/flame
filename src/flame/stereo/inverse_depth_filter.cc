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
 * @file inverse_depth_filter.cc
 * @author W. Nicholas Greene
 * @date 2016-09-17 00:38:42 (Sat)
 */

#include "flame/stereo/inverse_depth_filter.h"

#include <limits>

#include "flame/utils/image_utils.h"

namespace flame {

namespace stereo {

namespace inverse_depth_filter {

bool predict(const stereo::EpipolarGeometry<float>& epigeo,
             float process_var_factor,
             const cv::Point2f& u_ref, float mu, float var,
             cv::Point2f* u_cmp, float* mu_pred,
             float* var_pred) {
  // Project pixel into new image.
  epigeo.project(u_ref, mu, u_cmp, mu_pred);
  if (*mu_pred < 0.0f) {
    // Point is behind camera.
    *mu_pred = 0.0f;
    *var_pred = 1e10;
    return false;
  }

  // Compute new variance using LSD-SLAM model.
  // Large idepth -> point is near -> large increase in variance.
  // Small idepth -> point is far -> small increase in variance.
  float var_factor4 = *mu_pred / mu;
  var_factor4 *= var_factor4;
  var_factor4 *= var_factor4;

  if (mu < 1e-6) {
    // If mu == 0, then var_factor4 is inf.
    var_factor4 = 1;
  }
  *var_pred = process_var_factor * var_factor4 * var; // + params_.process_var;

  return true;
}

bool getSearchRegion(const Params& params,
                     const stereo::EpipolarGeometry<float>& epigeo,
                     int width, int height,
                     const cv::Point2f& u_ref,
                     float mu_prior, float var_prior,
                     cv::Point2f* start, cv::Point2f* end, cv::Point2f* epi) {
  // Compute idepth search region.
  float id_min = params.idepth_min;
  float id_max = params.idepth_max;

  if (!std::isnan(mu_prior) && !std::isnan(var_prior)) {
    // Center around mean.
    float sigma = sqrt(var_prior);
    id_min = mu_prior - params.search_sigma * sigma;
    id_max = mu_prior + params.search_sigma * sigma;
  }

  // Clip to idepth bounds.
  id_min = (id_min < params.idepth_min) ? params.idepth_min : id_min;
  id_max = (id_max > params.idepth_max) ? params.idepth_max : id_max;
  if (id_max < id_min) {
    // Search region outside of bounds.
    return false;
  }

  // Get pixels in comparison image corresponding to search region.
  *start = epigeo.project(u_ref, id_min);
  *end = epigeo.project(u_ref, id_max);

  // Compute epipolar direction.
  cv::Point2f diff(*end - *start);
  float epilength = sqrt(diff.x * diff.x + diff.y * diff.y);
  if (epilength <= 0) {
    // ABORT! ABORT! ABORT!
    return false;
  }

  *epi = cv::Point2f(diff.x/epilength, diff.y/epilength);

  // Clip to valid region.
  // 1pixel border to be safe.
  cv::Rect valid_region(1, 1, width-2, height-2);

  FLAME_ASSERT(!std::isnan(start->x));
  FLAME_ASSERT(!std::isnan(start->y));
  FLAME_ASSERT(!std::isnan(end->x));
  FLAME_ASSERT(!std::isnan(end->y));

  float start_clipx, start_clipy;
  float end_clipx, end_clipy;
  if (!utils::clipLineLiangBarsky(valid_region.tl().x, valid_region.br().x,
                                  valid_region.tl().y, valid_region.br().y,
                                  start->x, start->y,
                                  end->x, end->y,
                                  &start_clipx, &start_clipy,
                                  &end_clipx, &end_clipy)) {
    // Epipolar line is entirely outside valid region.
    return false;
  }

  start->x = start_clipx;
  start->y = start_clipy;
  end->x = end_clipx;
  end->y = end_clipy;

  // Update epilength.
  diff = *end - *start;
  epilength = sqrt(diff.x * diff.x + diff.y * diff.y);
  if (epilength <= 0) {
    // ABORT! ABORT! ABORT!
    return false;
  }

  // Clip to disparity search region.
  if (epilength < params.epilength_min) {
    // Epilength is too short - pad.
    float pad = (params.epilength_min - epilength) / 2.0f;
    *start -= pad * (*epi);
    *end += pad * (*epi);
  }

  if (epilength > params.epilength_max) {
    epilength = params.epilength_max;

    // Center at search region midpoint.
    // *start = mid - (epilength / 2) * (*epi);
    // *end = mid + (epilength / 2) * (*epi);

    // Clip far end.
    // *start = (*end) - epilength * (*epi);

    // Clip near end.
    *end = (*start) + epilength * (*epi);
  }

  FLAME_ASSERT(!std::isnan(start->x));
  FLAME_ASSERT(!std::isnan(start->y));
  FLAME_ASSERT(!std::isnan(end->x));
  FLAME_ASSERT(!std::isnan(end->y));

  // Check clipping again.
  if (!utils::clipLineLiangBarsky(valid_region.tl().x, valid_region.br().x,
                                  valid_region.tl().y, valid_region.br().y,
                                  start->x, start->y,
                                  end->x, end->y,
                                  &start_clipx, &start_clipy,
                                  &end_clipx, &end_clipy)) {
    // Epipolar line is entirely outside valid region.
    return false;
  }

  start->x = start_clipx;
  start->y = start_clipy;
  end->x = end_clipx;
  end->y = end_clipy;

  return true;
}

Status search(const Params& params,
              const stereo::EpipolarGeometry<float>& epigeo,
              float rescale_factor,
              const cv::Mat1b& img_ref,
              const cv::Mat1b& img_cmp,
              const cv::Point2f& u_ref,
              const cv::Point2f& u_start,
              const cv::Point2f& u_end,
              cv::Point2f* u_cmp) {
  // Get epilines.
  cv::Point2f epi = u_end - u_start;
  float norm = sqrt(epi.x*epi.x + epi.y*epi.y);
  epi.x /= norm;
  epi.y /= norm;

  cv::Point2f epi_ref;
  epigeo.referenceEpiline(u_ref, &epi_ref);

  FLAME_ASSERT(params.win_size % 2 == 1);
  FLAME_ASSERT(params.win_size == 5);

  // Check ref patch bounds.
  FLAME_ASSERT((u_ref.x - 2*epi_ref.x*rescale_factor) >= 0);
  FLAME_ASSERT((u_ref.x + 2*epi_ref.x*rescale_factor) < img_ref.cols - 1);
  FLAME_ASSERT((u_ref.y - 2*epi_ref.y*rescale_factor) >= 0);
  FLAME_ASSERT((u_ref.y + 2*epi_ref.y*rescale_factor) < img_ref.rows - 1);

  // Compute reference patch.
  cv::Mat1f ref_patch(1, 5);
  ref_patch(0, 0) = utils::bilinearInterp<uint8_t, float>(img_ref,
                                                          u_ref.x - 2*epi_ref.x*rescale_factor,
                                                          u_ref.y - 2*epi_ref.y*rescale_factor);
  ref_patch(0, 1) = utils::bilinearInterp<uint8_t, float>(img_ref,
                                                          u_ref.x - epi_ref.x*rescale_factor,
                                                          u_ref.y - epi_ref.y*rescale_factor);
  ref_patch(0, 2) = utils::bilinearInterp<uint8_t, float>(img_ref, u_ref.x, u_ref.y);
  ref_patch(0, 3) = utils::bilinearInterp<uint8_t, float>(img_ref,
                                                          u_ref.x + epi_ref.x*rescale_factor,
                                                          u_ref.y + epi_ref.y*rescale_factor);
  ref_patch(0, 4) = utils::bilinearInterp<uint8_t, float>(img_ref,
                                                          u_ref.x + 2*epi_ref.x*rescale_factor,
                                                          u_ref.y + 2*epi_ref.y*rescale_factor);

  // Check gradient along this patch to make sure there's information to
  // constrain matching.
  float ref_patch_grad = 0.0f;
  for (int ii = 1; ii < ref_patch.cols; ++ii) {
    float abs_grad = utils::fast_abs(ref_patch(ii) - ref_patch(ii - 1));
    if (abs_grad > ref_patch_grad) {
      ref_patch_grad = abs_grad;
    }
  }
  if (ref_patch_grad < params.min_grad_mag) {
    return Status::FAIL_REF_PATCH_GRADIENT;
  }

  // Do line stereo.
  float residual = std::numeric_limits<float>::max();
  auto result = stereo::line_stereo::match(rescale_factor, ref_patch, img_cmp,
                                           u_start, u_end, u_cmp, &residual,
                                           params.sparams);

  if (result != stereo::line_stereo::Status::SUCCESS) {
    // Stereo match not successful.
    bool verbose = false;
    if (verbose) {
      fprintf(stderr, "epi_ref = (%f, %f)\n", epi_ref.x, epi_ref.y);
      fprintf(stderr, "InverseDepthFilter[FAIL=%i]: Stereo match unsuccessful for pixel (%f, %f), start = (%f, %f), end = (%f, %f)!\n",
              result, u_ref.x, u_ref.y, u_start.x, u_start.y, u_end.x, u_end.y);
    }
  }

  if (result == stereo::line_stereo::Status::FAIL_AMBIGUOUS_MATCH) {
    return Status::FAIL_AMBIGUOUS_MATCH;
  } else if (result == stereo::line_stereo::Status::FAIL_MAX_COST) {
    return Status::FAIL_MAX_COST;
  } else if (result != stereo::line_stereo::Status::SUCCESS) {
    fprintf(stderr, "inverse_depth_filter::search: Unrecognized status!\n");
    FLAME_ASSERT(false);
  }

  return Status::SUCCESS;
}

bool update(float mu_pred, float var_pred,
            float mu_meas, float var_meas,
            float* mu_post, float* var_post,
            float outlier_sigma_thresh) {
  if (!std::isnan(mu_pred) && (mu_pred > 0.0f)) {
    // Fu-sion...HA!
    float w = var_pred + var_meas;
    *mu_post = (var_meas * mu_pred + var_pred * mu_meas) / w;
    *var_post = (var_pred * var_meas) / w;
  } else {
    // First detection, set to measurement.
    *mu_post = mu_meas;
    *var_post = var_meas;
  }

  // Perform Chi-square/mahlanobis distance test.
  float res = (mu_meas - mu_pred);
  float dist = res * res / var_pred;
  if (dist > outlier_sigma_thresh * outlier_sigma_thresh) {
    // Measurement is far away from prediction. Probably an outlier.
    bool verbose = false;
    if (verbose) {
      fprintf(stderr, "InverseDepthFilter[ERROR]: Measurement rejected pred = (%f, %f) meas = (%f, %f) res = %f sigma_dist = %f!\n",
              mu_pred, var_pred, mu_meas, var_meas, res, dist);
    }
    return false;
  }

  // Make sure idepth is >= 0.
  *mu_post = (*mu_post <= 0) ? 0.0f : *mu_post;

  FLAME_ASSERT(!std::isnan(*mu_post));
  FLAME_ASSERT(!std::isnan(*var_post));
  FLAME_ASSERT(*mu_post >= 0);
  FLAME_ASSERT(*var_post >= 0);

  return true;
}

}  // namespace inverse_depth_filter

}  // namespace stereo

}  // namespace flame
