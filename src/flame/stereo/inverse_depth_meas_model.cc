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
 * @file inverse_depth_meas_model.cc
 * @author W. Nicholas Greene
 * @date 2016-07-19 21:03:02 (Tue)
 */

#include "flame/stereo/inverse_depth_meas_model.h"

#include <stdio.h>

#include "flame/utils/assert.h"
#include "flame/utils/image_utils.h"

namespace flame {

namespace stereo {

InverseDepthMeasModel::InverseDepthMeasModel(const Eigen::Matrix3f& K,
                                             const Eigen::Matrix3f& Kinv,
                                             const Params& params) :
    inited_geo_(false),
    inited_imgs_(false),
    params_(params),
    img_ref_(),
    img_cmp_(),
    gradx_ref_(),
    grady_ref_(),
    gradx_cmp_(),
    grady_cmp_(),
    T_ref_to_cmp_(),
    epigeo_(K, Kinv) {}

bool InverseDepthMeasModel::idepth(const cv::Point2f& u_ref,
                                   const cv::Point2f& u_cmp,
                                   float* mu, float* var) const {
  FLAME_ASSERT(inited_geo_ && inited_imgs_);

  // Compute the inverse depth mean.
  cv::Point2f u_inf, epi;
  float disp = epigeo_.disparity(u_ref, u_cmp, &u_inf, &epi);

  // printf("epi = %f, %f\n", epi.x, epi.y);
  // printf("disp = %f\n", disp);

  if (disp < 1e-3) {
    if (params_.verbose) {
      fprintf(stderr, "IDepthMeasModel[ERROR]: NegativeDisparity: u_ref = (%f, %f), u_cmp = (%f, %f), u_inf = (%f, %f), disp = %f\n",
              u_ref.x, u_ref.y, u_cmp.x, u_cmp.y, u_inf.x, u_inf.y, disp);
    }

    // No dispariy = idepth information.
    *mu = 0.0f;
    *var = 1e10;
    return false;
  }

  *mu = epigeo_.disparityToInverseDepth(u_ref, u_inf, epi, disp);
  if (*mu < 0.0f) {
    if (params_.verbose) {
      fprintf(stderr, "IDepthMeasModel[ERROR]: NegativeIDepth: u_ref = (%f, %f), u_cmp = (%f, %f), u_inf = (%f, %f), disp = %f, idepth = %f\n",
              u_ref.x, u_ref.y, u_cmp.x, u_cmp.y, u_inf.x, u_inf.y, disp, *mu);
    }

    // No depth information.
    *mu = 0.0f;
    *var = 1e10;
    return false;
  }

  // Get the image gradient.
  cv::Point2f offset(params_.win_size/2+1, params_.win_size/2+1);
  float gx = utils::bilinearInterp<float, float>(gradx_cmp_,
                                                 u_cmp.x + offset.x,
                                                 u_cmp.y + offset.y);
  float gy = utils::bilinearInterp<float, float>(grady_cmp_,
                                                 u_cmp.x + offset.x,
                                                 u_cmp.y + offset.y);

  float gnorm = sqrt(gx * gx + gy * gy);
  if (gnorm < 1e-3) {
    if (params_.verbose) {
      fprintf(stderr, "IDepthMeasModel[ERROR]: NoGradient: u_ref = (%f, %f), u_cmp = (%f, %f), u_inf = (%f, %f), disp = %f, idepth = %f, grad = (%f, %f)\n",
              u_ref.x, u_ref.y, u_cmp.x, u_cmp.y, u_inf.x, u_inf.y, disp, *mu, gx, gy);
    }

    // No depth information.
    *mu = 0.0f;
    *var = 1e10;
    return false;
  }

  float ngx = gx / gnorm;
  float ngy = gy / gnorm;

  // printf("grad = %f, %f\n", gx, gy);

  // Compute geometry disparity variance.
  float epi_dot_ngrad = ngx * epi.x + ngy * epi.y;
  float geo_var = params_.epipolar_line_var / (epi_dot_ngrad * epi_dot_ngrad);

  if (utils::fast_abs(epi_dot_ngrad) < 1e-3) {
    if (params_.verbose) {
      fprintf(stderr, "IDepthMeasModel[ERROR]: NoEpiDotGradient: u_ref = (%f, %f), u_cmp = (%f, %f), u_inf = (%f, %f), disp = %f, idepth = %f, grad = (%f, %f), epi = (%f, %f)\n",
              u_ref.x, u_ref.y, u_cmp.x, u_cmp.y, u_inf.x, u_inf.y, disp, *mu, gx, gy, epi.x, epi.y);
    }

    // No depth information.
    *mu = 0.0f;
    *var = 1e10;
    return false;
  }

  // Compute photometric disparity variance.
  float epi_dot_grad = gx * epi.x + gy * epi.y;
  float photo_var = 2 * params_.pixel_var / (epi_dot_grad * epi_dot_grad);

  // Compute disparity to inverse depth scaling factor.
  float disp_min = disp - disp / 10;
  float disp_max = disp + disp / 10;
  float idepth_min = epigeo_.disparityToInverseDepth(u_ref, u_inf, epi, disp_min);
  float idepth_max = epigeo_.disparityToInverseDepth(u_ref, u_inf, epi, disp_max);

  float alpha = (idepth_max - idepth_min) / (disp_max - disp_min);

  float meas_var = alpha * alpha * (geo_var + photo_var);

  FLAME_ASSERT(!std::isnan(meas_var));
  FLAME_ASSERT(!std::isinf(meas_var));

  // printf("var1 = %f\n", meas_var);

  // float angle = Eigen::AngleAxisf(T_ref_to_cmp_.unit_quaternion()).angle();
  // float baseline = T_ref_to_cmp_.translation().squaredNorm();
  // meas_var = 0.000001f / baseline + 0.0001f / cos(angle);
  *var = meas_var;

  // printf("var2 = %f\n", meas_var);

  return true;
}

}  // namespace stereo

}  // namespace flame
