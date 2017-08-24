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
 * @file line_stereo.h
 * @author W. Nicholas Greene
 * @date 2017-06-16 20:41:47 (Fri)
 */

#pragma once

#include <limits>

#include <opencv2/core.hpp>

#include "flame/utils/assert.h"
#include "flame/utils/image_utils.h"

namespace flame {

namespace stereo {

namespace line_stereo {

enum Status {
  SUCCESS = 0,
  FAIL_AMBIGUOUS_MATCH = 1,
  FAIL_MAX_COST = 2,
};

/**
 * @brief Parameter struct for line_stereo::match().
 */
struct Params {
  float max_cost = 1300.0f; // Value for 5-sample SSD.

  bool do_subpixel = true;

  float sample_dist = 1.0f; // Distance in pixels between samples.

  // For a valid match, the second best cost must be greater than this factor
  // times the best cost.
  float second_best_factor = 1.5f;

  bool verbose = false;
};

/**
 * @brief Match a 1D linear patch along the epipolar line.
 *
 * The comparison patch will be updating using a sliding window.
 *
 * **Derived from LSD-SLAM.**
 *
 * @param[in] ref_patch 1D reference patch.
 * @param[in] rescale_factor Warp of reference patch.
 * @param[in] start Start of search region in second image.
 * @param[in] end End of search region in second image.
 * @param[out] u_cmp Matched pixel.
 * @param[in] params Parameter struct.
 */
inline Status match(float rescale_factor,
                    const cv::Mat1f& ref_patch,
                    const cv::Mat1b& img_cmp,
                    const cv::Point2f& start, const cv::Point2f& end,
                    cv::Point2f* u_cmp, float* residual,
                    const Params& params = Params()) {
  FLAME_ASSERT(ref_patch.rows == 1);
  FLAME_ASSERT(ref_patch.cols == 5);

  // These should be warped based on rescale_factor.
  float realVal_m2 = ref_patch(0, 0);
  float realVal_m1 = ref_patch(0, 1);
  float realVal = ref_patch(0, 2);
  float realVal_p1 = ref_patch(0, 3);
  float realVal_p2 = ref_patch(0, 4);

  // calculate increments in which we will step through the epipolar line.
  // they are sampleDist (or half sample dist) long
  float incx = end.x - start.x;
  float incy = end.y - start.y;
  float eplLength = sqrt(incx*incx + incy*incy);

  incx *= params.sample_dist / eplLength;
  incy *= params.sample_dist / eplLength;

  // from here on:
  // - pInf: search start-point
  // - p0: search end-point
  // - incx, incy: search steps in pixel
  // - eplLength, min_idepth, max_idepth: determines search-resolution, i.e. the result's variance.
  float cpx = start.x;
  float cpy = start.y;

  float val_cp_m2 = utils::bilinearInterp<uint8_t, float>(img_cmp,
                                                          cpx - 2.0f * incx,
                                                          cpy - 2.0f * incy);
  float val_cp_m1 = utils::bilinearInterp<uint8_t, float>(img_cmp,
                                                          cpx - incx,
                                                          cpy - incy);
  float val_cp = utils::bilinearInterp<uint8_t, float>(img_cmp, cpx, cpy);
  float val_cp_p1 = utils::bilinearInterp<uint8_t, float>(img_cmp,
                                                          cpx + incx,
                                                          cpy + incy);
  float val_cp_p2;

  /*
   * Subsequent exact minimum is found the following way:
   * - assuming lin. interpolation, the gradient of Error at p1 (towards p2) is given by
   *   dE1 = -2sum(e1*e1 - e1*e2)
   *   where e1 and e2 are summed over, and are the residuals (not squared).
   *
   * - the gradient at p2 (coming from p1) is given by
   * 	 dE2 = +2sum(e2*e2 - e1*e2)
   *
   * - linear interpolation => gradient changes linearely; zero-crossing is hence given by
   *   p1 + d*(p2-p1) with d = -dE1 / (-dE1 + dE2).
   *
   * => I for later exact min calculation, I need sum(e_i*e_i),sum(e_{i-1}*e_{i-1}),sum(e_{i+1}*e_{i+1})
   *    and sum(e_i * e_{i-1}) and sum(e_i * e_{i+1}),
   *    where i is the respective winning index.
   */

  // walk in equally sized steps, starting at depth=infinity.
  int loopCounter = 0;
  float best_match_x = -1;
  float best_match_y = -1;
  float best_match_err = std::numeric_limits<float>::max();
  float best_patch[5];
  float second_best_match_err = std::numeric_limits<float>::max();

  // best pre and post errors.
  float best_match_errPre = std::numeric_limits<float>::quiet_NaN();
  float best_match_errPost = std::numeric_limits<float>::quiet_NaN();
  float best_match_DiffErrPre = std::numeric_limits<float>::quiet_NaN();
  float best_match_DiffErrPost = std::numeric_limits<float>::quiet_NaN();
  bool bestWasLastLoop = false;

  float eeLast = -1;  // final error of last comp.

  // alternating intermediate vars
  float e1A = std::numeric_limits<float>::quiet_NaN();
  float e1B = std::numeric_limits<float>::quiet_NaN();
  float e2A = std::numeric_limits<float>::quiet_NaN();
  float e2B = std::numeric_limits<float>::quiet_NaN();
  float e3A = std::numeric_limits<float>::quiet_NaN();
  float e3B = std::numeric_limits<float>::quiet_NaN();
  float e4A = std::numeric_limits<float>::quiet_NaN();
  float e4B = std::numeric_limits<float>::quiet_NaN();
  float e5A = std::numeric_limits<float>::quiet_NaN();
  float e5B = std::numeric_limits<float>::quiet_NaN();

  int loopCBest = -1;
  int loopCSecond = -1;
  while (((incx < 0) == (cpx > end.x) && (incy < 0) == (cpy > end.y))
         || loopCounter == 0) {
    // interpolate one new point
    val_cp_p2 = utils::bilinearInterp<uint8_t, float>(img_cmp,
                                                      cpx + 2 * incx,
                                                      cpy + 2 * incy);

    // hacky but fast way to get error and differential error: switch buffer variables for last loop.
    float ee = 0.0f;
    if (loopCounter % 2 == 0) {
      // calc error and accumulate sums.
      e1A = val_cp_p2 - realVal_p2;
      ee += e1A*e1A;
      e2A = val_cp_p1 - realVal_p1;
      ee += e2A*e2A;
      e3A = val_cp - realVal;
      ee += e3A*e3A;
      e4A = val_cp_m1 - realVal_m1;
      ee += e4A*e4A;
      e5A = val_cp_m2 - realVal_m2;
      ee += e5A*e5A;
    } else {
      // calc error and accumulate sums.
      e1B = val_cp_p2 - realVal_p2;
      ee += e1B*e1B;
      e2B = val_cp_p1 - realVal_p1;
      ee += e2B*e2B;
      e3B = val_cp - realVal;
      ee += e3B*e3B;
      e4B = val_cp_m1 - realVal_m1;
      ee += e4B*e4B;
      e5B = val_cp_m2 - realVal_m2;
      ee += e5B*e5B;
    }

    // do I have a new winner??
    // if so: set.
    if (ee < best_match_err) {
      // put to second-best
      second_best_match_err = best_match_err;
      loopCSecond = loopCBest;

      // set best.
      best_match_err = ee;
      loopCBest = loopCounter;

      best_match_errPre = eeLast;
      best_match_DiffErrPre = e1A*e1B + e2A*e2B + e3A*e3B + e4A*e4B + e5A*e5B;
      best_match_errPost = -1;
      best_match_DiffErrPost = -1;

      best_match_x = cpx;
      best_match_y = cpy;
      bestWasLastLoop = true;

      best_patch[0] = val_cp_m2;
      best_patch[1] = val_cp_m1;
      best_patch[2] = val_cp;
      best_patch[3] = val_cp_p1;
      best_patch[4] = val_cp_p2;
    } else {
      // otherwise: the last might be the current winner, in which case i have to save these values.
      if (bestWasLastLoop) {
        best_match_errPost = ee;
        best_match_DiffErrPost = e1A*e1B + e2A*e2B + e3A*e3B +
            e4A*e4B + e5A*e5B;
        bestWasLastLoop = false;
      }

      // collect second-best:
      // just take the best of all that are NOT equal to current best.
      if (ee < second_best_match_err) {
        second_best_match_err = ee;
        loopCSecond = loopCounter;
      }
    }

    // shift everything one further.
    eeLast = ee;
    val_cp_m2 = val_cp_m1;
    val_cp_m1 = val_cp;
    val_cp = val_cp_p1;
    val_cp_p1 = val_cp_p2;

    cpx += incx;
    cpy += incy;

    loopCounter++;
  }

  *residual = best_match_err;

  if (best_match_err > 4.0f * params.max_cost) {
    // Best error exceeded threshold.
    if (params.verbose) {
      fprintf(stderr, "line_stereo::match[FAIL]: best_cost (%f) is greater than threshold (%f)\n",
              best_match_err, 4.0f * params.max_cost);
      fprintf(stderr, "ref = [%.2f, %.2f, %.2f, %.2f, %.2f], cmp = [%.2f, %.2f, %.2f, %.2f, %.2f]\n",
              ref_patch(0, 0), ref_patch(0, 1), ref_patch(0, 2), ref_patch(0, 3), ref_patch(0, 4),
              best_patch[0], best_patch[1], best_patch[2], best_patch[3], best_patch[4]);
    }
    return Status::FAIL_MAX_COST;
  }

  // Check if best match is significantly better than second best.
  if ((utils::fast_abs(loopCBest - loopCSecond) > 1.0f) &&
      (params.second_best_factor * best_match_err > second_best_match_err)) {
    if (params.verbose) {
      fprintf(stderr, "line_stereo::match[FAIL]: Ambiguous match (best_cost %f, second_best_cost = %f, ratio = %f)\n",
              best_match_err, second_best_match_err, second_best_match_err / best_match_err);
      fprintf(stderr, "ref = [%.2f, %.2f, %.2f, %.2f, %.2f], cmp = [%.2f, %.2f, %.2f, %.2f, %.2f]\n",
              ref_patch(0, 0), ref_patch(0, 1), ref_patch(0, 2), ref_patch(0, 3), ref_patch(0, 4),
              best_patch[0], best_patch[1], best_patch[2], best_patch[3], best_patch[4]);
    }
    return Status::FAIL_AMBIGUOUS_MATCH;
  }

  bool didSubpixel = false;
  if (params.do_subpixel) {
    // ================== compute exact match =========================
    // compute gradients (they are actually only half the real gradient)
    float gradPre_pre = -(best_match_errPre - best_match_DiffErrPre);
    float gradPre_this = +(best_match_err - best_match_DiffErrPre);
    float gradPost_this = -(best_match_err - best_match_DiffErrPost);
    float gradPost_post = +(best_match_errPost - best_match_DiffErrPost);

    // final decisions here.
    bool interpPost = false;
    bool interpPre = false;

    // if one is oob: return false.
    if (best_match_errPre < 0 || best_match_errPost < 0) {
      // stats->num_stereo_invalid_atEnd++;
    } else if ((gradPost_this < 0) ^ (gradPre_this < 0)) {
      // - if zero-crossing occurs exactly in between (gradient Inconsistent),
      // return exact pos, if both central gradients are small compared to their counterpart.
      if (gradPost_this * gradPost_this > 0.1f * 0.1f * gradPost_post * gradPost_post ||
          gradPre_this*gradPre_this > 0.1f*0.1f*gradPre_pre*gradPre_pre) {
        // stats->num_stereo_invalid_inexistantCrossing++;
      }
    } else if ((gradPre_pre < 0) ^ (gradPre_this < 0)) {
      // if pre has zero-crossing
      // if post has zero-crossing
      if ((gradPost_post < 0) ^ (gradPost_this < 0)) {
        // stats->num_stereo_invalid_twoCrossing++;
      } else {
        interpPre = true;
      }
    } else if ((gradPost_post < 0) ^ (gradPost_this < 0)) {
      // if post has zero-crossing
      interpPost = true;
    } else {
      // if none has zero-crossing
      // stats->num_stereo_invalid_noCrossing++;
    }

    // DO interpolation!
    // minimum occurs at zero-crossing of gradient, which is a straight line => easy to compute.
    // the error at that point is also computed by just integrating.
    if (interpPre) {
      float d = gradPre_this / (gradPre_this - gradPre_pre);
      best_match_x -= d*incx;
      best_match_y -= d*incy;
      best_match_err = best_match_err - 2*d*gradPre_this -
        (gradPre_pre - gradPre_this)*d*d;
      didSubpixel = true;
    } else if (interpPost) {
      float d = gradPost_this / (gradPost_this - gradPost_post);
      best_match_x += d*incx;
      best_match_y += d*incy;
      best_match_err = best_match_err + 2*d*gradPost_this +
        (gradPost_post - gradPost_this)*d*d;
      didSubpixel = true;
    } else {
    }
  }

  *residual = best_match_err;

  // sample_dist is the distance in pixel at which the realVal's were sampled
  float sampleDist = params.sample_dist * rescale_factor;

  float gradAlongLine = 0;
  float tmp = realVal_p2 - realVal_p1;
  gradAlongLine += tmp * tmp;
  tmp = realVal_p1 - realVal;
  gradAlongLine += tmp * tmp;
  tmp = realVal - realVal_m1;
  gradAlongLine += tmp * tmp;
  tmp = realVal_m1 - realVal_m2;
  gradAlongLine += tmp * tmp;

  gradAlongLine /= sampleDist * sampleDist;

  // check if interpolated error is OK. use evil hack to allow more error if there is a lot of gradient.
  if (best_match_err > params.max_cost + sqrtf(gradAlongLine) * 20) {
    if (params.verbose) {
      fprintf(stderr, "line_stereo::match[FAIL]: best_cost (%f) is greater than threshold (%f)\n",
              best_match_err, params.max_cost + sqrtf(gradAlongLine) * 20);
      fprintf(stderr, "ref = [%.2f, %.2f, %.2f, %.2f, %.2f], cmp = [%.2f, %.2f, %.2f, %.2f, %.2f]\n",
              ref_patch(0, 0), ref_patch(0, 1), ref_patch(0, 2), ref_patch(0, 3), ref_patch(0, 4),
              best_patch[0], best_patch[1], best_patch[2], best_patch[3], best_patch[4]);
    }
    return Status::FAIL_MAX_COST;
  }

  // if (params.verbose) {
  //   fprintf(stderr, "line_stereo::match[SUCCESS]\n");
  //   fprintf(stderr, "ref = [%.2f, %.2f, %.2f, %.2f, %.2f], cmp = [%.2f, %.2f, %.2f, %.2f, %.2f]\n",
  //           ref_patch(0, 0), ref_patch(0, 1), ref_patch(0, 2), ref_patch(0, 3), ref_patch(0, 4),
  //           best_patch[0], best_patch[1], best_patch[2], best_patch[3], best_patch[4]);
  // }

  u_cmp->x = best_match_x;
  u_cmp->y = best_match_y;

  return Status::SUCCESS;
}

}  // namespace line_stereo

}  // namespace stereo

}  // namespace flame
