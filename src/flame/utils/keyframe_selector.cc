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
 * @file keyframe_selector.cc
 * @author W. Nicholas Greene
 * @date 2017-08-18 20:36:40 (Fri)
 */

#include "flame/utils/keyframe_selector.h"

#include <iostream>

#include <algorithm>
#include <vector>
#include <limits>

#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/exception/diagnostic_information.hpp>

#include "flame/utils/assert.h"

// Some typedefs
namespace bgeo = boost::geometry;
typedef bgeo::model::d2::point_xy<float> BPoint;
typedef bgeo::model::polygon<BPoint> BPolygon;

namespace {

/**
 * \brief Fast round to int.
 *
 * std::round() is notoriously slow (I think because it needs to handle all
 * edge cases). This is a faster implementation I found here:
 * https://stackoverflow.com/questions/485525/round-for-float-in-c
 */
int fast_roundf(float r) {
  return (r > 0.0f) ? (r + 0.5f) : (r - 0.5f);
}

}  // namespace

namespace flame {

namespace utils {

KeyFrameSelector::KeyFrameSelector(const Matrix3f& K, int max_kfs,
                                   float new_kf_thresh) :
    kf_times_(),
    kf_imgs_(),
    kf_poses_(),
    K_(K),
    Kinv_(K.inverse()),
    num_kfs_(0),
    max_kfs_(max_kfs),
    new_kf_thresh_(new_kf_thresh) {}

int KeyFrameSelector::select(double new_time, const Image1b& new_img,
                             const SE3f& new_pose) {
  int best_idx = -1;
  float best_score = std::numeric_limits<float>::lowest();
  if (num_kfs_ > 0) {
    // Walk through keyframes and select best one.
    for (int ii = 0; ii < num_kfs_; ++ii) {
      float score_ii = score(new_img.cols, new_img.rows,
                             K_, Kinv_, kf_poses_[ii].inverse() * new_pose);

      bool debug_print_kf_score_info = false; // TODO(wng): Make this dynamic.
      if (debug_print_kf_score_info) {
        printf("kf score(%i) = %f, num_kfs = %i, %i, baseline = %f\n",
               ii, score_ii, num_kfs_, kf_times_.size(),
               (new_pose.translation() - kf_poses_[ii].translation()).norm());
      }

      if (score_ii > best_score) {
        best_score = score_ii;
        best_idx = ii;
      }
    }
  }

  bool debug_print_best_kfs_idx = false; // TODO(wng): Make this dynamic.
  if (debug_print_best_kfs_idx) {
    printf("best_idx = %i, score = %f\n", best_idx, best_score);
  }

  // Pick a new keyframe based on distance from last keyframe.
  if ((num_kfs_ == 0) ||
      ((new_pose.translation() - kf_poses_.back().translation()).norm() >
       new_kf_thresh_)) {
    // Add keyframe to pool.
    kf_times_.push_back(new_time);
    kf_imgs_.push_back(new_img);
    kf_poses_.push_back(new_pose);

    if (kf_times_.size() > max_kfs_) {
      kf_times_.pop_front();
      kf_imgs_.pop_front();
      kf_poses_.pop_front();
    }

    num_kfs_ = kf_times_.size();
    best_idx--;
  }

  return best_idx;
}

void KeyFrameSelector::getKeyFrame(int idx, double* time, Image1b* img,
                                   SE3f* pose) {
  FLAME_ASSERT(idx >= 0);
  FLAME_ASSERT(idx < max_kfs_);

  *time = kf_times_[idx];
  *img = kf_imgs_[idx];
  *pose = kf_poses_[idx];

  return;
}

/**
 * Adapted from MobileFusion - Ondruska et al.
 */
float KeyFrameSelector::score(int width, int height,
                              const Matrix3f& K, const Matrix3f& Kinv,
                              const SE3f& new_to_ref,
                              float min_depth, float max_depth,
                              float max_disparity) {
  /*==================== Compute orientation score ====================*/
  // Sparse features are sensitive to rotation. Try to find a keyframe that has
  // the same orientation.
  float angle = AngleAxisf(new_to_ref.unit_quaternion()).angle();
  float S_orientation = 0.5 * (cos(angle) + 1); // Use cos to get around wrapping to (-pi, pi).

  // Apply hard check for vastly different orientations.
  float cos_angle_thresh = 0.5 * (cos(60.0f * M_PI / 180.0f) + 1);
  if (S_orientation < cos_angle_thresh) {
    // printf("Inward normal does not align with viewing direction from reference image!\n");
    return std::numeric_limits<float>::lowest();
  }

  /*==================== Copmute overlap score ====================*/
  // Assume structure at max_depth relative to new_img. Project into ref_img and
  // compute the overlap.
  Vector3f u_new_corners[4];
  u_new_corners[0] << 0.0f, 0.0f, 1.0f;
  u_new_corners[1] << 0.0f, height-1, 1.0f;
  u_new_corners[2] << width-1, height-1, 1.0f;
  u_new_corners[3] << width-1, 0.0f, 1.0f;

  Vector3f u_ref_corners[4];
  for (int ii = 0; ii < 4; ++ii) {
    u_ref_corners[ii] = (K * (new_to_ref.unit_quaternion() *
                              (max_depth * (Kinv * u_new_corners[ii])) +
                              new_to_ref.translation()));
    u_ref_corners[ii] /= u_ref_corners[ii](2);
  }

  // Compute area of intersection using boost.
  // Extra point is needed to close polygon.
  std::vector<BPoint> new_pts(5);
  std::vector<BPoint> ref_pts(5);
  for (int ii = 0; ii < 4; ++ii) {
    new_pts[ii] = BPoint(u_new_corners[ii](0), u_new_corners[ii](1));
    ref_pts[ii] = BPoint(u_ref_corners[ii](0), u_ref_corners[ii](1));
  }
  new_pts[4] = new_pts[0];
  ref_pts[4] = ref_pts[0];

  BPolygon new_poly;
  bgeo::assign_points(new_poly, new_pts);
  BPolygon ref_poly;
  bgeo::assign_points(ref_poly, ref_pts);

  float S_overlap = 0.0f;

  // Check if reference polygon (u_ref_corners) intersects itself.
  bool ref_poly_self_intersection = bgeo::intersects(ref_poly);
  if (ref_poly_self_intersection) {
    fprintf(stderr, "KeyFrameSelector: Reference polygon self-intersection detected!\n");
    return std::numeric_limits<float>::lowest();
  }

  // Check if the reference polygon and new polygon do not intersect.
  bool new_ref_intersection = bgeo::intersects(new_poly, ref_poly);
  if (!new_ref_intersection) {
    fprintf(stderr, "KeyFrameSelector: No intersection detected!\n");
    return std::numeric_limits<float>::lowest();
  }

  // Use try-catch block. Sometimes boost throws a
  // boost::geometry::overlay_invalid_input_exception exception for some reason.
  float area = 0.0f;
  try {
    std::deque<BPolygon> intersection_poly;
    bgeo::intersection(new_poly, ref_poly, intersection_poly);

    // Check if intersection...intersects itself.
    bool self_intersects = bgeo::intersects(intersection_poly[0]);
    if (self_intersects) {
      fprintf(stderr, "KeyFrameSelector: Self intersection detected2!\n");
      return std::numeric_limits<float>::lowest();
    }

    area = bgeo::area(intersection_poly[0]);
  } catch (...) {
    fprintf(stderr, "WARNING: Caught exception from boost::intersection!\n");
    return std::numeric_limits<float>::lowest();
  }

  S_overlap = area / ((width-1) * (height-1));

  /*==================== Compute disparity score ====================*/
  // Assume structure at min_depth relative to new_img. This will generate
  // maximum disparity. Compare maximum disparity to allowable maximum
  // disparity (better if they're about the same).
  Vector3f u_new_test(width / 4, height / 4, 1.0f); // Pick point not on optical axis.

  // Point at infinite depth.
  Vector3f p_ref_inf = K * new_to_ref.unit_quaternion() * Kinv * u_new_test;
  p_ref_inf /= p_ref_inf(2);

  // Point at minimum depth.
  Vector3f p_ref_min(K * (new_to_ref.unit_quaternion() *
                          (min_depth * (Kinv * u_new_test)) +
                          new_to_ref.translation()));
  p_ref_min /= p_ref_min(2);

  float disparity = (p_ref_min - p_ref_inf).norm();

  float S_disparity = -fabs(1.0f - disparity / max_disparity);

  bool debug_print_score = false;
  if (debug_print_score) {
    printf("S_orient = %f, S_overlap = %f, S_disparity = %f\n",
           S_orientation, S_overlap, S_disparity);
  }

  return S_orientation + S_overlap + S_disparity;
}

}  // namespace utils

}  // namespace flame
