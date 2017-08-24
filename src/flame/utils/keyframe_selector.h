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
 * @file keyframe_selector.h
 * @author W. Nicholas Greene
 * @date 2017-08-18 20:35:30 (Fri)
 */

#pragma once

#include <deque>
#include <tuple>

#include "flame/types.h"

namespace flame {

namespace utils {

/**
 * \brief Selects a comparison image from a pool of keyframes.
 */
class KeyFrameSelector final {
 public:
  /**
   * \brief Constructor.
   *
   * @param[in] max_keyframes Max num keyframes to maintain in pool.
   * @param[in] new_keyframe_thresh Threshold to create new keyframe.
   */
  KeyFrameSelector(const Matrix3f& K, int max_kfs, float new_kf_thresh);

  ~KeyFrameSelector() = default;

  KeyFrameSelector() = delete;
  KeyFrameSelector(const KeyFrameSelector& rhs) = delete;
  KeyFrameSelector& operator=(const KeyFrameSelector& rhs) = delete;

  KeyFrameSelector(KeyFrameSelector&& rhs) = delete;
  KeyFrameSelector& operator=(KeyFrameSelector&& rhs) = delete;

  /**
   * \brief Select a keyframe index for this new image.
   *
   * Will add new image to keyframe pool if it exceeds threshold.
   *
   * If no suitable keyframe found (e.g. no keyframes in pool yet, not enough
   * baseline yet, etc.), will return -1.
   *
   * @return Index of keyframe in pool.
   */
  int select(double new_time, const Image1b& new_img,
             const SE3f& new_pose);

  /**
   * \brief Get keyframe corresponding to idx.
   */
  void getKeyFrame(int idx, double* time, Image1b* img, SE3f* pose);

  /**
   * \brief Return number of keyframes in pool.
   */
  int size() { return num_kfs_; }

  /**
   * \brief Get the score associated with using ref_img as the keyframe.
   */
  static float score(int width, int height,
                     const Matrix3f& K, const Matrix3f& Kinv,
                     const SE3f& new_to_ref,
                     float min_depth = 1.0f, float max_depth = 10.0f,
                     float max_disparity = 128.0f);

 private:
  // Store the keyframe pool in circular buffers.
  std::deque<double> kf_times_;
  std::deque<Image1b> kf_imgs_;
  std::deque<SE3f> kf_poses_;

  // Camera matrix.
  Matrix3f K_;
  Matrix3f Kinv_;

  // Number of keyframes currently in pool.
  int num_kfs_;

  // Maximum number of keyframes to maintain in pool.
  int max_kfs_;

  // Threshold to create new keyframe.
  float new_kf_thresh_;
};

}  // namespace utils

}  // namespace flame
