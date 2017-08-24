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
 * @file frame.h
 * @author W. Nicholas Greene
 * @date 2017-03-30 19:35:44 (Thu)
 */

#pragma once

#include <memory>
#include <vector>

#include "flame/types.h"
#include "flame/utils/triangulator.h"

namespace flame {

namespace utils {

/**
 * @brief Struct to hold various image data.
 */
struct Frame {
  using Ptr = std::shared_ptr<Frame>;
  using ConstPtr = std::shared_ptr<const Frame>;

  explicit Frame(int num_lvls) :
      id(0), pose(),
      img(num_lvls), gradx(num_lvls), grady(num_lvls),
      img_pad(num_lvls), gradx_pad(num_lvls), grady_pad(num_lvls) {}

  // Create a frame pointer object from a pose and raw image.
  static Frame::Ptr create(const Sophus::SE3f& pose, const cv::Mat1b& img,
                           int id, int num_levels, int border);

  uint32_t id; // Image number/ID.
  SE3f pose; // Pose of this image.
  ImagePyramidb img; // Image pyramid.
  ImagePyramidf gradx; // Horizontal gradient pyramid.
  ImagePyramidf grady; // Vertical gradient pyramid.
  ImagePyramidb img_pad; // Padded pyramids for LK.
  ImagePyramidf gradx_pad;
  ImagePyramidf grady_pad;
  ImagePyramidf idepthmap; // Dense inverse depthmap.

  // Mesh stuff.
  std::vector<Triangle> tris;
  std::vector<Edge> edges;
  std::vector<Point2f> vtx;
  std::vector<float> vtx_idepths;
  std::vector<float> vtx_w1;
  std::vector<float> vtx_w2;
};

}  // namespace utils

}  // namespace flame
