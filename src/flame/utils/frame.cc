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
 * @file frame.cc
 * @author W. Nicholas Greene
 * @date 2017-03-30 19:37:04 (Thu)
 */

#include "flame/utils/frame.h"

#include <limits>

#include "flame/utils/pyramids.h"

namespace flame {

namespace utils {

Frame::Ptr Frame::create(const SE3f& pose, const Image1b& img,
                         int id, int num_levels, int border) {
  Frame::Ptr frame = std::make_shared<Frame>(num_levels);

  frame->id = id;
  frame->pose = pose;

  // Create image pyramid.
  ImagePyramid img_pyr_tmp(utils::getGaussianPyramid(img, num_levels));
  // Need to cast down pyramid.
  frame->img = ImagePyramidb(img_pyr_tmp.begin(), img_pyr_tmp.end());

  // Create gradient pyramid.
  utils::getGradientPyramid(frame->img, &(frame->gradx), &(frame->grady));

  frame->idepthmap.resize(num_levels);

  for (int lvl = 0; lvl < num_levels; ++lvl) {
    // Create padded versions.
    int rlvl = frame->img[lvl].rows;
    int clvl = frame->img[lvl].cols;
    frame->img_pad[lvl].create(rlvl + 2 * border, clvl + 2 * border);
    frame->gradx_pad[lvl].create(rlvl + 2 * border, clvl + 2 * border);
    frame->grady_pad[lvl].create(rlvl + 2 * border, clvl + 2 * border);

    cv::copyMakeBorder(frame->img[lvl], frame->img_pad[lvl],
                       border, border, border, border, cv::BORDER_REFLECT_101);
    cv::copyMakeBorder(frame->gradx[lvl], frame->gradx_pad[lvl],
                       border, border, border, border, cv::BORDER_CONSTANT);
    cv::copyMakeBorder(frame->grady[lvl], frame->grady_pad[lvl],
                       border, border, border, border, cv::BORDER_CONSTANT);

    // Create idepthmap.
    frame->idepthmap[lvl] = Image1f(rlvl, clvl,
                                    std::numeric_limits<float>::quiet_NaN());
  }

  return frame;
}

}  // namespace utils

}  // namespace flame
