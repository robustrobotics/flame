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
 * @file pyramids.h<flame>
 * @author W. Nicholas Greene
 * @date 2017-08-18 20:07:46 (Fri)
 */

#pragma once

#include <vector>

#include "flame/utils/image_utils.h"

namespace flame {

namespace utils {

/**
 * \brief Create a Gaussian pyramid.
 *
 * The input image is placed into the lowest level.
 *
 * @param[in] img Base level image.
 * @param[in] num_lvl Number of levels in pyramid.
 * @return Image pyramid.
 */
inline ImagePyramid getGaussianPyramid(const cv::Mat& img, int num_lvl = 5) {
  ImagePyramid ret(num_lvl);

  ret[0] = img;
  for (int lvl = 1; lvl < num_lvl; ++lvl) {
    cv::pyrDown(ret[lvl - 1], ret[lvl]);
  }

  return ret;
}

/**
 * \brief Arrange an image pyramid for display in a single image.
 *
 * Arranges the levels of the image pyramid next to each other for display.
 *
 * @param[in] pyr Input pyramid.
 * @param[out] out Single output image.
 */
void getPyramidDisplay(const ImagePyramid& pyr, cv::Mat* out);

/**
 * \brief Create a pyramid of image gradients.
 *
 * @param[in] pyr Image pyramid with 0 being the lowest level.
 * @param[out] gradx Horiztonal gradient.
 * @param[out] grady Vertical gradient.
 * @return Gradient pyramid.
 */
template <typename ChannelType, typename GradientType>
inline void getGradientPyramid(const ImagePyramid_<ChannelType>& pyr,
                               ImagePyramid_<GradientType>* gradx,
                               ImagePyramid_<GradientType>* grady) {
  FLAME_ASSERT(pyr.size() > 0);

  if (gradx->empty()) {
    gradx->resize(pyr.size());
  }
  if (grady->empty()) {
    grady->resize(pyr.size());
  }

  for (int lvl = 0; lvl < pyr.size(); ++lvl) {
    getCentralGradient<ChannelType, GradientType>(pyr[lvl], &((*gradx)[lvl]),
                                                  &((*grady)[lvl]));
  }

  return;
}

/**
 * \brief Return the magnitude of the image gradients.
 * @param[in] gradx Horizontal gradient.
 * @param[in] grady Vertical gradient.
 * @param[out] mag Magnitude.
 */
template <typename GradientType>
inline void getGradientMagPyramid(const ImagePyramid_<GradientType>& gradx,
                                  const ImagePyramid_<GradientType>& grady,
                                  ImagePyramid_<GradientType>* mag) {
  FLAME_ASSERT(!gradx.empty());
  FLAME_ASSERT(!grady.empty());
  FLAME_ASSERT(gradx.size() == grady.size());

  if (mag->empty()) {
    mag->resize(gradx.size());
  }

  for (int lvl = 0; lvl < gradx.size(); ++lvl) {
    getGradientMag(gradx[lvl], grady[lvl], &((*mag)[lvl]));
  }

  return;
}

/**
 * \brief Apply 3x3 max filter to a pyramid.
 *
 * @param[in] pyr Image pyramid.
 */
template <typename ChannelType>
void applyMaxFilter3Pyramid(ImagePyramid_<ChannelType>* pyr) {
  for (int lvl = 0; lvl < pyr->size(); ++lvl) {
    applyMaxFilter3(&((*pyr)[lvl]));
  }
}

}  // namespace utils

}  // namespace flame
