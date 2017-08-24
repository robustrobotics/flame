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
 * @file pyramids.cc
 * @author W. Nicholas Greene
 * @date 2017-08-18 20:09:30 (Fri)
 */

#include "flame/utils/pyramids.h"

namespace flame {

namespace utils {

void getPyramidDisplay(const ImagePyramid& pyr, cv::Mat* out) {
  int rows = pyr[0].rows;
  int cols = pyr[0].cols;

  if (out->empty()) {
    out->create(rows, cols*1.5, pyr[0].type());
    (*out) = cv::Scalar(0);
  }

  cv::Rect curr_roi(0, 0, cols, rows);
  for (int ii = 0; ii < pyr.size(); ii++) {
    pyr[ii].copyTo((*out)(curr_roi));

    if (ii % 4 == 0) {
      /* Place new image on the right/top. */
      curr_roi.x += pyr[ii].cols;
      curr_roi.width = pyr[ii].cols/2;
      curr_roi.height = pyr[ii].rows/2;
    } else if (ii % 4 == 1) {
      /* Place new image on the bottom/right. */
      curr_roi.x += pyr[ii].cols/2;
      curr_roi.y += pyr[ii].rows;
      curr_roi.width = pyr[ii].cols/2;
      curr_roi.height = pyr[ii].rows/2;

    } else if (ii % 4 == 2) {
      /* Place new image on the left/bottom. */
      curr_roi.x -= pyr[ii].cols/2;
      curr_roi.y += pyr[ii].rows/2;
      curr_roi.width = pyr[ii].cols/2;
      curr_roi.height = pyr[ii].rows/2;

    } else if (ii % 4 == 3) {
      /* Place new image on the top/left. */
      curr_roi.y -= pyr[ii].rows/2;
      curr_roi.width = pyr[ii].cols/2;
      curr_roi.height = pyr[ii].rows/2;
    }
  }

  return;
}

}  // namespace utils

}  // namespace flame
