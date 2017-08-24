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
 * @file triangulator.h
 * @author W. Nicholas Greene
 * @date 2017-08-18 18:57:10 (Fri)
 */

#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

namespace flame {

using Vertex = cv::Point2f;
using Triangle = cv::Vec3i;
using Edge = cv::Vec2i;

namespace utils {

/**
 * \brief CRTP interface for 2D triangulation.
 */
template <typename Derived>
class Triangulator {
 public:
  /**
   * \brief Triangulate a set of 2D vertices.
   *
   * @param[in] vertices 2D vertices to triangulate.
   * @param[in] triangles Output triangles.
   */
  void triangulate(const std::vector<Vertex>& vertices,
                   std::vector<Triangle>* triangles) {
    static_cast<Derived*>(this)->triangulate(vertices, triangles);
  }
};

}  // namespace utils

}  // namespace flame
