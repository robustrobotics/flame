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
 * @file delaunay.h
 * @author W. Nicholas Greene
 * @date 2017-08-18 20:33:42 (Fri)
 */

// Adapted from cv_utils.cpp from fast-stereo (Sudeep Pillai).

#pragma once

#include <vector>

// Delaunay triangulation
#include <flame/external/triangle/triangle.h>

#include <flame/utils/triangulator.h>

namespace flame {

namespace utils {

/**
 * \brief Class that implements Delaunay triangulation.
 */
class Delaunay final : public Triangulator<Delaunay> {
 public:
  Delaunay() = default;
  ~Delaunay() = default;

  Delaunay(const Delaunay& rhs) = delete;
  Delaunay& operator=(const Delaunay& rhs) = delete;

  Delaunay(Delaunay&& rhs) = default;
  Delaunay& operator=(Delaunay&& rhs) = default;

  /**
   * \brief Triangulate a set of 2D vertices.
   *
   * @param[in] vertices 2D vertices to triangulate.
   * @param[in] triangles Output triangles.
   */
  void triangulate(const std::vector<Vertex>& vertices,
                   std::vector<Triangle>* triangles);

  /**
   * @brief Triangulate a set of 2D vertices.
   *
   * @param[in] vertices 2D vertices to triangulate.
   */
  void triangulate(const std::vector<Vertex>& vertices);

  // Accessors.
  const std::vector<Triangle>& triangles() const { return triangles_; }
  const std::vector<Edge>& edges() const { return edges_; }
  const std::vector<Triangle>& neighbors() const { return neighbors_; }

 private:
  void cleanup();
  void getTriangles(std::vector<Triangle>* triangles);
  void getNeighbors();
  void getEdges();

  struct triangulateio out_;

  std::vector<Triangle> triangles_;
  std::vector<Triangle> neighbors_;
  std::vector<Edge> edges_;
};

}  // namespace utils

}  // namespace flame
