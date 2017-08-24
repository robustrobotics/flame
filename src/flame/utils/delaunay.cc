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
 * @file delaunay.cc
 * @author W. Nicholas Greene
 * @date 2017-08-18 20:48:05 (Fri)
 */

// Adapted from cv_utils.cpp from fast-stereo (Sudeep Pillai).

#include "flame/utils/delaunay.h"

namespace flame {

namespace utils {

void Delaunay::triangulate(const std::vector<Vertex>& support,
                           std::vector<Triangle>* triangles) {

  // input/output structure for triangulation
  struct triangulateio in;
  int32_t k;

  // inputs
  in.numberofpoints = support.size();
  in.pointlist = (float*)malloc(in.numberofpoints*2*sizeof(float)); // NOLINT
  k = 0;
  for (int32_t i = 0; i < support.size(); i++) {
    in.pointlist[k++] = support[i].x;
    in.pointlist[k++] = support[i].y;
  }
  in.numberofpointattributes = 0;
  in.pointattributelist      = NULL;
  in.pointmarkerlist         = NULL;
  in.numberofsegments        = 0;
  in.numberofholes           = 0;
  in.numberofregions         = 0;
  in.regionlist              = NULL;

  // outputs
  out_.pointlist              = NULL;
  out_.pointattributelist     = NULL;
  out_.pointmarkerlist        = NULL;
  out_.trianglelist           = NULL;
  out_.triangleattributelist  = NULL;
  out_.neighborlist           = NULL;
  out_.segmentlist            = NULL;
  out_.segmentmarkerlist      = NULL;
  out_.edgelist               = NULL;
  out_.edgemarkerlist         = NULL;

  // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary markers)
  char parameters[] = "zneQB";
  ::triangulate(parameters, &in, &out_, NULL);
  free(in.pointlist);

  getTriangles(triangles);
  getNeighbors();
  getEdges();
  cleanup();

  return;
}

void Delaunay::triangulate(const std::vector<Vertex>& vertices) {
  triangulate(vertices, &triangles_);
  return;
}

void Delaunay::cleanup() {
  // free memory used for triangulation
  free(out_.pointlist);
  free(out_.trianglelist);
  free(out_.edgelist);
  free(out_.neighborlist);

  out_.pointlist = NULL;
  out_.trianglelist = NULL;
  out_.edgelist = NULL;
  out_.neighborlist = NULL;

  return;
}

void Delaunay::getTriangles(std::vector<Triangle>* triangles) {
  // put resulting triangles into vector tri
  triangles->resize(out_.numberoftriangles);
  int k = 0;
  for (int32_t i = 0; i < out_.numberoftriangles; i++) {
    (*triangles)[i] = Triangle(out_.trianglelist[k],
                               out_.trianglelist[k+1],
                               out_.trianglelist[k+2]);
    k+=3;
  }
  return;
}

void Delaunay::getNeighbors() {
  // put neighboring triangles into vector tri
  neighbors_.resize(out_.numberoftriangles);
  int k = 0;
  for (int32_t i = 0; i < out_.numberoftriangles; i++) {
    neighbors_[i] = Triangle(out_.neighborlist[k],
                            out_.neighborlist[k+1],
                            out_.neighborlist[k+2]);
    k+=3;
  }
  return;
}

void Delaunay::getEdges()  {
  // put resulting edges into vector
  edges_.resize(out_.numberofedges);
  int k = 0;
  for (int32_t i = 0; i < out_.numberofedges; i++) {
    edges_[i] = Edge(out_.edgelist[k], out_.edgelist[k+1]);
    k+=2;
  }
  return;
}

}  // namespace utils

}  // namespace flame
