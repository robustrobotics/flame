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
 * @file nltgv2_l1_graph_regularizer.h
 * @author W. Nicholas Greene
 * @date 2016-10-06 20:43:29 (Thu)
 */

#pragma once

#include <vector>
#include <memory>

#include <boost/graph/adjacency_list.hpp>

#include "flame/utils/image_utils.h"

namespace flame {

namespace optimizers {

/**
 * @namespace This namespace includes types and functions to minimize a
 * NLTGV2-L1 problem defined on a Boost graph/mesh.
 *
 * An NLTGV2-L1 problem is a cost functional of the following form:
 *
 *  min_x F(Kx) + G(x)
 *
 *  x = Primal variable
 *  q = Dual variable
 *  K = Continuous, linear operator between X and Q
 *  G = Convex lower-semicontinuous (lsc) function
 *  F = Convex lsc function.
 *  F* = Convex conjugate of F (also convex lsc function)
 *
 * Typically the F(Kx) is smoothness term and the G term is a data fitting term.
 *
 * Here the F(Kx) function is NLTGV2, which stands for Non-Local Total
 * Generalized Variation^2, which is a generalized version of the Total
 * Variation semi-norm. It promotes solutions that are polynomials of degree < 2
 * (i.e. affine or planar solutions).
 *
 * The G(x) function is (weighted) L1, which measures the L1 norm between the
 * primal variable x and observed data.
 *
 * The optimization is carried out using the Chambolle-Pock primal-dual
 * algorithm (2010). See Bredies 2010, Ranftl 2014, Pinies 2015 for more
 * information.
 *
 * The Boost Graph Library is used to set up the problem. To use, create a Graph
 * object and add vertices and edges as needed. Run the step(...) function to
 * perform an optimization step.
 */
namespace nltgv2_l1_graph_regularizer {

/**
 * @brief Struct that defines a vertex in the graph.
 */
struct VertexData {
  cv::Point2f pos; // Position.
  float x = 0.0f; // Main primal variable.
  float w1 = 0.0f; // Plane parameters.
  float w2 = 0.0f;;

  float x_bar = 0.0f; // Extragradient varaibles.
  float w1_bar = 0.0f;
  float w2_bar = 0.0f;

  float x_prev = 0.0f; // Previous x value.
  float w1_prev = 0.0f;
  float w2_prev = 0.0f;

  float data_term = 0.0f; // Data term.
  float data_weight = 1.0f; // Weight on data term.
};

/**
 * @brief Struct that defines an edge in the graph.
 */
struct EdgeData {
  float alpha = 1.0f; // Edge weights.
  float beta = 1.0f;
  float q1 = 0.0f; // Dual variables.
  float q2 = 0.0f;
  float q3 = 0.0f;
  bool valid = true; // False if this edge should be removed.
};

/**
 * @brief Graph representation using the Boost Graph Library.
 */
using Graph =
    boost::adjacency_list<boost::hash_setS, // Edges will be stored in a hash map.
                          boost::hash_setS, // Vertices will be stored in a hash map
                          boost::undirectedS, // Undirected graph
                          VertexData, // Data stored at each vertex
                          EdgeData>; // Data stored at each edge

// These descriptors are essentially handles to the vertices and edges.
using VertexHandle = boost::graph_traits<Graph>::vertex_descriptor;
using EdgeHandle = boost::graph_traits<Graph>::edge_descriptor;

/**
 * @brief Parameter struct.
 */
struct Params {
  float data_factor = 0.1f; // lambda in the TV literature.
  float step_x = 0.001f; // Primal step size.
  float step_q = 125.0f; // Dual step size.
  float theta = 0.25f; // Extra gradient step size.

  float x_min = 0.0f; // Feasible set.
  float x_max = 10.0f;
};

/**
 * @brief Performs a full optimization step.
 */
void step(const Params& params, Graph* graph);

/**
 * @brief Return the smoothness cost (i.e. the NLTGV2 part).
 */
float smoothnessCost(const Params& params, const Graph& graph);

/**
 * @brief Return the data cost (i.e. the weighted L1 part).
 */
float dataCost(const Params& params, const Graph& graph);

/**
 * @brief Return the total cost of the current solution.
 */
inline float cost(const Params& params, const Graph& graph) {
  return smoothnessCost(params, graph) + dataCost(params, graph);
}

namespace internal {

/**
 * @brief Perform a gradient ascent step in the dual.
 */
void dualStep(const Params& params, Graph* graph);

/**
 * @brief Perform a gradient descent step in the primal.
 */
void primalStep(const Params& params, Graph* graph);

/**
 * @brief Perform an extra-gradient step in the primal.
 */
void extraGradientStep(const Params& params, Graph* graph);

// Proximal operator for convex conjugate of NLTGV2 regularizer.
inline float proxNLTGV2Conj(float step, float q) {
  float absq = utils::fast_abs(q);
  float new_q = q / (absq > 1 ? absq : 1);
  FLAME_ASSERT(!std::isnan(new_q));
  return new_q;
}

// Proximal operator for L2 data term.
inline float proxL1(float x_min, float x_max, float step_x, float data_weight,
             float x, float data) {
  float diff = x - data;
  float thresh = step_x * data_weight;

  float new_x = 0.0f;
  if (diff > thresh) {
    new_x = x - thresh;
  } else if (diff < -thresh) {
    new_x = x + thresh;
  } else {
    new_x = data;
  }

  // Project back onto the feasible set.
  new_x = (new_x < x_min) ? x_min : new_x;
  new_x = (new_x > x_max) ? x_max : new_x;
  return new_x;
}

}  // namespace internal

}  // namespace nltgv2_l1_graph_regularizer

}  // namespace optimizers

}  // namespace flame
