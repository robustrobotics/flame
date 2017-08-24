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
 * @file params.h
 * @author W. Nicholas Greene
 * @date 2017-03-30 19:20:57 (Thu)
 */

#pragma once

#include "flame/stereo/inverse_depth_filter.h"
#include "flame/stereo/inverse_depth_meas_model.h"
#include "flame/stereo/line_stereo.h"

#include "flame/optimizers/nltgv2_l1_graph_regularizer.h"

namespace flame {

/**
 * @brief Parameter struct.
 */
struct Params {
  Params() {}
  // Threshold on gradient magnitude.
  float min_grad_mag = 5.0f;

  // Screen out low-gradient points after projecting to new image.
  bool do_grad_check_after_projection = false;

  int num_levels = 5; // Number of pyramid levels.

  // Feature Detection params.
  bool continuous_detection = true; // Continually detect features.
  int detection_win_size = 16; // Features are detected on grid with cells of size (win_size x win_size).
  bool do_letterbox = false; // Only process middle rows of image.
  float min_error = 100.0f;

  int photo_error_num_pfs = 30; // Number of pfs to consider for photo error computation.

  // Measurement model params.
  stereo::InverseDepthMeasModel::Params zparams;

  // IDepth filter params.
  float rescale_factor_min = 0.7f;
  float rescale_factor_max = 1.4f;
  float idepth_init = 0.01f;
  float idepth_var_init = 0.5f * 0.5f;
  float idepth_var_max = 0.5f * 0.5f; // Vertex is removed if variance exceeds this.
  int max_dropouts = 5; // Maximum number of failed tracking attempts.
  float outlier_sigma_thresh = 3.0; // Measurements >= this are rejected.
  float min_baseline = 0.01f; // Min baseline for idepth update.
  bool do_meas_fusion = true; // If false, measurements will not be fused.
  stereo::inverse_depth_filter::Params fparams;

  // Filter oblique triangles (only for display purposes).
  bool do_oblique_triangle_filter = true;
  float oblique_normal_thresh = 1.39626; // 80 degrees. Max angle between
  // surface normal and viewing ray.
  float oblique_idepth_diff_factor = 0.35f; // Filter triangle if (max_idepth -
                                            // min_idepth)/max_idepth is greater
                                            // than thresh.
  float oblique_idepth_diff_abs = 0.1f; // Filter triangle if (max_idepth -
                                        // min_idepth) is greater than thresh.

  // Filter triangles that have very long edges (only for display purposes).
  bool do_edge_length_filter = true;
  float edge_length_thresh = 0.333f; // As fraction of image width.

  // Filter triangles that are far away (only for display purposes).
  bool do_idepth_triangle_filter = true;
  float min_triangle_idepth = 0.01f;

  // Regularizer params.
  float min_height = 0.1f;
  float max_height = 4;
  float idepth_var_max_graph = 1e-2f; // Max var of feat to add to graph.
  bool adaptive_data_weights = false; // Set data weights to 1/var instead of 1.
  bool init_with_prediction = false; // Initialize idepth using predicted value.
  bool rescale_data = false;
  bool check_sticky_obstacles = false; // Check if idepths are being sucked towards the camera because of sticky obstacles.
  bool do_nltgv2 = true;
  optimizers::nltgv2_l1_graph_regularizer::Params rparams;

  // OpenMP params.
  int omp_num_threads = 4;
  int omp_chunk_size = 256;

  // Stereo params.
  stereo::line_stereo::Params sparams;

  // Used to scale idepths before they are colormapped. For example, if
  // average scene depth is >> 1m, then coloring by idepth will not provide
  // much dynamic range. Coloring by scene_color_scale * idepth (if
  // scene_color_scale > 1) is much more informative.
  float scene_color_scale = 1.0f;

  // Debugging parameters.
  bool debug_quiet = false;
  bool debug_print_timing_update = true;
  bool debug_print_timing_update_locking = true;
  bool debug_print_timing_frame_creation = true;
  bool debug_print_timing_gradients = true;
  bool debug_print_timing_poseframe = true;
  bool debug_print_timing_detection = true;
  bool debug_print_timing_detection_loop = true;
  bool debug_print_timing_graph_sync_loop = true;
  bool debug_print_timing_update_idepths = true;
  bool debug_print_timing_project_features = true;
  bool debug_print_timing_project_graph = true;
  bool debug_print_timing_sync_graph = true;
  bool debug_print_timing_triangulate = true;
  bool debug_print_timing_normals = true;
  bool debug_print_timing_interpolate = true;
  bool debug_print_timing_median_filter = true;
  bool debug_print_timing_lowpass_filter = true;
  bool debug_print_timing_oblique_triangle_filter = true;
  bool debug_print_timing_edge_length_filter = true;
  bool debug_print_timing_idepth_triangle_filter = true;
  bool debug_print_verbose_errors = false;
  bool debug_draw_detections = false;
  bool debug_draw_wireframe = false;
  bool debug_draw_features = false;
  bool debug_draw_matches = false;
  bool debug_draw_photo_error = false;
  bool debug_draw_normals = false;
  bool debug_draw_idepthmap = false;
  bool debug_draw_text_overlay = true;
  bool debug_flip_images = false;
};

}  // namespace flame
