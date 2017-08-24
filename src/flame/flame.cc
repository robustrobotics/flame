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
 * @file flame.cc
 * @author W. Nicholas Greene
 * @date 2016-09-16 13:17:54 (Fri)
 */

#include "flame/flame.h"

#include <stdio.h>
#include <limits>
#include <vector>
#include <random>
#include <unordered_set>
#include <atomic>

#include <opencv2/highgui/highgui.hpp>

#include "flame/stereo/epipolar_geometry.h"

#include "flame/utils/visualization.h"
#include "flame/utils/image_utils.h"
#include "flame/utils/keyframe_selector.h"

namespace flame {

namespace dgraph = optimizers::nltgv2_l1_graph_regularizer;

Flame::Flame(int width, int height,
             const Matrix3f& K,
             const Matrix3f& Kinv,
             const Params& params) :
    stats_(),
    inited_(false),
    num_data_updates_(0),
    num_regularizer_updates_(0),
    params_(params),
    width_(width),
    height_(height),
    K_(K),
    Kinv_(Kinv),
    epigeo_(K, Kinv),
    num_imgs_(0),
    fnew_(nullptr),
    fprev_(nullptr),
    update_mtx_(),
    pfs_(),
    pfs_mtx_(),
    curr_pf_(nullptr),
    detection_thread_(),
    detection_queue_(),
    detection_queue_mtx_(),
    detection_queue_cv_(),
    new_feats_(),
    new_feats_mtx_(),
    photo_error_(height, width, std::numeric_limits<float>::quiet_NaN()),
    feat_count_(0),
    feats_(),
    feats_in_curr_(),
    graph_(),
    graph_scale_(1.0f),
    graph_thread_(),
    graph_mtx_(),
    feat_to_vtx_(),
    vtx_to_feat_(),
    triangulator_(),
    triangulator_mtx_(),
    tri_validity_(),
    vtx_(),
    vtx_idepths_(),
    vtx_w1_(),
    vtx_w2_(),
    vtx_normals_(),
    idepthmap_(height, width, std::numeric_limits<float>::quiet_NaN()),
    w1_map_(height, width, std::numeric_limits<float>::quiet_NaN()),
    w2_map_(height, width, std::numeric_limits<float>::quiet_NaN()),
    debug_img_detections_(height, width),
    debug_img_wireframe_(height, width),
    debug_img_features_(height, width),
    debug_img_matches_(height, width),
    debug_img_normals_(height, width),
    debug_img_idepthmap_(height, width) {
  // Start graph optimization thread.
  if (params_.do_nltgv2) {
    graph_thread_ = std::thread([this]() {
        while (true) {
          if (inited_) {
            graph_mtx_.lock();
            dgraph::step(params_.rparams, &(graph_));
            num_regularizer_updates_++;
            graph_mtx_.unlock();
          }

          std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        }
      });
  }

  // Start feature detection thread.
  detection_thread_ = std::thread([this]() { detectionLoop(); });

  return;
}

Flame::~Flame() {
  std::lock_guard<std::mutex> guard(graph_mtx_);
  graph_thread_.join();
  detection_thread_.join();
  return;
}

bool Flame::update(double time, uint32_t img_id,
                   const Sophus::SE3f& T_new,
                   const Image1b& img_new,
                   bool is_poseframe,
                   const Image1f& idepths_true) {
  stats_.tick("update");

  stats_.tick("update_locking");

  std::lock_guard<std::recursive_mutex> update_lock(update_mtx_);

  stats_.tock("update_locking");
  if (!params_.debug_quiet && params_.debug_print_timing_update_locking) {
    printf("Flame/update_locking = %f ms\n", stats_.timings("update_locking"));
  }

  /*==================== Create frames ====================*/
  stats_.tick("frame_creation");

  fprev_ = fnew_;

  // Create frame from new image.
  int border = params_.fparams.win_size;
  fnew_ = utils::Frame::create(T_new, img_new, img_id, 1, border);

  // Remember to increment counter.
  num_imgs_++;

  if (is_poseframe) {
    // Add to poseframes.
    pfs_mtx_.lock();
    pfs_[fnew_->id] = fnew_;
    pfs_mtx_.unlock();

    curr_pf_ = fnew_;
  }

  stats_.tock("frame_creation");
  if (!params_.debug_quiet && params_.debug_print_timing_frame_creation) {
    printf("Flame/frame_creation = %f ms\n", stats_.timings("frame_creation"));
  }

  if (num_imgs_ < 2) {
    // Don't do anything until we have 2 images.
    return false;
  }

  if (is_poseframe && !inited_ && (feats_.size() == 0)) {
    // Create initial detection data.
    DetectionData data;

    // Fill in reference frame info. We need to make a deep copy of idepthmap
    // (and pose, but that copy is deep by default) so that we don't need to
    // worry about locking/unlocking.
    data.ref = *curr_pf_;
    for (int ii = 0; ii < curr_pf_->idepthmap.size(); ++ii) {
      data.ref.idepthmap[ii] = curr_pf_->idepthmap[ii].clone();
    }

    data.prev = *fprev_;

    pfs_mtx_.lock();
    int num_pfs = pfs_.size();
    pfs_mtx_.unlock();

    // Fill in comparison frame info.
    utils::Frame::ConstPtr fcmp;
    if (num_pfs <= 1) {
      // Not enough poseframes yet. Use previous frame as comparison frame.
      fcmp = fprev_;
    } else {
      // Pick best comparison frame.
      std::lock_guard<std::mutex> lock(pfs_mtx_);
      fcmp = getPoseFrame(params_, K_, Kinv_,
                          pfs_, *curr_pf_,
                          params_.photo_error_num_pfs, &stats_);

      // Make sure we don't pick the same pf.
      if (fcmp->id >= curr_pf_->id) {
        fcmp = fprev_; // Use previous pf instead.
      }
    }

    FLAME_ASSERT(fcmp != nullptr);
    FLAME_ASSERT(fcmp->id != curr_pf_->id);
    FLAME_ASSERT(fcmp->id < curr_pf_->id);

    data.cmp = *fcmp;
    for (int ii = 0; ii < fcmp->idepthmap.size(); ++ii) {
      data.cmp.idepthmap[ii] = fcmp->idepthmap[ii].clone();
    }

    // Project features into current frame.
    if (feats_.size() > 0) {
      pfs_mtx_.lock();
      projectFeatures(params_, K_, Kinv_, pfs_, *curr_pf_, &feats_, &feats_in_curr_,
                      &stats_);
      pfs_mtx_.unlock();

      // Fill in features projected into poseframe.
      data.ref_xy.resize(feats_in_curr_.size());
      for (int ii = 0; ii < feats_in_curr_.size(); ++ii) {
        data.ref_xy[ii] = feats_in_curr_[ii].xy;
      }
    }

    // Add to detection_queue.
    detection_queue_mtx_.lock();
    detection_queue_.push_back(data);
    detection_queue_mtx_.unlock();

    // Signal detection thread that it has new poseframes for which to detect
    // features. Note that we need to unlock the detection_queue_mtx before sending
    // the signal so that we don't wake up a thread and then immediately block.
    detection_queue_cv_.notify_one();
  }

  // Load epipolar geometry between previous and new frame.
  Sophus::SE3f T_prev_to_new = fnew_->pose.inverse() * fprev_->pose;
  epigeo_.loadGeometry(T_prev_to_new.unit_quaternion(),
                       T_prev_to_new.translation());

  /*==================== Add new features ====================*/
  new_feats_mtx_.lock();
  if ((feats_.size() == 0) && (new_feats_.size() == 0)) {
    // No features to add.
    new_feats_mtx_.unlock();
    return false;
  }

  feats_.insert(feats_.end(), new_feats_.begin(), new_feats_.end());
  new_feats_.clear();

  new_feats_mtx_.unlock();

  /*==================== Update features ====================*/
  // Update depth estimates.
  pfs_mtx_.lock();
  bool idepth_success = updateFeatureIDepths(params_, K_, Kinv_, pfs_, *fnew_,
                                             *curr_pf_, &feats_, &stats_,
                                             &debug_img_matches_);
  pfs_mtx_.unlock();

  if (!idepth_success) {
    // No idepths could be updated.
    // TODO(wng): Not sure what to do here.
  }

  // Project features into current frame.
  pfs_mtx_.lock();
  projectFeatures(params_, K_, Kinv_, pfs_, *fnew_, &feats_, &feats_in_curr_,
                  &stats_);
  pfs_mtx_.unlock();

  if (feats_.size() < 3) {
    // Not enough detections.
    if (!params_.debug_quiet) {
      fprintf(stderr, "Flame[Error]: Too few feats (%i) to triangulate!\n",
              feats_.size());
    }
    // Clear everything.
    clear();
    return false;
  }

  if (params_.debug_draw_features) {
    drawFeatures(params_, fnew_->img[0], feats_in_curr_, &stats_,
                 &debug_img_features_);
  }

  /*==================== Get current smoothed solution  ====================*/
  epigeo_.loadGeometry(T_prev_to_new.unit_quaternion(),
                       T_prev_to_new.translation());

  // Project graph into new frame.
  graph_mtx_.lock();
  projectGraph(params_, epigeo_, *fnew_, &graph_, graph_scale_,
               &feat_to_vtx_, &vtx_to_feat_, &stats_);
  graph_mtx_.unlock();

  /*==================== Synchronize graph ====================*/
  // Update and synchronize graph and features.
  graph_mtx_.lock();
  pfs_mtx_.lock();
  triangulator_mtx_.lock();
  bool sync_success = syncGraph(params_, Kinv_, pfs_, idepthmap_, feats_,
                                feats_in_curr_, &triangulator_,
                                &graph_, graph_scale_,
                                &feat_to_vtx_, &vtx_to_feat_, &stats_);
  triangulator_mtx_.unlock();
  pfs_mtx_.unlock();
  graph_mtx_.unlock();

  if (!sync_success) {
    // Not enough detections.
    if (!params_.debug_quiet) {
      fprintf(stderr, "Flame[Error]: Could not synchronize graph with features.\n");
    }
    return false;
  }

  if (params_.rescale_data) {
    std::lock_guard<std::mutex> lock(graph_mtx_);

    // Rescale graph.
    float idepth_sum = 0.0f;
    Graph::vertex_iterator vit, end;
    boost::tie(vit, end) = boost::vertices(graph_);
    for ( ; vit != end; ++vit) {
      const auto& vtx = graph_[*vit];
      idepth_sum += vtx.data_term * graph_scale_;
    }
    float new_scale = idepth_sum / num_vertices(graph_);

    boost::tie(vit, end) = boost::vertices(graph_);
    for ( ; vit != end; ++vit) {
      auto& vtx = graph_[*vit];
      vtx.x = vtx.x * graph_scale_ / new_scale;
      vtx.x_bar = vtx.x_bar * graph_scale_ / new_scale;
      vtx.x_prev = vtx.x_prev * graph_scale_ / new_scale;
      vtx.data_term = vtx.data_term * graph_scale_ / new_scale;
    }
    params_.rparams.data_factor *= new_scale / graph_scale_;
    graph_scale_ = new_scale;
  }

  // Copy current triangulation.
  triangulator_mtx_.lock();
  triangles_curr_ = triangulator_.triangles();
  edges_curr_ = triangulator_.edges();
  triangulator_mtx_.unlock();

  if (triangles_curr_.size() == 0) {
    // No triangles.
    return false;
  }

  // Extract smoothed idepths from graph.
  graph_mtx_.lock();
  vtx_.clear();
  vtx_idepths_.clear();
  vtx_w1_.clear();
  vtx_w2_.clear();
  vtx_normals_.clear();

  Graph::vertex_iterator vit, end;
  boost::tie(vit, end) = boost::vertices(graph_);
  for ( ; vit != end; ++vit) {
    const auto& vtx = graph_[*vit];
    vtx_.push_back(vtx.pos);
    vtx_idepths_.push_back(vtx.x * graph_scale_);
    vtx_w1_.push_back(vtx.w1);
    vtx_w2_.push_back(vtx.w2);
  }
  graph_mtx_.unlock();

  // getVertexNormals(params_, K_, vtx_, vtx_idepths_, vtx_w1_, vtx_w2_, &vtx_normals_,
  //                  &stats_);
  getVertexNormals(params_, Kinv_, vtx_, vtx_idepths_, triangles_curr_, &vtx_normals_,
                   &stats_);

  /*==================== Do triangle filtering for display ====================*/
  tri_validity_.assign(triangles_curr_.size(), true);
  if (params_.do_oblique_triangle_filter) {
    // Filter oblique triangles. This is only for display purposes. The actual
    // graph will contain all the triangles.
    obliqueTriangleFilter(params_, Kinv_, vtx_, vtx_idepths_,
                          triangles_curr_, &tri_validity_, &stats_);
  }

  if (params_.do_edge_length_filter) {
    // Filter triangles with long edges. This is only for display purposes. The
    // actual graph will contain all the triangles.
    edgeLengthFilter(params_, width_, height_, vtx_, triangles_curr_,
                     &tri_validity_, &stats_);
  }

  if (params_.do_idepth_triangle_filter) {
    idepthTriangleFilter(params_, vtx_idepths_, triangles_curr_,
                         &tri_validity_, &stats_);
  }

  /*==================== Interpolate to dense idepthmap ====================*/
  stats_.tick("interpolate");
  std::vector<bool> vtx_validity(vtx_.size(), true);
  idepthmap_ = std::numeric_limits<float>::quiet_NaN();
  std::vector<bool> tri_validity_true(triangles_curr_.size(), true);
  utils::interpolateMesh(triangles_curr_, vtx_, vtx_idepths_,
                         vtx_validity, tri_validity_true, &idepthmap_);

  if (is_poseframe) {
    // Give idepthmap to poseframe.
    curr_pf_->idepthmap[0] = idepthmap_.clone();
    curr_pf_->tris = triangles_curr_;
    curr_pf_->edges = edges_curr_;
    curr_pf_->vtx = vtx_;
    curr_pf_->vtx_idepths = vtx_idepths_;
    curr_pf_->vtx_w1 = vtx_w1_;
    curr_pf_->vtx_w2 = vtx_w2_;
  }

  // Compute coverage stats.
  int coverage = 0;
  for (int ii = 0; ii < height_; ++ii) {
    for (int jj = 0; jj < width_; ++jj) {
      if (!std::isnan(idepthmap_(ii, jj))) {
        coverage++;
      }
    }
  }
  stats_.set("coverage", static_cast<float>(coverage) / (width_ * height_));

  stats_.tock("interpolate");
  if (!params_.debug_quiet && params_.debug_print_timing_interpolate) {
    printf("Flame/interpolate = %f ms\n", stats_.timings("interpolate"));
  }

  /*==================== Add frames to detection queue ====================*/
  if (is_poseframe) {
    // Create detection data.
    DetectionData data;

    // Fill in reference frame info. We need to make a deep copy of idepthmap
    // (and pose, but that copy is deep by default) so that we don't need to
    // worry about locking/unlocking.
    data.ref = *curr_pf_;
    for (int ii = 0; ii < curr_pf_->idepthmap.size(); ++ii) {
      data.ref.idepthmap[ii] = curr_pf_->idepthmap[ii].clone();
    }

    data.prev = *fprev_;

    // Fill in comparison frame info.
    pfs_mtx_.lock();
    utils::Frame::ConstPtr fcmp = getPoseFrame(params_, K_, Kinv_,
                                               pfs_, *curr_pf_,
                                               params_.photo_error_num_pfs, &stats_);
    pfs_mtx_.unlock();

    FLAME_ASSERT(fcmp->id != curr_pf_->id);

    data.cmp = *fcmp;
    for (int ii = 0; ii < fcmp->idepthmap.size(); ++ii) {
      data.cmp.idepthmap[ii] = fcmp->idepthmap[ii].clone();
    }

    // Fill in features projected into poseframe.
    data.ref_xy.resize(feats_in_curr_.size());
    for (int ii = 0; ii < feats_in_curr_.size(); ++ii) {
      data.ref_xy[ii] = feats_in_curr_[ii].xy;
    }

    // Add to detection_queue.
    detection_queue_mtx_.lock();
    detection_queue_.push_back(data);
    detection_queue_mtx_.unlock();

    // Signal detection thread that it has new poseframes for which to detect
    // features. Note that we need to unlock the detection_queue_mtx before sending
    // the signal so that we don't wake up a thread and then immediately block.
    detection_queue_cv_.notify_one();
  }

  /*==================== Draw stuff ====================*/
  if (params_.debug_draw_wireframe) {
    drawWireframe(params_, fnew_->img[0], triangles_curr_, edges_curr_,
                  tri_validity_, vtx_, vtx_idepths_, &stats_,
                  &debug_img_wireframe_);
  }

  if (params_.debug_draw_normals) {
    w1_map_ = std::numeric_limits<float>::quiet_NaN();
    w2_map_ = std::numeric_limits<float>::quiet_NaN();
    utils::interpolateMesh(triangles_curr_, vtx_, vtx_w1_,
                           vtx_validity, tri_validity_true, &w1_map_);
    utils::interpolateMesh(triangles_curr_, vtx_, vtx_w2_,
                           vtx_validity, tri_validity_true, &w2_map_);
    drawNormals(params_, K_, fnew_->img[0], idepthmap_, w1_map_, w2_map_,
                &debug_img_normals_);
  }

  if (params_.debug_draw_idepthmap) {
    drawInverseDepthMap(params_, fnew_->img[0], idepthmap_, &stats_,
                        &debug_img_idepthmap_);
  }

  /*==================== Time stuff ====================*/
  // Compute two measures of throughput in Hz. The first is the actual number of
  // frames per second, the second is the theoretical maximum fps based on the
  // runtime. They are not necessarily the same - the former takes external
  // latencies into account.

  // Compute maximum fps based on runtime.
  double fps_max = 0.0f;
  if ((stats_.stats("fps_max") > 0.0f) && (stats_.timings("update") > 0.0f)) {
    fps_max = 1.0f / (0.99 * 1.0f/stats_.stats("fps_max") +
                      0.01 * stats_.timings("update")/1000.0f);
  } else if (stats_.timings("update") > 0.0f) {
    fps_max = 1000.0f / stats_.timings("update");
  }
  stats_.set("fps_max", fps_max);

  // Compute actual fps (overall throughput of system).
  stats_.tock("fps");
  double fps = 0.0;
  if ((stats_.stats("fps") > 0.0f) && (stats_.timings("fps") > 0.0f)) {
    fps = 1.0f / (0.99 * 1.0f/stats_.stats("fps") +
                  0.01 * stats_.timings("fps")/1000.0f);
  } else if (stats_.timings("fps") > 0.0f) {
    fps = 1000.0f / stats_.timings("fps");
  }
  stats_.set("fps", fps);
  stats_.tick("fps");

  inited_ = true;
  num_data_updates_++;

  stats_.tock("update");
  if (!params_.debug_quiet && params_.debug_print_timing_update) {
    printf("Flame/update(%lu, %lu) = %4.1fms/%.1fHz (%.1fHz)\n",
           num_data_updates_, num_regularizer_updates_, stats_.timings("update"),
           stats_.stats("fps_max"), stats_.stats("fps"));
  }

  return true;
}

void Flame::prunePoseFrames(const std::vector<uint32_t>& pfs_to_keep) {
  // Locking the update_mtx_ might be overkill. We really just need to lock
  // feats_, new_feats_, and pfs_, and detection_queue_.
  std::lock_guard<std::recursive_mutex> update_lock(update_mtx_);
  std::lock_guard<std::mutex> detection_lock(detection_queue_mtx_);
  std::lock_guard<std::mutex> pfs_lock(pfs_mtx_);
  std::lock_guard<std::mutex> new_feats_lock(new_feats_mtx_);

  FrameIDToFrame pruned_pfs;
  bool found_curr_pf = false; // Make sure the current poseframe is in the list.
  for (int ii = 0; ii < pfs_to_keep.size(); ++ii) {
    if (pfs_.count(pfs_to_keep[ii]) > 0) {
      pruned_pfs[pfs_to_keep[ii]] = pfs_[pfs_to_keep[ii]];
      if (pfs_to_keep[ii] == curr_pf_->id) {
        found_curr_pf = true;
      }
    }
  }

  if (!found_curr_pf) {
    if (!params_.debug_quiet) {
      fprintf(stderr, "Flame[FAIL]: Current poseframe is not in to_keep list. Not updating poses.\n");
    }
    return;
  }

  // Remove any detection data items if any pfs are deleted.
  std::deque<DetectionData> pruned_detection_data;
  for (const auto& data : detection_queue_) {
    if ((pruned_pfs.count(data.ref.id) > 0) &&
        (pruned_pfs.count(data.cmp.id) > 0)) {
      // Both pfs still exist. Add to queue.
      pruned_detection_data.push_back(data);
    }
  }
  detection_queue_.swap(pruned_detection_data);

  if (pruned_pfs.size() == 0) {
    // No more pfs. Reset.
    clear();
    return;
  }

  int row_offset = 0;
  if (params_.do_letterbox) {
    // Only detect features in middle third of image.
    row_offset = height_/3;
  }

  // Move features whose parent pose has been deleted to new oldest pf.
  int border = params_.rescale_factor_max * params_.fparams.win_size / 2 + 1;
  cv::Rect valid_region(border, border + row_offset,
                        width_ - 2*border, height_ - 2*border - 2*row_offset);
  auto pfit = pruned_pfs.crbegin(); // Pointer to now oldest pf.
  for (int ii = 0; ii < feats_.size(); ++ii) {
    auto& feat = feats_[ii];
    if (pruned_pfs.count(feat.frame_id) == 0) {
      stereo::EpipolarGeometry<float> epipf(K_, Kinv_);
      Sophus::SE3f T_old_to_new = pfit->second->pose.inverse() *
          pfs_[feat.frame_id]->pose;
      epipf.loadGeometry(T_old_to_new.unit_quaternion(),
                         T_old_to_new.translation());

      cv::Point2f u_pf;
      float idepth_pf, var_pf;
      bool move_success =
          stereo::inverse_depth_filter::predict(epipf,
                                                params_.fparams.process_var_factor,
                                                feat.xy,
                                                feat.idepth_mu,
                                                feat.idepth_var,
                                                &u_pf, &idepth_pf, &var_pf);

      feat.frame_id = pfit->second->id;
      feat.xy = u_pf;
      float old_idepth = feat.idepth_mu;
      feat.idepth_mu = idepth_pf;

      // Project idepth variance.
      float var_factor4 = idepth_pf / old_idepth;
      var_factor4 *= var_factor4;
      var_factor4 *= var_factor4;

      if (idepth_pf < 1e-6) {
        // If feat_ref.idepth_mu == 0, then var_factor4 is inf.
        var_factor4 = 1;
      }
      feat.idepth_var *= var_factor4;

      if (!move_success || !valid_region.contains(u_pf)) {
        // Move wasn't successful or pixel projected OOB.
        feat.valid = false;
        continue;
      }
    }
    FLAME_ASSERT(pruned_pfs.count(feat.frame_id) > 0);
    FLAME_ASSERT(pruned_pfs.count(feats_[ii].frame_id) > 0);
  }

  // Do the same for new features.
  std::vector<FeatureWithIDepth> pruned_new_feats;
  pruned_new_feats.reserve(new_feats_.size()); // Copy instead of marking invalid.
  for (int ii = 0; ii < new_feats_.size(); ++ii) {
    auto& feat = new_feats_[ii];
    if (pruned_pfs.count(feat.frame_id) == 0) {
      stereo::EpipolarGeometry<float> epipf(K_, Kinv_);
      Sophus::SE3f T_old_to_new = pfit->second->pose.inverse() *
          pfs_[feat.frame_id]->pose;
      epipf.loadGeometry(T_old_to_new.unit_quaternion(),
                         T_old_to_new.translation());

      cv::Point2f u_pf;
      float idepth_pf, var_pf;
      bool move_success =
          stereo::inverse_depth_filter::predict(epipf,
                                                params_.fparams.process_var_factor,
                                                feat.xy,
                                                feat.idepth_mu,
                                                feat.idepth_var,
                                                &u_pf, &idepth_pf, &var_pf);

      if (!move_success || !valid_region.contains(u_pf)) {
        // Remove this feature.
        continue;
      }

      feat.frame_id = pfit->second->id;
      feat.xy = u_pf;
      float old_idepth = feat.idepth_mu;
      feat.idepth_mu = idepth_pf;

      // Project idepth variance.
      float var_factor4 = idepth_pf / old_idepth;
      var_factor4 *= var_factor4;
      var_factor4 *= var_factor4;

      if (idepth_pf < 1e-6) {
        // If feat_ref.idepth_mu == 0, then var_factor4 is inf.
        var_factor4 = 1;
      }
      feat.idepth_var *= var_factor4;
    }

    // Push back updated feature.
    pruned_new_feats.push_back(feat);
  }
  new_feats_.swap(pruned_new_feats);

  // Swap the maps.
  pfs_.swap(pruned_pfs);

  return;
}

void Flame::detectionLoop() {
  while (true) {
    // Wait for new items.
    std::unique_lock<std::mutex> lock(detection_queue_mtx_);
    detection_queue_cv_.wait(lock, [this]() {
        return detection_queue_.size() > 0;
      });

    if (!params_.debug_quiet) {
      printf("I HAS %i POSEFRAMES\n", detection_queue_.size());
    }

    while (detection_queue_.size() > 0) {
      stats_.tick("detection_loop");

      // Grab poseframe.
      const auto& data = detection_queue_.front();

      // Detect new features if this is a poseframe.
      std::vector<cv::Point2f> new_feats;
      if (params_.continuous_detection ||
          (!params_.continuous_detection && (num_data_updates_ < 1))) {
        detectFeatures(params_, K_, Kinv_,
                       data.ref, data.prev, data.cmp, data.ref.idepthmap[0],
                       data.ref_xy, &photo_error_,
                       &new_feats, &stats_, &debug_img_detections_);
      }

      // Add new features to list.
      new_feats_mtx_.lock();
      for (int ii = 0; ii < new_feats.size(); ++ii) {
        FeatureWithIDepth newf;
        newf.id = feat_count_++;
        newf.frame_id = data.ref.id;
        newf.xy = new_feats[ii];
        newf.idepth_var = params_.idepth_var_init;
        newf.valid = true;
        newf.num_updates = 0;

        // Initialize idepth from dense idepthmap if possible.
        newf.idepth_mu =  params_.idepth_init;
        int x = utils::fast_roundf(newf.xy.x);
        int y = utils::fast_roundf(newf.xy.y);
        if (!std::isnan(data.ref.idepthmap[0](y, x))) {
          newf.idepth_mu = data.ref.idepthmap[0](y, x);
        }

        new_feats_.push_back(newf);
      }
      new_feats_mtx_.unlock();

      // Remove poseframe from featureless list.
      detection_queue_.pop_front();

      stats_.tock("detection_loop");
      if (!params_.debug_quiet && params_.debug_print_timing_detection_loop) {
        printf("Flame/detection_loop = %f ms\n",
               stats_.timings("detection_loop"));
      }
    }

    std::this_thread::sleep_for(std::chrono::nanoseconds(1));
  }

  return;
}

utils::Frame::ConstPtr Flame::getPoseFrame(const Params& params,
                                           const Matrix3f& K,
                                           const Matrix3f& Kinv,
                                           const FrameIDToFrame& pfs,
                                           const utils::Frame& fnew,
                                           int max_pfs,
                                           utils::StatsTracker* stats) {
  stats->tick("poseframe");

  // Pick past pf to use for comparison.
  float best_score = std::numeric_limits<float>::lowest();
  utils::Frame::Ptr pf = nullptr;
  int best_idx = 0;

  // We walk backwards through the pfs (i.e. backwarfs in time), considering a
  // maximum of max_pfs and take the one with the best score.
  auto pfi = pfs.crbegin(); // Const reverse iterator.
  auto rend = pfs.crend();
  for (int ii = 0; ii < max_pfs; ++ii) {
    if (pfi->second->id != fnew.id) {
      float score =
          utils::KeyFrameSelector::score(fnew.img[0].cols, fnew.img[0].rows,
                                         K, Kinv,
                                         pfi->second->pose.inverse() * fnew.pose,
                                         1.0f, 50.0f);
      if (score > best_score) {
        best_score = score;
        pf = pfi->second;
        best_idx = ii;
      }
    }

    pfi++;
    if (pfi == rend) {
      break;
    }
  }

  stats->tock("poseframe");
  if (!params.debug_quiet && params.debug_print_timing_poseframe) {
    printf("Flame/poseframe = %f ms, picked %i/%i\n",
           stats->timings("poseframe"), best_idx, max_pfs);
  }

  return pf;
}

void Flame::detectFeatures(const Params& params,
                           const Matrix3f& K,
                           const Matrix3f& Kinv,
                           const utils::Frame& fref,
                           const utils::Frame& fprev,
                           const utils::Frame& fcmp,
                           const Image1f& idepthmap,
                           const std::vector<cv::Point2f>& curr_feats,
                           Image1f* error,
                           std::vector<cv::Point2f>* features,
                           utils::StatsTracker* stats,
                           Image3b* debug_img) {
  // Sample points.
  stats->tick("detection");

  int width = fref.img[0].cols;
  int height = fref.img[0].rows;

  int row_offset = 0;
  if (params.do_letterbox) {
    // Only detect features in middle third of image.
    row_offset = height/3;
  }

  // Images aren't padded - accessing pixels outside this region may result in a
  // segfault.
  int border = params.rescale_factor_max * params.fparams.win_size / 2 + 1;
  cv::Rect valid_region(border, border + row_offset,
                        width - 2*border, height - 2*border - 2*row_offset);

  stereo::EpipolarGeometry<float> epigeo(K, Kinv);

  //   /*==================== Compute photo error ====================*/
  //   Sophus::SE3f T_ref_to_cmp(fcmp.pose.inverse() * fref.pose);
  //   epigeo.loadGeometry(T_ref_to_cmp.unit_quaternion(),
  //                       T_ref_to_cmp.translation());

  //   // Compute photo metric error for high gradient points.
  //   if (error->empty()) {
  //     // Initialize output to nans.
  //     *error = Image1f(height, width, std::numeric_limits<float>::quiet_NaN());
  //   } else {
  //     error->setTo(std::numeric_limits<float>::quiet_NaN());
  //   }

  // #pragma omp parallel for collapse(2) num_threads(params.omp_num_threads) schedule(static, params.omp_chunk_size) // NOLINT
  //   for (uint32_t ii = border; ii < height - border; ++ii) {
  //     for (uint32_t jj = border; jj < width - border; ++jj) {
  //       float idepth = idepthmap(ii, jj);

  //       if (std::isnan(idepth)) {
  //         continue;
  //       }

  //       // Project pixel into poseframe.
  //       cv::Point2f u_ref(jj, ii);
  //       cv::Point2f u_cmp = epigeo.project(u_ref, idepth);

  //       if (!valid_region.contains(u_cmp)) {
  //         // Skip if u_cmp is not in the valid region.
  //         continue;
  //       }

  //       uint8_t curr_val = fref.img[0](ii, jj);
  //       float cmp_val = utils::bilinearInterp<uint8_t, float>(fcmp.img[0],
  //                                                             u_cmp.x,
  //                                                             u_cmp.y);
  //       float cost = utils::fast_abs(cmp_val - curr_val);
  //       (*error)(ii, jj) = cost;
  //     }
  //   }

  //   // Compute error integral image.
  //   Image1f error_int(height, width, std::numeric_limits<float>::quiet_NaN());
  //   for (int ii = 0; ii < height; ++ii) {
  //     for (int jj = 0; jj < width; ++jj) {
  //       float err_ii_jj = (*error)(ii, jj);
  //       if (std::isnan(err_ii_jj)) {
  //         err_ii_jj = 0.0f;
  //       }

  //       if ((ii == 0) || (jj == 0)) {
  //         error_int(ii, jj) = err_ii_jj;
  //       } else if (ii == 0) {
  //         error_int(ii, jj) = err_ii_jj + error_int(ii, jj - 1);
  //       } else if (jj == 0) {
  //         error_int(ii, jj) = err_ii_jj + error_int(ii - 1, jj);
  //       } else {
  //         error_int(ii, jj) = err_ii_jj +
  //             error_int(ii - 1, jj) + error_int(ii, jj - 1) -
  //             error_int(ii - 1, jj - 1);
  //       }
  //     }
  //   }

  //   // Compute box filtered error using integral image.
  //   Image1f error_smooth(height, width, std::numeric_limits<float>::quiet_NaN());
  //   int win_size = 11;
  //   if (width < 640) {
  //     win_size = 5; // Hack until I make this a param.
  //   }

  //   int half_win = win_size/2;
  //   for (int ii = half_win; ii < height - half_win; ++ii) {
  //     for (int jj = half_win; jj < width - half_win; ++jj) {
  //       error_smooth(ii, jj) = error_int(ii + half_win, jj + half_win) +
  //           error_int(ii - half_win, jj - half_win) -
  //           error_int(ii - half_win, jj + half_win) -
  //           error_int(ii + half_win, jj - half_win);
  //       error_smooth(ii, jj) /= win_size * win_size;
  //     }
  //   }

  //   // Set error to smoothed error.
  //   *error = error_smooth;

  //   // Compute error histogram.
  //   int num_valid = 0;
  //   std::vector<int> hist(256, 0);
  //   for (int ii = border; ii < height - border; ++ii) {
  //     for (int jj = border; jj < width - border; ++jj) {
  //       float err = (*error)(ii, jj);
  //       if (!std::isnan(err)) {
  //         hist[utils::fast_roundf(err)]++;
  //         num_valid++;
  //       }
  //     }
  //   }

  //   // Get median.
  //   int median = -1;

  //   float ptile_prob = 0.99;
  //   int ptile = -1;
  //   int count = 0;
  //   for (int ii = 0; ii < hist.size(); ++ii) {
  //     count += hist[ii];
  //     if ((median < 0) && (count > 0.5f * num_valid)) {
  //       median = ii;
  //     } else if ((ptile < 0) && (count > ptile_prob * num_valid)) {
  //       ptile = ii;
  //       break;
  //     }
  //   }

  //   float total_error = 0.0f;
  //   int num_valid_final = 0;
  //   for (int ii = border; ii < height - border; ++ii) {
  //     for (int jj = border; jj < width - border; ++jj) {
  //       float err = (*error)(ii, jj);
  //       if (!std::isnan(err)) {
  //         total_error += err;
  //         num_valid_final++;
  //       }
  //     }
  //   }
  //   stats->set("total_photo_error", total_error);

  //   float avg_error = 0.0f;
  //   if (num_valid > 0) {
  //     avg_error = total_error / num_valid_final;
  //   }
  //   stats->set("avg_photo_error", avg_error);

  // /*==================== NLTGV2 Score ====================*/
  // Image1f idepthmap2(fref.img[0].rows, fref.img[0].cols,
  //                      std::numeric_limits<float>::quiet_NaN());
  // Image1f w1_map(fref.img[0].rows, fref.img[0].cols,
  //                  std::numeric_limits<float>::quiet_NaN());
  // Image1f w2_map(fref.img[0].rows, fref.img[0].cols,
  //                  std::numeric_limits<float>::quiet_NaN());
  // std::vector<bool> vtx_validity(fref.vtx.size(), true);
  // std::vector<bool> tri_validity(fref.tris.size(), true);
  // utils::interpolateMesh(fref.tris, fref.vtx, fref.vtx_idepths,
  //                        vtx_validity, tri_validity, &idepthmap2);
  // utils::interpolateMesh(fref.tris, fref.vtx, fref.vtx_w1,
  //                        vtx_validity, tri_validity, &w1_map);
  // utils::interpolateMesh(fref.tris, fref.vtx, fref.vtx_w2,
  //                        vtx_validity, tri_validity, &w2_map);

  // float alpha = 1.0f;
  // float beta = 1.0f;
  // Image1f nltgv2(height, width, 0.0f);
  // for (int ii = 0; ii < height - 1; ++ii) {
  //   for (int jj = 0; jj < width - 1; ++jj) {
  //     float cost = 0.0f;

  //     float id = idepthmap2(ii, jj);
  //     float w1 = w1_map(ii, jj);
  //     float w2 = w2_map(ii, jj);

  //     // Compute contribution from horizontal neighbor.
  //     float idh = idepthmap2(ii, jj + 1);
  //     float w1h = w1_map(ii, jj + 1);
  //     float w2h = w2_map(ii, jj + 1);

  //     cv::Point2f horz_diff(-1.0f, 0.0f);
  //     cost += alpha * utils::fast_abs(id - idh - w1 * horz_diff.x - w2 * horz_diff.y);
  //     cost += beta * utils::fast_abs(w1 - w1h) + beta * utils::fast_abs(w2 - w2h);

  //     // Compute contribution from vertical neighbor.
  //     float idv = idepthmap2(ii + 1, jj);
  //     float w1v = w1_map(ii + 1, jj);
  //     float w2v = w2_map(ii + 1, jj);

  //     cv::Point2f vert_diff(0.0f, -1.0f);
  //     cost += alpha * utils::fast_abs(id - idv - w1 * vert_diff.x - w2 * vert_diff.y);
  //     cost += beta * utils::fast_abs(w1 - w1v) + beta * utils::fast_abs(w2 - w2v);

  //     if (!std::isnan(cost)) {
  //       nltgv2(ii, jj) = cost;
  //     }
  //   }
  // }

  // // Blur cost.
  // Image1f nltgv2_blurred(height, width, 0.0f);
  // cv::GaussianBlur(nltgv2, nltgv2_blurred, cv::Size(33, 33), 0);

  // /*==================== Detect features with coarse/fine pass ====================*/
  // // Compute mask where we already have features.
  // int clvl = params.coarse_level;
  // int hclvl = height >> clvl;
  // int wclvl = width >> clvl;
  // Image1b cmask(hclvl, wclvl, 255);

  // int flvl = params.fine_level;
  // int hflvl = height >> flvl;
  // int wflvl = width >> flvl;
  // Image1b fmask(hflvl, wflvl, 255);

  // for (uint32_t ii = 0; ii < curr_feats.size(); ++ii) {
  //   // Fill in coarse mask.
  //   uint32_t x_clvl = curr_feats[ii].x / (1 << clvl);
  //   uint32_t y_clvl = curr_feats[ii].y / (1 << clvl);
  //   cmask(y_clvl, x_clvl) = 0;

  //   // Fill in fine mask.
  //   uint32_t x_flvl = curr_feats[ii].x / (1 << flvl);
  //   uint32_t y_flvl = curr_feats[ii].y / (1 << flvl);
  //   fmask(y_flvl, x_flvl) = 0;
  // }

  // // Load epipolar geometry from prev to ref.
  // Sophus::SE3f T_ref_to_prev(fprev.pose.inverse() * fref.pose);
  // epigeo.loadGeometry(T_ref_to_prev.unit_quaternion(),
  //                     T_ref_to_prev.translation());

  // // Coarse pass.
  // Image1f best_gradsc(hclvl, wclvl, 0.0f);
  // std::vector<cv::Point2f> best_pxc(hclvl * wclvl);
  // float grad_thresh2 = params.min_grad_mag * params.min_grad_mag;
  // for (uint32_t ii = border; ii < height - border; ++ii) {
  //   for (uint32_t jj = border; jj < width - border; ++jj) {
  //     int ii_flvl = ii >> flvl;
  //     int jj_flvl = jj >> flvl;

  //     if (fmask(ii_flvl, jj_flvl) == 0) {
  //       // Already have feature here.
  //       continue;
  //     }

  //     // Check gradient magnitude.
  //     float gx = fref.gradx[0](ii, jj);
  //     float gy = fref.grady[0](ii, jj);
  //     float gmag2 = gx*gx + gy*gy;
  //     if (gmag2 < grad_thresh2) {
  //       continue;
  //     }

  //     // Check gradient magnitude in epipolar direction.
  //     cv::Point2f epi_ref;
  //     epigeo.referenceEpiline(cv::Point2f(ii, jj), &epi_ref);

  //     float epigrad2 = gx * epi_ref.x + gy * epi_ref.y;
  //     epigrad2 *= epigrad2;

  //     if (epigrad2 < grad_thresh2) {
  //       // Gradient isn't large enough, skip.
  //       continue;
  //     }

  //     int ii_clvl = ii >> clvl;
  //     int jj_clvl = jj >> clvl;
  //     int idx_clvl = ii_clvl * wclvl + jj_clvl;

  //     // Fill in best gradients.
  //     if (epigrad2 >= best_gradsc(ii_clvl, jj_clvl)) {
  //       best_gradsc(ii_clvl, jj_clvl) = epigrad2;
  //       best_pxc[idx_clvl] = cv::Point2f(jj, ii);
  //     }
  //   }
  // }

  // // Now extract detections of grid.
  // std::vector<cv::KeyPoint> kps;
  // for (int ii = 0; ii < hclvl; ++ii) {
  //   for (int jj = 0; jj < wclvl; ++jj) {
  //     int idx = ii * wclvl + jj;
  //     if ((cmask(ii, jj) > 0) && (best_gradsc(ii, jj) > 0)) {
  //       kps.push_back(cv::KeyPoint(best_pxc[idx], 1));

  //       int ii_flvl = static_cast<int>(best_pxc[idx].y) >> flvl;
  //       int jj_flvl = static_cast<int>(best_pxc[idx].x) >> flvl;
  //       fmask(ii_flvl, jj_flvl) = 0;
  //     }
  //   }
  // }

  // // Fine pass.
  // Image1f score(height, width, std::numeric_limits<float>::quiet_NaN());
  // Image1f best_gradsf(hflvl, wflvl, 0.0f);
  // std::vector<cv::Point2f> best_pxf(hflvl * wflvl);
  // for (uint32_t ii = border; ii < height - border; ++ii) {
  //   for (uint32_t jj = border; jj < width - border; ++jj) {
  //     int ii_flvl = ii >> flvl;
  //     int jj_flvl = jj >> flvl;
  //     int idx_flvl = ii_flvl * wflvl + jj_flvl;

  //     if (fmask(ii_flvl, jj_flvl) == 0) {
  //       // Already have feature here.
  //       continue;
  //     }

  //     // Check gradient magnitude.
  //     float gx = fref.gradx[0](ii, jj);
  //     float gy = fref.grady[0](ii, jj);
  //     float gmag2 = gx*gx + gy*gy;
  //     if (gmag2 < grad_thresh2) {
  //       continue;
  //     }

  //     // Check gradient magnitude in epipolar direction.
  //     cv::Point2f epi_ref;
  //     epigeo.referenceEpiline(cv::Point2f(ii, jj), &epi_ref);

  //     float epigrad2 = gx * epi_ref.x + gy * epi_ref.y;
  //     epigrad2 *= epigrad2;

  //     if (epigrad2 < grad_thresh2) {
  //       // Gradient isn't large enough, skip.
  //       continue;
  //     }

  //     // Grab nltgv2 score.
  //     float nltgv2_cost = nltgv2_blurred(ii, jj);
  //     if (nltgv2_cost < 0.005f) {
  //       continue;
  //     }

  //     score(ii, jj) = nltgv2_cost;

  //     // Fill in best gradients.
  //     if (nltgv2_cost >= best_gradsf(ii_flvl, jj_flvl)) {
  //       best_gradsf(ii_flvl, jj_flvl) = nltgv2_cost;
  //       best_pxf[idx_flvl] = cv::Point2f(jj, ii);
  //     }
  //   }
  // }

  // // Now extract detections of grid.
  // for (int ii = 0; ii < hflvl; ++ii) {
  //   for (int jj = 0; jj < wflvl; ++jj) {
  //     int idx = ii * wflvl + jj;
  //     if ((fmask(ii, jj) > 0) && (best_gradsf(ii, jj) > 0)) {
  //       kps.push_back(cv::KeyPoint(best_pxf[idx], 1));
  //     }
  //   }
  // }

  /*==================== Detect features on grid with single pass ====================*/
  // Compute mask where we already have features.
  int win_size = params.detection_win_size;
  int hclvl = utils::fast_ceil(static_cast<float>(height) / win_size);
  int wclvl = utils::fast_ceil(static_cast<float>(width) / win_size);
  Image1b cmask(hclvl, wclvl, 255);

  for (uint32_t ii = 0; ii < curr_feats.size(); ++ii) {
    // Fill in coarse mask.
    uint32_t x_clvl = curr_feats[ii].x / win_size;
    uint32_t y_clvl = curr_feats[ii].y / win_size;
    cmask(y_clvl, x_clvl) = 0;
  }

  // Load epipolar geometry from prev to ref.
  Sophus::SE3f T_ref_to_prev(fprev.pose.inverse() * fref.pose);
  epigeo.loadGeometry(T_ref_to_prev.unit_quaternion(),
                      T_ref_to_prev.translation());

  // Coarse pass.
  Image1f score(height, width, std::numeric_limits<float>::quiet_NaN());
  Image1f best_gradsc(hclvl, wclvl, 0.0f);
  std::vector<cv::Point2f> best_pxc(hclvl * wclvl);
  float grad_thresh2 = params.min_grad_mag * params.min_grad_mag;
  for (uint32_t ii = border + row_offset; ii < height - border - row_offset; ++ii) {
    for (uint32_t jj = border; jj < width - border; ++jj) {
      // Check gradient magnitude.
      float gx = fref.gradx[0](ii, jj);
      float gy = fref.grady[0](ii, jj);
      float gmag2 = gx*gx + gy*gy;
      if (gmag2 < grad_thresh2) {
        continue;
      }

      // Check gradient magnitude in epipolar direction.
      cv::Point2f epi_ref;
      epigeo.referenceEpiline(cv::Point2f(ii, jj), &epi_ref);

      float epigrad = gx * epi_ref.x + gy * epi_ref.y;
      float epigrad2 = epigrad * epigrad;

      if (epigrad2 < grad_thresh2) {
        // Gradient isn't large enough, skip.
        continue;
      }

      int ii_clvl = static_cast<float>(ii) / win_size;
      int jj_clvl = static_cast<float>(jj) / win_size;
      int idx_clvl = ii_clvl * wclvl + jj_clvl;

      // Fill in best gradients.
      if (epigrad2 >= best_gradsc(ii_clvl, jj_clvl)) {
        best_gradsc(ii_clvl, jj_clvl) = epigrad2;
        best_pxc[idx_clvl] = cv::Point2f(jj, ii);
      }

      // Fill in score (for visualization).
      score(ii, jj) = utils::fast_abs(epigrad);
    }
  }

  // Now extract detections of grid.
  std::vector<cv::KeyPoint> kps;
  for (int ii = 0; ii < hclvl; ++ii) {
    for (int jj = 0; jj < wclvl; ++jj) {
      int idx = ii * wclvl + jj;
      if ((cmask(ii, jj) > 0) && (best_gradsc(ii, jj) > 0)) {
        kps.push_back(cv::KeyPoint(best_pxc[idx], 1));
      }
    }
  }

  cv::KeyPoint::convert(kps, *features);

  stats->tock("detection");
  if (!params.debug_quiet && params.debug_print_timing_detection) {
    printf("Flame/detection(%i) = %f ms\n",
           features->size(), stats->timings("detection"));
  }

  if (params.debug_draw_detections) {
    drawDetections(params, fref.img[0], score, 0.005f, kps, stats,
                   debug_img);
  }

  return;
}

bool Flame::updateFeatureIDepths(const Params& params,
                                 const Matrix3f& K,
                                 const Matrix3f& Kinv,
                                 const FrameIDToFrame& pfs,
                                 const utils::Frame& fnew,
                                 const utils::Frame& curr_pf,
                                 std::vector<FeatureWithIDepth>* feats,
                                 utils::StatsTracker* stats,
                                 Image3b* debug_img) {
  stats->tick("update_idepths");

  int debug_feature_radius = 4 * fnew.img[0].cols / 320; // For drawing features.

  if (params.debug_draw_matches) {
    cv::cvtColor(fnew.img[0], *debug_img, cv::COLOR_GRAY2RGB);
  }

  bool success = false;
  int num_total_updates = 0;

  // Count failure types
  std::atomic<int> num_fail_max_var(0);
  std::atomic<int> num_fail_max_dropouts(0);
  std::atomic<int> num_ref_patch(0);
  std::atomic<int> num_amb_match(0);
  std::atomic<int> num_max_cost(0);

#pragma omp parallel for num_threads(params.omp_num_threads) schedule(static, params.omp_chunk_size) // NOLINT
  for (int ii = 0; ii < feats->size(); ++ii) {
    stereo::EpipolarGeometry<float> epigeo(K, Kinv);

    FeatureWithIDepth& fii = (*feats)[ii];

    // Load geometry.
    Sophus::SE3f T_ref_to_new = fnew.pose.inverse() * pfs.at(fii.frame_id)->pose;
    Sophus::SE3f T_new_to_ref = pfs.at(fii.frame_id)->pose.inverse() * fnew.pose;
    epigeo.loadGeometry(T_ref_to_new.unit_quaternion(),
                        T_ref_to_new.translation());

    // Check baseline.
    float baseline = T_ref_to_new.translation().norm();
    if (baseline < params.min_baseline) {
      // Not enough baseline. Skip.
      continue;
    }

    /*==================== Track feature in new image ====================*/
    cv::Point2f flow;
    float residual;
    bool track_success = trackFeature(params, K, Kinv, pfs, epigeo, fnew,
                                      curr_pf, &fii, &flow, &residual,
                                      debug_img);

    // Count failure types.
    if (fii.search_status == stereo::inverse_depth_filter::Status::FAIL_REF_PATCH_GRADIENT) {
      ++num_ref_patch;
    } else if (fii.search_status == stereo::inverse_depth_filter::Status::FAIL_AMBIGUOUS_MATCH) {
      ++num_amb_match;
    } else if (fii.search_status == stereo::inverse_depth_filter::Status::FAIL_MAX_COST) {
      ++num_max_cost;
    } else {
      // Unknown status.
    }

    if (!track_success) {
      fii.idepth_var *= params.fparams.process_fail_var_factor;
      if (fii.idepth_var > params.idepth_var_max) {
        fii.valid = false;
        ++num_fail_max_var;

        if (params.debug_draw_matches) {
          cv::Scalar color(0, 255, 0); // Green for max var.
          float blah;
          cv::Point2f fii_cmp;
          epigeo.project(fii.xy, fii.idepth_mu, &fii_cmp, &blah);
          cv::circle(*debug_img, cv::Point2i(fii_cmp.x + 0.5f, fii_cmp.y + 0.5f),
                     debug_feature_radius, color);
        }
      }

      fii.num_dropouts++;
      if (fii.num_dropouts > params.max_dropouts) {
        fii.valid = false;
        ++num_fail_max_dropouts;

        if (params.debug_draw_matches) {
          cv::Scalar color(255, 0, 0); // Blue for max dropouts.
          float blah;
          cv::Point2f fii_cmp;
          epigeo.project(fii.xy, fii.idepth_mu, &fii_cmp, &blah);
          cv::circle(*debug_img, cv::Point2i(fii_cmp.x + 0.5f, fii_cmp.y + 0.5f),
                     debug_feature_radius, color);
        }
      }

      continue;
    }

    /*==================== Update idepth ====================*/
    // Load stuff into meas model.
    stereo::InverseDepthMeasModel model(K, Kinv, params.zparams);
    auto& pfii = pfs.at(fii.frame_id);
    model.loadGeometry(pfii->pose, fnew.pose);
    model.loadPaddedImages(pfii->img_pad[0], fnew.img_pad[0],
                           pfii->gradx_pad[0],
                           pfii->grady_pad[0],
                           fnew.gradx_pad[0], fnew.grady_pad[0]);

    // Generate measurement.
    float mu_meas, var_meas;
    bool sense_success = model.idepth(fii.xy, flow, &mu_meas, &var_meas);

    if (!sense_success) {
      if (!params.debug_quiet && params.debug_print_verbose_errors) {
        fprintf(stderr, "FAIL:Sense: u_ref = (%f, %f), id = %f, var = %f\n",
                fii.xy.x, fii.xy.y, fii.idepth_mu, fii.idepth_var);
      }

      cv::Scalar color;
      fii.idepth_var *= params.fparams.process_fail_var_factor;
      if (fii.idepth_var > params.idepth_var_max) {
        fii.valid = false;
        ++num_fail_max_var;


        if (params.debug_draw_matches) {
          cv::Scalar color(0, 255, 0); // Green for max var.
          float blah;
          cv::Point2f fii_cmp;
          epigeo.project(fii.xy, fii.idepth_mu, &fii_cmp, &blah);
          cv::circle(*debug_img, cv::Point2i(fii_cmp.x + 0.5f, fii_cmp.y + 0.5f),
                     debug_feature_radius, color);
        }
      }

      fii.num_dropouts++;
      if (fii.num_dropouts > params.max_dropouts) {
        fii.valid = false;
        ++num_fail_max_dropouts;

        if (params.debug_draw_matches) {
          cv::Scalar color(255, 0, 0); // Blue for max dropouts.
          float blah;
          cv::Point2f fii_cmp;
          epigeo.project(fii.xy, fii.idepth_mu, &fii_cmp, &blah);
          cv::circle(*debug_img, cv::Point2i(fii_cmp.x + 0.5f, fii_cmp.y + 0.5f),
                     debug_feature_radius, color);
        }
      }

      continue;
    }

    // Fuse.
    float mu_post, var_post;
    bool fuse_success =
        stereo::inverse_depth_filter::update(fii.idepth_mu,
                                             fii.idepth_var,
                                             mu_meas, var_meas,
                                             &mu_post, &var_post,
                                             params.outlier_sigma_thresh);

    if (!fuse_success) {
      if (!params.debug_quiet && params.debug_print_verbose_errors) {
        fprintf(stderr, "FAIL:Fuse: mu_meas = %f, var_meas = %f\n",
                mu_meas, var_meas);
      }

      cv::Scalar color;
      fii.idepth_var *= params.fparams.process_fail_var_factor;
      if (fii.idepth_var > params.idepth_var_max) {
        fii.valid = false;
        ++num_fail_max_var;

        if (params.debug_draw_matches) {
          cv::Scalar color(0, 255, 0); // Green for max var.
          float blah;
          cv::Point2f fii_cmp;
          epigeo.project(fii.xy, fii.idepth_mu, &fii_cmp, &blah);
          cv::circle(*debug_img, cv::Point2i(fii_cmp.x + 0.5f, fii_cmp.y + 0.5f),
                     debug_feature_radius, color);
        }
      }

      fii.num_dropouts++;
      if (fii.num_dropouts > params.max_dropouts) {
        fii.valid = false;
        ++num_fail_max_dropouts;

        if (params.debug_draw_matches) {
          cv::Scalar color(255, 0, 0); // Blue for max dropouts.
          float blah;
          cv::Point2f fii_cmp;
          epigeo.project(fii.xy, fii.idepth_mu, &fii_cmp, &blah);
          cv::circle(*debug_img, cv::Point2i(fii_cmp.x + 0.5f, fii_cmp.y + 0.5f),
                     debug_feature_radius, color);
        }
      }

      continue;
    }

    if (params.do_meas_fusion) {
      fii.idepth_mu = mu_post;
      fii.idepth_var = var_post;
    } else {
      fii.idepth_mu = mu_meas;
      fii.idepth_var = var_meas;
    }

    fii.valid = true;
    fii.num_updates++;
    fii.num_dropouts = 0;
    num_total_updates++;

    success = true;
  }

  // Fill in some stats.
  stats->set("num_idepth_updates", num_total_updates);
  stats->set("num_fail_max_var", num_fail_max_var.load());
  stats->set("num_fail_max_dropouts", num_fail_max_dropouts.load());
  stats->set("num_fail_ref_patch_grad", num_ref_patch.load());
  stats->set("num_fail_ambiguous_match", num_amb_match.load());
  stats->set("num_fail_max_cost", num_max_cost.load());

  if (params.debug_draw_matches) {
    if (params.debug_flip_images) {
      // Flip image for display.
      cv::flip(*debug_img, *debug_img, -1);
    }

    if (params.debug_draw_text_overlay) {
      // Print some info.
      char buf[200];
      snprintf(buf, sizeof(buf), "%i updates, %i fails (%i ref_patch_grad, %i, amb_match, %i max_cost)",
               num_total_updates, feats->size() - num_total_updates,
               num_ref_patch.load(), num_amb_match.load(), num_max_cost.load());
      float font_scale = 0.6 / (640.0f / debug_img->cols);
      int font_thickness = 2.0f / (640.0f / debug_img->cols) + 0.5f;
      cv::putText(*debug_img, buf,
                  cv::Point(10, debug_img->rows - 5),
                  cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(200, 200, 250),
                  font_thickness, 8);
    }
  }

  stats->tock("update_idepths");
  if (!params.debug_quiet && params.debug_print_timing_update_idepths) {
    printf("Flame/update_idepths(%i) = %f ms\n",
           feats->size(), stats->timings("update_idepths"));
  }

  return success;
}

bool Flame::trackFeature(const Params& params,
                         const Matrix3f& K,
                         const Matrix3f& Kinv,
                         const FrameIDToFrame& pfs,
                         const stereo::EpipolarGeometry<float>& epigeo,
                         const utils::Frame& fnew,
                         const utils::Frame& curr_pf,
                         FeatureWithIDepth* feat,
                         cv::Point2f* flow,
                         float* residual,
                         Image3b* debug_img) {
  int debug_feature_radius = fnew.img[0].cols / 320; // For drawing features.
  cv::Point2i debug_feature_offset(debug_feature_radius, debug_feature_radius);

  /*==================== Predict feature in new image ====================*/
  cv::Point2f u_cmp;
  float idepth_cmp, var_cmp;
  bool pred_success =
      stereo::inverse_depth_filter::predict(epigeo,
                                            params.fparams.process_var_factor,
                                            feat->xy,
                                            feat->idepth_mu,
                                            feat->idepth_var,
                                            &u_cmp, &idepth_cmp, &var_cmp);

  if (!pred_success) {
    // TOOD(wng): Dropout count.
    return false;
  }

  /*==================== Use LSD-SLAM style direct search ====================*/
  int width = fnew.img[0].cols;
  int height = fnew.img[0].rows;

  int row_offset = 0;
  if (params.do_letterbox) {
    // Only detect features in middle third of image.
    row_offset = height/3;
  }

  int border = params.rescale_factor_max * params.fparams.win_size / 2 + 1;
  cv::Rect valid_region(border, border + row_offset,
                        width - 2*border, height - 2*border - 2*row_offset);

  // Compute rescale factor. This is how much to grow/shrink the reference
  // patch based on the difference in idepth between the reference and
  // comparison frames (i.e. travel along the optical axis).
  float rescale_factor = 1.0f;
  if ((feat->idepth_mu > 0.0f) && (idepth_cmp > 0.0f)) {
    rescale_factor = idepth_cmp / feat->idepth_mu;
  }

  FLAME_ASSERT(!std::isnan(rescale_factor));
  FLAME_ASSERT(rescale_factor > 0);

  if ((rescale_factor <= params.rescale_factor_min) ||
      (rescale_factor >= params.rescale_factor_max)) {
    // Warp on reference patch is too large - i.e. idepth difference between
    // reference frame and comparison frame is too large. Move the feature to
    // the most recent pf.
    bool verbose = false;
    if (verbose) {
      fprintf(stderr, "Flame[FAIL]: bad rescale_factor = %f, prior_idepth = %f, idepth_cmp = %f\n",
              rescale_factor, feat->idepth_mu, idepth_cmp);
    }

    if (verbose) {
      fprintf(stderr, "Flame[WARNING]: Moving feature from u_ref = (%f, %f) idepth = %f to u_cmp = (%f, %f) idepth = %f\n",
              feat->xy.x, feat->xy.y, feat->idepth_mu,
              u_cmp.x, u_cmp.y, idepth_cmp);
    }

    // If this feature has converged already, move it so that it's parent
    // pose is the most recent poseframe frame rather than throw it away.
    stereo::EpipolarGeometry<float> epipf(K, Kinv);
    Sophus::SE3f T_old_to_new = curr_pf.pose.inverse() * pfs.at(feat->frame_id)->pose;
    epipf.loadGeometry(T_old_to_new.unit_quaternion(),
                       T_old_to_new.translation());

    cv::Point2f u_pf;
    float idepth_pf, var_pf;
    bool move_success =
        stereo::inverse_depth_filter::predict(epipf,
                                              params.fparams.process_var_factor,
                                              feat->xy,
                                              feat->idepth_mu,
                                              feat->idepth_var,
                                              &u_pf, &idepth_pf, &var_pf);
    if (!move_success || !valid_region.contains(u_pf)) {
      feat->valid = false;
      if (params.debug_draw_matches) {
        // Failed move in brown.
        cv::Point2i u_cmpi(u_cmp.x + 0.5f, u_cmp.y + 0.5f);
        cv::rectangle(*debug_img, u_cmpi - debug_feature_offset,
                      u_cmpi + debug_feature_offset, cv::Scalar(0, 51, 102), -1);
      }
      return false;
    }

    feat->frame_id = curr_pf.id;
    feat->xy = u_pf;
    float old_idepth = feat->idepth_mu;
    feat->idepth_mu = idepth_pf;

    // Project idepth variance.
    float var_factor4 = idepth_pf / old_idepth;
    var_factor4 *= var_factor4;
    var_factor4 *= var_factor4;

    if (idepth_pf < 1e-6) {
      // If feat_ref.idepth_mu == 0, then var_factor4 is inf.
      var_factor4 = 1;
    }
    feat->idepth_var *= var_factor4;

    if (params.debug_draw_matches) {
      // Successful move in magenta.
      cv::Point2i u_cmpi(u_cmp.x + 0.5f, u_cmp.y + 0.5f);
      cv::rectangle(*debug_img, u_cmpi - debug_feature_offset,
                    u_cmpi + debug_feature_offset, cv::Scalar(255, 0, 255), -1);
    }

    return false;
  }

  cv::Point2f u_start, u_end, epi;
  bool region_success =
      stereo::inverse_depth_filter::getSearchRegion(params.fparams, epigeo,
                                                    width, height, feat->xy,
                                                    feat->idepth_mu, feat->idepth_var,
                                                    &u_start, &u_end, &epi);
  if (!region_success) {
    if (params.debug_draw_matches) {
      // Failed search region in black.
      cv::Point2i u_cmpi(u_cmp.x + 0.5f, u_cmp.y + 0.5f);
      cv::rectangle(*debug_img, u_cmpi - debug_feature_offset,
                    u_cmpi + debug_feature_offset, cv::Scalar(0, 0, 0), -1);
    }
    return false;
  }

  int padding = (fnew.img_pad[0].rows - fnew.img[0].rows) / 2;
  cv::Point2f offset(padding, padding);

  if (!valid_region.contains(feat->xy)) {
    if (!params.debug_quiet) {
      printf("Flame[WARNING]: Feature outside bounds: feat->xy = %f, %f\n",
             feat->xy.x, feat->xy.y);
    }
    return false;
  }
  // FLAME_ASSERT(valid_region.contains(feat->xy));

  auto search_success =
      stereo::inverse_depth_filter::search(params.fparams, epigeo, rescale_factor,
                                           pfs.at(feat->frame_id)->img_pad[0],
                                           fnew.img_pad[0],
                                           feat->xy + offset, u_start + offset,
                                           u_end + offset, &u_cmp);
  feat->search_status = search_success;

  // Parse output.
  if (search_success != stereo::inverse_depth_filter::SUCCESS) {
    if (params.debug_draw_matches) {
      // Color failure by error status.
      cv::Vec3b color;

      if (search_success == stereo::inverse_depth_filter::FAIL_REF_PATCH_GRADIENT) {
        color = cv::Vec3b(255, 255, 0); // Cyan.
        if (feat->num_updates == 0) {
          color = cv::Vec3b(255, 255, 255); // White.
        }
      } else if (search_success == stereo::inverse_depth_filter::FAIL_AMBIGUOUS_MATCH) {
        color = cv::Vec3b(0, 0, 255); // Red.
      } else if (search_success == stereo::inverse_depth_filter::FAIL_MAX_COST) {
        color = cv::Vec3b(0, 255, 255); // Yellow.
      } else if (search_success != stereo::inverse_depth_filter::SUCCESS) {
        fprintf(stderr, "inverse_depth_filter::search: Unrecognized status!\n");
        FLAME_ASSERT(false);
        return false;
      }

      cv::Point2i u_cmpi(u_cmp.x + 0.5f, u_cmp.y + 0.5f);
      cv::rectangle(*debug_img, u_cmpi - debug_feature_offset,
                    u_cmpi + debug_feature_offset,
                    cv::Scalar(color[0], color[1], color[2]), -1);

      auto colormap = [&color](float a) { return color; };
      utils::applyColorMapLine(u_start, u_end, 1, 1, colormap, 0.5, debug_img);
    }

    return false;
  }

  *flow = u_cmp - offset;

  // if (params.debug_draw_matches) {
  //   cv::Point2i flowi(flow->x + 0.5f, flow->y + 0.5f);
  //   // cv::circle(*debug_img, cv::Point2i(flow->x + 0.5f, flow->y + 0.5f),
  //   //            2, cv::Scalar(0, 255, 0));

  //   // cv::Vec3b color(0, 255, 0);

  //   cv::Vec3b color = utils::blendColor(cv::Vec3b(255, 0, 0),
  //                                       cv::Vec3b(0, 255, 0),
  //                                       feat->num_updates, 0, 30);
  //   // cv::Vec3b color = utils::jet(feat->num_updates, 0, 30);
  //   cv::rectangle(*debug_img, flowi - debug_feature_offset,
  //                 flowi + debug_feature_offset,
  //                 cv::Scalar(color[0], color[1], color[2]), -1);

  //   auto colormap = [&color](float a) { return color; };
  //   utils::applyColorMapLine(u_start, u_end, 1, 1, colormap, 0.5, debug_img);
  // }

  return true;
}

void Flame::projectFeatures(const Params& params,
                            const Matrix3f& K,
                            const Matrix3f& Kinv,
                            const FrameIDToFrame& pfs,
                            const utils::Frame& fcur,
                            std::vector<FeatureWithIDepth>* feats,
                            std::vector<FeatureWithIDepth>* feats_in_curr,
                            utils::StatsTracker* stats) {
  stats->tick("project_features");

  feats_in_curr->resize(feats->size());

  int row_offset = 0;
  if (params.do_letterbox) {
    // Only detect features in middle third of image.
    row_offset = fcur.img[0].rows/3;
  }

  int border = params.rescale_factor_max * params.fparams.win_size / 2 + 1;
  cv::Rect_<float> valid_region(border, border  + row_offset,
                                fcur.img[0].cols - 2*border,
                                fcur.img[0].rows - 2*border - 2*row_offset);

#pragma omp parallel for num_threads(params.omp_num_threads) schedule(static, params.omp_chunk_size) // NOLINT
  for (int ii = 0; ii < feats->size(); ++ii) {
    FeatureWithIDepth& feat_ref = (*feats)[ii];
    FeatureWithIDepth& feat_cur = (*feats_in_curr)[ii];

    // Load geometry.
    stereo::EpipolarGeometry<float> epigeo(K, Kinv);
    auto& fref = pfs.at(feat_ref.frame_id);
    Sophus::SE3f T_ref_to_cur = fcur.pose.inverse() * fref->pose;
    Sophus::SE3f T_cur_to_ref = fref->pose.inverse() * fcur.pose;
    epigeo.loadGeometry(T_ref_to_cur.unit_quaternion(),
                        T_ref_to_cur.translation());

    if (!feat_ref.valid) {
      feat_cur.valid = false;
      continue;
    }

    // Project feature point and idepth mean and check that it lies within
    // current frame.
    cv::Point2f xy_cur;
    float idepth_cur;
    epigeo.project(feat_ref.xy, feat_ref.idepth_mu, &xy_cur, &idepth_cur);

    if (!valid_region.contains(xy_cur) || (idepth_cur < 0.0f)) {
      // Point projected outside of image or behind camera.
      feat_ref.valid = false;
      feat_cur.valid = false;
      continue;
    }

    FLAME_ASSERT(xy_cur.x >= 0);
    FLAME_ASSERT(xy_cur.x < fcur.img[0].cols);
    FLAME_ASSERT(xy_cur.y >= 0);
    FLAME_ASSERT(xy_cur.y < fcur.img[0].rows);

    // Update feature in current frame.
    feat_cur.valid = true;
    feat_cur.id = feat_ref.id;
    feat_cur.frame_id = fcur.id;
    feat_cur.xy = xy_cur;
    feat_cur.idepth_mu = idepth_cur;
    feat_cur.num_updates = feat_ref.num_updates;

    // Project idepth variance.
    float var_factor4 = feat_cur.idepth_mu / feat_ref.idepth_mu;
    var_factor4 *= var_factor4;
    var_factor4 *= var_factor4;

    if (feat_ref.idepth_mu < 1e-6) {
      // If feat_ref.idepth_mu == 0, then var_factor4 is inf.
      var_factor4 = 1;
    }
    feat_cur.idepth_var = var_factor4 * feat_ref.idepth_var;
  }

  // Remove invalid features.
  std::vector<FeatureWithIDepth> featsv;
  std::vector<FeatureWithIDepth> feats_in_currv;
  featsv.reserve(feats->size());
  feats_in_currv.reserve(feats_in_curr->size());
  for (int ii = 0; ii < feats->size(); ++ii) {
    const FeatureWithIDepth& feat_ref = (*feats)[ii];
    const FeatureWithIDepth& feat_curr = (*feats_in_curr)[ii];

    if (feat_ref.valid) {
      featsv.push_back(feat_ref);
      feats_in_currv.push_back(feat_curr);
      FLAME_ASSERT(feat_curr.valid);
    } else {
      FLAME_ASSERT(!feat_curr.valid);
    }
  }
  feats->swap(featsv);
  feats_in_curr->swap(feats_in_currv);

  stats->tock("project_features");
  if (!params.debug_quiet && params.debug_print_timing_project_features) {
    printf("Flame/project_features = %f ms\n",
           stats->timings("project_features"));
  }

  return;
}

void Flame::projectGraph(const Params& params,
                         const stereo::EpipolarGeometry<float>& epigeo,
                         const utils::Frame& fnew,
                         Graph* graph,
                         float graph_scale,
                         FeatureToVtx* feat_to_vtx,
                         VtxToFeature* vtx_to_feat,
                         utils::StatsTracker* stats) {
  stats->tick("project_graph");

  int width = fnew.img[0].cols;
  int height = fnew.img[0].rows;

  int row_offset = 0;
  if (params.do_letterbox) {
    // Only detect features in middle third of image.
    row_offset = height/3;
  }

  int border = params.rescale_factor_max * params.fparams.win_size / 2 + 1;
  cv::Rect_<float> valid_region(border, border + row_offset,
                                width - 2*border,
                                height - 2*border - 2*row_offset);

  // Mark vertices we should remove.
  std::unordered_set<VertexHandle> vtx_to_remove;

  // Loop over vertices and project.
  Graph::vertex_iterator vit, end;
  boost::tie(vit, end) = boost::vertices(*graph);
  for ( ; vit != end; ++vit) {
    auto& vtx = (*graph)[*vit];

    // Project into new frame.
    cv::Point2f u_new;
    float idepth_new;
    epigeo.project(vtx.pos, vtx.x * graph_scale, &u_new, &idepth_new);
    vtx.pos = u_new;
    vtx.x = idepth_new / graph_scale;

    if (!valid_region.contains(u_new) || (idepth_new < 0.0f)) {
      // Projected outside of image or behind camera.
      vtx_to_remove.insert(*vit);
      continue;
    }

    if (params.do_grad_check_after_projection) {
      // Check gradient.
      float gx = utils::bilinearInterp<float, float>(fnew.gradx[0], u_new.x,
                                                     u_new.y);
      float gy = utils::bilinearInterp<float, float>(fnew.grady[0], u_new.x,
                                                     u_new.y);
      if (gx*gx + gy*gy < params.min_grad_mag * params.min_grad_mag) {
        // Projected to a point without gradient.
        vtx_to_remove.insert(*vit);
      }
    }
  }

  // Remove marked vertices.
  for (auto vtx : vtx_to_remove) {
    boost::clear_vertex(vtx, *graph); // Remove connected edges.
    boost::remove_vertex(vtx, *graph); // Remove vertex.

    int feat_id = (*vtx_to_feat)[vtx];
    feat_to_vtx->erase(feat_id);
    vtx_to_feat->erase(vtx);
  }

  stats->tock("project_graph");
  if (!params.debug_quiet && params.debug_print_timing_project_graph) {
    printf("Flame/project_graph = %f ms\n",
           stats->timings("project_graph"));
  }

  return;
}

bool Flame::syncGraph(const Params& params,
                      const Matrix3f& Kinv,
                      const FrameIDToFrame& pfs,
                      const Image1f& idepthmap,
                      const std::vector<FeatureWithIDepth>& feats,
                      const std::vector<FeatureWithIDepth>& feats_in_curr,
                      utils::Delaunay* triangulator,
                      Graph* graph,
                      float graph_scale,
                      FeatureToVtx* feat_to_vtx,
                      VtxToFeature* vtx_to_feat,
                      utils::StatsTracker* stats) {
  stats->tick("sync_graph");

  /*==================== Preprocessing ====================*/
  // Create set of feature ids to update graph with.
  std::unordered_set<int> feats_to_update;
  feats_to_update.reserve(feats.size());
  std::unordered_map<int, int> feat_id_to_idx; // Map from feature ID to feature index in vectors.
  feat_id_to_idx.reserve(feats.size());
  for (int ii = 0; ii < feats.size(); ++ii) {
    const FeatureWithIDepth& feat = feats[ii];

    feat_id_to_idx[feat.id] = ii;

    float idepth = feat.idepth_mu;
    float var = feat.idepth_var;

    FLAME_ASSERT(idepth >= 0.0f);

    // Project into world.
    Vector3f pix(feat.xy.x, feat.xy.y, 1.0f);
    pix /= idepth;
    Vector3f xyz(Kinv * pix);
    xyz = pfs.at(feat.frame_id)->pose * xyz;

    if (feat.valid && (var < params.idepth_var_max_graph)
        && (-xyz(1) >= params.min_height) && (-xyz(1) <= params.max_height)) {
      feats_to_update.insert(feat.id);
    }
  }

  // Mark vertices to remove.
  std::unordered_set<VertexHandle> vtx_to_remove;

  /*==================== Update existing vertices ====================*/
  Graph::vertex_iterator vit, end;
  boost::tie(vit, end) = boost::vertices(*graph);
  for ( ; vit != end; ++vit) {
    auto& vtx = (*graph)[*vit];

    // First check if this vertex's associated feature is still valid.
    int feat_id = vtx_to_feat->at(*vit);
    if (feats_to_update.count(feat_id) == 0) {
      // This vertex no longer has an associated features. Remove it.
      vtx_to_remove.insert(*vit);
    }

    const FeatureWithIDepth& feat = feats_in_curr[feat_id_to_idx[feat_id]];

    // Update vertex data.
    vtx.pos = feat.xy;
    vtx.data_term = feat.idepth_mu / graph_scale;
    vtx.data_weight = (params.adaptive_data_weights) ?
        1.0f/feat.idepth_var : 1.0f;

    if (!params.do_nltgv2) {
      // Not doing regularization. Set graph data to feature.
      vtx.x = feat.idepth_mu / graph_scale;
    }

    if (params.check_sticky_obstacles && (vtx.x - vtx.data_term > 0.25f)) {
      // Smoothed solution is being sucked towards camera. Reset it.
      vtx.x = vtx.data_term;
    }

    // Remove associated feature from the update list.
    feats_to_update.erase(feat_id);
  }

  /*==================== Remove marked vertices ====================*/
  for (auto vtx : vtx_to_remove) {
    boost::clear_vertex(vtx, *graph); // Remove connected edges.
    boost::remove_vertex(vtx, *graph); // Remove vertex.

    int feat_id = vtx_to_feat->at(vtx);
    feat_to_vtx->erase(feat_id);
    vtx_to_feat->erase(vtx);
  }

  /*==================== Add new vertices to graph ====================*/
  for (int feat_id : feats_to_update) {
    const FeatureWithIDepth feat = feats_in_curr[feat_id_to_idx[feat_id]];

    // Add new vertex to graph.
    VertexHandle vtx_ii = boost::add_vertex(dgraph::VertexData(), *graph);
    (*feat_to_vtx)[feat_id] = vtx_ii;
    (*vtx_to_feat)[vtx_ii] = feat_id;

    // Initialize vertex data.
    auto& vdata = (*graph)[vtx_ii];
    vdata.pos = feat.xy;
    vdata.data_term = feat.idepth_mu / graph_scale;
    vdata.data_weight = (params.adaptive_data_weights) ?
        1.0f/feat.idepth_var : 1.0f;

    vdata.x = feat.idepth_mu / graph_scale;
    vdata.x_bar = vdata.x;
    vdata.x_prev = vdata.x;
  }

  /*==================== Retriangulate ====================*/
  std::vector<cv::Point2f> vtx_xy;
  vtx_xy.reserve(boost::num_vertices(*graph));
  VtxIdxToHandle vtx_idx_to_handle;
  boost::tie(vit, end) = boost::vertices(*graph);
  int count = 0;
  for ( ; vit != end; ++vit) {
    auto& vtx = (*graph)[*vit];
    vtx_xy.push_back(vtx.pos);
    vtx_idx_to_handle[count] = *vit;
    count++;
  }

  if (vtx_xy.size() < 3) {
    // Not enough detections.
    if (!params.debug_quiet) {
      fprintf(stderr, "Flame[Error]: Too few vertices (%lu) to triangulate!\n",
              static_cast<uint32_t>(vtx_xy.size()));
    }
    return false;
  }

  triangulate(params, vtx_xy, triangulator, stats);

  /*==================== Update graph edges ====================*/
  // Set all edges to invalid.
  Graph::edge_iterator eit, eend;
  boost::tie(eit, eend) = boost::edges(*graph);
  for ( ; eit != eend; ++eit) {
    (*graph)[*eit].valid = false;
  }

  // Add new edges.
  for (int ii = 0; ii < triangulator->edges().size(); ++ii) {
    VertexHandle vtx_ii = vtx_idx_to_handle[triangulator->edges()[ii][0]];
    VertexHandle vtx_jj = vtx_idx_to_handle[triangulator->edges()[ii][1]];

    // Compute edge length.
    cv::Point2f u_ii = vtx_xy[triangulator->edges()[ii][0]];
    cv::Point2f u_jj = vtx_xy[triangulator->edges()[ii][1]];
    cv::Point2f diff(u_ii - u_jj);
    float edge_length = sqrt(diff.x*diff.x + diff.y*diff.y);

    if (!boost::edge(vtx_ii, vtx_jj, *graph).second) {
      // Add edge to graph if new.
      boost::add_edge(vtx_ii, vtx_jj, dgraph::EdgeData(), *graph);
    }

    // Initialize edge data.
    const auto& epair = boost::edge(vtx_ii, vtx_jj, *graph);
    auto& edata = (*graph)[epair.first];
    edata.alpha = 1.0f / edge_length;
    edata.beta = 1.0f;
    edata.valid = true;
  }

  // Remove edges no longer part of triangulation.
  std::vector<EdgeHandle> edges_to_remove; // Can't use a hashmap because hash
  // for edge_descriptors are weird.
  edges_to_remove.reserve(boost::num_edges(*graph));
  boost::tie(eit, eend) = boost::edges(*graph);
  for ( ; eit != eend; ++eit) {
    if (!(*graph)[*eit].valid) {
      edges_to_remove.push_back(*eit);
    }
  }
  for (auto& edge : edges_to_remove) {
    boost::remove_edge(edge, *graph);
  }

  FLAME_ASSERT(triangulator->edges().size() == boost::num_edges(*graph));

  /*==================== Initialize idepth for new vertices ====================*/
  // Needs to happen after triangulation so that we can use neighbor idepths to
  // initialize.
  for (int feat_id : feats_to_update) {
    VertexHandle vtx_ii = (*feat_to_vtx)[feat_id];
    auto& vdata = (*graph)[vtx_ii];

    float init_idepth = feats_in_curr[feat_id_to_idx[feat_id]].idepth_mu;

    if (params.init_with_prediction) {
      // Set initial idepth to predicted value from dense idepthmap or from
      // neighbors of graph.
      init_idepth = idepthmap(vdata.pos.y + 0.5f, vdata.pos.x + 0.5f);
      if (std::isnan(init_idepth)) {
        // If the predicted value in the idepthmap is invalid, compute the mean
        // idepth of the neighbors.
        Graph::adjacency_iterator nit, end;
        boost::tie(nit, end) = boost::adjacent_vertices(vtx_ii, (*graph));
        float idepth_sum = 0.0f;
        int valid_neighbor_count = 0;
        for ( ; nit != end; ++nit) {
          const auto& vneighdata = (*graph)[*nit];
          if (vneighdata.data_weight > 0.0f) {
            idepth_sum += vneighdata.x * graph_scale;
            valid_neighbor_count++;
          }
        }

        if (valid_neighbor_count > 0) {
          init_idepth = idepth_sum / valid_neighbor_count;
        } else {
          // No valid neighbors. Set to data value.
          init_idepth = feats_in_curr[feat_id_to_idx[feat_id]].idepth_mu;
        }
      }
    }

    vdata.x = init_idepth / graph_scale;
    vdata.x_bar = vdata.x;
    vdata.x_prev = vdata.x;
  }

  // Set some statistics.
  int num_vertices = boost::num_vertices(*graph);
  stats->set("num_feats", feats.size());
  stats->set("num_vtx", num_vertices);
  stats->set("num_tris", triangulator->triangles().size());
  stats->set("num_edges", triangulator->edges().size());

  float smoothness_cost = dgraph::smoothnessCost(params.rparams, *graph);
  float data_cost = dgraph::dataCost(params.rparams, *graph);
  stats->set("nltgv2_total_smoothness_cost", smoothness_cost);
  stats->set("nltgv2_avg_smoothness_cost", smoothness_cost/num_vertices);
  stats->set("nltgv2_total_data_cost", data_cost);
  stats->set("nltgv2_avg_data_cost", data_cost/num_vertices);

  stats->tock("sync_graph");
  if (!params.debug_quiet && params.debug_print_timing_sync_graph) {
    printf("Flame/sync_graph = %f ms, smooth = %f, data = %f\n",
           stats->timings("sync_graph"),
           stats->stats("nltgv2_avg_smoothness_cost"),
           stats->stats("nltgv2_avg_data_cost"));
  }

  return true;
}

void Flame::triangulate(const Params& params,
                        const std::vector<cv::Point2f>& vertices,
                        utils::Delaunay* triangulator,
                        utils::StatsTracker* stats) {
  stats->tick("triangulate");

  triangulator->triangulate(vertices);

  stats->tock("triangulate");
  if (!params.debug_quiet && params.debug_print_timing_triangulate) {
    printf("Flame/triangulate = %f ms\n",
           stats->timings("triangulate"));
  }

  return;
}

void Flame::obliqueTriangleFilter(const Params& params,
                                  const Matrix3f& Kinv,
                                  const std::vector<cv::Point2f>& vertices,
                                  const std::vector<float>& idepths,
                                  const std::vector<Triangle>& triangles,
                                  std::vector<bool>* validity,
                                  utils::StatsTracker* stats) {
  stats->tick("oblique_triangle_filter");

  if (validity->size() != triangles.size()) {
    validity->assign(triangles.size(), true);
  }

  for (int ii = 0; ii < triangles.size(); ++ii) {
    // NOTE: Triangle spits out points in clock-wise order.
    float id0 = idepths[triangles[ii][0]];
    float id1 = idepths[triangles[ii][1]];
    float id2 = idepths[triangles[ii][2]];

    Vector3f p0(vertices[triangles[ii][0]].x,
                vertices[triangles[ii][0]].y,
                1.0f);
    p0 = Kinv * p0 / id0;

    Vector3f p1(vertices[triangles[ii][1]].x,
                vertices[triangles[ii][1]].y,
                1.0f);
    p1 = Kinv * p1 / id1;

    Vector3f p2(vertices[triangles[ii][2]].x,
                vertices[triangles[ii][2]].y,
                1.0f);
    p2 = Kinv * p2 / id2;

    // Inward-facing normal.
    Vector3f delta1(p1 - p0);
    Vector3f delta2(p2 - p0);
    Vector3f normal(delta1.cross(delta2));
    normal.normalize();

    // Compute angle diff between inward normal and viewing ray through center
    // of triangle.
    Vector3f ray((p0 + p1 + p2)/3);
    ray.normalize();

    float angle = fabs(acos(ray.dot(normal)));
    if (angle > params.oblique_normal_thresh) {
      (*validity)[ii] = false;
    }

    // Compute idepth difference between triangle corners.
    float min_id = (id0 < id1) ? id0 : id1;
    min_id = (min_id < id2) ? min_id : id2;

    float max_id = (id0 > id1) ? id0 : id1;
    max_id = (max_id > id2) ? max_id : id2;
    FLAME_ASSERT(max_id >= min_id);

    if ((max_id - min_id)/max_id > params.oblique_idepth_diff_factor) {
      (*validity)[ii] = false;
    }

    if (max_id - min_id > params.oblique_idepth_diff_abs) {
      (*validity)[ii] = false;
    }
  }

  stats->tock("oblique_triangle_filter");

  if (!params.debug_quiet &&
      params.debug_print_timing_oblique_triangle_filter) {
    printf("Flame/oblique_triangle_filter = %f ms\n",
           stats->timings("oblique_triangle_filter"));
  }

  return;
}

void Flame::edgeLengthFilter(const Params& params,
                             int width, int height,
                             const std::vector<cv::Point2f>& vertices,
                             const std::vector<Triangle>& triangles,
                             std::vector<bool>* validity,
                             utils::StatsTracker* stats) {
  stats->tick("edge_length_filter");

  if (validity->size() != triangles.size()) {
    validity->assign(triangles.size(), true);
  }

  float dist_thresh2 = params.edge_length_thresh * width;
  dist_thresh2 *= dist_thresh2;

  for (int ii = 0; ii < triangles.size(); ++ii) {
    cv::Point2f vtx0 = vertices[triangles[ii][0]];
    cv::Point2f vtx1 = vertices[triangles[ii][1]];
    cv::Point2f vtx2 = vertices[triangles[ii][2]];

    cv::Point2f diff01(vtx0 - vtx1);
    cv::Point2f diff02(vtx0 - vtx2);
    cv::Point2f diff12(vtx1 - vtx2);

    float dist01 = diff01.x * diff01.x + diff01.y * diff01.y;
    float dist02 = diff02.x * diff02.x + diff02.y * diff02.y;
    float dist12 = diff12.x * diff12.x + diff12.y * diff12.y;

    if ((dist01 > dist_thresh2) || (dist02 > dist_thresh2) ||
        (dist12 > dist_thresh2)) {
      (*validity)[ii] = false;
    }
  }

  stats->tock("edge_length_filter");

  if (!params.debug_quiet &&
      params.debug_print_timing_edge_length_filter) {
    printf("Flame/edge_length_filter = %f ms\n",
           stats->timings("edge_length_filter"));
  }

  return;
}

void Flame::idepthTriangleFilter(const Params& params,
                                 const std::vector<float>& idepths,
                                 const std::vector<Triangle>& triangles,
                                 std::vector<bool>* validity,
                                 utils::StatsTracker* stats) {
  stats->tick("idepth_triangle_filter");

  if (validity->size() != triangles.size()) {
    validity->assign(triangles.size(), true);
  }

  for (int ii = 0; ii < triangles.size(); ++ii) {
    float id0 = idepths[triangles[ii][0]];
    float id1 = idepths[triangles[ii][1]];
    float id2 = idepths[triangles[ii][2]];

    float mean_idepth = (id0 + id1 + id2) / 3;
    if (mean_idepth < params.min_triangle_idepth) {
      (*validity)[ii] = false;
    }
  }

  stats->tock("idepth_triangle_filter");

  if (!params.debug_quiet &&
      params.debug_print_timing_idepth_triangle_filter) {
    printf("Flame/idepth_triangle_filter = %f ms\n",
           stats->timings("idepth_triangle_filter"));
  }

  return;
}

void Flame::drawDetections(const Params& params,
                           const Image1b& img,
                           const Image1f& score,
                           float max_score,
                           const std::vector<cv::KeyPoint>& kps,
                           utils::StatsTracker* stats,
                           Image3b* debug_img) {
  Image3b img_rgb;
  cv::cvtColor(img, img_rgb, cv::COLOR_GRAY2RGB);

  auto colormap = [max_score](float v, cv::Vec3b c) {
    if (!std::isnan(v)) {
      return utils::jet(v, 0, max_score);
    } else {
      return c;
    }
  };
  if (debug_img->empty()) {
    debug_img->create(score.rows, score.cols);
  }
  debug_img->setTo(cv::Vec3b(0, 0, 0));

  utils::applyColorMap<float>(score, colormap, debug_img);

  // Blend source image and score image.
  *debug_img = 0.7*(*debug_img) + 0.3*img_rgb;

  // cv::drawKeypoints(*debug_img, kps, *debug_img);

  if (params.debug_flip_images) {
    // Flip image for display.
    cv::flip(*debug_img, *debug_img, -1);
  }

  if (params.debug_draw_text_overlay) {
    // Print some info.
    char buf[200];
    snprintf(buf, sizeof(buf), "%4.1fms, %lu new feats, avg_err = %.2f",
             stats->timings("update"), kps.size(),
             stats->stats("avg_photo_score"));
    float font_scale = 0.6 / (640.0f / debug_img->cols);
    int font_thickness = 2.0f / (640.0f / debug_img->cols) + 0.5f;
    cv::putText(*debug_img, buf,
                cv::Point(10, debug_img->rows - 5),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(200, 200, 250),
                font_thickness, 8);
  }

  return;
}

void Flame::drawWireframe(const Params& params,
                          const Image1b& img_curr,
                          const std::vector<Triangle>& triangles,
                          const std::vector<Edge>& edges,
                          const std::vector<bool>& tri_validity,
                          const std::vector<cv::Point2f>& vertices,
                          const std::vector<float>& idepths,
                          utils::StatsTracker* stats,
                          Image3b* debug_img) {
  // auto colormap = [&params](float v) {
  //   return utils::idepthColor(v * params.scene_color_scale);
  // };
  auto colormap = [&params](float v) {
    return utils::jet(v * params.scene_color_scale, 0.0f, 2.0f);
  };

  std::vector<bool> vtx_validity(vertices.size(), true);
  cv::cvtColor(img_curr, *debug_img, cv::COLOR_GRAY2RGB);
  float alpha = 0.5;
  utils::drawColorMappedWireframe(triangles, vertices, idepths,
                                  vtx_validity, tri_validity,
                                  colormap, alpha, debug_img);

  if (params.debug_flip_images) {
    // Flip image for display.
    cv::flip(*debug_img, *debug_img, -1);
  }

  if (params.debug_draw_text_overlay) {
    // Print some info.
    char buf[200];
    snprintf(buf, sizeof(buf), "%4.1fms/%.1fHz (%.1fHz), %lu vtx, %lu tris, %lu edges",
             stats->timings("update"), stats->stats("fps_max"), stats->stats("fps"),
             idepths.size(), triangles.size(), edges.size());
    float font_scale = 0.6 / (640.0f / debug_img->cols);
    int font_thickness = 2.0f / (640.0f / debug_img->cols) + 0.5f;
    cv::putText(*debug_img, buf,
                cv::Point(10, debug_img->rows - 5),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(200, 200, 250),
                font_thickness, 8);
  }

  return;
}

void Flame::drawFeatures(const Params& params,
                         const Image1b& img_curr,
                         const std::vector<FeatureWithIDepth>& feats_in_curr,
                         utils::StatsTracker* stats,
                         Image3b* debug_img) {
  cv::cvtColor(img_curr, *debug_img, cv::COLOR_GRAY2RGB);

  int num_valid = 0;
  int num_converged = 0;
  int num_invalid = 0;
  for (int ii = 0; ii < feats_in_curr.size(); ++ii) {
    const FeatureWithIDepth& feat = feats_in_curr[ii];

    cv::Point2i xyi(feat.xy.x + 0.5f,
                    feat.xy.y + 0.5f);
    // Color by idepth.
    // cv::Vec3b color = utils::idepthColor(feat.idepth_mu *
    //                                      params.scene_color_scale);
    cv::Vec3b color = utils::jet(feat.idepth_mu * params.scene_color_scale,
                                 0.0f, 2.0f);

    // Color by variance.
    // cv::Vec3b color = utils::jet(sqrt(1.0f/feat.idepth_var),
    //                              sqrt(1.0f/params.idepth_var_max),
    //                              10*sqrt(1.0f/params.idepth_var_max_graph));

    // Color by number of successful updates.
    // cv::Vec3b color = utils::jet(feat.num_updates, 0, 30);

    // Color by feature ID.
    // cv::Vec3b color((879879*feat.id) % 255,
    //                 (25234543*feat.id) % 255,
    //                 (54645376*feat.id) % 255);

    if (feat.idepth_var < params.idepth_var_max_graph) {
      // cv::circle(*debug_img, xyi, 3, cv::Scalar(color[0], color[1], color[2]), -1);
      cv::rectangle(*debug_img, xyi - cv::Point2i(2, 2), xyi + cv::Point2i(2, 2),
                    cv::Scalar(color[0], color[1], color[2]), -1);
      num_converged++;
    } else {
      // Color unconverged features black.
      // cv::circle(*debug_img, xyi, 3, cv::Scalar(0, 0, 0));
      // cv::rectangle(*debug_img, xyi - cv::Point2i(2, 2), xyi + cv::Point2i(2, 2),
      //               cv::Scalar(0, 0, 0));
      num_valid++;
    }
  }

  if (params.debug_flip_images) {
    // Flip image for display.
    cv::flip(*debug_img, *debug_img, -1);
  }

  if (params.debug_draw_text_overlay) {
    // Print some info.
    char buf[200];
    snprintf(buf, sizeof(buf), "%4.1fms/%.1fHz (%.1fHz), %i converged, %i valid, %i invalid",
             stats->timings("update"), stats->stats("fps_max"), stats->stats("fps"),
             num_converged, num_valid, num_invalid);
    float font_scale = 0.6 / (640.0f / debug_img->cols);
    int font_thickness = 2.0f / (640.0f / debug_img->cols) + 0.5f;
    cv::putText(*debug_img, buf,
                cv::Point(10, debug_img->rows - 5),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(200, 200, 250),
                font_thickness, 8);
  }

  return;
}

void Flame::getVertexNormals(const Params& params,
                             const Matrix3f& K,
                             const std::vector<cv::Point2f>& vtx,
                             const std::vector<float>& idepths,
                             const std::vector<float>& w1,
                             const std::vector<float>& w2,
                             std::vector<Vector3f>* normals,
                             utils::StatsTracker* stats) {
  stats->tick("normals");

  normals->resize(vtx.size());
#pragma omp parallel for num_threads(params.omp_num_threads) schedule(static, params.omp_chunk_size) // NOLINT
  for (int ii = 0; ii < vtx.size(); ++ii) {
    (*normals)[ii] = planeParamToNormal(K, vtx[ii], idepths[ii], w1[ii], w2[ii]);
  }

  stats->tock("normals");
  if (!params.debug_quiet && params.debug_print_timing_normals) {
    printf("Flame/normals(%i) = %f ms\n",
           vtx.size(), stats->timings("normals"));
  }

  return;
}

void Flame::getVertexNormals(const Params& params,
                             const Matrix3f& Kinv,
                             const std::vector<cv::Point2f>& vtx,
                             const std::vector<float>& idepths,
                             const std::vector<Triangle>& triangles,
                             std::vector<Vector3f>* normals,
                             utils::StatsTracker* stats) {
  stats->tick("normals");

  FLAME_ASSERT(vtx.size() == idepths.size());

  std::vector<int> counts(vtx.size(), 0);
  normals->resize(vtx.size());
#pragma omp parallel for num_threads(params.omp_num_threads) schedule(static, params.omp_chunk_size) // NOLINT
  for (int ii = 0; ii < normals->size(); ++ii) {
    (*normals)[ii](0) = 0.0f;
    (*normals)[ii](1) = 0.0f;
    (*normals)[ii](2) = 0.0f;
  }

  // Walk through triangles and compute averaged vertex normals.
  for (int ii = 0; ii < triangles.size(); ++ii) {
    FLAME_ASSERT(triangles[ii][0] < idepths.size());
    FLAME_ASSERT(triangles[ii][1] < idepths.size());
    FLAME_ASSERT(triangles[ii][2] < idepths.size());

    // NOTE: Triangle spits out points in clock-wise order.
    float id0 = idepths[triangles[ii][0]];
    float id1 = idepths[triangles[ii][1]];
    float id2 = idepths[triangles[ii][2]];

    if ((id0 <= 0) || (id1 <= 0) || (id2 <= 0)) {
      continue;
    }

    // FLAME_ASSERT(id0 > 0);
    // FLAME_ASSERT(id1 > 0);
    // FLAME_ASSERT(id2 > 0);

    Vector3f p0(vtx[triangles[ii][0]].x,
                vtx[triangles[ii][0]].y,
                1.0f);
    p0 = Kinv * p0 / id0;

    Vector3f p1(vtx[triangles[ii][1]].x,
                vtx[triangles[ii][1]].y,
                1.0f);
    p1 = Kinv * p1 / id1;

    Vector3f p2(vtx[triangles[ii][2]].x,
                vtx[triangles[ii][2]].y,
                1.0f);
    p2 = Kinv * p2 / id2;

    // Outward-facing normal.
    Vector3f delta1(p1 - p0);
    Vector3f delta2(p2 - p0);
    Vector3f normal(delta2.cross(delta1));
    normal.normalize();

    (*normals)[triangles[ii][0]] =
        (counts[triangles[ii][0]] * (*normals)[triangles[ii][0]] + normal) /
        (counts[triangles[ii][0]] + 1);
    (*normals)[triangles[ii][0]].normalize();
    counts[triangles[ii][0]]++;

    (*normals)[triangles[ii][1]] =
        (counts[triangles[ii][1]] * (*normals)[triangles[ii][1]] + normal) /
        (counts[triangles[ii][1]] + 1);
    (*normals)[triangles[ii][1]].normalize();
    counts[triangles[ii][1]]++;

    (*normals)[triangles[ii][2]] =
        (counts[triangles[ii][2]] * (*normals)[triangles[ii][2]] + normal) /
        (counts[triangles[ii][2]] + 1);
    (*normals)[triangles[ii][2]].normalize();
    counts[triangles[ii][2]]++;
  }

  stats->tock("normals");
  if (!params.debug_quiet && params.debug_print_timing_normals) {
    printf("Flame/normals(%i) = %f ms\n",
           vtx.size(), stats->timings("normals"));
  }

  return;
}


Vector3f Flame::planeParamToNormal(const Matrix3f& K,
                                   const cv::Point2f& u_ref,
                                   float idepth, float w1,
                                   float w2) {
  double a = w1 * u_ref.x + w2 * u_ref.y - w1*K(0, 0) - w2*K(1, 1);
  double b = K(0, 0)*K(0, 0)*w1*w1 + K(1, 1)*K(1, 1)*w2*w2 +
      (idepth - a)*(idepth - a);

  double d = 1.0/sqrt(b);

  double nx = K(0, 0) * w1 * d;
  double ny = K(1, 1) * w2 * d;
  double nz = (idepth - a) * d;
  Vector3f normal(nx, ny, nz);
  normal.normalize();

  // Make normal face outward.
  normal *= -1;

  return normal;
}

// Extract normal vector of plane from w components.
// See notes on 2016.09.29 (wng).
void Flame::drawNormals(const Params& params,
                        const Matrix3f& K,
                        const Image1b& img_curr,
                        const Image1f& idepthmap,
                        const Image1f& w1_map,
                        const Image1f& w2_map,
                        Image3b* debug_img) {
  cv::cvtColor(img_curr, *debug_img, cv::COLOR_GRAY2RGB);

  for (int ii = 0; ii < img_curr.rows; ++ii) {
    for (int jj = 0; jj < img_curr.cols; ++jj) {
      float idepth = idepthmap(ii, jj);
      double w1 = w1_map(ii, jj);
      double w2 = w2_map(ii, jj);

      Vector3f normal(planeParamToNormal(K, cv::Point2f(jj, ii),
                                         idepth, w1, w2));

      if (normal(2) > 0.0f) {
        (*debug_img)(ii, jj) = utils::normalMap(normal(0), normal(1), normal(2));
      }
    }
  }

  if (params.debug_flip_images) {
    // Flip image for display.
    cv::flip(*debug_img, *debug_img, -1);
  }

  return;
}

void Flame::drawInverseDepthMap(const Params& params,
                                const Image1b& img_curr,
                                const Image1f& idepthmap,
                                utils::StatsTracker* stats,
                                Image3b* debug_img) {
  cv::cvtColor(img_curr, *debug_img, cv::COLOR_GRAY2RGB);

  auto colormap = [&params](float v, cv::Vec3b c) {
    if (!std::isnan(v)) {
      return utils::jet(v * params.scene_color_scale, 0.0f, 2.0f);
    } else {
      return c;
    }
  };

  utils::applyColorMap<float>(idepthmap, colormap, debug_img);

  if (params.debug_flip_images) {
    // Flip image for display.
    cv::flip(*debug_img, *debug_img, -1);
  }

  if (params.debug_draw_text_overlay) {
    // Print some info.
    char buf[200];
    snprintf(buf, sizeof(buf), "%4.1fms/%.1fHz (%.1fHz), %f coverage",
             stats->timings("update"), stats->stats("fps_max"), stats->stats("fps"),
             stats->stats("coverage"));
    float font_scale = 0.6 / (640.0f / debug_img->cols);
    int font_thickness = 2.0f / (640.0f / debug_img->cols) + 0.5f;
    cv::putText(*debug_img, buf,
                cv::Point(10, debug_img->rows - 5),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(200, 200, 250),
                font_thickness, 8);
  }

  return;
}

}  // namespace flame
