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
 * @file flame.h
 * @author W. Nicholas Greene
 * @date 2016-09-16 13:08:11 (Fri)
 */

#pragma once

#include <vector>
#include <deque>
#include <memory>
#include <map>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <limits>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "flame/types.h"
#include "flame/params.h"

#include "flame/optimizers/nltgv2_l1_graph_regularizer.h"

#include "flame/stereo/inverse_depth_filter.h"
#include "flame/stereo/inverse_depth_meas_model.h"
#include "flame/stereo/epipolar_geometry.h"

#include "flame/utils/frame.h"
#include <flame/utils/stats_tracker.h>
#include "flame/utils/delaunay.h"

namespace flame {

// The main optimization graph.
using Graph = optimizers::nltgv2_l1_graph_regularizer::Graph;

// This type is used to access a vertex in the graph.
using VertexHandle = optimizers::nltgv2_l1_graph_regularizer::VertexHandle;

// This type is used to access an edge in the graph.
using EdgeHandle = optimizers::nltgv2_l1_graph_regularizer::EdgeHandle;

// Convenience aliases.
using FeatureToVtx = std::unordered_map<uint32_t, VertexHandle>;
using VtxToFeature = std::unordered_map<VertexHandle, uint32_t>;
using VtxIdxToHandle = std::unordered_map<uint32_t, VertexHandle>;
using VtxHandleToIdx = std::unordered_map<VertexHandle, uint32_t>;

// Needs to be an ordered map so that we can itereate in frame_id order in
// getPoseFrame.
using FrameIDToFrame = std::map<uint32_t, utils::Frame::Ptr>;

/**
 * @brief Struct to hold data needed for feature detection.
 */
struct DetectionData {
  explicit DetectionData(int num_lvls = 1):
      ref(num_lvls), prev(num_lvls), cmp(num_lvls), ref_xy() {}

  utils::Frame ref; // Reference frame.
  utils::Frame prev; // Previous frame.
  utils::Frame cmp; // Comparison frame.
  std::vector<Point2f> ref_xy; // Current feature locations in ref frame.
};

/**
 * @brief Struct to hold feature data.
 */
struct FeatureWithIDepth {
  uint32_t id = 0;
  uint32_t frame_id = 0;
  Point2f xy;
  float idepth_mu = 0.0f;
  float idepth_var = 0.0f;
  bool valid = false;
  uint32_t num_updates = 0;
  uint32_t num_dropouts = 0;
  stereo::inverse_depth_filter::Status search_status =
      stereo::inverse_depth_filter::Status::SUCCESS;
};

/**
 * @brief Struct to hold data needed to syncronize graph with raw features.
 */
struct SyncData {
  std::vector<FeatureWithIDepth> feats;
};

/**
 * @brief Class that implements the Fast Lightweight Mesh Estimation (FLaME)
 * algorithm.
 */
class Flame final {
 public:
  /**
   * @brief Constructor.
   *
   * @param[in] width Image width in pixels.
   * @param[in] height Image height in pixels.
   * @param[in] K Camera intrinsics matrix.
   * @param[Kinv] Kinv Inverse of the camera intrinsics matrix.
   * @param[in] params Parameter struct.
   */
  Flame(int width, int height,
        const Matrix3f& K, const Matrix3f& Kinv,
        const Params& params = Params());
  ~Flame();

  Flame(const Flame& rhs) = delete;
  Flame& operator=(const Flame& rhs) = delete;

  Flame(Flame&& rhs) = default;
  Flame& operator=(Flame&& rhs) = default;

  /**
   * @brief Estimate inverse depthmesh for a new image.
   *
   * @param[in] time Timestamp.
   * @param[in] img_id Image ID.
   * @param[in] T_new Pose of new image.
   * @param[in] img_new New image.
   * @param[in] is_poseframe Whether this image is a poseframe.
   * @param[in] idepths_true True inverse depths (for debugging).
   * @return True if update successful. Outputs are only valid if returns True.
   */
  bool update(double time, uint32_t img_id, const Sophus::SE3f& T_new,
              const Image1b& img_new, bool is_poseframe,
              const Image1f& idepths_true = Image1f());

  /**
   * @brief Update the poseframe poses.
   *
   * @param[in] pf_ids PoseFrame IDs.
   * @param[in] pf_poses New poses for the poseframes.
   */
  void updatePoseFramePoses(const std::vector<uint32_t>& pf_ids,
                           const std::vector<Sophus::SE3f>& pf_poses) {
    std::lock_guard<std::mutex> pfs_lock(pfs_mtx_);
    for (int ii = 0; ii < pf_ids.size(); ++ii) {
      if (pfs_.count(pf_ids[ii]) > 0) {
        pfs_[pf_ids[ii]]->pose =  pf_poses[ii];
      }
    }
    return;
  }

  /**
   * @brief Prune poseframes.
   *
   * Features defined relative to a removed poseframe will be transferred to the
   * latest poseframe.
   *
   * @param[in] pfs_to_keep IDs of the poseframes to keep.
   */
  void prunePoseFrames(const std::vector<uint32_t>& pfs_to_keep);

  /**
   * @brief Clear everything.
   */
  void clear() {
    std::lock_guard<std::recursive_mutex> lock(update_mtx_);

    inited_ = false;

    feats_.clear();

    new_feats_mtx_.lock();
    new_feats_.clear();
    new_feats_mtx_.unlock();

    graph_mtx_.lock();
    graph_.clear();
    graph_mtx_.unlock();

    vtx_.clear();
    vtx_idepths_.clear();
    vtx_w1_.clear();
    vtx_w2_.clear();

    // vtx_idx_to_handle_.clear();
    // vtx_handle_to_idx_.clear();
    return;
  }

  /**
   * @brief Return the current dense inverse depthmap.
   */
  const Image1f& getInverseDepthMap() {
    std::lock_guard<std::recursive_mutex> lock(update_mtx_);
    return idepthmap_;
  }

  /**
   * @brief Return the filtered inverse depthmap.
   *
   * Applies any triangle filtering (e.g. oblique triangles, etc.).
   */
  void getFilteredInverseDepthMap(Image1f* idepthmap) {
    std::lock_guard<std::recursive_mutex> lock(update_mtx_);
    if (idepthmap->empty()) {
      idepthmap->create(height_, width_);
    }
    *idepthmap = std::numeric_limits<float>::quiet_NaN();

    std::vector<bool> vtx_validity(vtx_.size(), true);
    utils::interpolateMesh(triangles_curr_, vtx_, vtx_idepths_,
                           vtx_validity, tri_validity_, idepthmap);
    return;
  }

  /**
   * @brief Get the current inverse depthmesh.
   */
  void getInverseDepthMesh(std::vector<Point2f>* vertices,
                           std::vector<float>* idepths,
                           std::vector<Vector3f>* normals,
                           std::vector<Triangle>* triangles,
                           std::vector<bool>* tri_validity,
                           std::vector<Edge>* edges) {
    std::lock_guard<std::recursive_mutex> lock(update_mtx_);

    *vertices = vtx_;
    *idepths = vtx_idepths_;
    *normals = vtx_normals_;
    *triangles = triangles_curr_;
    *edges = edges_curr_;
    *tri_validity = tri_validity_;

    return;
  }

  /**
   * @brief Get the raw, unregularized inverse depth estimates relative to
   * current frame.
   */
  void getRawIDepths(std::vector<Point2f>* vertices,
                     std::vector<float>* idepths_mu,
                     std::vector<float>* idepths_var) {
    std::lock_guard<std::recursive_mutex> lock(update_mtx_);

    vertices->clear();
    idepths_mu->clear();
    idepths_var->clear();

    for (int ii = 0; ii < feats_in_curr_.size(); ++ii) {
      if (feats_in_curr_[ii].valid) {
        vertices->push_back(feats_in_curr_[ii].xy);
        idepths_mu->push_back(feats_in_curr_[ii].idepth_mu);
        idepths_var->push_back(feats_in_curr_[ii].idepth_var);
      }
    }

    return;
  }

  /**
   * @brief Return stats object.
   *
   * TODO(wng): This should be const.
   */
  utils::StatsTracker& stats() {
    std::lock_guard<std::recursive_mutex> lock(update_mtx_);
    return stats_;
  }

  // Access debug images.
  const Image3b& getDebugImageDetections() {
    std::lock_guard<std::mutex> lock(detection_queue_mtx_);
    return debug_img_detections_;
  }
  const Image3b& getDebugImageWireframe() {
    std::lock_guard<std::recursive_mutex> lock(update_mtx_);
    return debug_img_wireframe_;
  }
  const Image3b& getDebugImageFeatures() {
    std::lock_guard<std::recursive_mutex> lock(update_mtx_);
    return debug_img_features_;
  }
  const Image3b& getDebugImageMatches() {
    std::lock_guard<std::recursive_mutex> lock(update_mtx_);
    return debug_img_matches_;
  }
  const Image3b& getDebugImageNormals() {
    std::lock_guard<std::recursive_mutex> lock(update_mtx_);
    return debug_img_normals_;
  }
  const Image3b& getDebugImageInverseDepthMap() {
    std::lock_guard<std::recursive_mutex> lock(update_mtx_);
    return debug_img_idepthmap_;
  }

 private:
  // Main detection loop.
  void detectionLoop();

  // Synchronizes graph with updated features.
  void graphSyncLoop();

  // Get the best poseframe for frame fnew.
  static utils::Frame::ConstPtr getPoseFrame(const Params& params,
                                             const Matrix3f& K,
                                             const Matrix3f& Kinv,
                                             const FrameIDToFrame& pfs,
                                             const utils::Frame& fnew,
                                             int max_pfs,
                                             utils::StatsTracker* stats);

  static void detectFeatures(const Params& params,
                             const Matrix3f& K,
                             const Matrix3f& Kinv,
                             const utils::Frame& fref,
                             const utils::Frame& fprev,
                             const utils::Frame& fcmp,
                             const Image1f& idepthmap,
                             const std::vector<Point2f>& curr_feats,
                             Image1f* error,
                             std::vector<Point2f>* features,
                             utils::StatsTracker* stats,
                             Image3b* debug_img);

  // Update the depth estimates.
  static bool updateFeatureIDepths(const Params& params,
                                   const Matrix3f& K,
                                   const Matrix3f& Kinv,
                                   const FrameIDToFrame& pfs,
                                   const utils::Frame& fnew,
                                   const utils::Frame& curr_pf,
                                   std::vector<FeatureWithIDepth>* feats,
                                   utils::StatsTracker* stats,
                                   Image3b* debug_img);

  // Track a single feature in the new image.
  static bool trackFeature(const Params& params,
                           const Matrix3f& K,
                           const Matrix3f& Kinv,
                           const FrameIDToFrame& pfs,
                           const stereo::EpipolarGeometry<float>& epigeo,
                           const utils::Frame& fnew,
                           const utils::Frame& curr_pf,
                           FeatureWithIDepth* feat,
                           Point2f* flow, float* residual,
                           Image3b* debug_img);

  // Project features into current frame.
  static void projectFeatures(const Params& params,
                              const Matrix3f& K,
                              const Matrix3f& Kinv,
                              const FrameIDToFrame& pfs,
                              const utils::Frame& fcur,
                              std::vector<FeatureWithIDepth>* feats,
                              std::vector<FeatureWithIDepth>* feats_in_curr,
                              utils::StatsTracker* stats);

  // Project graph into new frame.
  static void projectGraph(const Params& params,
                           const stereo::EpipolarGeometry<float>& epigeo,
                           const utils::Frame& fnew,
                           Graph* graph,
                           float graph_scale,
                           FeatureToVtx* feat_to_vtx,
                           VtxToFeature* vtx_to_feat,
                           utils::StatsTracker* stats);

  // Update and synchronize graph and features. After this function, the
  // triangulation in triangulator should be valid.
  static bool syncGraph(const Params& params,
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
                        utils::StatsTracker* stats);

  // Triangulate raw vertices.
  static void triangulate(const Params& params,
                          const std::vector<Point2f>& vertices,
                          utils::Delaunay* triangulator,
                          utils::StatsTracker* stats);

  // Filter oblique triangles.
  static void obliqueTriangleFilter(const Params& params,
                                    const Matrix3f& Kinv,
                                    const std::vector<Point2f>& vertices,
                                    const std::vector<float>& idepths,
                                    const std::vector<Triangle>& triangles,
                                    std::vector<bool>* validity,
                                    utils::StatsTracker* stats);

  // Filter triangles with long edges.
  static void edgeLengthFilter(const Params& params,
                               int width, int height,
                               const std::vector<Point2f>& vertices,
                               const std::vector<Triangle>& triangles,
                               std::vector<bool>* validity,
                               utils::StatsTracker* stats);

  // Filter triangles based on idepth.
  static void idepthTriangleFilter(const Params& params,
                                   const std::vector<float>& idepths,
                                   const std::vector<Triangle>& triangles,
                                   std::vector<bool>* validity,
                                   utils::StatsTracker* stats);

  // Draw feature detections.
  static void drawDetections(const Params& params,
                             const Image1b& img,
                             const Image1f& score,
                             float max_score,
                             const std::vector<cv::KeyPoint>& kps,
                             utils::StatsTracker* stats,
                             Image3b* debug_img);

  // Draw colormapped wireframe.
  static void drawWireframe(const Params& params,
                            const Image1b& img_curr,
                            const std::vector<Triangle>& triangles,
                            const std::vector<Edge>& edges,
                            const std::vector<bool>& tri_validity,
                            const std::vector<Point2f>& vertices,
                            const std::vector<float>& idepths,
                            utils::StatsTracker* stats,
                            Image3b* debug_img);

  // Draw features.
  static void drawFeatures(const Params& params,
                           const Image1b& img_curr,
                           const std::vector<FeatureWithIDepth>& feats_in_curr,
                           utils::StatsTracker* stats,
                           Image3b* debug_img);

  // Get vertex normals using w parameters.
  static void getVertexNormals(const Params& params,
                               const Matrix3f& K,
                               const std::vector<Point2f>& vtx,
                               const std::vector<float>& idepths,
                               const std::vector<float>& w1,
                               const std::vector<float>& w2,
                               std::vector<Vector3f>* normals,
                               utils::StatsTracker* stats);

  // Get vertex normals using triangles.
  static void getVertexNormals(const Params& params,
                               const Matrix3f& Kinv,
                               const std::vector<Point2f>& vtx,
                               const std::vector<float>& idepths,
                               const std::vector<Triangle>& triangles,
                               std::vector<Vector3f>* normals,
                               utils::StatsTracker* stats);

  // Get inward pointing normal.
  static Vector3f planeParamToNormal(const Matrix3f& K,
                                     const Point2f& u_ref,
                                     float idepth, float w1, float w2);

  static void drawNormals(const Params& params,
                          const Matrix3f& K,
                          const Image1b& img_curr, const Image1f& idepthmap,
                          const Image1f& w1_map, const Image1f& w2_map,
                          Image3b* debug_img);

  // Draw dense inverse depth map.
  static void drawInverseDepthMap(const Params& params,
                                  const Image1b& img_curr,
                                  const Image1f& idepthmap,
                                  utils::StatsTracker* stats,
                                  Image3b* debug_img);

  utils::StatsTracker stats_;
  Params params_;

  bool inited_;
  uint32_t num_data_updates_;
  uint32_t num_regularizer_updates_;

  int width_;
  int height_;

  Matrix3f K_;
  Matrix3f Kinv_;

  stereo::EpipolarGeometry<float> epigeo_;

  uint32_t num_imgs_;
  utils::Frame::Ptr fnew_; // New frame.
  utils::Frame::Ptr fprev_; // Previous frame.

  // Lock when performing an update or accessing internals.
  std::recursive_mutex update_mtx_;

  // PoseFrames.
  FrameIDToFrame pfs_; // Main container for pfs.
  std::mutex pfs_mtx_; // Locks pfs_ container.
  utils::Frame::Ptr curr_pf_; // Pointer to the current poseframe.

  // Feature detection stuff.
  std::thread detection_thread_;
  std::deque<DetectionData> detection_queue_; // Queue that holds DetectionData. Need deque instead of queue so we can iterate.
  std::mutex detection_queue_mtx_; // Locks detection_queue_.
  std::condition_variable detection_queue_cv_; // Signals when there are new items in detection queue.

  std::vector<FeatureWithIDepth> new_feats_; // Newly detected features.
  std::mutex new_feats_mtx_; // Locks new_feats_ container.

  Image1f photo_error_; // Photometric error.

  // Raw depth estimates.
  uint32_t feat_count_; // Running count of features. Used to create feature ID.
  std::vector<FeatureWithIDepth> feats_; // Raw features.
  std::vector<FeatureWithIDepth> feats_in_curr_; // Feature projected into current frame.

  // The main optimization graph.
  Graph graph_;
  float graph_scale_; // Scale of input data. IDepths are scaled to have mean 1.
  std::thread graph_thread_; // Thread that optimizes graph.
  std::mutex graph_mtx_; // Protects the graph data.

  // Maps between feature IDs and vertex handles for the current graph.
  FeatureToVtx feat_to_vtx_;
  VtxToFeature vtx_to_feat_;

  utils::Delaunay triangulator_; // Performs Delaunay triangulation.
  std::mutex triangulator_mtx_;  // Locks triangulator.
  std::vector<bool> tri_validity_; // False if triangle is oblique or too big.
  std::vector<Triangle> triangles_curr_; // Local copy of current triangulation.
  std::vector<Edge> edges_curr_; // Local copy of current edges.

  std::vector<Point2f> vtx_; // Positions of regularized depthmesh.
  std::vector<float> vtx_idepths_; // Regularized idepths.
  std::vector<float> vtx_w1_; // Plane parameters.
  std::vector<float> vtx_w2_;
  std::vector<Vector3f> vtx_normals_;

  Image1f idepthmap_; // Dense idepthmap.
  Image1f w1_map_; // Dense plane parameters.
  Image1f w2_map_; // Dense plane parameters.

  // // Debug images.
  Image3b debug_img_detections_;
  Image3b debug_img_wireframe_;
  Image3b debug_img_features_;
  Image3b debug_img_matches_;
  Image3b debug_img_normals_;
  Image3b debug_img_idepthmap_;
};

}  // namespace flame
