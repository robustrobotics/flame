/**
 * This file is part of FLaME.
 * Copyright (C) 2017 W. Nichoilas Greene (wng@csail.mit.edu)
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
 * @file epipolar_geometry.h
 * @author W. Nicholas Greene
 * @date 2017-08-18 19:17:13 (Fri)
 */

#pragma once

#include <iostream>

#include <opencv2/core/core.hpp>

#include "flame/types.h"
#include "flame/utils/assert.h"
#include "flame/utils/image_utils.h"

namespace flame {

namespace stereo {

/**
 * \brief Class that represents an epipolar geometry setup.
 *
 * Useful for epipolar geometry queries (epipolar lines, projections, depth,
 * etc.)
 */
template <typename Scalar>
class EpipolarGeometry final {
  // Convenience aliases.
  using Point2s = cv::Point_<Scalar>; // TODO(wng): Use core types.
  using Vector3s = Vector3<Scalar>;
  using Matrix3s = Matrix3<Scalar>;
  using Quaternions = Quaternion<Scalar>;

 public:
  /**
   * \brief Constructor.
   *
   * The comparison camera is the camera that pixels are projected onto to form
   * epipolar lines and compute disparity. The reference camera is the camera
   * that depths are define relative to.
   *
   * @param[in] K Camera intrinsic matrix.
   * @param[in] Kinv Inverse camera intrinsic matrix.
   */
  EpipolarGeometry(const Matrix3s& K, const Matrix3s& Kinv) :
      K_(K),
      Kinv_(Kinv),
      q_ref_to_cmp_(),
      t_ref_to_cmp_(),
      t_cmp_to_ref_(),
      KRKinv3_(),
      Kt_(),
      epipole_() {}
  EpipolarGeometry() = default;
  ~EpipolarGeometry() = default;

  EpipolarGeometry(const EpipolarGeometry& rhs) = default;
  EpipolarGeometry& operator=(const EpipolarGeometry& rhs) = default;

  EpipolarGeometry(EpipolarGeometry&& rhs) = default;
  EpipolarGeometry& operator=(EpipolarGeometry&& rhs) = default;

  /**
   * \brief Load an epipolar geometry setup.
   *
   * @param[in] q_ref_to_cmp Rotation from reference to comparison frame.
   * @param[in] t_ref_to_cmp Translation from reference to compariosn frame.
   */
  void loadGeometry(const Quaternions& q_ref_to_cmp,
                    const Vector3s& t_ref_to_cmp) {
    q_ref_to_cmp_ = q_ref_to_cmp;
    t_ref_to_cmp_ = t_ref_to_cmp;
    t_cmp_to_ref_ = -(q_ref_to_cmp_.inverse() * t_ref_to_cmp_);
    KRKinv_ = K_ * q_ref_to_cmp_.toRotationMatrix() * Kinv_;
    KRKinv3_ = KRKinv_.row(2);
    Kt_ = K_ * t_ref_to_cmp_;

    if (t_ref_to_cmp_(2) > 0) {
      // Precompute epipole.
      epipole_.x = (K_(0, 0) * t_ref_to_cmp_(0) + K_(0, 2) * t_ref_to_cmp_(2)) /
          t_ref_to_cmp_(2);
      epipole_.y = (K_(1, 1) * t_ref_to_cmp_(1) + K_(1, 2) * t_ref_to_cmp_(2)) /
          t_ref_to_cmp_(2);
    }
    return;
  }

  /**
   * \brief Helper function for perspective projection.
   *
   * @param[in] K Camera intrinsic matrix.
   * @param[in] q Orientation of camera in world.
   * @param[in] t Translation of camera in world.
   * @param[in] p Point to project in world.
   * @return Projected pixel.
   */
  static Point2s project(const Matrix3s& K, const Quaternions& q,
                         const Vector3s& t, const Vector3s& p) {
    Vector3s p_cam(q.inverse() * (p - t));
    return Point2s((K(0, 0) * p_cam(0) + K(0, 2) * p_cam(2)) / p_cam(2),
                   (K(1, 1) * p_cam(1) + K(1, 2) * p_cam(2)) / p_cam(2));
  }

  /**
   * \brief Project pixel u_ref into comparison frame assuming inverse depth.
   *
   * @param[in] u_ref Reference pixel.
   * @param[in] idepth Inverse depth.
   */
  Point2s project(const Point2s& u_ref, Scalar idepth) const {
    FLAME_ASSERT(idepth >= 0.0f);
    if (idepth == 0.0f) {
      Point2s u_max;
      maxDepthProjection(u_ref, &u_max);
      return u_max;
    }

    Scalar depth = 1.0f / idepth;
    Vector3s u_ref_hom(u_ref.x * depth, u_ref.y * depth, depth);
    Vector3s u_cmp_hom(KRKinv_ * u_ref_hom + Kt_);

    FLAME_ASSERT(utils::fast_abs(u_cmp_hom(2)) > 0.0f);

    Scalar inv_u_cmp_hom2 = 1.0f / u_cmp_hom(2);
    return Point2s(u_cmp_hom(0) * inv_u_cmp_hom2, u_cmp_hom(1) * inv_u_cmp_hom2);
  }

  /**
   * \brief Project pixel u_ref into comparison frame assuming inverse depth.
   *
   * @param[in] u_ref Reference pixel.
   * @param[in] idepth Inverse depth.
   * @param[out] u_cmp Projected pixel.
   * @param[out] new_idepth New inverse depth.
   */
  void project(const Point2s& u_ref, Scalar idepth,
               Point2s* u_cmp, Scalar* new_idepth) const {
    FLAME_ASSERT(idepth >= 0.0f);
    if (idepth == 0.0f) {
      Point2s u_max;
      maxDepthProjection(u_ref, &u_max);
      *u_cmp = u_max;
      *new_idepth = 0.0f;
      return;
    }

    Scalar depth = 1.0f / idepth;
    Vector3s p_ref(Kinv_(0, 0) * u_ref.x + Kinv_(0, 2),
                   Kinv_(1, 1) * u_ref.y + Kinv_(1, 2),
                   1.0f);
    p_ref *= depth;

    Vector3s p_cmp(q_ref_to_cmp_ * p_ref + t_ref_to_cmp_);
    Vector3s u_cmp3(K_(0, 0) * p_cmp(0) + K_(0, 2) * p_cmp(2),
                    K_(1, 1) * p_cmp(1) + K_(1, 2) * p_cmp(2),
                    p_cmp(2));
    FLAME_ASSERT(fabs(u_cmp3(2)) > 0.0f);

    *new_idepth = 1.0f / p_cmp(2);
    u_cmp->x = u_cmp3(0) * (*new_idepth);
    u_cmp->y = u_cmp3(1) * (*new_idepth);
    return;
  }

  /**
   * \brief Compute projection of pixel with infinite depth.
   *
   * Compute the projection of pixel u_ref into the comparison image assuming
   * infinite depth.
   *
   * @param u_ref[in] Pixel in the reference image to project.
   * @param u_inf[out] The projection corresponding to infinite depth.
   */
  void maxDepthProjection(const Point2s& u_ref, Point2s* u_inf) const {
    Vector3s u_ref_hom(u_ref.x, u_ref.y, 1.0f);
    Vector3s u_cmp_hom(KRKinv_ * u_ref_hom);

    FLAME_ASSERT(fabs(u_cmp_hom(2)) > 0.0f);

    Scalar inv_u_cmp_hom2 = 1.0f / u_cmp_hom(2);
    u_inf->x = u_cmp_hom(0) * inv_u_cmp_hom2;
    u_inf->y = u_cmp_hom(1) * inv_u_cmp_hom2;
    return;
  }

  /**
   * \brief Compute the epiline endpoint.
   *
   * There are several ways to compute the epiline endpoint. The most natural
   * way is if the reference camera is in front of the comparison camera in the
   * comparison camera frame (that is t_ref_to_cmp_z > 0). Then the epiline
   * endpoint is just the epipole (i.e. the projection of the reference camera
   * into the comparison camera). This corresponds to the point in the world
   * having 0 depth.
   *
   * Things get more complicated however if t_ref_to_cmp_z <= 0 (i.e. the
   * reference camera lies at the same z or behind the comparison camera).

   * If t_ref_to_cmp_z = 0, then the epipole lies at infinity and all epilines
   * are parallel (typically the case for a traditional stereo setup). In this
   * case, the vector (fx * t_ref_to_cmp_x, fy * t_ref_to_cmp_y) is parallel to
   * the epiline. Given the infinity point, the minimum depth point is simply
   * the infinite point plus a large value times this vector.
   *
   * If t_ref_to_cmp_z < 0, then the minimum possible depth of the point must be
   * 0 in order to be projected into both camera. In this case, we simply
   * compute the depth in the reference frame such * that the point has depth 1
   * in the comparison frame and then project this * point into the comparison
   * camera
   *
   * It is also possible to compute the epiline using the fundamental matrix
   * F. If epiline is parameterized by the implicit equation l^T u_cmp = 0,
   * where u_cmp are homogenous pixels in the comparison image, then l = F
   * u_ref. This formulation, however, does not give the *direction* of the
   * epiline from far depth to near depth, which is what we would like.
   *
   * @param u_ref[in] Pixel in the reference image to project.
   * @param u_min[out] The projection corresponding to minimum depth.
   */
  void minDepthProjection(const Point2s& u_ref, Point2s* u_min) const {
    if (t_ref_to_cmp_(2) > 0) {
      *u_min = epipole_;
    } else if (t_ref_to_cmp_(2) == 0) {
      // Compute epiline direction.
      Point2s epi(K_(0, 0) * t_ref_to_cmp_(0), K_(1, 1) * t_ref_to_cmp_(1));
      Point2s u_inf;
      maxDepthProjection(u_ref, &u_inf);
      *u_min = u_inf + 1e6 * epi;
    } else {
      // Compute depth in the ref frame such that point has depth 1 in comparison
      // frame.
      Vector3s qp_ref(Kinv_(0, 0) * u_ref.x + Kinv_(0, 2),
                      Kinv_(1, 1) * u_ref.y + Kinv_(1, 2),
                      1.0f);
      qp_ref = q_ref_to_cmp_ * qp_ref;
      Scalar min_depth = (1.0f - t_ref_to_cmp_(2)) / qp_ref(2);

      Vector3s p_cmp(min_depth * qp_ref + t_ref_to_cmp_);
      FLAME_ASSERT(p_cmp(2) > 0.0f);

      u_min->x = (K_(0, 0) * p_cmp(0) + K_(0, 2) * p_cmp(2)) / p_cmp(2);
      u_min->y = (K_(1, 1) * p_cmp(1) + K_(1, 2) * p_cmp(2)) / p_cmp(2);
    }

    return;
  }

  /**
   * \brief Compute epipolar line corresponding to pixel.
   *
   * Computes the epipolar line of pixel u_ref in the cmp image. The line points
   * from infinite depth to minimum depth and is computed by finding the pixels
   * corresponding to infinite depth and minimum depth in the comparison image.
   *
   * It is also possible to compute the epiline using the fundamental matrix
   * F. If epiline is parameterized by the implicit equation l^T u_cmp = 0,
   * where u_cmp are homogenous pixels in the comparison image, then l = F
   * u_ref. This formulation, however, does not give the *direction* of the
   * epiline from far depth to near depth, which is what we would like.
   *
   * @param u_ref[in] Reference pixel.
   * @param u_inf[out] Start of epipolar line (point of infinite depth).
   & @param epi[out] Epipolar unit vector.
  */
  void epiline(const Point2s& u_ref, Point2s* u_inf, Point2s* epi) const {
    Point2s u_zero;
    minDepthProjection(u_ref, &u_zero);
    maxDepthProjection(u_ref, u_inf);
    *epi = u_zero - *u_inf;
    Scalar norm2 = epi->x*epi->x + epi->y*epi->y;

    if (norm2 > 1e-10) {
      Scalar inv_norm = 1.0f / sqrt(norm2);
      epi->x *= inv_norm;
      epi->y *= inv_norm;
    } else {
      // If u_zero == u_inf, then epi mag is 0.
      epi->x = 0.0f;
      epi->y = 0.0f;
    }

    return;
  }

  /**
   * @brief Return the epiline that corresponds to u_ref in the reference
   * image. This is the projection of the epipolar plane onto the reference
   * image at u_ref. It points from near depth to far depth (opposite of what's
   * returned from epiline).
   *
   * @param[in] u_ref Reference pixel.
   * @param[out] epi Epipolar line.
   */
  void referenceEpiline(const Point2s& u_ref, Point2s* epi) const {
    // Get epiline in reference image for the template.
    // calculate the plane spanned by the two camera centers and the point (x,y,1)
    // intersect it with the keyframe's image plane (at depth = 1)
    // This is the epipolar line in the keyframe.
    Point2s epi_ref;
    epi_ref.x = -K_(0, 0) * t_cmp_to_ref_(0) +
        t_cmp_to_ref_(2)*(u_ref.x - K_(0, 2));
    epi_ref.y = -K_(1, 1) * t_cmp_to_ref_(1) +
        t_cmp_to_ref_(2)*(u_ref.y - K_(1, 2));

    Scalar epi_ref_norm2 = epi_ref.x * epi_ref.x + epi_ref.y * epi_ref.y;
    FLAME_ASSERT(epi_ref_norm2 > 0);
    Scalar inv_epi_ref_norm = 1.0f / sqrt(epi_ref_norm2);
    epi_ref.x *= inv_epi_ref_norm;
    epi_ref.y *= inv_epi_ref_norm;

    *epi = epi_ref;

    return;
  }

  /**
   * \brief Compute disparity from pixel correspondence.
   *
   * @param u_ref[in] Reference pixel.
   * @param u_cmp[in] Comparison pixel.
   * @param u_inf[out] Endpoint of epipolar line (point of infinite depth).
   * @param epi[out] Epipolar line direction.
   * @param disparity[out] Disparity.
   */
  Scalar disparity(const Point2s& u_ref, const Point2s& u_cmp,
                   Point2s* u_inf, Point2s* epi) const {
    epiline(u_ref, u_inf, epi);
    return epi->x*(u_cmp.x - u_inf->x) + epi->y*(u_cmp.y - u_inf->y);
  }
  Scalar disparity(const Point2s& u_ref, const Point2s& u_cmp) const {
    Point2s u_inf, epi;
    return disparity(u_ref, u_cmp, &u_inf, &epi);
  }
  Scalar disparity(const Point2s& u_ref, const Point2s& u_cmp,
                   const Point2s& u_inf, const Point2s& epi) const {
    return epi.x*(u_cmp.x - u_inf.x) + epi.y*(u_cmp.y - u_inf.y);
  }

  /**
   * \brief Compute depth from disparity.
   *
   * @param u_ref[in] Reference pixel
   * @param u_inf[in] Epipolar line start point (infinite depth).
   * @param epi[in] Epipolar line direction.
   * @param disparity[in] Disparity.
   * @return depth
   */
  Scalar disparityToDepth(const Point2s& u_ref, const Point2s& u_inf,
                          const Point2s& epi, const Scalar disparity) const {
    FLAME_ASSERT(disparity >= 0.0f);
    Scalar w = KRKinv3_(0) * u_ref.x + KRKinv3_(1) * u_ref.y + KRKinv3_(2);
    Point2s A(w * disparity * epi);
    Point2s b(Kt_(0) - Kt_(2)*(u_inf.x + disparity * epi.x),
              Kt_(1) - Kt_(2)*(u_inf.y + disparity * epi.y));

    Scalar ATA = A.x*A.x + A.y*A.y;
    Scalar ATb = A.x*b.x + A.y*b.y;

    FLAME_ASSERT(ATA > 0.0f);

    return ATb/ATA;
  }

  /**
   * \brief Compute inverse depth from disparity.
   *
   * @param KR_ref_to_cmpKinv3 Third row of K*R_ref_to_cmp*Kinv.
   * @param Kt_ref_to_cmp[in] K * translation from ref to cmp.
   * @param u_ref[in] Reference pixel
   * @param u_inf[in] Epipolar line start point (infinite depth).
   * @param epi[in] Epipolar line direction.
   * @param disparity[in] Disparity.
   * @return inverse depth
   */
  Scalar disparityToInverseDepth(const Point2s& u_ref, const Point2s& u_inf,
                                 const Point2s& epi,
                                 const Scalar disparity) const {
    FLAME_ASSERT(disparity >= 0.0f);
    Scalar w = KRKinv3_(0) * u_ref.x + KRKinv3_(1) * u_ref.y + KRKinv3_(2);
    Point2s A(Kt_(0) - Kt_(2)*(u_inf.x + disparity * epi.x),
              Kt_(1) - Kt_(2)*(u_inf.y + disparity * epi.y));
    Point2s b(w * disparity * epi);

    Scalar ATA = A.x*A.x + A.y*A.y;
    Scalar ATb = A.x*b.x + A.y*b.y;

    FLAME_ASSERT(ATA > 0.0f);

    return ATb/ATA;
  }

  // Accessors.
  const Matrix3s& K() const { return K_; }
  const Matrix3s& Kinv() const { return Kinv_; }
  const Quaternions& getRefToCmpQuat() const { return q_ref_to_cmp_; }
  const Vector3s& getRefToCmpTrans() const { return t_ref_to_cmp_; }

 private:
  // Camera parameters.
  Matrix3s K_;
  Matrix3s Kinv_;

  // Geometry.
  Quaternions q_ref_to_cmp_;
  Vector3s t_ref_to_cmp_;
  Vector3s t_cmp_to_ref_;
  Matrix3s KRKinv_;
  Vector3s KRKinv3_;
  Vector3s Kt_;
  Point2s epipole_; // Projection of cmp camera in ref camera.
};

}  // namespace stereo

}  // namespace flame
