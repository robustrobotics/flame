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
 * @file types.h
 * @author W. Nicholas Greene
 * @date 2017-06-16 16:49:05 (Fri)
 */

#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <sophus/se3.hpp>

namespace flame {

/*==================== Image types ====================*/
// Point2.
template <typename T>
using Point2 = cv::Point_<T>;

using Point2i = Point2<int>;
using Point2f = Point2<float>;
using Point2d = Point2<double>;

// Vec3b/Vec4b.
using Vec3b = cv::Vec3b;
using Vec4b = cv::Vec4b;

// Image.
template <typename T>
using Image = cv::Mat_<T>;

using Image1b = Image<uint8_t>;
using Image3b = Image<Vec3b>;
using Image4b = Image<Vec4b>;
using Image1f = Image<float>;

// Pyramids.
using ImagePyramid = std::vector<cv::Mat>;

template <typename T>
using ImagePyramid_ = std::vector<Image<T> >;

using ImagePyramidb = ImagePyramid_<uint8_t>;
using ImagePyramid3b = ImagePyramid_<Vec3b>;
using ImagePyramidf = ImagePyramid_<float>;
using ImagePyramidd = ImagePyramid_<double>;

/*==================== Matrix/Vector types ====================*/
template <typename T, int rows, int cols, int options = Eigen::RowMajor>
using Matrix = Eigen::Matrix<T, rows, cols, options>;

template <typename T>
using Matrix2 = Matrix<T, 2, 2>;

template <typename T>
using Matrix3 = Matrix<T, 3, 3>;

template <typename T>
using Matrix4 = Matrix<T, 4, 4>;

template <typename T, int rows>
using Vector = Matrix<T, rows, 1, Eigen::ColMajor>;

template <typename T>
// cppcheck-suppress constStatement
using Vector2 = Vector<T, 2>;

template <typename T>
// cppcheck-suppress constStatement
using Vector3 = Vector<T, 3>;

template <typename T>
// cppcheck-suppress constStatement
using Vector4 = Vector<T, 4>;

using Matrix2f = Matrix2<float>;
using Matrix3f = Matrix3<float>;
using Matrix4f = Matrix4<float>;
using Matrix2d = Matrix2<double>;
using Matrix3d = Matrix3<double>;
using Matrix4d = Matrix4<double>;

using Vector2f = Vector2<float>;
using Vector3f = Vector3<float>;
using Vector4f = Vector4<float>;
using Vector2d = Vector2<double>;
using Vector3d = Vector3<double>;
using Vector4d = Vector4<double>;

/*==================== Quaternion types. ====================*/
template <typename T>
using Quaternion = Eigen::Quaternion<T>;

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

/*==================== AngleAxis types ====================*/
template <typename T>
using AngleAxis = Eigen::AngleAxis<T>;

using AngleAxisf = AngleAxis<float>;
using AngleAxisd = AngleAxis<double>;

/*==================== Lie group types. ====================*/
template <typename T>
using SE3 = Sophus::SE3Group<T>;

using SE3f = SE3<float>;
using SE3d = SE3<double>;

}  // namespace flame
