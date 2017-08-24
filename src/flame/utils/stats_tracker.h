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
 * @file stats_tracker.h
 * @author W. Nicholas Greene
 * @date 2017-08-18 19:43:19 (Fri)
 */

#pragma once

#include <string>
#include <chrono>
#include <unordered_map>
#include <limits>
#include <mutex>

namespace flame {

namespace utils {

typedef std::chrono::high_resolution_clock clock;
typedef std::chrono::duration<double, std::milli> msec;

class StatsTracker final {
 public:
  /**
   * \brief Constructor.
   *
   * @param[in] prefix Prefix to all keys.
   */
  explicit StatsTracker(const std::string& prefix = "") :
      prefix_(prefix),
      start_times_(),
      timings_(),
      stats_() {}

  ~StatsTracker() = default;

  StatsTracker(const StatsTracker& rhs) = delete;
  StatsTracker& operator=(const StatsTracker& rhs) = delete;

  StatsTracker(const StatsTracker&& rhs) = delete;
  StatsTracker& operator=(const StatsTracker&& rhs) = delete;

  /**
   * \brief Return copy of internal timing stats map.
   *
   * Return a copy of the data, otherwise you would need to lock the mutex
   * before getting a reference.
   */
  std::unordered_map<std::string, double> timings() {
    std::lock_guard<std::mutex> lock(mtx_);
    return timings_;
  }

  /**
   * \brief Return a copy of internal stats map.
   *
   * Return a copy of the data, otherwise you would need to lock the mutex
   * before getting a reference.
   */
  std::unordered_map<std::string, double> stats() {
    std::lock_guard<std::mutex> lock(mtx_);
    return stats_;
  }

  /**
   * \brief Return current timing for key.
   */
  double timings(const std::string& key) {
    std::lock_guard<std::mutex> lock(mtx_);
    return timings_[prefix_ + key];
  }

  /**
   * \brief Return current stat for key.
   */
  double stats(const std::string& key) {
    std::lock_guard<std::mutex> lock(mtx_);
    return stats_[prefix_ + key];
  }

  /**
   * \brief Start timer for key.
   */
  void tick(const std::string& key) {
    std::lock_guard<std::mutex> lock(mtx_);
    start_times_[prefix_ + key] = clock::now();
    return;
  }

  /**
   * \brief Update timing for key.
   *
   * Updates timing with latest time since last call to tick(key).
   */
  double tock(const std::string& key) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto search = start_times_.find(prefix_ + key);
    if (search != start_times_.end()) {
      msec ms = clock::now() - search->second;
      timings_[prefix_ + key] = ms.count();
      return ms.count();
    }

    return std::numeric_limits<double>::quiet_NaN();
  }

  /**
   * \brief Set stat to x.
   */
  void set(const std::string& key, double x = 0) {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_[prefix_ + key] = x;
    return;
  }

  /**
   * \brief Add x to stat.
   */
  double add(const std::string& key, double x) {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_[prefix_ + key] += x;
    return stats_[prefix_ + key];
  }

  /**
   * \brief Return a handle to the underlying mutex.
   */
  std::mutex& mutex() { return mtx_; }

  /**
   * \brief Return a copy of the prefix string.
   */
  std::string prefix() { return prefix_; }

  void reset() {
    start_times_.clear();
    timings_.clear();
    stats_.clear();
    return;
  }

 private:
  std::string prefix_;
  std::unordered_map<std::string, std::chrono::time_point<clock> > start_times_;
  std::unordered_map<std::string, double> timings_;
  std::unordered_map<std::string, double> stats_;
  std::mutex mtx_;
};

}  // namespace utils

}  // namespace flame
