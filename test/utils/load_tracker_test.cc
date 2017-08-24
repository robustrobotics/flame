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
 * @file load_tracker_test.cc
 * @author W. Nicholas Greene
 * @date 2017-02-04 19:56:23 (Sat)
 */

#include "flame/utils/load_tracker.h"

#include "gtest/gtest.h"

namespace flame {

namespace utils {

/**
 * @brief Query load periodically while sleeping.
 *
 * Not sure how to effectively test. What I did during development is pick a PID
 * and compare output with htop.
 */
TEST(LoadTrackerTest, IdleTest) {
  LoadTracker load_tracker(getpid());

  int dt_ms = 500;
  int num_iters = 5;
  for (int ii = 0; ii < num_iters; ++ii) {
    Load max_load, sys_load, pid_load;

    load_tracker.get(&max_load, &sys_load, &pid_load);

    printf("max_load.cpu = %f\n", max_load.cpu);
    printf("max_load.mem = %lu\n", max_load.mem);
    printf("max_load.swap = %lu\n", max_load.swap);

    printf("sys_load.cpu = %f\n", sys_load.cpu);
    printf("sys_load.mem = %lu\n", sys_load.mem);
    printf("sys_load.swap = %lu\n", sys_load.swap);

    printf("pid_load.cpu = %f\n", pid_load.cpu);
    printf("pid_load.mem = %lu\n", pid_load.mem);
    printf("pid_load.swap = %lu\n", pid_load.swap);

    std::this_thread::sleep_for(std::chrono::milliseconds(dt_ms));
  }

  return;
}

}  // namespace utils

}  // namespace flame
