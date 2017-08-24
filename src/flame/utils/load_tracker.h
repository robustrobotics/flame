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
 * @file load_tracker.h
 * @author W. Nicholas Greene
 * @date 2017-02-04 17:41:45 (Sat)
 */

#pragma once

#include <unistd.h>

#include <asm/param.h> // For the Jiffy constant HZ.

#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <thread>

#include "flame/utils/assert.h"

namespace flame {

namespace utils {

typedef std::chrono::high_resolution_clock clock;
typedef std::chrono::duration<double, std::milli> msec;

/**
 * @brief Struct to hold load information.
 */
struct Load {
  float cpu = 0; // CPU load between 0 and number of processors.
  uint64_t mem = 0; // Memory load in megabytes/mebibytes (1024^2 bytes).
  uint64_t swap = 0; // Swap load in megabytes/mebibytes (1024^2 bytes).
};

/**
 * @brief Struct to hold measured process CPU times in clock ticks.
 *
 * Obtained from columns 13-16 in /proc/pid/stat.
 */
struct ProcessCPUMeasurement {
  uint64_t utime = 0; // Time spent in user mode.
  uint64_t stime = 0; // Time spent in kernel mode.
  uint64_t cutime = 0; // Time children spent in user mode.
  uint64_t cstime = 0; // Time children spent in kernel mode.
};

/**
 * @brief Struct to hold measured system CPU times in jiffies (0.01 seconds).
 *
 * Obtained from columns 1-7 in /proc/stat.
 */
struct SystemCPUMeasurement {
  uint64_t user = 0; // Time processes spent in user mode.
  uint64_t nice = 0; // Time niced processes spent in user mode.
  uint64_t system = 0; // Time processes spent in kernel mode.
  uint64_t idle = 0; // Time spent idle.
  uint64_t iowait = 0; // Time spent waiting for IO to complete.
  uint64_t irq = 0; // Time spent servicing interrupts.
  uint64_t softirq = 0; // Time spent servicing softirqs.
};

/**
 * @brief Read /proc/<pid>/stat to generate CPU timing measurement.
 */
inline void getProcessCPUMeasurement(int pid, ProcessCPUMeasurement* meas) {
  std::ifstream file("/proc/" + std::to_string(pid) + "/stat",
                     std::ifstream::in);
  std::string line;
  std::getline(file, line);
  std::istringstream lstream(line);

  // Skip the first 13 columns.
  for (int ii = 0; ii < 13; ++ii) {
    std::string word;
    lstream >> word;
  }

  // Read columns 14-17.
  lstream >> meas->utime; // Time spent in user mode.
  lstream >> meas->stime; // Time spent in kernel mode.
  lstream >> meas->cutime; // Time spent in child processes in user mode.
  lstream >> meas->cstime; // Time spent in child processes in kernel mode.

  return;
}

/**
 * @brief Read /proc/stat to generate system CPU timing measurement.
 */
inline void getSystemCPUMeasurement(SystemCPUMeasurement* meas) {
  std::ifstream file("/proc/stat", std::ifstream::in);
  std::string line;
  std::getline(file, line);
  std::istringstream lstream(line);

  // Skip first column.
  std::string blah;
  lstream >> blah;
  FLAME_ASSERT(blah == "cpu");

  // Read next 7 columns.
  lstream >> meas->user; // Time processes spent in user mode.
  lstream >> meas->nice; // Time niced processes spent in user mode.
  lstream >> meas->system; // Time processes spent in kernel mode.
  lstream >> meas->idle; // Time spent idle.
  lstream >> meas->iowait; // Time spent waiting for IO to complete.
  lstream >> meas->irq; // Time spent servicing interrupts.
  lstream >> meas->softirq; // Time spent servicing softirqs.

  return;
}

/**
 * @brief Get the CPU load on the system given two time measurements.
 */
inline void getSystemCPULoad(const SystemCPUMeasurement& prev,
                             const SystemCPUMeasurement& curr,
                             float dt, int num_procs,
                             Load* sys_load) {
  // Compute the updated system load. We take the difference of the sum of the
  // two measurements (except idle time) and then divide by the Jiffy constant
  // (HZ) and the sampling rate, which gives us the load expressed as a number
  // between 0 and num_procs.
  float cpu_sys_total = curr.user + curr.nice + curr.system + curr.iowait +
      curr.irq + curr.softirq;
  float cpu_sys_total_prev = prev.user + prev.nice + prev.system + prev.iowait +
      prev.irq + prev.softirq;

  FLAME_ASSERT(cpu_sys_total >= cpu_sys_total_prev);

  sys_load->cpu = (cpu_sys_total - cpu_sys_total_prev) / dt / HZ;

  if (sys_load->cpu < 0) {
    sys_load->cpu = 0;
  }
  if (sys_load->cpu > num_procs) {
    sys_load->cpu = num_procs;
  }

  return;
}

/**
 * @brief Get the CPU load from a single process given two time measurements.
 */
inline void getProcessCPULoad(const ProcessCPUMeasurement& prev,
                              const ProcessCPUMeasurement& curr,
                              float dt, int num_procs,
                              int ticks_per_sec, Load* pid_load) {
  // Compute updated process load. We take the difference of the sum of the
  // two measurements and then divide by the clock speed (ticks_per_sec_) and
  // the sampling rate (dt_), which gives us the load expressed as a number
  // between 0 and num_procs_.
  float cpu_pid_total = curr.utime + curr.stime + curr.cutime + curr.cstime;
  float cpu_pid_total_prev = prev.utime + prev.stime + prev.cutime + prev.cstime;

  FLAME_ASSERT(cpu_pid_total >= cpu_pid_total_prev);

  pid_load->cpu = (cpu_pid_total - cpu_pid_total_prev) / dt / ticks_per_sec;

  // Check bounds. Numbers may be slightly off because of timing accuracy.
  if (pid_load->cpu < 0) {
    pid_load->cpu = 0;
  }
  if (pid_load->cpu > num_procs) {
    pid_load->cpu = num_procs;
  }

  return;
}

/**
 * @brief Read /proc/meminfo to compute system memory loads.
 */
inline void getSystemMemoryLoad(Load* max_load, Load* sys_load) {
  std::ifstream file("/proc/meminfo", std::ifstream::in);
  std::string line;

  // Loop through file and grab relevant fields.
  uint64_t mem_total_kb = 0;
  uint64_t mem_free_kb = 0;
  uint64_t buffers_kb = 0;
  uint64_t cached_kb = 0;
  uint64_t swap_free_kb = 0;
  uint64_t swap_total_kb = 0;
  int count = 0;
  while (std::getline(file, line)) {
    std::stringstream lstream(line);
    std::string field;
    lstream >> field;

    if (field == "MemTotal:") {
      lstream >> mem_total_kb;
      max_load->mem = mem_total_kb >> 10; // Convert from kB to MB.
      count++;
    } else if (field == "MemFree:") {
      lstream >> mem_free_kb;
      count++;
    } else if (field == "Buffers:") {
      lstream >> buffers_kb;
      count++;
    } else if (field == "Cached:") {
      lstream >> cached_kb;
      count++;
    } else if (field == "SwapTotal:") {
      lstream >> swap_total_kb;
      max_load->swap = swap_total_kb >> 10; // Convert from kB to MB.
      count++;
    } else if (field == "SwapFree:") {
      lstream >> swap_free_kb;
      count++;
    }
  }

  FLAME_ASSERT(count == 6);

  // Substract to get used swap.
  sys_load->swap = (swap_total_kb - swap_free_kb) >> 10; // Convert to MB.

  // Physical memory used is total - free - buffers - cached. This number
  // corresponds to the green bar in htop:
  // https://stackoverflow.com/questions/41224738/how-to-calculate-memory-usage-from-proc-meminfo-like-htop
  sys_load->mem = (mem_total_kb - mem_free_kb - buffers_kb -
                   cached_kb) >> 10; // Convert to MB.

  return;
}

/**
 * @brief Read /proc/<pid>/status to generate process memory load.
 */
inline void getProcessMemoryLoad(int pid, Load* load) {
  // Grab a process CPU load measurement.
  std::ifstream file("/proc/" + std::to_string(pid) + "/status",
                     std::ifstream::in);
  std::string line;

  // Loop through file and get relevant fields.
  int count = 0;
  uint64_t mem_pid_kb = 0;
  uint64_t swap_pid_kb = 0;
  while (std::getline(file, line)) {
    std::stringstream lstream(line);
    std::string field;
    lstream >> field;
    if (field == "VmRSS:") {
      lstream >> mem_pid_kb;
      load->mem = mem_pid_kb >> 10; // Convert to MB.
      count++;
    } else if (field == "VmSwap:") {
      lstream >> swap_pid_kb;
      load->swap = swap_pid_kb >> 10; // Convert to MB.
      count++;
    }
  }

  FLAME_ASSERT(count == 2);

  return;
}

/**
 * @brief A class to track a process's CPU and memory load.
 *
 * We read the following files to compute load:
 *   - /proc/stat for system CPU load
 *   - /proc/meminfo for system memory load
 *   - /proc/<pid>/stat for CPU load for given pid.
 *   - /proc/<pid>/status for memory load for given pid.
 *
 * CPU load information in the above files is expressed in ticks or jiffies
 * spent executing code in either user or kernel mode. To get actual load we
 * need to take two measurements from the files, subtract them, and then scale
 * the result, which will be a number between 0 and the number of processors
 * (technically number of concurrent threads supported).
 *
 * Memory load is pretty much read directly from the files.
 *
 * Usage:
 * @code{.cpp}
 * #include <flame/flame.h>
 *
 *   LoadTracker load_tracker(pid);
 *
 *   Load max_load, sys_load, pid_load;
 *   load_tracker.get(&max_load, &sys_load, &pid_load); // Grab current load info.
 *
 *   printf("system CPU load = %f / %.0f\n", sys_load.cpu, max_load.cpu);
 *   printf("system memory load = %lu / %lu MB\n", sys_load.mem, max_load.mem);
 *   printf("system swap load = %lu / %lu MB\n", sys_load.swap, max_load.swap);
 *   printf("process CPU load = %f / %.0f\n", pid_load.cpu, max_load.cpu);
 *   printf("process memory load = %lu / %lu MB\n", pid_load.mem, max_load.mem);
 *   printf("process swap load = %lu / %lu MB\n", pid_load.swap, max_load.swap);
 * @endcode
 *
 * Based on:
 * https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
 * https://stackoverflow.com/questions/16726779/how-do-i-get-the-total-cpu-usage-of-an-application-from-proc-pid-stat
 */
class LoadTracker final {
 public:
  /**
   * #brief Constructor.
   *
   * Tracks load for given PID.
   *
   * @param[in] pid Process ID.
   */
  explicit LoadTracker(uint32_t pid = getpid()) :
      last_update_(),
      num_procs_(std::thread::hardware_concurrency()),
      ticks_per_sec_(sysconf(_SC_CLK_TCK)),
      pid_(pid),
      cpu_pid_meas_(),
      cpu_sys_meas_(),
      max_load_(),
      sys_load_(),
      pid_load_() {
    // Initialize CPU load measurements.
    getSystemCPUMeasurement(&cpu_sys_meas_);
    getProcessCPUMeasurement(pid_, &cpu_pid_meas_);
    max_load_.cpu = num_procs_;
    return;
  }

  ~LoadTracker() = default;

  LoadTracker(const LoadTracker& rhs) = delete;
  LoadTracker& operator=(const LoadTracker& rhs) = delete;

  LoadTracker(LoadTracker&& rhs) = default;
  LoadTracker& operator=(LoadTracker&& rhs) = default;

  /**
   * @brief Get the current load information.
   *
   * @param[out] max_load Maximum possible load for this system.
   * @param[out] sys_load Current system load.
   * @param[out] pid_load Current process load.
   */
  void get(Load* max_load, Load* sys_load, Load* pid_load);

 private:
  clock::time_point last_update_; // Time of last update.

  int num_procs_; // Number of processors.
  uint32_t ticks_per_sec_; // Clock speed.

  int pid_; // Process ID.
  ProcessCPUMeasurement cpu_pid_meas_; // Last process CPU load measurement.
  SystemCPUMeasurement cpu_sys_meas_; // Last system CPU load measurement.

  Load max_load_; // Maximum possible load for system.
  Load sys_load_; // Current system load.
  Load pid_load_; // Current load for PID.
};

inline void LoadTracker::get(Load* max_load, Load* sys_load, Load* pid_load) {
  // Get time since last update.
  msec dt_ms = clock::now() - last_update_;
  float dt = dt_ms.count() / 1000; // in seconds.

  SystemCPUMeasurement cpu_sys_meas_prev(cpu_sys_meas_); // Save old measurement.
  getSystemCPUMeasurement(&cpu_sys_meas_); // Get new measurement.
  getSystemCPULoad(cpu_sys_meas_prev, cpu_sys_meas_, dt, num_procs_, &sys_load_);

  ProcessCPUMeasurement cpu_pid_meas_prev(cpu_pid_meas_); // Save old measurement.
  getProcessCPUMeasurement(pid_, &cpu_pid_meas_); // Get new measurement.
  getProcessCPULoad(cpu_pid_meas_prev, cpu_pid_meas_, dt, num_procs_,
                    ticks_per_sec_, &pid_load_);

  getSystemMemoryLoad(&max_load_, &sys_load_);
  getProcessMemoryLoad(pid_, &pid_load_);

  last_update_ = clock::now();

  *max_load = max_load_;
  *sys_load = sys_load_;
  *pid_load = pid_load_;

  return;
}

}  // namespace utils

}  // namespace flame
