/**
 * @file debugUtils.h
 * @brief Debugging and utility macros for conditional testing, timing, and assertions.
 *
 * This header provides macros and helper functions to facilitate debugging,
 * performance timing, and controlled assertions in CUDA/C++ applications.
 * When `DEBUGGING` is enabled, additional diagnostic checks and utilities
 * are included; otherwise, macros are defined as no-ops for optimized builds.
 */

#pragma once

 //------------------------------------------------------------------------------
 // Control Macros
 //------------------------------------------------------------------------------

#define SHOULD_NOT_GET_HERE false  ///< Fallback assertion for unexpected code paths.

//------------------------------------------------------------------------------
// Conditional Compilation for Debugging
//------------------------------------------------------------------------------
#ifdef DEBUGGING

// Optional testing and timing flags (disabled unless explicitly defined).
//#define TESTING true
#ifndef TESTING
#define TESTING false
#endif

//#define TIMING true
#ifndef TIMING
#define TIMING false
#endif

//------------------------------------------------------------------------------
// Includes
//------------------------------------------------------------------------------
#include <iostream>
#include <cstdlib>
#include <string>
#include <algorithm>

#if TESTING
#include <filesystem>
#endif

#ifndef NDEBUG
#define NOMINMAX
#include <Windows.h>
#undef LoadString
#endif

//------------------------------------------------------------------------------
// Function Declarations
//------------------------------------------------------------------------------

/**
 * @brief Convert a string to uppercase.
 * @param str Input string.
 * @return Uppercase version of the input string.
 */
std::string toUpperCase(const std::string& str);

/**
 * @brief Custom assertion handler for debugging.
 * @param error Error message or failed condition.
 * @param file  Source file name where the assertion failed.
 * @param line  Line number of the failed assertion.
 */
void customAssert(const std::string& error, const std::string& file, int line);

/**
 * @brief Triggers a breakpoint or halts program execution during debugging.
 */
void breakExecution();

/**
 * @brief Create a folder if it does not already exist.
 * @param folderPath Path to the folder.
 * @return True if folder created or already exists, false otherwise.
 */
bool createFolder(const std::string& folderPath);

/**
 * @brief Set the simulation grid or array size depending on configuration flags.
 * @param size Reference to the size variable to set.
 * @param flag Configuration flag.
 * @param Nz   Number of cells along the z-axis.
 * @param N    Total number of cells.
 * @return True on success, false if invalid parameters are detected.
 */
bool setSize(int& size, const int flag, const int Nz, const int N);

//------------------------------------------------------------------------------
// Assertion Macro
//------------------------------------------------------------------------------

/**
 * @brief Assertion macro for debugging; calls customAssert and breakExecution on failure.
 */
#define ASSERT(condition)                                                      \
do {                                                                          \
    (void)((!!(condition)) ||                                                  \
           (customAssert(#condition, __FILE__, (unsigned)(__LINE__)),          \
            breakExecution(), 0));                                             \
} while (0)

 //------------------------------------------------------------------------------
 // Size Setting Macro
 //------------------------------------------------------------------------------

 /**
  * @brief Wrapper around setSize() with assertion on failure.
  */
#define SET_SIZE(size, flag, Nz, N)                                            \
do {                                                                          \
    if (!setSize(size, flag, Nz, N)) { ASSERT(SHOULD_NOT_GET_HERE); }          \
} while (0)

  //------------------------------------------------------------------------------
  // Timing Macro (conditional on TIMING flag)
  //------------------------------------------------------------------------------

#if TIMING
/**
 * @brief Simple execution timer macro for profiling code blocks.
 */
#define timer(x)                                                               \
do {                                                                          \
    auto start = std::chrono::high_resolution_clock::now();                    \
    x;                                                                         \
    auto stop = std::chrono::high_resolution_clock::now();                     \
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); \
    printf("%s time (ms): %f\n", #x, elapsed.count());                         \
} while (0)
#else
#define timer(x) x;
#endif // TIMING

//------------------------------------------------------------------------------
// Non-debugging Mode (Production Build)
//------------------------------------------------------------------------------
#else // !DEBUGGING

#define SAVE_FIELD(name, fieldPtr, geometry)
#define ASSERT(condition) (void)(!!(condition))

/**
 * @brief Simplified SET_SIZE macro for non-debugging mode (no assertions).
 */
#define SET_SIZE(size, flag, Nz, N)                                            \
    (void)setSize(size, flag, Nz, N)

#define timer(x) x;

#endif // DEBUGGING
