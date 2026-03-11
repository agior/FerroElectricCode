/**
 * @file Debugging.cpp
 * @brief Implements debugging and utility functions for error handling, validation, and folder management.
 *
 * This file provides platform-dependent assertion handling, execution interruption,
 * folder creation utilities, and basic string transformation helpers used throughout
 * the simulation framework.
 */

#include "Debugging.hpp"
#include <algorithm>     // For std::transform
#include <cctype>        // For std::toupper
#include <filesystem>    // For folder management
#include <iostream>      // For std::cerr

 // ============================================================================
 // Utility: Convert String to Uppercase
 // ============================================================================
 /**
  * @brief Converts a given string to uppercase.
  *
  * @param str Input string.
  * @return Uppercase version of the input string.
  *
  * @details
  * Used to format error messages in assertion and logging routines.
  */
std::string toUpperCase(const std::string& str)
{
    std::string temp(str.size(), '\0');
    std::transform(str.begin(), str.end(), temp.begin(),
        [](unsigned char c) { return std::toupper(c); });
    return temp;
}

// ============================================================================
// Custom Assertion Handler
// ============================================================================
/**
 * @brief Prints a detailed error message when an assertion fails.
 *
 * @param error Description of the failed condition.
 * @param file  Name of the source file where the assertion occurred.
 * @param line  Line number where the assertion failed.
 *
 * @details
 * This function formats and outputs a standardized assertion message
 * including the file name and line number. It does **not** terminate execution.
 */
void customAssert(const std::string& error, const std::string& file, int line)
{
    std::string errorString =
        "ASSERTION FAILED :: " + toUpperCase(error) +
        "\nIN FILE :: \"" + file + "\" AT LINE :: " + std::to_string(line) + "\n";

    std::cerr << errorString;
}

// ============================================================================
// Break Execution for Debugging
// ============================================================================
/**
 * @brief Halts execution for debugging purposes.
 *
 * @details
 * - In debug builds, triggers a debugger break if one is attached (Windows).
 * - In release builds, exits the program immediately.
 *
 * @note On non-Windows platforms, `IsDebuggerPresent()` may not be available.
 *       Consider using platform-agnostic methods for portability.
 */
void breakExecution()
{
#ifndef NDEBUG
    if (IsDebuggerPresent())
        __debugbreak();
    else
        exit(-1);
#else
    exit(-1);
#endif
}

// ============================================================================
// Folder Creation Utility
// ============================================================================
/**
 * @brief Creates a folder if it does not already exist (only in testing mode).
 *
 * @param folderPath Path to the folder to be created.
 * @return `true` if the folder exists or was created successfully, otherwise `false`.
 *
 * @note This function only performs actual directory creation when the `TESTING`
 *       macro is defined; otherwise, it returns `true` unconditionally.
 */
bool createFolder(const std::string& folderPath)
{
#if TESTING
    if (std::filesystem::exists(std::filesystem::path(folderPath))) {
        return true;
    }
    return std::filesystem::create_directories(std::filesystem::path(folderPath));
#else
    return true;
#endif
}

// ============================================================================
// Parameter Size Setter
// ============================================================================
/**
 * @brief Determines the appropriate vector size based on a configuration flag.
 *
 * @param size Reference to the output size variable.
 * @param flag Configuration flag determining the size mode.
 * @param Nz   Number of cells along the z-dimension.
 * @param N    Total number of grid elements.
 * @return `true` if the size was set successfully, `false` otherwise.
 *
 * @details
 * The function sets `size` as follows:
 * - **flag = 1:** `size = 1` (uniform value)
 * - **flag = 2:** `size = Nz` (layer-dependent)
 * - **flag = 3:** `size = N`  (full grid)
 * - Otherwise: returns `SHOULD_NOT_GET_HERE`
 */
bool setSize(int& size, const int flag, const int Nz, const int N)
{
    switch (flag)
    {
    case 1: size = 1;  break;
    case 2: size = Nz; break;
    case 3: size = N;  break;
    default: return SHOULD_NOT_GET_HERE;
    }
    return true;
}
