# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_catch_and_throw_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED catch_and_throw_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(catch_and_throw_FOUND FALSE)
  elseif(NOT catch_and_throw_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(catch_and_throw_FOUND FALSE)
  endif()
  return()
endif()
set(_catch_and_throw_CONFIG_INCLUDED TRUE)

# output package information
if(NOT catch_and_throw_FIND_QUIETLY)
  message(STATUS "Found catch_and_throw: 0.0.0 (${catch_and_throw_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'catch_and_throw' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${catch_and_throw_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(catch_and_throw_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${catch_and_throw_DIR}/${_extra}")
endforeach()
