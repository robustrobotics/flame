# Build main library.
set(FLAME_SRCS
  flame/flame.h
  flame/flame.cc

  flame/params.h
  flame/types.h

  flame/external/triangle/triangle.h
  flame/external/triangle/triangle.cpp

  flame/optimizers/nltgv2_l1_graph_regularizer.h
  flame/optimizers/nltgv2_l1_graph_regularizer.cc

  flame/stereo/epipolar_geometry.h
  flame/stereo/inverse_depth_filter.h
  flame/stereo/inverse_depth_filter.cc
  flame/stereo/inverse_depth_meas_model.h
  flame/stereo/inverse_depth_meas_model.cc
  flame/stereo/line_stereo.h

  flame/utils/assert.h
  flame/utils/delaunay.h
  flame/utils/delaunay.cc
  flame/utils/frame.h
  flame/utils/frame.cc
  flame/utils/image_utils.h
  flame/utils/image_utils.cc
  flame/utils/keyframe_selector.h
  flame/utils/keyframe_selector.cc
  flame/utils/load_tracker.h
  flame/utils/pyramids.h
  flame/utils/pyramids.cc
  flame/utils/rasterization.h
  flame/utils/rasterization.cc
  flame/utils/stats_tracker.h
  flame/utils/triangulator.h
  flame/utils/visualization.h)

add_library(${PROJECT_NAME} SHARED ${FLAME_SRCS})

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY ${PROJECT_NAME} DESTINATION include
  FILES_MATCHING PATTERN "*.h")
