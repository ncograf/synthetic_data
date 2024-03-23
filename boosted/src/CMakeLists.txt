# Include modules
# add_subdirectory(lag_prod_mean)


# create main module
add_library(boosted_stats MODULE boosted_stats.cpp)
target_link_libraries(boosted_stats ${Boost_LIBRARIES} Python3::NumPy)
set_target_properties(boosted_stats PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(boosted_stats PRIVATE ${Python3_INCLUDE_DIRS})

# copy output
add_custom_command(TARGET boosted_stats POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:boosted_stats> "${CMAKE_SOURCE_DIR}/boost_modules")
