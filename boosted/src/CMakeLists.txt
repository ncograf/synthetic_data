# Include modules
add_subdirectory(lag_prod_mean)


# create main module
add_library(boosted_stats MODULE boosted_stats.cpp)
target_link_libraries(boosted_stats ${Boost_LIBRARIES} Python3::NumPy lag_prod_mean)
set_target_properties(boosted_stats PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(boosted_stats PRIVATE ${Python3_INCLUDE_DIRS})
