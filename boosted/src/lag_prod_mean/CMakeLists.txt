add_library(lag_prod_mean STATIC lag_prod_mean.h lag_prod_mean.cpp)

target_link_libraries(lag_prod_mean ${Boost_LIBRARIES} Python3::NumPy)
set_target_properties(lag_prod_mean PROPERTIES LINKER_LANGUAGE CXX FOLDER "src/")
target_include_directories(lag_prod_mean PUBLIC ${Python3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
