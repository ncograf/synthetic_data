#!/bin/bash
echo "---------------------------------------"
echo "Compile C++ code used in the repository"
echo "---------------------------------------"
echo
echo "The script first installs dependencies with Hunter."
echo "This populates the HUNTER_ROOT directory which defaults to '~/.hunter'."
echo "Moreover, it creates the directory 'boost_modules'
in the current directory which contains the shared library compatible with
the python code. A second directory 'build' contains the build artefacts
from the compilation."
echo
echo "The process will be logged to the file log_setup."
echo "Setup CMake and compile Boostlib ..."

mkdir build -p > log_setup 2>&1 && \
cd build >> log_setup  2>&1 && \
poetry run cmake .. >> ../log_setup 2>&1 && \
echo "Compile project code ..." && \
poetry run make >> ../log_setup 2>&1
if [ $? -eq 0 ]
then
    echo "Successful build!"
else
    echo "Build failed, check log_setup for details"
fi
