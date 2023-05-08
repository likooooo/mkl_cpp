mkdir build
cd build
if [ "debug" == "$1" ];then
    BuildType=Debug
else
    BuildType=Release
fi
cmake ..  -DCMAKE_BUILD_TYPE=$BuildType
cmake --build . --config $BuildType && ctest

