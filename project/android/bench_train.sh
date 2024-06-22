set -e
ABI="arm64-v8a"
CLEAN=""

BUILD_DIR=build_64
ANDROID_DIR=/data/local/tmp/mnn

function build_android_bench() {

    cd $BUILD_DIR
    cmake ../../../ \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_STL=c++_static \
    -DMNN_USE_LOGCAT=OFF \
    -DMNN_OPENCL=ON \
    -DMNN_OPENCL_PROFILE=ON \
    -DMNN_BUILD_TRAIN=ON \
    -DMNN_TRAIN_DEBUG=OFF \
    -DMNN_BUILD_OPENCV=OFF \
    -DMNN_USE_OPENCV=OFF \
    -DMNN_BUILD_BENCHMARK=ON \
    -DMNN_INTERNAL=OFF \
    -DMNN_USE_SSE=OFF \
    -DMNN_SUPPORT_BF16=OFF \
    -DMNN_BUILD_TEST=ON \
    -DANDROID_NATIVE_API_LEVEL=android-21  \
    -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
    -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=.
    make -j
}

function bench_android() {
    build_android_bench
    find . -name "*.so" | while read solib; do
        adb push $solib  $ANDROID_DIR
    done
    adb push runTrainDemo.out $ANDROID_DIR
    adb push timeProfile.out $ANDROID_DIR
    adb push run_test.out $ANDROID_DIR
    adb shell chmod 0777 $ANDROID_DIR/runTrainDemo.out

    # adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/run_test.out op/Conv2DBackPropTest 3"
    # adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/run_test.out  op/Conv2DBackPropFilter 3"
#    adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/runTrainDemo.out MnistTrain test"
#    adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/runTrainDemo.out FSRCNNTrain $ANDROID_DIR/dataset/train_images $ANDROID_DIR/dataset/test_images"
    adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/runTrainDemo.out ViTTrain $ANDROID_DIR/dataset/train_images $ANDROID_DIR/dataset/train_images/train_label.txt"
    # adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/runTrainDemo.out MobilenetV2Train $ANDROID_DIR/train_dataset/train_images $ANDROID_DIR/train_dataset/train.txt $ANDROID_DIR/test_dataset/test_images $ANDROID_DIR/test_dataset/test.txt"
#     adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/runTrainDemo.out ImageDatasetDemo $ANDROID_DIR/train_dataset/ $ANDROID_DIR/train_dataset/train.txt"
    # adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/runTrainDemo.out ImageDatasetDemo $ANDROID_DIR/test_dataset/test_images $ANDROID_DIR/test_dataset/test.txt"

    # adb shell "LD_LIBRARY_PATH=$ANDROID_DIR $ANDROID_DIR/runTrainDemo.out NNGradV4 3"
}

bench_android
