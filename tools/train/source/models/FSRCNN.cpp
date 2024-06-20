#include <algorithm>
#include <iostream>
#include "FSRCNN.hpp"

namespace MNN {
namespace Train {
namespace Model {
using namespace MNN::Express;

// Draft the FSRCNN structure

//// Feature Extraction
class _Feature_Extraction : public Module {
public:
    _Feature_Extraction(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, int padding = 1, bool depthwise = false);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv2d_fe;
    std::shared_ptr<Module> prelu; // TODO:
};

_Feature_Extraction::_Feature_Extraction(std::vector<int> inputOutputChannels, int kernelSize, int stride, int padding, bool depthwise) {
    int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];
    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.channel = {inputChannels, outputChannels};
//    convOption.padMode = Express::SAME;
    convOption.stride = {stride, stride};
    convOption.pads = {padding, padding};
    convOption.depthwise = depthwise;
    conv2d_fe.reset(NN::Conv(convOption, true, std::shared_ptr<Initializer>(Initializer::MSRA())));

    registerModel({conv2d_fe});
}
std::vector<Express::VARP> _Feature_Extraction::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;

    VARP x = inputs[0];
    x = conv2d_fe->forward(x);
    return {x};
}

std::shared_ptr<Module> FeatureExtraction(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, int padding = 1, bool depthwise = false) {
    return std::shared_ptr<Module>(new _Feature_Extraction(inputOutputChannels, kernelSize, stride, padding, depthwise));
}

//// Shrinking
class _Shrinking: public Module {
public:
    _Shrinking(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, int padding = 1, bool depthwise = false);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv2d_shrink;
};

_Shrinking::_Shrinking(std::vector<int> inputOutputChannels, int kernelSize, int stride, int padding,
                       bool depthwise) {
    int inChannels = inputOutputChannels[0], outChannels = inputOutputChannels[1];
    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.channel = {inChannels, outChannels};
    convOption.stride = {stride, stride};
    convOption.pads = {padding, padding};
    convOption.depthwise = depthwise;

    conv2d_shrink.reset(NN::Conv(convOption, true, std::shared_ptr<Initializer>(Initializer::MSRA())));

    registerModel({conv2d_shrink});
}

std::vector<Express::VARP> _Shrinking::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x = conv2d_shrink->forward(x);
    return {x};
}

std::shared_ptr<Module> Shrinking(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, int padding = 1, bool depthwise = false) {
    return std::shared_ptr<Module>(new _Shrinking(inputOutputChannels, kernelSize, stride, padding, depthwise));
}

///// Mapping


//// Expanding

//// Deconvolution


/////////////////////////////////////
//// FSRCNN Model
FSRCNN::FSRCNN(int num_channels, int d, int s, int m, int upscale_factor) {

    // 1. Feature Extraction
    // TODO: Double check the encoder_layer
    feature_extraction = FeatureExtraction({1, 56}, 5, 1, 2);

    // 2. Shrinking
    drop_out.reset(NN::Dropout(0.1));

    // 3. Mapping
    // TODO: Double check the encoder_layer
    for (int i=0; i<12; i++) {
        encoder_layers.emplace_back(EncoderBlock());
    }

    // 4. Expanding

    // 5. Deconvolution
    // TODO: Double check the parameters of Linear block
    linear.reset(NN::Linear(768, 1000, false));

    registerModel({conv_proj, drop_out, linear});
    registerModel(encoder_layers);
}

std::vector<Express::VARP> FSRCNN::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x = feature_extraction->forward(x);
    x = shrinking->forward(x);
    for (int i = 0; i < mapping.size(); i++) {
        x = mapping[i]->forward(x);
    }
    x = expanding->forward(x);
    x = deconvolution->forward(x);
    return {x};
}

} // namespace Model
} // namespace Train
} // namespace MNN
