#include <algorithm>
#include <iostream>
#include "ViT.hpp"

namespace MNN {
namespace Train {
namespace Model {
using namespace MNN::Express;

// Draft the ViT structure

//// Conv2d
class _Conv2d : public Module {
public:
    _Conv2d(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, bool depthwise = false);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv2d;
};

_Conv2d::_Conv2d(std::vector<int> inputOutputChannels, int kernelSize, int stride, bool depthwise) {
    int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];
    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.channel = {inputChannels, outputChannels};
    convOption.padMode = Express::SAME;
    convOption.stride = {stride, stride};
    convOption.depthwise = depthwise;
    conv2d.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    registerModel({conv2d});
}
std::vector<Express::VARP> _Conv2d::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;

    VARP x = inputs[0];
    x = conv2d->forward(x);
    return {x};
}

std::shared_ptr<Module> Conv2d(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, bool depthwise = false) {
    return std::shared_ptr<Module>(new _Conv2d(inputOutputChannels, kernelSize, stride, depthwise));
}


//// Attention
class _Attention : public Module {
public:
    _Attention(int in_dim, int Attention_dim);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
    std::shared_ptr<Module> matmul0;
};

_Attention::_Attention(int hidden_dim, int num_heads) {
    matmul0.reset(NN::Linear(hidden_dim, hidden_dim*3));
    registerModel({matmul0});
}

// TODO: Current working
std::vector<Express::VARP> _Attention::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = matmul0->forward(x);
    // yy = _Slice(yy, _Const(sliceStartData, {4}, NCHW), _Const(sliceEndData, {4}, NCHW));
    auto q = _Slice(x,                       0, _Const(768, {4}, NCHW));
    auto k = _Slice(x, _Const(768, {4}, NCHW) , _Const(768, {4}, NCHW));
    auto v = _Slice(x, _Const(1536, {4}, NCHW), _Const(768, {4}, NCHW));
    q = _Transpose(_Reshape(q, {197, 12, 64}), {1,0,2});
    k = _Transpose(_Reshape(q, {197, 12, 64}), {1,2,0});
    v = _Transpose(_Reshape(q, {197, 12, 64}), {1,0,2});
    q = _Divide(q, _Const(8, {4}, NCHW));
    auto qk = _Softmax()
    return {x};
}

std::shared_ptr<Module> Attention(int hidden_dim, int num_heads) {
    return std::shared_ptr<Module>(new _Attention(hidden_dim, num_heads));
}

//// MLPBlock = Linear + GELU + Dropout + Linear + Dropout
class _MLP : public Module {
public:
    _MLP(int in_dim, int mlp_dim);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> linear0;
    std::shared_ptr<Module> dropout0;
    std::shared_ptr<Module> linear1;
    std::shared_ptr<Module> dropout1;
};

_MLP::_MLP(int in_dim, int mlp_dim) {
    linear0.reset(NN::Linear(in_dim, mlp_dim));
    dropout0.reset(NN::Dropout(0.1));
    linear1.reset(NN::Linear(mlp_dim, in_dim));
    dropout1.reset(NN::Dropout(0.1));

    registerModel({linear0, dropout0, linear1, dropout1});
}

std::vector<Express::VARP> _MLP::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = linear0->forward(x);
    x = _Gelu(x);
    x = dropout0->forward(x);
    x = linear1->forward(x);
    x = dropout1->forward(x);
    return {x};
}

std::shared_ptr<Module> MLP(int in_dim, int mlp_dim) {
    return std::shared_ptr<Module>(new _MLP(in_dim, mlp_dim));
}


//// EncoderBlock = LayerNorm(Skip) + MultiheadAttention + Dropout + LayerNorm(Skip) + MLPBlock
class _EncoderBlock:public Module{
public:
    _EncoderBlock();
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> dropout_encoder_block;
    std::shared_ptr<Module> attention_block;
    std::shared_ptr<Module> mlp_block;
};

_EncoderBlock::_EncoderBlock() {

    // LayerNorm(Skip)
    // MultiheadAttention
    // Dropout
    // LayerNorm(Skip)
    // MLPBlock
    attention_block = Attention(768, 12);
    dropout_encoder_block.reset(NN::Dropout(0.1));
    mlp_block = MLP(768, 3072);

    // registerModel
    registerModel({attention_block, dropout_encoder_block, mlp_block});
}

std::vector<Express::VARP> _EncoderBlock::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x = attention_block->forward(x);
    x = mlp_block->forward(x);
    return {x};
}

std::shared_ptr<Module> EncoderBlock(){
    return std::shared_ptr<Module>(new _EncoderBlock());
}

ViT::ViT(int numClasses, int patch_size, int num_layers, int num_heads, int hidden_dim, int mlp_dim) {

    // 1. conv_proj
    conv_proj = Conv2d({3, 768}, 16, 16);

    // 2. dropout
    drop_out.reset(NN::Dropout(0.0));

    // 3. encoder_layers
    // TODO: Double check the encoder_layer
    for (int i=0; i<12; i++) {
        encoder_layers.emplace_back(EncoderBlock());
    }

    // 4. last_layer_norm. NOTE: Currently we skip LayerNorm temporarily

    // 5. Final Linear Block
    linear.reset(NN::Linear(768, 1000, true));

    registerModel({conv_proj, drop_out, linear});
    registerModel(encoder_layers);
}

std::vector<Express::VARP> ViT::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x = conv_proj->forward(x);
    x = drop_out->forward(x);
    
    // TODO: To chain blocks
    for (int i=0; i<12; i++) {
        x = encoder_layers[i]->forward(x);
    }

//    3. last_layer_norm. NOTE: Currently we skip LayerNorm temporarily
//    x = last_layer_norm->forward(x);
    x = linear->forward(x);
    return {x};
}

} // namespace Model
} // namespace Train
} // namespace MNN
