//
// Created by huzq85 on 6/23/24.
//
#include <algorithm>
#include <iostream>
#include "ViT.hpp"
#include <MNN/expr/ExprCreator.hpp>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <cmath>
#include <iostream>
#include <random>

namespace MNN {
    namespace Train {
        namespace Model {

            using namespace MNN::Express;

            class PatchEmbedding : public Module {
            public:
                PatchEmbedding(int img_size, int patch_size, int in_channels, int embed_dim) {
                    mImgSize = img_size;
                    mPatchSize = patch_size;
                    mInChannels = in_channels;
                    mEmbedDim = embed_dim;
                    int num_patches = (img_size / patch_size) * (img_size / patch_size);
                    cls_token = _Const(0.0f, {1, 1, embed_dim}, NCHW);
                    pos_embedding = _Const(0.0f, {1, num_patches + 1, embed_dim}, NCHW);

//                    patch_proj = _Conv({in_channels, embed_dim, patch_size, patch_size}, {embed_dim}, {patch_size, patch_size});
                    int inputChannels = in_channels, outputChannels = embed_dim;
                    NN::ConvOption convOption;
                    convOption.kernelSize = {patch_size, patch_size};
                    convOption.channel = {inputChannels, outputChannels};
                    convOption.padMode = Express::SAME;
                    convOption.stride = {16, 16};
                    convOption.depthwise = false;
                    patch_proj.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

                    registerModel({patch_proj});
                }

                virtual std::vector<Express::VARP>  onForward(const std::vector<Express::VARP> &inputs) {
                    using namespace std;
                    VARP x = inputs[0];
                    int batch_size = x->getInfo()->dim[0];
                    //cout << "DEBUGGING: batch_size = " << batch_size << endl;
                    // Patchify the input image
                    VARP patches = patch_proj->forward(x);
                    //cout << endl << "DEBUGGING: x dim size1 = " << patches->getInfo()->dim.size() << endl;
                    //cout << "DEBUGGING: x shape = " << patches->getInfo()->dim.at(0) << " " << patches->getInfo()->dim.at(1) << " " << patches->getInfo()->dim.at(2) << " " << patches->getInfo()->dim.at(3) << endl;

                    patches = _Reshape(patches, {batch_size, -1, mEmbedDim});
                    //cout << endl << "DEBUGGING: x dim size1 = " << patches->getInfo()->dim.size() << endl;
                    //cout << "DEBUGGING: x shape = " << patches->getInfo()->dim.at(0) << " " << patches->getInfo()->dim.at(1) << " " << patches->getInfo()->dim.at(2) << " " << endl;

                    // Add the classification token
//                    VARP cls_tokens = _Tile(cls_token, _Const(1.0, {batch_size, 1, 1}));
//                    //cout << endl << "DEBUGGING: cls_tokens dim size1 = " << cls_tokens->getInfo()->dim.size() << endl;
//                    //cout << "DEBUGGING: x shape = " << cls_tokens->getInfo()->dim.at(0) << " " << cls_tokens->getInfo()->dim.at(1) << " " << cls_tokens->getInfo()->dim.at(2) << " " << endl;

                    patches = _Concat({cls_token, patches}, 1);
                    //cout << endl << "DEBUGGING: cls_tokens dim size1 = " << patches->getInfo()->dim.size() << endl;
                    //cout << "DEBUGGING: x shape = " << patches->getInfo()->dim.at(0) << " " << patches->getInfo()->dim.at(1) << " " << patches->getInfo()->dim.at(2) << " " << endl;

                    // Add position embeddings
                    patches = patches + pos_embedding;
                    return {patches};
                }

            private:
                int mImgSize;
                int mPatchSize;
                int mInChannels;
                int mEmbedDim;
                std::shared_ptr<Module> patch_proj;
                VARP cls_token;
                VARP pos_embedding;
            };

            class MultiHeadAttention : public Module {
            public:
                MultiHeadAttention(int embed_dim, int num_heads) {
                    mEmbedDim = embed_dim;
                    mNumHeads = num_heads;
                    mHeadDim = embed_dim / num_heads;

                    queryProj.reset(NN::Linear(embed_dim, embed_dim, true));
                    keyProj.reset(NN::Linear(embed_dim, embed_dim, true));
                    valueProj.reset(NN::Linear(embed_dim, embed_dim, true));
                    outProj.reset(NN::Linear(embed_dim, embed_dim, true));

                    registerModel({queryProj, keyProj, valueProj, outProj});
                }

                virtual std::vector<Express::VARP>  onForward(const std::vector<Express::VARP> &inputs) {
                    VARP x = inputs[0];
                    int batch_size = x->getInfo()->dim[0];
                    int seq_len = x->getInfo()->dim[1];

                    // Linear projections
                    VARP queries = queryProj->forward(_Reshape(x, {batch_size * seq_len, mEmbedDim}));
                    queries = _Reshape(queries, {batch_size, seq_len, mNumHeads, mHeadDim});
                    queries = _Transpose(queries, {0, 2, 1, 3});

                    VARP keys = keyProj->forward(_Reshape(x, {batch_size * seq_len, mEmbedDim}));
                    keys = _Reshape(keys, {batch_size, seq_len, mNumHeads, mHeadDim});
                    keys = _Transpose(keys, {0, 2, 1, 3});

                    VARP values = valueProj->forward(_Reshape(x, {batch_size * seq_len, mEmbedDim}));
                    values = _Reshape(values, {batch_size, seq_len, mNumHeads, mHeadDim});
                    values = _Transpose(values, {0, 2, 1, 3});

                    // Scaled dot-product attention
                    VARP scores = _MatMul(queries, _Transpose(keys, {0, 1, 3, 2}));
                    VARP scale = _Scalar(sqrt(static_cast<float>(mHeadDim)));
                    scores = _Divide(scores, scale);

                    VARP attn_weights = _Softmax(scores, -1);
                    VARP attn_output = _MatMul(attn_weights, values);

                    // Concatenate attention outputs
                    attn_output = _Transpose(attn_output, {0, 2, 1, 3});
                    attn_output = _Reshape(attn_output, {batch_size, seq_len, mEmbedDim});

                    // Final linear projection
                    VARP output = outProj->forward(attn_output);

                    return {output};
                }

            private:
                int mEmbedDim;
                int mNumHeads;
                int mHeadDim;
                std::shared_ptr<Module> queryProj;
                std::shared_ptr<Module> keyProj;
                std::shared_ptr<Module> valueProj;
                std::shared_ptr<Module> outProj;
            };

            class TransformerBlock : public Module {
            public:
                TransformerBlock(int embed_dim, int num_heads, int mlp_dim) {
                    self_attn = std::make_shared<MultiHeadAttention>(embed_dim, num_heads);
                    mlp_linear1.reset(NN::Linear(embed_dim, mlp_dim));
                    mlp_linear2.reset(NN::Linear(mlp_dim, embed_dim));
//                    norm1 = _LayerNorm(embed_dim);
//                    mlp = _Sequential({
//                                              mlp_linear1.reset(NN::Linear(embed_dim, mlp_dim)),
//                                              _Relu(),
//                                              mlp_linear2.reset(NN::Linear(mlp_dim, embed_dim))
//                                      });
//                    norm2 = _LayerNorm(embed_dim);;
                }

                virtual std::vector<Express::VARP>  onForward(const std::vector<Express::VARP> &inputs) {
//                    VARP attn_output = self_attn->forward(x);
//                    x = norm1->forward(x + attn_output);
//                    VARP mlp_output = mlp->forward(x);
//                    x = norm2->forward(x + mlp_output);

                    VARP x = inputs[0];
//                    VARP attn_output = self_attn->forward(x);
//                    x = mlp->forward(x + attn_output);
//                    x = norm1->forward(x + attn_output);
//                    x = norm2->forward(x + mlp_output);
                    x = self_attn->forward(x);
                    x = mlp_linear1->forward(x);
                    x = mlp_linear2->forward(x);
                    return {x};
                }

            private:
                std::shared_ptr<Module> self_attn;
//                std::shared_ptr<Module> norm1;
                std::shared_ptr<Module> mlp;
                std::shared_ptr<Module> mlp_linear1;
                std::shared_ptr<Module> mlp_linear2;
//                std::shared_ptr<Module> norm2;
            };



            ViT::ViT(int img_size, int patch_size, int in_channels, int embed_dim, int num_heads, int mlp_dim, int num_layers, int num_classes) {
                patch_embed = std::make_shared<PatchEmbedding>(img_size, patch_size, in_channels, embed_dim);
                for (int i = 0; i < num_layers; ++i) {
                    transformer_blocks.push_back(std::make_shared<TransformerBlock>(embed_dim, num_heads, mlp_dim));
                }
                cls_head.reset(NN::Linear(embed_dim, num_classes));

                registerModel({patch_embed, cls_head});
                registerModel(transformer_blocks);
            }

            std::vector<Express::VARP> ViT::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace std;
                VARP x = inputs[0];
                //cout << endl << "DEBUGGING: x dim size1 = " << x->getInfo()->dim.size() << endl;
                //cout << "DEBUGGING: x shape = " << x->getInfo()->dim.at(0) << " " << x->getInfo()->dim.at(1) << " " << x->getInfo()->dim.at(2) << " " << x->getInfo()->dim.at(3) << endl;
                x = patch_embed->forward(x);
                //cout << endl << "DEBUGGING: x dim size2 = " << x->getInfo()->dim.size() << endl;


                for (auto& block : transformer_blocks) {
                    x = block->forward(x);
                    //cout << "DEBUGGING: x shape = " << x->getInfo()->dim.at(0) << " " << x->getInfo()->dim.at(1) << " " << x->getInfo()->dim.at(2) << " " << endl;

                }

//                VARP cls_token = _Slice(x, 0, _Const(1, {x->getInfo()->dim[0], 1, x->getInfo()->dim[2]}));
                x = cls_head->forward(x);

                //cout << endl << "DEBUGGING: x dim size1 = " << x->getInfo()->dim.size() << endl;

                //cout << "DEBUGGING: x shape = " << x->getInfo()->dim.at(0) << " " << x->getInfo()->dim.at(1) << " " << x->getInfo()->dim.at(2) << endl;

                return {x};
            }

        } // namespace Model
    } // namespace Train
} // namespace MNN
