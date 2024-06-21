//
//  FSRCNNUtils.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef FSRCNNUtils_hpp
#define FSRCNNUtils_hpp

#include <MNN/expr/Module.hpp>
#include <string>

class FSRCNNUtils {
public:
    static void train(std::shared_ptr<MNN::Express::Module> model, const int numClasses, const int addToLabel,
                      std::string trainImagesFolder,
                      std::string testImagesFolder, const int quantBits = 8);
};

#endif
