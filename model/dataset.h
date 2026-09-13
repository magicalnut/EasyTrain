#ifndef DATASET_H
#define DATASET_H
#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

// 在线增广配置
struct AugConfig {
    bool   hflip       = true;
    bool   random_crop = true;
    double rotate_deg  = 15.0;
    double brightness  = 0.2;
};

struct Dataset {
    std::vector<std::wstring> paths_;
    std::vector<int64_t>      labels_;
    int64_t classes = 0;
    size_t size() const { return paths_.size(); }
};

Dataset load_dataset(const std::wstring& src, bool is_csv);

// 冒烟用：往 out_dir 生成 N 张随机 .bmp（classes 个子目录），再按文件夹加载返回。
Dataset random_dataset(const std::wstring& out_dir, int N, int C, int H, int W, int classes);

// 懒加载一张：读图 → 增广 → resize → 归一化 → [C,H,W]（train=false 只 resize+归一化）
torch::Tensor sample_at(const Dataset& d, size_t i, int C, int H, int W,const AugConfig& aug, bool train);

#endif // DATASET_H
