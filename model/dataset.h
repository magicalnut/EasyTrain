#ifndef DATASET_H
#define DATASET_H
#pragma once

#include <torch/torch.h>
#include <string>

struct Dataset {
    torch::Tensor x;      // [N, C, H, W] float，/255 归一化到 [0,1]
    torch::Tensor y;      // [N] long，类别索引
    int64_t classes = 0;
};

// 从文件夹(每类一子目录) 或 CSV(path,label) 加载分类数据集。
// C/H/W 取「输入」页的值；C 决定读成几通道（1=灰度，3=彩色）。
Dataset load_dataset(const std::wstring& src, int C, int H, int W, bool is_csv);

// 冒烟用：随机数据 N 样本 / classes 类
Dataset random_dataset(int N, int C, int H, int W, int classes);


#endif // DATASET_H
