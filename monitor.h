#ifndef MONITOR_H
#define MONITOR_H
#pragma once

#include "plot_widget.h"

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTimer>
#include <QThread>

class Monitor : public QWidget
{
    Q_OBJECT
public:
    Monitor(QWidget *parent = nullptr);
    ~Monitor();
private:
    QHBoxLayout *hbox1;
    QVBoxLayout *vbox1;
    QVBoxLayout *vbox2;

    QLabel *label_cpu;
    QLabel *label_memory_cpu;
    QLabel *label_gpu;
    QLabel *label_memory_gpu;
    Plot_Widget *cpu_plot;
    Plot_Widget *memory_cpu_plot;
    Plot_Widget *gpu_plot;
    Plot_Widget *memory_gpu_plot;

    QTimer *timer;
    QThread *thread1;
    QThread *thread2;
    QThread *thread3;
};

#endif // MONITOR_H
