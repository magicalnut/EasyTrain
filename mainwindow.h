#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#pragma once

#include "monitor.h"
#include "model/test.h"

#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>

class MainWindow : public QWidget
{
    Q_OBJECT
public:
    MainWindow(Monitor *monitor,QWidget *parent = nullptr);
    ~MainWindow();
private:
    Monitor *monitor;

    QPushButton *button1;
    QPushButton *button2;

    QVBoxLayout *mainlayout;

    Test *test;

signals:
};

#endif // MAINWINDOW_H
