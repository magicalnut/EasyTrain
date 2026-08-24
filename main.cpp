#include "mainwindow.h"

#include <QApplication>
#include <QThread>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Monitor monitor;
    MainWindow w(&monitor);
    w.show();
    return QApplication::exec();
}
