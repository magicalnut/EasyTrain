#include "monitor.h"

#include <QProcess>

#if defined (__linux__)
#include <fstream>
#include <sstream>
#elif defined (_WIN32)
#include <windows.h>
#endif

class Worker:public QObject{
    Q_OBJECT
public:
    Worker(QObject *parent=nullptr):QObject(parent){}
    ~Worker(){}
    std::string res1;
    std::string res2;
    std::string res3="%";
    QString to_QString(){
        return (res1+res2+res3).c_str();
    }
signals:
    void done();
};

class Monitor_cpu:public Worker{
public:
    Monitor_cpu(QObject *parent=nullptr){
        res1="cpu:";
    }
    ~Monitor_cpu(){}
    void do_work(){
#if defined (__linux__)
        static uint64_t prevIdle = 0, prevTotal = 0;

        std::ifstream f("/proc/stat");
        std::string cpu;
        uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
        f >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

        uint64_t idleTime  = idle + iowait;
        uint64_t totalTime = user + nice + system + idle + iowait + irq + softirq + steal;

        double usage = 0.0;
        if (prevTotal != 0) {
            uint64_t dIdle  = idleTime  - prevIdle;
            uint64_t dTotal = totalTime - prevTotal;
            if (dTotal > 0)
                usage = 100.0 * (1.0 - (float)dIdle / dTotal);
        }
        prevIdle  = idleTime;
        prevTotal = totalTime;
        res2=std::to_string(usage);
#elif defined (_WIN32)
        static ULONGLONG prevIdle = 0, prevKernel = 0, prevUser = 0;

        FILETIME idle, kernel, user;
        GetSystemTimes(&idle, &kernel, &user);

        auto toU64 = [](FILETIME ft) -> ULONGLONG {
            return (ULONGLONG)ft.dwHighDateTime << 32 | ft.dwLowDateTime;
        };

        ULONGLONG idleT   = toU64(idle);
        ULONGLONG kernelT = toU64(kernel);
        ULONGLONG userT   = toU64(user);

        float usage = 0.0f;
        if (prevIdle != 0) {
            ULONGLONG dIdle   = idleT   - prevIdle;
            ULONGLONG dKernel = kernelT - prevKernel;
            ULONGLONG dUser   = userT   - prevUser;
            ULONGLONG dTotal  = dKernel + dUser;

            if (dTotal > 0)
                usage = 100.0f * (1.0f - (float)dIdle / dTotal);
        }
        prevIdle   = idleT;
        prevKernel = kernelT;
        prevUser   = userT;
        res2=std::to_string(usage);
#endif
        emit done();
    }
};

class Monitor_memory_cpu:public Worker{
public:
    Monitor_memory_cpu(QObject *parent=nullptr){
        res1="内存:";
    }
    ~Monitor_memory_cpu(){}
    void do_work(){
#if defined (__linux__)
        std::ifstream f("/proc/meminfo");
        std::string key;
        long value = 0, total = 0, available = 0;
        std::string unit;

        while (f >> key >> value >> unit) {       // "MemTotal:" "8167556" "kB"
            if (key == "MemTotal:")      total     = value;
            if (key == "MemAvailable:")  available = value;
            if (total && available) break;
        }

        if (total == 0) return 0.0f;
        res2=std::to_string( 100.0 * (total - available) / total);
#elif defined (_WIN32)
        MEMORYSTATUSEX mem = {sizeof(mem)};
        GlobalMemoryStatusEx(&mem);
        double used =(mem.ullTotalPhys-mem.ullAvailPhys);
        double total = mem.ullTotalPhys;
        res2=std::to_string(used*100.0/total);
#endif
        emit done();
    }
};

class Monitor_gpu:public Worker{
public:
    Monitor_gpu(QObject *parent=nullptr){
        p=new QProcess(this);
        connect(p,&QProcess::finished,this,[this](){
            if(p->exitCode()!=0){
                res1="-1";
                res2="-1";
                return ;
            }
            QString out=p->readAllStandardOutput().trimmed();
            QStringList parts = out.split(',');
            if(parts.size()<3){
                res1="-1";
                res2="-1";
                return ;
            }
            res1=parts[0].trimmed().toStdString();
            res2=std::to_string(parts[1].trimmed().toDouble()/parts[2].trimmed().toDouble()*100.0);
            emit done();
        });
    }
    ~Monitor_gpu(){}
    QProcess *p;
    void do_work(){
        p->start("nvidia-smi",
                 QStringList() << "--query-gpu=utilization.gpu,memory.used,memory.total"
                               << "--format=csv,noheader,nounits");

    }
    std::pair<QString,QString> to_QString(){
        return {("gpu:"+res1+res3).c_str(),("显存:"+res2+res3).c_str()};
    }
};

Monitor::Monitor(QWidget *parent)
    : QWidget{parent}
{
    this->resize(600,600);

    hbox1=new QHBoxLayout(this);
    this->setLayout(hbox1);

    vbox1=new QVBoxLayout();
    vbox2=new QVBoxLayout();
    hbox1->addLayout(vbox1);
    hbox1->addLayout(vbox2);

    label_cpu=new QLabel("1",this);
    label_memory_cpu=new QLabel("2",this);
    label_gpu=new QLabel("3",this);
    label_memory_gpu=new QLabel("4",this);
    cpu_plot=new Plot_Widget(60,this);
    memory_cpu_plot=new Plot_Widget(60,this);
    gpu_plot=new Plot_Widget(60,this);
    memory_gpu_plot=new Plot_Widget(60,this);
    vbox1->addWidget(label_cpu);
    vbox1->addWidget(cpu_plot);
    vbox1->addWidget(label_memory_cpu);
    vbox1->addWidget(memory_cpu_plot);
    vbox2->addWidget(label_gpu);
    vbox2->addWidget(gpu_plot);
    vbox2->addWidget(label_memory_gpu);
    vbox2->addWidget(memory_gpu_plot);

    timer=new QTimer(this);
    Monitor_cpu *monitor_cpu=new Monitor_cpu();
    Monitor_memory_cpu *monitor_memory_cpu=new Monitor_memory_cpu();
    Monitor_gpu *monitor_gpu=new Monitor_gpu();
    thread1=new QThread(this);
    thread2=new QThread(this);
    thread3=new QThread(this);
    monitor_cpu->moveToThread(thread1);
    monitor_memory_cpu->moveToThread(thread2);
    monitor_gpu->moveToThread(thread3);
    connect(timer,&QTimer::timeout,monitor_cpu,&Monitor_cpu::do_work);
    connect(timer,&QTimer::timeout,monitor_memory_cpu,&Monitor_memory_cpu::do_work);
    connect(timer,&QTimer::timeout,monitor_gpu,&Monitor_gpu::do_work);
    connect(monitor_cpu,&Worker::done,[this,monitor_cpu](){
        label_cpu->setText(monitor_cpu->to_QString());
        cpu_plot->add_point(QString(monitor_cpu->res2.c_str()).toDouble());
    });
    connect(monitor_memory_cpu,&Worker::done,[this,monitor_memory_cpu](){
        label_memory_cpu->setText(monitor_memory_cpu->to_QString());
        memory_cpu_plot->add_point(QString(monitor_memory_cpu->res2.c_str()).toDouble());
    });
    connect(monitor_gpu,&Worker::done,[this,monitor_gpu](){
        std::pair<QString,QString>temp=monitor_gpu->to_QString();
        label_gpu->setText(temp.first);
        label_memory_gpu->setText(temp.second);
        gpu_plot->add_point(QString(monitor_gpu->res1.c_str()).toDouble());
        memory_gpu_plot->add_point(QString(monitor_gpu->res2.c_str()).toDouble());
    });
    thread1->start();
    thread2->start();
    thread3->start();
    timer->start(1000);
}

Monitor::~Monitor(){
    timer->stop();
    thread1->quit();
    thread2->quit();
    thread3->quit();
    thread1->wait(1000);
    thread2->wait(1000);
    thread3->wait(1000);
    thread1->deleteLater();
    thread2->deleteLater();
    thread3->deleteLater();
}

#include "monitor.moc"