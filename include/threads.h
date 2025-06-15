#ifndef BOUNDEDQUEUE_H
#define BOUNDEDQUEUE_H
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <iostream>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <tbb/concurrent_queue.h>
// #include "yolov8.hpp"
#include "yolov8_batch.h"
namespace fs = ghc::filesystem;

// 线程安全有界队列
template<typename T>
class BoundedQueue {
public:
    BoundedQueue(size_t cap) : cap_(cap), stopped_(false) {
    }

    // Producer 调用：阻塞直到有空位或停止
    void push(const T &item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [&] { return queue_.size() < cap_ || stopped_; });
        if (stopped_) return;
        queue_.push(item);
        // 只唤醒一个等待 pop 的消费者线程
        cv_not_empty_.notify_one();
    }

    // Consumer 调用：阻塞直到有数据或停止
    bool pop(T &item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [&] { return !queue_.empty() || stopped_; });
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        // 只唤醒一个等待 push 的生产者线程
        cv_not_full_.notify_one();
        return true;
    }

    // 通知所有线程停止，并唤醒所有阻塞
    void stop() { {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

private:
    std::queue<T> queue_;
    size_t cap_;
    bool stopped_;
    std::mutex mutex_;
    std::condition_variable cv_not_full_;
    std::condition_variable cv_not_empty_;
};

// 添加辅助函数：调整图像大小，保持横纵比，最长边不超过maxSize
cv::Mat resizeKeepAspectRatio(const cv::Mat &input, const int maxSize = 640) {
    int width = input.cols;
    int height = input.rows;

    // 如果图像已经小于等于最大尺寸，则不需要调整
    if (width <= maxSize && height <= maxSize)
        return input.clone();

    double scale;
    if (width >= height)
        scale = static_cast<double>(maxSize) / width;
    else
        scale = static_cast<double>(maxSize) / height;
    int newWidth = static_cast<int>(width * scale);
    int newHeight = static_cast<int>(height * scale);

    cv::Mat resized;
    cv::resize(input, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
    return resized;
}

// 获取当前时间字符串
std::string getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// 添加保存检测结果到文件的功能
void saveDetectionResults(const std::vector<std::vector<Object> > &allObjects,
                          const std::string &outputDir,
                          const std::vector<std::string> &fileNames,
                          const std::vector<std::string> &classNames) {
    std::string summaryFile = outputDir + "/summary.txt";
    std::ofstream summary(summaryFile);
    if (summary.is_open()) {
        summary << "检测时间: " << getCurrentTimeString() << std::endl;
        summary << "总帧数: " << allObjects.size() << std::endl;

        int totalObjects = 0;
        std::map<std::string, int> objectCounts;

        for (size_t i = 0; i < allObjects.size(); i++) {
            const auto &objects = allObjects[i];
            totalObjects += objects.size();

            for (const auto &obj: objects) {
                objectCounts[classNames[obj.label]]++;
            }
        }
        summary << "检测到的目标总数: " << totalObjects << std::endl;
        summary << "各类别统计:" << std::endl;
        for (const auto &pair: objectCounts) {
            summary << pair.first << ": " << pair.second << std::endl;
        }
        summary.close();
    }
}

enum class PrintMode {
    INFO, // 使用 std::cout
    ERROR // 使用 std::cerr
};

void safe_print(PrintMode mode = PrintMode::INFO, std::string args = "", bool flush = false) {
    static std::mutex print_mutex;
    std::lock_guard<std::mutex> lock(print_mutex);
    // 折叠表达式把参数都插入到 ostringstream
    if (mode == PrintMode::ERROR) {
        std::cerr << args << std::endl;
    } else {
        if (flush)
            std::cout << args << std::flush;
        else
            std::cout << args << std::endl;
    }
}

// 兼容原有调用方式的重载版本（默认使用 std::cout）
void safe_print(std::string args) {
    safe_print(PrintMode::INFO, args, false);
}


// 生产者：读取视频帧
void frameProducer(const std::string &videoPath,
                   tbb::concurrent_bounded_queue<cv::Mat> &queue,
                   std::atomic<bool> &done,
                   int batchSize = 3) {
    cv::VideoCapture cap(videoPath);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int pushedFrames = 0;
    cv::Mat image;

    if (!cap.isOpened()) {
        safe_print(PrintMode::ERROR, "无法打开视频: " + videoPath);
        done = true;
        return;
    }
    std::vector<cv::Mat> frameBatch;
    frameBatch.reserve(batchSize);

    while (true) {
        // 批量读取帧
        frameBatch.clear();
        for (int i = 0; i < batchSize; ++i) {
            if (!cap.read(image))
                break; // 视频读取完毕
            frameBatch.push_back(resizeKeepAspectRatio(image, 640).clone());
        }
        if (frameBatch.empty()) // 如果没有读到任何帧，说明视频已结束
            break;

        // 批量预处理（可以使用并行处理提高效率）
#pragma omp parallel for num_threads(preprocessThreads)
        for (int i = 0; i < frameBatch.size(); ++i) {
            // 这里可以添加更多预处理操作
            // 比如图像增强、归一化等
        }
        for (auto &frame: frameBatch) {// 批量推送到队列
            queue.push(std::move(frame)); // 使用移动语义避免拷贝
            pushedFrames++;
        }
    }
    // while (cap.read(image)) {
    //     image = resizeKeepAspectRatio(image, 640);
    //     queue.push(image.clone()); // 阻塞直至有空余位置
    //     pushedFrames++;
    // }
    safe_print("生产者线程已完成，入队帧数: " + std::to_string(pushedFrames) + "/" + std::to_string(totalFrames) + " 帧");
    done = true;
}

// 用于保存图像和路径信息的结构体
struct ImageSaveItem {
    cv::Mat image;
    std::string path;
};

// 消费者：并发推理并保存结果
void workerConsumer(int id,
                    const std::string &enginePath,
                    tbb::concurrent_bounded_queue<cv::Mat> &queue,
                    std::atomic<bool> &done,
                    std::atomic<int> &frameCounter,
                    cv::Size size,
                    const std::string &videoName,
                    const std::string &resultDir,
                    const std::vector<std::string> &CLASS_NAMES,
                    const std::vector<std::vector<unsigned int> > &COLORS,
                    const std::vector<std::string> &targetClasses,
                    tbb::concurrent_bounded_queue<ImageSaveItem> &saveQueue, std::atomic<bool> &writerDone) {
    // 每线程独立加载 engine 和上下文
    YOLOv8 detector(enginePath);
    detector.make_pipe(true);
    detector.createClassFilter(CLASS_NAMES, targetClasses);

    // 批处理相关参数
    const int batchSize = 3; // 可根据GPU内存调整批大小
    std::vector<cv::Mat> batchImages;
    std::vector<int> batchFrameIDs;

    cv::Mat image, res;
    std::vector<Object> objs;
    int localCount = 0;
    // 用于存储所有帧的检测结果
    std::vector<std::vector<Object> > allFrameObjects;
    std::vector<std::string> frameNames;
    std::string frameName;
    std::string imageDir = resultDir + "/images";
    fs::create_directories(imageDir);
    while (true) {
        try {
            batchImages.clear();
            // 尝试填满一个批次
            while (batchImages.size() < batchSize) {
                if (!queue.try_pop(image)) {
                    // 如果队列为空且生产者已结束，跳出收集循环
                    if (done && queue.empty()) break;
                    // 如果已经有一些图像，不再等待更多，直接处理现有批次
                    if (!batchImages.empty()) break;
                    // 队列空但生产者未结束，等待一小段时间后继续尝试
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                } // 获取唯一的帧ID
                int currentFrameID = frameCounter.fetch_add(1);
                batchFrameIDs.push_back(currentFrameID);
                batchImages.push_back(image);
            }
            // 如果批次为空且生产者已结束，退出主循环
            if (batchImages.empty() && done) break;

            // 如果收集到了图像，进行批量处理
            if (!batchImages.empty()) {
                // 批量推理
                detector.copy_from_Mat_batch(batchImages, size);
                detector.infer(batchImages.size());

                std::vector<std::vector<Object> > batchObjs;
                detector.postprocess_batch(batchObjs, batchImages.size());

                // 处理每张图像的结果
                for (size_t i = 0; i < batchImages.size(); i++) {
                    // auto &objs = batchObjs[i];// 过滤类别
                    detector.filterObjectsByClass(objs);
                    // cv::Mat resImage = batchImages[i].clone();

                    if (!objs.empty()) {
                        // 绘制检测结果
                        YOLOv8::draw_objects(batchImages[i], objs, CLASS_NAMES, COLORS);
                        std::string frameName = videoName + "_frame" + std::to_string(batchFrameIDs[i]) +
                                                "_Worker" + std::to_string(id) + ".jpg";
                        std::string fullImagePath = imageDir + "/" + frameName;
                        saveQueue.push(ImageSaveItem{batchImages[i].clone(), fullImagePath});

                        // 记录结果
                        allFrameObjects.push_back(objs);
                        frameNames.push_back(frameName);
                        localCount++;
                    }
                }
            }
            // // 使用try_pop避免永久阻塞，允许检查退出条件
            // if (!queue.try_pop(image)) {
            //     //queue.pop(image)会一直不退出 去处理
            //     // 如果队列为空且生产者已结束，则退出循环
            //     if (done && queue.empty()) break;
            //     std::this_thread::sleep_for(std::chrono::milliseconds(1));
            //     continue;
            // }
            // 对每一帧，先获取唯一的帧ID，然后再处理
            // int currentFrameID = frameCounter.fetch_add(1); // 原子操作，确保每个帧有唯一ID
            //
            // // 收集一个批次的图像
            // batchImages.clear();
            // batchFrameIDs.clear();
            // // objs.clear();
            // detector.copy_from_Mat(image, size);
            // // detector.infer();
            // detector.infer(3);
            // detector.postprocess(objs);
            // detector.filterObjectsByClass(objs);
            //
            // if (objs.empty())
            //     continue;
            // //             res = std::move(image);
            // YOLOv8::draw_objects(image, objs, CLASS_NAMES, COLORS);
            // frameName = videoName + "_frame" + std::to_string(currentFrameID) + "_Worker" + std::to_string(id) + ".jpg";
            // std::string fullImagePath = imageDir + "/" + frameName; // 每���创建新的完整路径
            // saveQueue.push(std::move(ImageSaveItem{std::move(image), fullImagePath}));
            // // 使用移动语义代替复制，提高性能：临时对象、image 避免大对象成员的深拷贝
            //
            // allFrameObjects.push_back(objs);
            // frameNames.push_back(frameName); // 保存文件名以便后续使用
            //
            // // cv::imshow("Worker_" + std::to_string(id), res);
            // // if (cv::waitKey(1) == 27)
            // //     // 按ESC键退出
            // //     break;
        } catch (const std::exception &e) {
            safe_print(PrintMode::ERROR, "Worker " + std::to_string(id) + " 异常: " + e.what());
            // std::cerr << "Worker " << id << " 异常: " << e.what() << std::endl;
        }
        ++localCount;
    }
    safe_print(" Worker " + std::to_string(id) + " 处理了 " + std::to_string(localCount) + " 帧");
    saveDetectionResults(allFrameObjects, resultDir, frameNames, CLASS_NAMES);
}

// 图像写入线程函数
inline void imageWriterThread(tbb::concurrent_bounded_queue<ImageSaveItem> &saveQueue, std::atomic<bool> &writerDone) {
    // int savedCount = 0;
    while (true) {
        ImageSaveItem item;
        if (!saveQueue.try_pop(item)) {
            if (writerDone && saveQueue.empty()) break; //工作线程已完成 且 不会出现新的任务
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        // cv::imwrite(item.path, item.image);
        // 使用更高效的图像格式和压缩参数
        cv::imwrite(item.path, item.image, {cv::IMWRITE_JPEG_QUALITY, 90}); // 适度降低质量提高写入速度
        // savedCount++;
    }
    // safe_print("图像写入线程已完成，保存了 " + std::to_string(savedCount) + " 个图像");
}
#endif //BOUNDEDQUEUE_H
