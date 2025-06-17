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

// struct Batch {
//     std::vector<cv::Mat> frameBatch;
//     int size = 1; // 默认批大小为1
// };

// 生产者：读取视频帧
void frameProducer(const std::string &videoPath,
                   tbb::concurrent_bounded_queue<std::vector<cv::Mat> > &queue,
                   std::atomic<bool> &done,
                   int batchSize = 1) {
    cv::VideoCapture cap(videoPath);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int pushedFrames = 0;
    cv::Mat image;

    if (!cap.isOpened()) {
        safe_print(PrintMode::ERROR, "无法打开视频: " + videoPath);
        done = true;
        return;
    }
    std::vector<cv::Mat> batch;
    batch.reserve(batchSize);

    while (true) {
        // 打包一个batch
        batch.clear();
        for (int i = 1; i <= batchSize; ++i) {
            if (!cap.read(image))
                break; // 视频读取完毕
            pushedFrames++;
            batch.push_back(std::move(resizeKeepAspectRatio(image, 640)));
        }
        if (batch.empty()) // 如果没有读到任何帧，说明视频已结束
            break;
        queue.push(std::move(batch)); // 使用移动语义避免拷贝
    }
    safe_print("生产者线程已完成，入队帧数: " + std::to_string(pushedFrames) + "/" + std::to_string(totalFrames) + " 帧");
    done = true;
}

// 用于保存图像和路径信息的结构体
struct ImageSaveItem {
    std::vector<Object> objs;
    cv::Mat image;
    std::string path;
};

// 消费者：并发推理并保存结果
void workerConsumer(int id,
                    const std::string &enginePath,
                    tbb::concurrent_bounded_queue<std::vector<cv::Mat> > &queue,
                    std::atomic<bool> &done,
                    std::atomic<int> &frameCounter,
                    cv::Size size,
                    const std::string &videoName,
                    const std::string &resultDir,
                    const std::vector<std::string> &CLASS_NAMES,
                    const std::vector<std::string> &targetClasses,
                    tbb::concurrent_bounded_queue<ImageSaveItem> &saveQueue) {
    YOLOv8 detector(enginePath,id);
    detector.make_pipe(true);
    detector.createClassFilter(CLASS_NAMES, targetClasses);

    std::vector<cv::Mat> batch;

    cv::Mat image, res;
    std::vector<std::vector<Object> > batchObjs;
    std::vector<Object> objs;
    int inference_count = 0, n, currentFrameID;
    thread_local double total_gpu_time_ms = 0;
    // 用于存储所有帧的检测结果
    std::vector<std::vector<Object> > allFrameObjects;
    std::vector<std::string> frameNames;
    std::string frameName;
    std::string imageDir = resultDir + "/images";
    fs::create_directories(imageDir);
    while (true) {
        try {
            batch.clear();
            if (!queue.try_pop(batch)) {
                if (done && queue.empty()) break; // 如果队列为空且生产者已结束，跳出收集循环
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            if (batch.empty()) {
                if (done)
                    break;
            } else {
                n = static_cast<int>(batch.size());
                detector.copy_from_Mat_batch(batch, size);
                //
                auto infer_start = std::chrono::high_resolution_clock::now();
                detector.infer(n); // 批量推理
                auto infer_end = std::chrono::high_resolution_clock::now();
                //
                total_gpu_time_ms += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                        infer_end - infer_start).
                    count()) / 1000.0;

                batchObjs.clear();
                detector.postprocess_batch(batchObjs, n);
                detector.filterObjectsByClassBatch(batchObjs);

                // 拆出每张图像的结果 压入队列
                inference_count += n;
                for (int i = 0; i < n; i++) {
                    currentFrameID = frameCounter.fetch_add(1);
                    objs = batchObjs[i];

                    if (!objs.empty()) {
                        frameName = videoName + "_frame" + std::to_string(currentFrameID) +
                                    "_Worker" + std::to_string(id) + ".jpg";
                        saveQueue.push(std::move(ImageSaveItem{objs, std::move(batch[i]), imageDir + "/" + frameName}));
                        // 记录结果
                        allFrameObjects.push_back(std::move(objs));
                        frameNames.push_back(frameName);
                    }
                }
            }
        } catch (const std::exception &e) {
            safe_print(PrintMode::ERROR, "Worker " + std::to_string(id) + " 异常: " + e.what());
        }
    }
    safe_print(
        " Worker " + std::to_string(id) + " 处理了 " + std::to_string(inference_count) + " 帧，" + " avg GPU time: " +
        std::to_string(total_gpu_time_ms / inference_count) + " ms/frame");
    saveDetectionResults(allFrameObjects, resultDir, frameNames, CLASS_NAMES);
}

inline void imageWriterThread(tbb::concurrent_bounded_queue<ImageSaveItem> &saveQueue, std::atomic<bool> &writerDone,
                              const std::vector<std::string> &CLASS_NAMES,
                              const std::vector<std::vector<unsigned int> > &COLORS) {
    ImageSaveItem item;
    while (true) {
        if (!saveQueue.try_pop(item)) {
            if (writerDone && saveQueue.empty()) break; //工作线程已完成 且 不会出现新的任务
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        // 先给图片绘制检测结果
        YOLOv8::draw_objects(item.image, item.objs, CLASS_NAMES, COLORS);
        // cv::imwrite(item.path, item.image); 使用更高效的图像格式和压缩参数
        cv::imwrite(item.path, item.image, {cv::IMWRITE_JPEG_QUALITY, 90}); // 适度降低质量提高写入速度
    }
}
#endif //BOUNDEDQUEUE_H
