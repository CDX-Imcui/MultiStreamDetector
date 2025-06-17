#include "opencv2/opencv.hpp"
// #include "yolov8.hpp"
#include "yolov8_batch.h"
#include <chrono>
#include <fstream>
#include <threads.h>
#include <tbb/concurrent_queue.h>
namespace fs = ghc::filesystem;


const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
};


const std::vector<std::vector<unsigned int> > COLORS = {
    {0, 114, 189}, {217, 83, 25}, {237, 177, 32}, {126, 47, 142}, {119, 172, 48}, {77, 190, 238},
    {162, 20, 47}, {76, 76, 76}, {153, 153, 153}, {255, 0, 0}, {255, 128, 0}, {191, 191, 0},
    {0, 255, 0}, {0, 0, 255}, {170, 0, 255}, {85, 85, 0}, {85, 170, 0}, {85, 255, 0},
    {170, 85, 0}, {170, 170, 0}, {170, 255, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0},
    {0, 85, 128}, {0, 170, 128}, {0, 255, 128}, {85, 0, 128}, {85, 85, 128}, {85, 170, 128},
    {85, 255, 128}, {170, 0, 128}, {170, 85, 128}, {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128}, {255, 170, 128}, {255, 255, 128}, {0, 85, 255}, {0, 170, 255}, {0, 255, 255},
    {85, 0, 255}, {85, 85, 255}, {85, 170, 255}, {85, 255, 255}, {170, 0, 255}, {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255}, {255, 85, 255}, {255, 170, 255}, {85, 0, 0},
    {128, 0, 0}, {170, 0, 0}, {212, 0, 0}, {255, 0, 0}, {0, 43, 0}, {0, 85, 0},
    {0, 128, 0}, {0, 170, 0}, {0, 212, 0}, {0, 255, 0}, {0, 0, 43}, {0, 0, 85},
    {0, 0, 128}, {0, 0, 170}, {0, 0, 212}, {0, 0, 255}, {0, 0, 0}, {36, 36, 36},
    {73, 73, 73}, {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189}, {128, 128, 0}
};

// void checkSystemMemory() {
//     size_t free_memory, total_memory;
//     cudaMemGetInfo(&free_memory, &total_memory);
//     std::cout << "GPU 总内存: " << total_memory / 1024 / 1024 << " MB" << std::endl;
//     std::cout << "GPU 可用内存: " << free_memory / 1024 / 1024 << " MB" << std::endl;
//
//     std::ifstream meminfo("/proc/meminfo");
//     std::string line;
//     while (std::getline(meminfo, line)) {
//         if (line.find("MemTotal") != std::string::npos ||
//             line.find("MemFree") != std::string::npos ||
//             line.find("MemAvailable") != std::string::npos) {
//             std::cout << line << std::endl;
//         }
//     }
// }
void calculate_time(const std::string &info, const std::chrono::system_clock::time_point &start,
                    const std::chrono::system_clock::time_point &end) {
    auto ms = (double) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    if (ms >= 60000.0) {
        int total_ms = static_cast<int>(ms);
        int minutes = total_ms / 60000;
        int seconds = (total_ms % 60000) / 1000;
        int millis = total_ms % 1000;
        printf("%s: %d min %d s %d ms\n", info.c_str(), minutes, seconds, millis);
    } else if (ms >= 1000.0) {
        printf("%s: %.3f s\n", info.c_str(), ms / 1000.0);
    } else {
        printf("%s: %.3f ms\n", info.c_str(), ms);
    }
}

int main(int argc, char **argv) {
    // const std::string engine_file_path{argv[1]};
    // const fs::path path{argv[2]};
    // if (argc != 3) {
    //     fprintf(stderr, "Usage: %s [engine_path] [image_path/image_dir/video_path]\n", argv[0]);
    //     return -1;
    // }
    const std::string engine_file_path{"../models/yolov8s.engine"};
    // const std::string engine_file_path{"../yolov8s.engine"};
    const fs::path path{"../data/1.mp4"};
    std::vector<cv::String> imagePathList;
    bool isVideo{false};
    std::vector<Object> objs;
    cv::Mat res, image;
    cv::Size size = cv::Size{640, 640};

    if (fs::exists(path)) {
        std::string suffix = path.extension().string();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png") {
            imagePathList.push_back(path.string());
        } else if (suffix == ".mp4" || suffix == ".avi" || suffix == ".m4v" || suffix == ".mpeg" || suffix == ".mov"
                   || suffix == ".mkv") {
            isVideo = true;
        } else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    } else if (fs::is_directory(path)) {
        cv::glob(path.string() + "/*.jpg", imagePathList);
    }

    if (isVideo) {
        // 有界队列容量
        // BoundedQueue<cv::Mat> queue(1500);
        tbb::concurrent_bounded_queue<std::vector<cv::Mat> > queue; ///  Intel TBB
        // queue.set_capacity(10000); // 设置最大容量
        tbb::concurrent_bounded_queue<ImageSaveItem> saveQueue;
        std::atomic<bool> writerDone(false);

        std::atomic<bool> done{false};
        std::atomic<int> frameCounter{1};
        cv::Size inputSize{640, 640};
        std::string videoName = fs::path(path).filename().string();
        cv::VideoCapture cap(path.string());
        int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        safe_print(videoName + "共有" + std::to_string(totalFrames) + "帧");
        std::vector<std::string> targetClasses = {"person", "car"};
        // std::vector<std::string> targetClasses = { "car"};
        std::cout << "只检测以下类别: ";
        for (const auto &cls: targetClasses)
            std::cout << cls << " ";
        // safe_print();


        // 准备输出目录
        std::string resultDir = "../detection_results/" + videoName;
        if (fs::exists(resultDir)) {
            for (const auto &entry: fs::directory_iterator(resultDir))
                fs::remove_all(entry.path());
        } else fs::create_directories(resultDir);


        auto start = std::chrono::system_clock::now();
        // 启动生产者线程
        int batchSize = 1;
        std::thread producer(frameProducer, path, std::ref(queue), std::ref(done), batchSize);
        // 启动消费者线程池
        // int numThreads = static_cast<int>(std::thread::hardware_concurrency()) - 1;
        int inference_Threads = 1;
        std::vector<std::thread> workers;
        for (int id = 1; id <= inference_Threads; ++id)
            workers.emplace_back(workerConsumer, id,
                                 std::ref(engine_file_path), std::ref(queue), std::ref(done),
                                 std::ref(frameCounter), inputSize, videoName, resultDir,
                                 std::ref(CLASS_NAMES), std::ref(targetClasses), std::ref(saveQueue));

        std::vector<std::thread> writerThreads;
        int writerThreadCount = 7; // 写入线程数量
        for (int i = 0; i < writerThreadCount; ++i)
            writerThreads.emplace_back(imageWriterThread, std::ref(saveQueue), std::ref(writerDone),
                                       std::ref(CLASS_NAMES), std::ref(COLORS));

        while (!done || frameCounter.load() < totalFrames) {// 主线程显示进度
            safe_print(PrintMode::INFO, "\r已处理: " + std::to_string(frameCounter.load()) + " 帧", true);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        calculate_time("inference cost", start, std::chrono::system_clock::now());

        producer.join(); // 确保生产者线程完成
        safe_print("读取视频线程完成...");
        for (auto &w: workers)
            w.join();
        safe_print("推理完成...，等待图像写入线程完成...");

        writerDone = true; // 设置writerDone标志，通知写入线程可以退出
        for (auto &w: writerThreads)
            w.join();
        safe_print("save线程完成");

        auto end = std::chrono::system_clock::now();
        safe_print("处理完成，总共帧数: " + std::to_string(frameCounter.load()) + " 帧");
        calculate_time("all time cost", start, end);

        // 确保所有窗口都关闭
        cv::destroyAllWindows();
        // 添加短暂延时使窗口事件能被处理
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } else {
        auto yolov8 = new YOLOv8(engine_file_path);
        yolov8->make_pipe(true);
        for (auto &p: imagePathList) {
            objs.clear();
            image = cv::imread(p);
            image = resizeKeepAspectRatio(image, 640);
            yolov8->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8->infer();
            auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs);
            res = image.clone();
            yolov8->draw_objects(res, objs, CLASS_NAMES, COLORS);
            auto tc = (double) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
            delete yolov8;
        }
    }
    cv::destroyAllWindows();
    return 0;
}
