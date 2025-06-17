#ifndef DETECT_END2END_YOLOV8_HPP
#define DETECT_END2END_YOLOV8_HPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>
using namespace det;

/**
 * @class YOLOv8
 * @brief 基于TensorRT的YOLOv8目标检测类
 *
 * 该类封装了YOLOv8模型的推理过程，包括模型加载、预处理、推理和后处理等功能。
 * 支持从TensorRT引擎文件加载模型，并进行实时目标检测。
 */
class YOLOv8 {
public:
    /**
     * @brief 构造函数，加载TensorRT引擎文件
     * @param engine_file_path TensorRT引擎文件路径
     */
    explicit YOLOv8(const std::string &engine_file_path, int stream_id = 0);

    /**
     * @brief 析构函数，释放资源
     */
    ~YOLOv8();


    // 获取当前流的方法
    cudaStream_t getStream() const {
        return this->stream;
    }

    /**
     * @brief 创建推理管道并准备资源
     * @param warmup 是否预热模型，默认为true
     */
    void make_pipe(bool warmup = true);

    /**
     * @brief 从OpenCV Mat格式图像拷贝数据到模型输入
     * @param image 输入图像
     */
    void copy_from_Mat(const cv::Mat &image);

    /**
     * @brief 从OpenCV Mat格式图像拷贝数据到模型输入，并指定大小
     * @param image 输入图像
     * @param size 指定的输出大小
     */
    void copy_from_Mat(const cv::Mat &image, cv::Size &size);

    /**
     * @brief 批量从OpenCV Mat格式图像拷贝数据到模型输入
     * @param images 输入图像列表
     * @param size 指定的输出大小
     */
    void copy_from_Mat_batch(const std::vector<cv::Mat> &images, cv::Size &size);

    /**
     * @brief 图像预处理，执行letterbox操作
     * @param image 输入图像
     * @param out 预处理后的输出图像
     * @param size 目标大小
     */
    void letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size);

    /**
     * @brief 批量图像预处理，执行letterbox操作
     * @param images 输入图像列表
     * @param outputs 预处理后的输出图像列表
     * @param size 目标大小
     */
    void letterbox_batch(const std::vector<cv::Mat> &images, std::vector<cv::Mat> &outputs, cv::Size &size);

    /**
     * @brief 执行模型推理
     */
    void infer();

    /**
     * @brief 执行批量模型推理
     * @param batch_size 批量大小
     */
    void infer(int batch_size);

    /**
     * @brief 后处理，解析模型输出为检测结果
     * @param objs 存储检测到的目标对象
     */
    void postprocess(std::vector<Object> &objs);

    /**
     * @brief 批量后处理，解析模型输出为多张图像的检测结果
     * @param batch_objs 存储每张图像检测到的目标对象
     * @param batch_size 批量大小
     */
    void postprocess_batch(std::vector<std::vector<Object> > &batch_objs, int batch_size);

    std::vector<bool> createClassFilter(const std::vector<std::string> &classNames,
                                        const std::vector<std::string> &targetClasses);

    void filterObjectsByClass(std::vector<Object> &objects);

    void filterObjectsByClassBatch(std::vector<std::vector<Object> > &batch_objects);


    /**
     * @brief 在图像上绘制检测结果
     * @param image 原始图像
     * @param res 绘制结果的输出图像
     * @param objs 检测到的目标对象
     * @param CLASS_NAMES 类别名称列表
     * @param COLORS 类别对应的颜色列表
     */
    static void draw_objects(cv::Mat &res,
                             const std::vector<Object> &objs,
                             const std::vector<std::string> &CLASS_NAMES,
                             const std::vector<std::vector<unsigned int> > &COLORS);

    // 模型绑定信息
    int num_bindings; ///< 总绑定数量
    int num_inputs = 0; ///< 输入绑定数量
    int num_outputs = 0; ///< 输出绑定数量
    std::vector<Binding> input_bindings; ///< 输入绑定信息
    std::vector<Binding> output_bindings; ///< 输出绑定信息
    std::vector<void *> host_ptrs; ///< 主机内存指针
    std::vector<void *> device_ptrs; ///< 设备(GPU)内存指针

    PreParam pparam; ///< 预处理参数
    std::vector<PreParam> pparams; ///< 批量预处理参数

private:
    nvinfer1::ICudaEngine *engine = nullptr; ///< TensorRT引擎
    nvinfer1::IRuntime *runtime = nullptr; ///< TensorRT运行时
    nvinfer1::IExecutionContext *context = nullptr; ///< 执行上下文
    cudaStream_t stream = nullptr; ///< CUDA流
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR}; ///< TensorRT日志记录器
    std::vector<bool> filter = std::vector<bool>(80, false);
};

// inline void checkRuntime(cudaError_t code, const char *file = __FILE__, int line = __LINE__) {
//     if (code != cudaSuccess) {
//         const char *err_name = cudaGetErrorName(code);
//         const char *err_message = cudaGetErrorString(code);
//         printf("CUDA Runtime Error [%s:%d]: %s(%s)\n", file, line, err_name, err_message);
//         exit(-1);
//     }
// }

inline YOLOv8::YOLOv8(const std::string &engine_file_path,int stream_id) {
    // 打开TensorRT引擎文件
    std::ifstream file(engine_file_path, std::ios::binary);
    //assert 是 C++ 中的断言宏，用于在调试时检查某个条件是否为真。
    //如果条件为假（即为 0），程序会输出错误信息并终止执行。常用于捕捉不应该发生的程序状态。
    assert(file.good());

    // 获取文件大小
    file.seekg(0, std::ios::end); //将文件指针移动到文件末尾。
    auto size = file.tellg(); //获取当前文件指针的位置（即文件末尾），返回的就是文件的字节数，也就是文件大小。
    file.seekg(0, std::ios::beg); //再把文件指针移回文件开头，方便后续读取。

    // 读取引擎文件内容到内存
    char *trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // 初始化TensorRT插件库
    initLibNvInferPlugins(&this->gLogger, "");

    // 创建TensorRT运行时
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    // 从文件内容反序列化TensorRT引擎
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;

    // 创建独立的执行上下文
#ifdef TRT_10
    this->context = this->engine->createExecutionContext(1); // 使用标志1表示独立上下文
#else
    this->context = this->engine->createExecutionContext();
#endif
    assert(this->context != nullptr);

    // 使用特定ID创建非阻塞CUDA流 TODO
    std::string stream_name = "yolov8_stream_" + std::to_string(stream_id);
    CHECK(cudaStreamCreateWithFlags(&this->stream, cudaStreamNonBlocking));
#if CUDA_VERSION >= 11000
    cudaStreamAttrValue attrValue;
    attrValue.nameInfo.name = stream_name.c_str();
    CHECK(cudaStreamSetAttribute(this->stream, cudaStreamAttributeName, &attrValue));
#endif
    // cudaStreamCreate(&this->stream);

    // 获取绑定数量，适配不同TensorRT版本
#ifdef TRT_10
    this->num_bindings = this->engine->getNbIOTensors();
#else
    this->num_bindings = this->engine->getNbBindings();
#endif

    // 遍历所有绑定，收集输入输出信息
    for (int i = 0; i < this->num_bindings; ++i) {
        Binding binding;
        nvinfer1::Dims dims;

        // 根据TensorRT版本获取绑定信息
#ifdef TRT_10
        std::string        name  = this->engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());
#else
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string name = this->engine->getBindingName(i);
#endif
        binding.name = name;
        binding.dsize = type_to_size(dtype);

        // 判断当前绑定是输入还是输出
#ifdef TRT_10
        bool IsInput = engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
#else
        bool IsInput = engine->bindingIsInput(i);
#endif
        if (IsInput) {
            this->num_inputs += 1;

            // 获取并设置输入��度
#ifdef TRT_10
            dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            // 设置最大优化形状
            this->context->setInputShape(name.c_str(), dims);
#else
            dims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            // 设置最大优化形状
            this->context->setBindingDimensions(i, dims);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
        } else {
            // 获取输出维度
#ifdef TRT_10
            dims = this->context->getTensorShape(name.c_str());
#else
            dims = this->context->getBindingDimensions(i);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}


inline YOLOv8::~YOLOv8() {
    // 根据TensorRT版本不同，使用不同方式释放资源
#ifdef TRT_10
    delete this->context;
    delete this->engine;
    delete this->runtime;
#else
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
#endif
    // 销毁CUDA流
    cudaStreamDestroy(this->stream);
    // 释放所有设备(GPU)内存
    for (auto &ptr: this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }
    // 释放所有主机内存
    for (auto &ptr: this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

inline void YOLOv8::make_pipe(bool warmup) {
    // 为输入绑定分配设备内存
    for (auto &bindings: this->input_bindings) {
        void *d_ptr;
        // 异步分配GPU内存
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);

        // 设置TensorRT输入形状和地址(TensorRT 10版本专用)
#ifdef TRT_10
        auto name = bindings.name.c_str();
        this->context->setInputShape(name, bindings.dims);
        this->context->setTensorAddress(name, d_ptr);
#endif
    }

    // 为输出绑定分配设备和主机内存
    for (auto &bindings: this->output_bindings) {
        void *d_ptr, *h_ptr;

        size_t size = bindings.size * bindings.dsize;
        // 异步分配GPU内存
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        // 分配可锁页主机内存，用于高效的数据传输
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);

        // 设置TensorRT输出地址(TensorRT 10版本专用)
#ifdef TRT_10
        auto name = bindings.name.c_str();
        this->context->setTensorAddress(name, d_ptr);
#endif
    }

    // 如果需要预热模型
    if (warmup) {
        try {
            // 进行10次推理预热
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < this->input_bindings.size(); j++) {
                    auto &bindings = this->input_bindings[j];
                    size_t size = bindings.size * bindings.dsize;
                    void *h_ptr = malloc(size);
                    if (!h_ptr) {
                        printf("Warning: Failed to allocate memory for warmup\n");
                        continue;
                    }
                    // 使用零填充输入内存
                    memset(h_ptr, 0, size);
                    // 使用正确的设备指针索引
                    CHECK(cudaMemcpyAsync(this->device_ptrs[j], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                    // 确保异步操作完成后才释放内存
                    cudaStreamSynchronize(this->stream);
                    free(h_ptr);
                }
                // 执行一次推理
                this->infer();
                // 每次预热后同步一下，确保安全
                cudaStreamSynchronize(this->stream);
                // printf("Warmup %d/10 completed\n", i+1);
            }
            // printf("Model warmup 10 times completed successfully\n");
        } catch (const std::exception &e) {
            printf("Error during warmup: %s\n", e.what());
        }
    }
}

inline void YOLOv8::letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size) {
    // 获取目标尺寸和原始图像尺寸
    const float inp_h = size.height; // 目标高度
    const float inp_w = size.width; // 目标宽度
    float height = image.rows; // 原图高度
    float width = image.cols; // 原图宽度

    // 计算缩放比例，保持原始图像的宽高比
    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r); // 缩放后的宽度
    int padh = std::round(height * r); // 缩放后的高度

    // 如果尺寸发生变化，则进行缩放
    cv::Mat tmp;
    if ((int) width != padw || (int) height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    } else {
        tmp = image.clone();
    }

    // 计算需要添加的边框尺寸
    float dw = inp_w - padw; // 宽度方向上需要填充的像素数
    float dh = inp_h - padh; // 高度方向上需要填充的像素数

    // 计算上下左右各需要填充的像素数（保持图像居中）
    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    // 边框填充，使用灰色(114,114,114)作为填充色
    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    // 创建输出张量，形状为(1,3,inp_h,inp_w)，对应批次大小为1的RGB图像
    out.create({1, 3, (int) inp_h, (int) inp_w}, CV_32F);

    // 分离图像通道
    std::vector<cv::Mat> channels;
    cv::split(tmp, channels);

    // 创建指向输出数据的Mat视图，分别对应三个通道
    cv::Mat c0((int) inp_h, (int) inp_w, CV_32F, (float *) out.data);
    cv::Mat c1((int) inp_h, (int) inp_w, CV_32F, (float *) out.data + (int) inp_h * (int) inp_w);
    cv::Mat c2((int) inp_h, (int) inp_w, CV_32F, (float *) out.data + (int) inp_h * (int) inp_w * 2);

    // 将BGR通道转换为浮点数并归一化到[0,1]范围
    channels[0].convertTo(c2, CV_32F, 1 / 255.f); // B通道 -> 第3通道
    channels[1].convertTo(c1, CV_32F, 1 / 255.f); // G通道 -> 第2通道
    channels[2].convertTo(c0, CV_32F, 1 / 255.f); // R通道 -> 第1通道

    // 保存预处理参数，用于后处理时还原坐标
    this->pparam.ratio = 1 / r; // 缩放比例的倒数
    this->pparam.dw = dw; // 宽度方向上的填充量
    this->pparam.dh = dh; // 高度方向上的填充量
    this->pparam.height = height; // 原始图像高度
    this->pparam.width = width; // 原始图像宽度
}

inline void YOLOv8::copy_from_Mat(const cv::Mat &image) {
    // 声明一个临时变量存放预处理后的图像数据
    cv::Mat nchw;

    // 获取模型的输入尺寸
    auto &in_binding = this->input_bindings[0];
    int width = in_binding.dims.d[3];
    int height = in_binding.dims.d[2];
    cv::Size size{width, height};

    // 调用letterbox函数进行图像预处理
    this->letterbox(image, nchw, size);

    // 将预处理后的图像数据从主机内存拷贝到GPU设备内存
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));

    // 根据TensorRT版本不同，设置输入维度
#ifdef TRT_10
    auto name = this->input_bindings[0].name.c_str();
    this->context->setInputShape(name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    this->context->setTensorAddress(name, this->device_ptrs[0]);
#else
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});
#endif
}

inline void YOLOv8::copy_from_Mat(const cv::Mat &image, cv::Size &size) {
    // 声明一个临时变量存放预处理后的图像数据
    cv::Mat nchw;

    // 调用letterbox函数进行图像预处理，使用指定的尺寸
    this->letterbox(image, nchw, size);

    // 将预处理后的图像数据从主机内存拷贝到GPU设备内存
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));

    // 根据TensorRT版本不同，设置输入维度
#ifdef TRT_10
    auto name = this->input_bindings[0].name.c_str();
    this->context->setInputShape(name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    this->context->setTensorAddress(name, this->device_ptrs[0]);
#else
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
#endif
}

inline void YOLOv8::copy_from_Mat_batch(const std::vector<cv::Mat> &images, cv::Size &size) {
    // 声明一个临时变量存放预处理后的图像数据
    std::vector<cv::Mat> nchw_batch;
    // 清空之前批次的预处理参数
    this->pparams.clear();

    // 获取模型的输入尺寸
    auto &in_binding = this->input_bindings[0];
    int width = in_binding.dims.d[3];
    int height = in_binding.dims.d[2];
    cv::Size in_size{width, height};

    // 批量预处理：对每张图像执行letterbox操作
    this->letterbox_batch(images, nchw_batch, size);

    // 创建合并后的批量输入数据
    cv::Mat nchw;
    nchw.create({static_cast<int>(images.size()), 3, size.height, size.width}, CV_32F);

    // 将预处理后的多张图像数据合并
    float *dst_ptr = (float *) nchw.data;
    for (const auto &img: nchw_batch) {
        std::memcpy(dst_ptr, img.ptr<float>(), img.total() * sizeof(float));
        dst_ptr += img.total();
    }

    // 将合并后的数据从主机内存拷贝到GPU设备内存 - 这是关键的缺失步骤
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * sizeof(float), cudaMemcpyHostToDevice, this->stream));

    // 根据TensorRT版本不同，设置输入维度
#ifdef TRT_10
    auto name = this->input_bindings[0].name.c_str();
    this->context->setInputShape(name, nvinfer1::Dims{4, {static_cast<int>(images.size()), 3, size.height, size.width}});
    this->context->setTensorAddress(name, this->device_ptrs[0]);
#else
    this->context->setBindingDimensions(0, nvinfer1::Dims{
                                            4, {static_cast<int>(images.size()), 3, size.height, size.width}
                                        });
#endif
}

inline void YOLOv8::letterbox_batch(const std::vector<cv::Mat> &images, std::vector<cv::Mat> &outputs, cv::Size &size) {
    // 清空输出容器
    outputs.clear();
    outputs.reserve(images.size());

    // 获取目标尺寸
    const float inp_h = size.height; // 目标高度
    const float inp_w = size.width; // 目标宽度

    // 遍历每张输入图像
    for (const auto &image: images) {
        // 获取原始图像尺寸
        float height = image.rows; // 原图高度
        float width = image.cols; // 原图宽度

        // 计算缩放比例，保持原始图像的宽高比
        float r = std::min(inp_h / height, inp_w / width);
        int padw = std::round(width * r); // 缩放后的宽度
        int padh = std::round(height * r); // 缩放后的高度

        // 如果尺寸发生变化，则进行缩放
        cv::Mat tmp;
        if ((int) width != padw || (int) height != padh) {
            cv::resize(image, tmp, cv::Size(padw, padh));
        } else {
            tmp = image.clone();
        }

        // 计算需要添加的边框尺寸
        float dw = inp_w - padw; // 宽度方向上需要填充的像素数
        float dh = inp_h - padh; // 高度方向上需要填充的像素数

        // 计算上下左右各需要填充的像素数（保持图像居中）
        dw /= 2.0f;
        dh /= 2.0f;
        int top = int(std::round(dh - 0.1f));
        int bottom = int(std::round(dh + 0.1f));
        int left = int(std::round(dw - 0.1f));
        int right = int(std::round(dw + 0.1f));

        // 边框填充，使用灰色(114,114,114)作为填充色
        cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

        // 创建输出张量，形状为(1,3,inp_h,inp_w)，对应批次大小为1的RGB图像
        cv::Mat out;
        out.create({1, 3, (int) inp_h, (int) inp_w}, CV_32F);

        // 分离图像通道
        std::vector<cv::Mat> channels;
        cv::split(tmp, channels);

        // 创建指向输出数据的Mat视图，分别对应三个通道
        cv::Mat c0((int) inp_h, (int) inp_w, CV_32F, (float *) out.data);
        cv::Mat c1((int) inp_h, (int) inp_w, CV_32F, (float *) out.data + (int) inp_h * (int) inp_w);
        cv::Mat c2((int) inp_h, (int) inp_w, CV_32F, (float *) out.data + (int) inp_h * (int) inp_w * 2);

        // 将BGR通道转换为浮点数并归一化到[0,1]范围
        channels[0].convertTo(c2, CV_32F, 1 / 255.f); // B通道 -> 第3通道
        channels[1].convertTo(c1, CV_32F, 1 / 255.f); // G通道 -> 第2通道
        channels[2].convertTo(c0, CV_32F, 1 / 255.f); // R通道 -> 第1通道

        // 保存预处理参数，用于后处理时还原坐标
        PreParam param;
        param.ratio = 1 / r; // 缩放比例的倒数
        param.dw = dw; // 宽度方向上的填充量
        param.dh = dh; // 高度方向上的填充量
        param.height = height; // 原始图像高度
        param.width = width; // 原始图像宽度
        this->pparams.push_back(param);

        // 将预处理后的图像添加到输出列表
        outputs.push_back(out);
    }
}

inline void YOLOv8::infer() {
    // 根据TensorRT版本不同，调用不同的推理函数
#ifdef TRT_10
    this->context->enqueueV3(this->stream);
#else
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
#endif

    // 将推理结果从GPU设备内存拷贝回主机内存
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }

    // 同步CUDA流，等待所有异步操作完成
    cudaStreamSynchronize(this->stream);
}

inline void YOLOv8::infer(int batch_size) {
    // 检查批处理大小是否有效
    if (batch_size <= 0) {
        printf("警告: 批处理大小必须为正数，使用默认值1\n");
        batch_size = 1;
    }
    // 设置批处理维度
#ifdef TRT_10
    auto name = this->input_bindings[0].name.c_str();
    nvinfer1::Dims dims = this->context->getInputShape(name);
    dims.d[0] = batch_size; // 设置批处理大小
    this->context->setInputShape(name, dims);
    this->context->enqueueV3(this->stream);
#else
    nvinfer1::Dims dims = this->context->getBindingDimensions(0);
    dims.d[0] = batch_size; // 设置批处理大小
    this->context->setBindingDimensions(0, dims);

    // 确保TensorRT引擎能够处理设置的批处理大小
    if (!this->context->allInputDimensionsSpecified()) {
        printf("错误: 未完全指定输入维度\n");
        return;
    }

    // 执行推理
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
#endif

    // 将推理结果从GPU设备内存拷贝回主机内存
    for (int i = 0; i < this->num_outputs; i++) {
        // 计算批处理后的输出大小
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize * batch_size / this->
                       output_bindings[i].dims.d[0];
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    // 同步CUDA流，等待所有异步操作完成
    cudaStreamSynchronize(this->stream);
}

inline void YOLOv8::postprocess(std::vector<Object> &objs) {
    objs.clear(); // 清空输出对象列表

    // 解析模型输出，YOLOv8模型输出了4个张量：检测框数量、边界框坐标、置信度、类别
    int *num_dets = static_cast<int *>(this->host_ptrs[0]); // 检测到的目标数量
    auto *boxes = static_cast<float *>(this->host_ptrs[1]); // 边界框坐标数组
    auto *scores = static_cast<float *>(this->host_ptrs[2]); // 置信度分数数组
    int *labels = static_cast<int *>(this->host_ptrs[3]); // 类别标签数组

    // 获取预处理参数，用于还原原始图像坐标
    auto &dw = this->pparam.dw; // 宽度方向填充量
    auto &dh = this->pparam.dh; // 高度方向填充量
    auto &width = this->pparam.width; // 原始图像宽度
    auto &height = this->pparam.height; // 原始图像高度
    auto &ratio = this->pparam.ratio; // 缩放比例

    // 遍历所有检测到的目标
    for (int i = 0; i < num_dets[0]; i++) {
        // 指向当前边界框坐标的指针
        float *ptr = boxes + i * 4;

        // 获取边界框的左上角和右下角坐标，并减去填充量
        float x0 = *ptr++ - dw; // 左上角x坐标
        float y0 = *ptr++ - dh; // 左上角y坐标
        float x1 = *ptr++ - dw; // 右下角x坐标
        float y1 = *ptr - dh; // 右下角y坐标

        // 将坐标转换回原始图像尺寸，并确保不超出图像边界
        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);

        // 创建检测对象并填充信息
        Object obj;
        obj.rect.x = x0; // 边界框左上角x坐标
        obj.rect.y = y0; // 边界框左上角y坐标
        obj.rect.width = x1 - x0; // 边界框宽度
        obj.rect.height = y1 - y0; // 边界框高度
        obj.prob = *(scores + i); // 置信度分数
        obj.label = *(labels + i); // 类别标签

        // 将检测对象添加到结果列表
        objs.push_back(obj);
    }
}

inline void YOLOv8::postprocess_batch(std::vector<std::vector<Object> > &batch_objs, int batch_size) {
    // 清空并预分配输出对象列表
    batch_objs.clear();
    batch_objs.resize(batch_size);

    // 解析模型输出，YOLOv8模型输出了4个张量：检测框数量、边界框坐标、置信度、类别
    int *num_dets = static_cast<int *>(this->host_ptrs[0]); // 检测到的目标数量
    auto *boxes = static_cast<float *>(this->host_ptrs[1]); // 边界框坐标数组
    auto *scores = static_cast<float *>(this->host_ptrs[2]); // 置信度分数数组
    int *labels = static_cast<int *>(this->host_ptrs[3]); // 类别标签数组

    int max_det = this->output_bindings[1].dims.d[1]; // 从binding信息获取最大检测数

    for (int b = 0; b < batch_size; b++) {
        PreParam &pparam = this->pparams[b];
        auto &dw = pparam.dw, &dh = pparam.dh, &width = pparam.width, &height = pparam.height, &ratio = pparam.ratio;

        int num_detections_for_this_image = num_dets[b]; // 使用当前图片的检测数量

        // 计算指向当前图片数据的指针
        float *p_boxes_for_this_image = boxes + b * max_det * 4;
        float *p_scores_for_this_image = scores + b * max_det;
        int *p_labels_for_this_image = labels + b * max_det;

        for (int i = 0; i < num_detections_for_this_image; i++) {
            float *ptr = p_boxes_for_this_image + i * 4;
            float x0 = (*ptr++ - dw) * ratio;
            float y0 = (*ptr++ - dh) * ratio;
            float x1 = (*ptr++ - dw) * ratio;
            float y1 = (*ptr - dh) * ratio;

            Object obj;
            obj.rect.x = clamp(x0, 0.f, width);
            obj.rect.y = clamp(y0, 0.f, height);
            obj.rect.width = clamp(x1, 0.f, width) - obj.rect.x;
            obj.rect.height = clamp(y1, 0.f, height) - obj.rect.y;
            obj.prob = p_scores_for_this_image[i];
            obj.label = p_labels_for_this_image[i];
            batch_objs[b].push_back(obj);
        }
    }
}

// 创建目标类别过滤器
inline std::vector<bool> YOLOv8::createClassFilter(const std::vector<std::string> &classNames,
                                                   const std::vector<std::string> &targetClasses) {
    // 如果目标类别为空，则默认检测所有类别
    if (targetClasses.empty()) {
        std::fill(filter.begin(), filter.end(), true);
        return filter;
    }

    for (const auto &target: targetClasses) {
        auto it = std::find(classNames.begin(), classNames.end(), target);
        if (it != classNames.end()) {
            int index = std::distance(classNames.begin(), it);
            filter[index] = true;
        }
    }

    return filter;
}

// 根据类别过滤检测结果
inline void YOLOv8::filterObjectsByClass(std::vector<Object> &objects) {
    auto it = std::remove_if(objects.begin(), objects.end(),
                             [this](const Object &obj) {
                                 // 确保索引在有效范围内
                                 return obj.label >= this->filter.size() || !this->filter[obj.label];
                             });

    objects.erase(it, objects.end());
}

inline void YOLOv8::filterObjectsByClassBatch(std::vector<std::vector<Object> > &batch_objects) {
    for (auto &objects: batch_objects) {
        filterObjectsByClass(objects);
    }
}

inline void YOLOv8::draw_objects(cv::Mat &res,
                                 const std::vector<Object> &objs,
                                 const std::vector<std::string> &CLASS_NAMES,
                                 const std::vector<std::vector<unsigned int> > &COLORS) {
    // 遍历所有检测到的目标对象
    for (auto &obj: objs) {
        // 安全检查：确保类别索引在有效范围内
        size_t labelIndex = obj.label;
        if (labelIndex >= CLASS_NAMES.size() || labelIndex >= COLORS.size()) {
            continue; // 跳过无效类别的对象
        }
        // 根据类别选择对应的颜色
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);

        // 绘制边界框矩形
        cv::rectangle(res, obj.rect, color, 2);

        // 创建标签文本，包含类别名称和置信度百分比
        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        // 计算文本尺寸，用于绘制标签背景
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        // 计算标签位置，默认放在边界框左上角
        int x = (int) obj.rect.x;
        int y = (int) obj.rect.y + 1;

        // 确保标签不会超出图像边界
        if (y > res.rows) {
            y = res.rows;
        }

        // 绘制标签背景矩形（红色）
        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        // 绘制标签文本（白色）
        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}
#endif
