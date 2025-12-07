#ifndef FILTER_HPP
#define FILTER_HPP

#include <iostream>
#include <string>

// For threads and parallelism
#include <pthread.h>
#include <condition_variable>
#include <mutex>
#include <atomic>

// For neon
#include <arm_neon.h>

// For opencv
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/parallel/parallel_backend.hpp>

// Macros
constexpr int NUMTHREADS = 4;

enum class FilterType {
    PLAIN,
    GRAY,
    SOBEL
};

// Struct
struct ThreadArgs {
    const cv::Mat* src;                 // Input
    cv::Mat* dst;                       // Output (gray, sobel)
    int height, width;                  // Frame dimensions: Height, width
    int tid, totalThreads;              // Thread id and total threads
};

// Create a class that pools the threads to reusing threads across frames
class ThreadPool {
public:
    ThreadPool();
    ~ThreadPool();

    // Prevent copying
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Dispatch work to all threads and wait for completion
    void dispatch(void* (*workFunc)(void*), ThreadArgs* args);

private:
    static void* workerLoop(void* arg);

    struct WorkerContext {
        ThreadPool* pool;
        int tid;
    };

    pthread_t threads[NUMTHREADS];
    WorkerContext contexts[NUMTHREADS];
    ThreadArgs threadArgs[NUMTHREADS];

    void* (*currentWorkFunc)(void*);

    pthread_mutex_t mutex;
    pthread_cond_t workReady;
    pthread_cond_t workDone;

    std::atomic<int> activeWorkers{0};
    std::atomic<int> workGeneration{0};
    int lastSeenGeneration[NUMTHREADS] = {0};
    bool shutdown = false;
};

// Double buffer for frame processing
class FrameBuffer {
public:
    cv::Mat plain;
    cv::Mat gray;
    cv::Mat sobel;
    
    void allocate(int rows, int cols, FilterType filter);
    void release();
};

FilterType parseFilterType(const std::string& filter);
std::string getWindowName(const std::string& videoName, FilterType filter);

// Thread function prototypes
void *grayThread(void *args);
void *sobelThread(void *args);

// Function prototypes
void grayscale(const cv::Mat &frame, cv::Mat &dest, int start, int end);
void sobelFilter(const cv::Mat &frame, cv::Mat &dest, int start, int end);

#endif
