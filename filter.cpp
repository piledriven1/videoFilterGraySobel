#include "filter.hpp"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                    <<" <VIDEO_FILE> <FILTER_TYPE>" << std::endl
                    << "  FILTER_TYPE: plain | gray | sobel" <<std::endl;
        return 1;
    }

    const std::string videoPath = argv[1];
    const std::string filterStr = argv[2];

    // Parse and validate filter type
    if (filterStr != "plain" && filterStr != "gray" && filterStr != "sobel") {
        std::cerr << "Error: Invalid filter type '" << filterStr << std::endl
                  << "Valid options: plain | gray | sobel" << std::endl;
        return 1;
    }

    const FilterType filter = parseFilterType(filterStr);

    // Open video
    cv::VideoCapture video(videoPath);
    if (!video.isOpened()) {
        std::cerr << "Error: Could not open video file "
                    << videoPath << std::endl;
        return 1;
    }

    // Get video properties
    const int fps = std::max(1, static_cast<int>(video.get(cv::CAP_PROP_FPS)));
    const int totalFrames = video.get(cv::CAP_PROP_FRAME_COUNT);;
    const int frameWidth = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    const int frameHeight = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "Video: " << frameWidth << "x" << frameHeight  << " @ " << 
            fps << " fps, processing " << totalFrames << " frames" << std::endl;

    // Create window
    const std::string windowName = getWindowName(videoPath, filter);
    cv::namedWindow(windowName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(windowName, 750, 500);
    bool windowOpen = true;

    // Initialize thread pool (created once, reused for all frames)
    ThreadPool threadPool;
    ThreadArgs threadArgs[NUMTHREADS];

    // Double buffering: process frame N while displaying frame N-1
    FrameBuffer buffers[2];
    int currentBuffer = 0;
    bool firstFrame = true;

    // Pre-allocate buffers after reading first frame to get dimensions
    cv::Mat firstFrameMat;
    if (!video.read(firstFrameMat) || firstFrameMat.empty()) {
        std::cerr << "Error: Could not read first frame" << std::endl;
        return 1;
    }

    buffers[0].plain = firstFrameMat.clone();
    buffers[0].allocate(frameHeight, frameWidth, filter);
    buffers[1].allocate(frameHeight, frameWidth, filter);

    // Process frames
    for (int frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
        FrameBuffer& buf = buffers[currentBuffer];
        FrameBuffer& prevBuf = buffers[1 - currentBuffer];

        // Read next frame (except for first which we already read)
        if (frameIdx > 0) {
            if (!video.read(buf.plain) || buf.plain.empty()) {
                std::cerr << "Warning: End of video at frame "
                        << frameIdx << std::endl;
                break;
            }
        }

        // Process current frame based on filter type
        if (filter != FilterType::PLAIN) {
            // Set up thread arguments for grayscale
            for (int t = 0; t < NUMTHREADS; ++t) {
                threadArgs[t] = {
                    &buf.plain, &buf.gray, frameHeight,
                    frameWidth, t, NUMTHREADS
                };
            }

            // Dispatch grayscale conversion
            threadPool.dispatch(grayThread, threadArgs);

            // If sobel, also apply sobel filter
            if (filter == FilterType::SOBEL) {
                for (int t = 0; t < NUMTHREADS; ++t) {
                    threadArgs[t] = {
                        &buf.gray, &buf.sobel, frameHeight,
                        frameWidth, t, NUMTHREADS};
                }
                threadPool.dispatch(sobelThread, threadArgs);
            }
        }

        // Display previous frame (double buffering) or current if first frame
        if (!firstFrame) {
            const cv::Mat& displayFrame = \
                                (filter == FilterType::PLAIN) ? prevBuf.plain :
                                (filter == FilterType::GRAY)  ? prevBuf.gray :
                                prevBuf.sobel;
            cv::imshow(windowName, displayFrame);
        } else {
            firstFrame = false;
        }

        // Wait for frame timing
        if (cv::waitKey(1000 / fps) == 27) {  // ESC to exit early
            break;
        }

        // Close window if X is pressed
        if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1) {
            windowOpen = false;
            break;
        }

        // Swap buffers
        currentBuffer = 1 - currentBuffer;
    }

    if(windowOpen) {
        // Display last frame
        const FrameBuffer& lastBuf = buffers[1 - currentBuffer];
        const cv::Mat& lastFrame = \
                                (filter == FilterType::PLAIN) ? lastBuf.plain :
                                (filter == FilterType::GRAY)  ? lastBuf.gray :
                                lastBuf.sobel;
        cv::imshow(windowName, lastFrame);
        cv::waitKey(0);  // Wait for key press before closing
    }

    // Cleanup
    video.release();
    buffers[0].release();
    buffers[1].release();
    cv::destroyAllWindows();

    return 0;
}

ThreadPool::ThreadPool() {
    pthread_mutex_init(&mutex, nullptr);
    pthread_cond_init(&workReady, nullptr);
    pthread_cond_init(&workDone, nullptr);

    currentWorkFunc = nullptr;

    // Create worker threads
    for (int i = 0; i < NUMTHREADS; ++i) {
        contexts[i] = {this, i};
        pthread_create(&threads[i], nullptr, workerLoop, &contexts[i]);
    }
}

ThreadPool::~ThreadPool() {
    // Signal shutdown
    pthread_mutex_lock(&mutex);
    shutdown = true;
    pthread_cond_broadcast(&workReady);
    pthread_mutex_unlock(&mutex);

    // Wait for all threads to finish
    for (int i = 0; i < NUMTHREADS; ++i) {
        pthread_join(threads[i], nullptr);
    }

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&workReady);
    pthread_cond_destroy(&workDone);
}

void* ThreadPool::workerLoop(void* arg) {
    WorkerContext* ctx = static_cast<WorkerContext*>(arg);
    ThreadPool* pool = ctx->pool;
    int tid = ctx->tid;
    int localGeneration = 0;

    while (true) {
        pthread_mutex_lock(&pool->mutex);

        // Wait for new work or shutdown
        while (!pool->shutdown && pool->workGeneration.load() == localGeneration) {
            pthread_cond_wait(&pool->workReady, &pool->mutex);
        }

        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }

        // Capture work function and args
        void* (*workFunc)(void*) = pool->currentWorkFunc;
        ThreadArgs args = pool->threadArgs[tid];
        localGeneration = pool->workGeneration.load();

        pthread_mutex_unlock(&pool->mutex);

        // Execute work outside of lock
        if (workFunc) {
            workFunc(&args);
        }

        // Signal completion
        if (pool->activeWorkers.fetch_sub(1) == 1) {
            pthread_mutex_lock(&pool->mutex);
            pthread_cond_signal(&pool->workDone);
            pthread_mutex_unlock(&pool->mutex);
        }
    }

    return nullptr;
}

void ThreadPool::dispatch(void* (*workFunc)(void*), ThreadArgs* args) {
    pthread_mutex_lock(&mutex);

    // Set up work
    currentWorkFunc = workFunc;
    for (int i = 0; i < NUMTHREADS; ++i) {
        threadArgs[i] = args[i];
    }

    activeWorkers.store(NUMTHREADS);
    workGeneration.fetch_add(1);

    // Wake all workers
    pthread_cond_broadcast(&workReady);

    // Wait for completion
    while (activeWorkers.load() > 0) {
        pthread_cond_wait(&workDone, &mutex);
    }

    pthread_mutex_unlock(&mutex);
}

void FrameBuffer::allocate(int rows, int cols, FilterType filter) {
    if (filter == FilterType::GRAY || filter == FilterType::SOBEL) {
        gray.create(rows, cols, CV_8UC1);
        if (filter == FilterType::SOBEL) {
            sobel.create(rows, cols, CV_8UC1);
            // Zero out borders for sobel (they won't be computed)
            sobel.row(0).setTo(0);
            sobel.row(rows - 1).setTo(0);
            sobel.col(0).setTo(0);
            sobel.col(cols - 1).setTo(0);
        }
    }
}

void FrameBuffer::release() {
    plain.release();
    gray.release();
    sobel.release();
}

FilterType parseFilterType(const std::string& filter) {
    if (filter == "gray") return FilterType::GRAY;
    if (filter == "sobel") return FilterType::SOBEL;
    return FilterType::PLAIN;
}

std::string getWindowName(const std::string& videoName, FilterType filter) {
    switch (filter) {
        case FilterType::GRAY:  return videoName + "_grayscale";
        case FilterType::SOBEL: return videoName + "_sobel";
        default:                return videoName;
    }
}

void* grayThread(void* args) {
    ThreadArgs* arg = static_cast<ThreadArgs*>(args);

    int start = (arg->tid * arg->height) / arg->totalThreads;
    int end = ((arg->tid + 1) * arg->height) / arg->totalThreads;

    grayscale(*arg->src, *arg->dst, start, end);
    return nullptr;
}

void* sobelThread(void* args) {
    ThreadArgs* arg = static_cast<ThreadArgs*>(args);

    int start = (arg->tid * arg->height) / arg->totalThreads;
    int end = ((arg->tid + 1) * arg->height) / arg->totalThreads;

    sobelFilter(*arg->src, *arg->dst, start, end);
    return nullptr;
}

void grayscale(const cv::Mat& frame, cv::Mat& dest, int start, int end) {
    CV_Assert(frame.type() == CV_8UC3);
    
    // BT.709 coefficients scaled to 8-bit (sum = 255)
    // Y = 0.2126R + 0.7152G + 0.0722B
    constexpr uint16_t wB = 18;   // 0.0722 * 256 ≈ 18
    constexpr uint16_t wG = 183;  // 0.7152 * 256 ≈ 183
    constexpr uint16_t wR = 54;   // 0.2126 * 256 ≈ 54
    
    const int cols = frame.cols;
    
    for (int y = start; y < end; ++y) {
        const uint8_t* __restrict srow = frame.ptr<uint8_t>(y);
        uint8_t* __restrict dst = dest.ptr<uint8_t>(y);

        // Prefetch next row
        if (y + 1 < end) {
            __builtin_prefetch(frame.ptr<uint8_t>(y + 1), 0, 1);
        }

        int x = 0;

        // Process 32 pixels per iteration (unrolled)
        for (; x <= cols - 32; x += 32) {
            // Load 32 BGR pixels (96 bytes) as two batches
            uint8x16x3_t bgr0 = vld3q_u8(srow + 3 * x);
            uint8x16x3_t bgr1 = vld3q_u8(srow + 3 * (x + 16));

            // First batch of 16 pixels
            uint16x8_t b0_lo = vmovl_u8(vget_low_u8(bgr0.val[0]));
            uint16x8_t b0_hi = vmovl_u8(vget_high_u8(bgr0.val[0]));
            uint16x8_t g0_lo = vmovl_u8(vget_low_u8(bgr0.val[1]));
            uint16x8_t g0_hi = vmovl_u8(vget_high_u8(bgr0.val[1]));
            uint16x8_t r0_lo = vmovl_u8(vget_low_u8(bgr0.val[2]));
            uint16x8_t r0_hi = vmovl_u8(vget_high_u8(bgr0.val[2]));

            uint16x8_t y0_lo = vmulq_n_u16(r0_lo, wR);
            y0_lo = vmlaq_n_u16(y0_lo, g0_lo, wG);
            y0_lo = vmlaq_n_u16(y0_lo, b0_lo, wB);

            uint16x8_t y0_hi = vmulq_n_u16(r0_hi, wR);
            y0_hi = vmlaq_n_u16(y0_hi, g0_hi, wG);
            y0_hi = vmlaq_n_u16(y0_hi, b0_hi, wB);

            y0_lo = vshrq_n_u16(y0_lo, 8);
            y0_hi = vshrq_n_u16(y0_hi, 8);
            uint8x16_t out0 = vcombine_u8(vmovn_u16(y0_lo), vmovn_u16(y0_hi));

            // Second batch of 16 pixels
            uint16x8_t b1_lo = vmovl_u8(vget_low_u8(bgr1.val[0]));
            uint16x8_t b1_hi = vmovl_u8(vget_high_u8(bgr1.val[0]));
            uint16x8_t g1_lo = vmovl_u8(vget_low_u8(bgr1.val[1]));
            uint16x8_t g1_hi = vmovl_u8(vget_high_u8(bgr1.val[1]));
            uint16x8_t r1_lo = vmovl_u8(vget_low_u8(bgr1.val[2]));
            uint16x8_t r1_hi = vmovl_u8(vget_high_u8(bgr1.val[2]));

            uint16x8_t y1_lo = vmulq_n_u16(r1_lo, wR);
            y1_lo = vmlaq_n_u16(y1_lo, g1_lo, wG);
            y1_lo = vmlaq_n_u16(y1_lo, b1_lo, wB);

            uint16x8_t y1_hi = vmulq_n_u16(r1_hi, wR);
            y1_hi = vmlaq_n_u16(y1_hi, g1_hi, wG);
            y1_hi = vmlaq_n_u16(y1_hi, b1_hi, wB);

            y1_lo = vshrq_n_u16(y1_lo, 8);
            y1_hi = vshrq_n_u16(y1_hi, 8);
            uint8x16_t out1 = vcombine_u8(vmovn_u16(y1_lo), vmovn_u16(y1_hi));

            // Store 32 grayscale pixels
            vst1q_u8(dst + x, out0);
            vst1q_u8(dst + x + 16, out1);
        }

        // Process 16 pixels
        for (; x <= cols - 16; x += 16) {
            uint8x16x3_t bgr = vld3q_u8(srow + 3 * x);

            uint16x8_t b_lo = vmovl_u8(vget_low_u8(bgr.val[0]));
            uint16x8_t b_hi = vmovl_u8(vget_high_u8(bgr.val[0]));
            uint16x8_t g_lo = vmovl_u8(vget_low_u8(bgr.val[1]));
            uint16x8_t g_hi = vmovl_u8(vget_high_u8(bgr.val[1]));
            uint16x8_t r_lo = vmovl_u8(vget_low_u8(bgr.val[2]));
            uint16x8_t r_hi = vmovl_u8(vget_high_u8(bgr.val[2]));

            uint16x8_t y_lo = vmulq_n_u16(r_lo, wR);
            y_lo = vmlaq_n_u16(y_lo, g_lo, wG);
            y_lo = vmlaq_n_u16(y_lo, b_lo, wB);

            uint16x8_t y_hi = vmulq_n_u16(r_hi, wR);
            y_hi = vmlaq_n_u16(y_hi, g_hi, wG);
            y_hi = vmlaq_n_u16(y_hi, b_hi, wB);

            y_lo = vshrq_n_u16(y_lo, 8);
            y_hi = vshrq_n_u16(y_hi, 8);
            uint8x16_t out = vcombine_u8(vmovn_u16(y_lo), vmovn_u16(y_hi));

            vst1q_u8(dst + x, out);
        }

        // Scalar tail for any remaining pixels
        for (; x < cols; ++x) {
            const uint8_t B = srow[3 * x + 0];
            const uint8_t G = srow[3 * x + 1];
            const uint8_t R = srow[3 * x + 2];
            dst[x] = static_cast<uint8_t>((wB * B + wG * G + wR * R) >> 8);
        }
    }
}

void sobelFilter(const cv::Mat& frame, cv::Mat& dest, int start, int end) {
    CV_Assert(frame.type() == CV_8UC1);

    const int rows = frame.rows;
    const int cols = frame.cols;

    // Clamp to interior rows (skip top and bottom borders)
    const int ys = std::max(start, 1);
    const int ye = std::min(end, rows - 1);

    for (int y = ys; y < ye; ++y) {
        const uint8_t* __restrict t = frame.ptr<uint8_t>(y - 1);  // Top row
        const uint8_t* __restrict m = frame.ptr<uint8_t>(y);      // Middle row
        const uint8_t* __restrict b = frame.ptr<uint8_t>(y + 1);  // Bottom row
        uint8_t* __restrict d = dest.ptr<uint8_t>(y);

        // Prefetch next rows for better cache performance
        if (y + 2 < rows) {
            __builtin_prefetch(frame.ptr<uint8_t>(y + 2), 0, 1);
        }

        // Zero left border
        d[0] = 0;

        int x = 1;

        // NEON: Process 16 pixels per iteration
        // Need x-1 to x+16, so last valid start is cols-17
        for (; x <= cols - 17; x += 16) {
            // Load 3x3 neighborhood data
            uint8x16_t tL8 = vld1q_u8(t + (x - 1));
            uint8x16_t tC8 = vld1q_u8(t + x);
            uint8x16_t tR8 = vld1q_u8(t + (x + 1));

            uint8x16_t mL8 = vld1q_u8(m + (x - 1));
            uint8x16_t mR8 = vld1q_u8(m + (x + 1));

            uint8x16_t bL8 = vld1q_u8(b + (x - 1));
            uint8x16_t bC8 = vld1q_u8(b + x);
            uint8x16_t bR8 = vld1q_u8(b + (x + 1));

            // Helper lambdas for widening to signed 16-bit
            auto widen_lo = [](uint8x16_t v) { 
                return vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))); 
            };
            auto widen_hi = [](uint8x16_t v) { 
                return vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))); 
            };

            // Widen all values to 16-bit signed (low halves)
            int16x8_t tL_lo = widen_lo(tL8), tC_lo = widen_lo(tC8), tR_lo = widen_lo(tR8);
            int16x8_t mL_lo = widen_lo(mL8), mR_lo = widen_lo(mR8);
            int16x8_t bL_lo = widen_lo(bL8), bC_lo = widen_lo(bC8), bR_lo = widen_lo(bR8);

            // Widen all values to 16-bit signed (high halves)
            int16x8_t tL_hi = widen_hi(tL8), tC_hi = widen_hi(tC8), tR_hi = widen_hi(tR8);
            int16x8_t mL_hi = widen_hi(mL8), mR_hi = widen_hi(mR8);
            int16x8_t bL_hi = widen_hi(bL8), bC_hi = widen_hi(bC8), bR_hi = widen_hi(bR8);

            // Sobel Gx kernel: [-1  0  1]
            //                  [-2  0  2]
            //                  [-1  0  1]
            auto compute_gx = [](int16x8_t tL, int16x8_t tR, 
                                 int16x8_t mL, int16x8_t mR,
                                 int16x8_t bL, int16x8_t bR) {
                int16x8_t gx = vsubq_s16(tR, tL);                    // tR - tL
                gx = vaddq_s16(gx, vshlq_n_s16(vsubq_s16(mR, mL), 1)); // + 2*(mR - mL)
                gx = vaddq_s16(gx, vsubq_s16(bR, bL));               // + (bR - bL)
                return gx;
            };

            // Sobel Gy kernel: [-1 -2 -1]
            //                  [ 0  0  0]
            //                  [ 1  2  1]
            auto compute_gy = [](int16x8_t tL, int16x8_t tC, int16x8_t tR,
                                 int16x8_t bL, int16x8_t bC, int16x8_t bR) {
                int16x8_t gy = vsubq_s16(bL, tL);                    // bL - tL
                gy = vaddq_s16(gy, vshlq_n_s16(vsubq_s16(bC, tC), 1)); // + 2*(bC - tC)
                gy = vaddq_s16(gy, vsubq_s16(bR, tR));               // + (bR - tR)
                return gy;
            };

            // Compute gradients for low 8 pixels
            int16x8_t gx_lo = compute_gx(tL_lo, tR_lo, mL_lo, mR_lo, bL_lo, bR_lo);
            int16x8_t gy_lo = compute_gy(tL_lo, tC_lo, tR_lo, bL_lo, bC_lo, bR_lo);

            // Compute gradients for high 8 pixels
            int16x8_t gx_hi = compute_gx(tL_hi, tR_hi, mL_hi, mR_hi, bL_hi, bR_hi);
            int16x8_t gy_hi = compute_gy(tL_hi, tC_hi, tR_hi, bL_hi, bC_hi, bR_hi);

            // Magnitude approximation: |Gx| + |Gy| (faster than sqrt(Gx² + Gy²))
            uint16x8_t mag_lo = vaddq_u16(
                vreinterpretq_u16_s16(vabsq_s16(gx_lo)),
                vreinterpretq_u16_s16(vabsq_s16(gy_lo))
            );

            uint16x8_t mag_hi = vaddq_u16(
                vreinterpretq_u16_s16(vabsq_s16(gx_hi)),
                vreinterpretq_u16_s16(vabsq_s16(gy_hi))
            );

            // Saturate to 8-bit and store
            uint8x16_t out = vcombine_u8(vqmovn_u16(mag_lo), vqmovn_u16(mag_hi));
            vst1q_u8(d + x, out);
        }

        // Scalar tail for remaining interior pixels
        for (; x < cols - 1; ++x) {
            // Gx = [-1  0  1] * neighborhood
            //      [-2  0  2]
            //      [-1  0  1]
            int gx = -t[x-1] + t[x+1]
                   - 2*m[x-1] + 2*m[x+1]
                   - b[x-1] + b[x+1];

            // Gy = [-1 -2 -1] * neighborhood
            //      [ 0  0  0]
            //      [ 1  2  1]
            int gy = -t[x-1] - 2*t[x] - t[x+1]
                   + b[x-1] + 2*b[x] + b[x+1];

            // Magnitude with saturation
            int mag = std::abs(gx) + std::abs(gy);
            d[x] = static_cast<uint8_t>(std::min(mag, 255));
        }

        // Zero right border
        d[cols - 1] = 0;
    }
}
