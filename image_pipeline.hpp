#ifndef IMAGE_PIPELINE_H
#define IMAGE_PIPELINE_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <torch/torch.h>
#include "image_store.hpp"

// A frame owned by a caller on the training device. The prefetch pool keeps
// only CPU uint8 images/masks; device tensors are created by acquire().
struct Frame{
    torch::Tensor image; // uint8 [H,W,3]
    torch::Tensor mask;  // CPU pool: uint8; acquired: float [H,W] in {0,1}
    torch::Tensor edges; // float [H,W] in {0,1}, or undefined
    int imageId = -1;
    int level = 1;
};
// Owns the tensors independently of the prefetch pool and pipeline lifetime.
using FrameLease = std::shared_ptr<const Frame>;

class ImagePipeline{
public:
    ImagePipeline(ImageStore &store, const torch::Device &device, int numWorkers);
    ~ImagePipeline();

    void request(const Camera &cam, int level);
    FrameLease acquire(const Camera &cam, int level, bool wantEdges = false);
    void noteTrainStep(double seconds);
    int prefetchTarget() const { return target.load(); }

private:
    struct Slot;
    struct Request{ int imageId; int level; int width; int height; bool hasMask; };

    void workerLoop();
    Slot *reserveSlotLocked();
    void decodeInto(Slot &slot, const Request &r);
    void retuneLocked(bool waited);

    ImageStore &store;
    torch::Device device;
    std::vector<std::unique_ptr<Slot>> slots;
    std::deque<Request> queue;
    std::vector<std::thread> workers;
    std::mutex m;
    std::condition_variable queueCv;
    std::condition_variable readyCv;
    bool stopping = false;
    std::exception_ptr failure;

    std::atomic<int> target{2};
    int capacity = 4;
    double decodeEma = 0.0;
    double trainEma = 0.0;   // step time excluding time spent waiting for frames
    double waitAccum = 0.0;  // wait time since the last noteTrainStep
    int windowWaits = 0;     // acquire() calls that had to wait in this window
    int quietWindows = 0;
    int stepsTotal = 0;
    int stepsSinceRetune = 0;
    uint64_t useCounter = 0;
};

#endif
