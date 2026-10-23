#include <algorithm>
#include <chrono>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "image_pipeline.hpp"

namespace {

const int MIN_PREFETCH = 2;
const int MAX_PREFETCH = 12;
const int RETUNE_EVERY = 64;

double seconds(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b){
    return std::chrono::duration<double>(b - a).count();
}

}

struct ImagePipeline::Slot{
    enum State { FREE, DECODING, READY };
    State state = FREE;
    int imageId = -1;
    int level = 0;
    uint64_t lastUse = 0;
    std::shared_ptr<Frame> frame;
};

ImagePipeline::ImagePipeline(ImageStore &store, const torch::Device &device, int numWorkers)
    : store(store), device(device){
    target = MIN_PREFETCH;
    capacity = target.load() + 2;
    numWorkers = std::clamp(numWorkers, 1, capacity);
    for (int i = 0; i < numWorkers; i++){
        workers.emplace_back([this](){ workerLoop(); });
    }
}

ImagePipeline::~ImagePipeline(){
    {
        std::lock_guard<std::mutex> lock(m);
        stopping = true;
    }
    queueCv.notify_all();
    readyCv.notify_all();
    for (std::thread &t : workers) t.join();
}

void ImagePipeline::request(const Camera &cam, int level){
    std::lock_guard<std::mutex> lock(m);
    if (failure) std::rethrow_exception(failure);
    for (const auto &s : slots){
        if (s->state != Slot::FREE && s->imageId == cam.imageId && s->level == level) return;
    }
    for (const Request &r : queue){
        if (r.imageId == cam.imageId && r.level == level) return;
    }
    queue.push_back({ cam.imageId, level, cam.width, cam.height, cam.hasMask });
    queueCv.notify_one();
}

// Picks a slot for a new decode: a free one, a new one while under capacity,
// or the least recently used frame. Callers own their buffers independently.
ImagePipeline::Slot *ImagePipeline::reserveSlotLocked(){
    for (auto &s : slots){
        if (s->state == Slot::FREE) return s.get();
    }
    if (static_cast<int>(slots.size()) < capacity){
        slots.push_back(std::make_unique<Slot>());
        return slots.back().get();
    }
    Slot *victim = nullptr;
    for (auto &s : slots){
        if (s->state == Slot::READY && (!victim || s->lastUse < victim->lastUse)) victim = s.get();
    }
    return victim;
}

void ImagePipeline::workerLoop(){
    while (true){
        Request r;
        Slot *slot = nullptr;
        {
            std::unique_lock<std::mutex> lock(m);
            queueCv.wait(lock, [&](){ return stopping || failure || !queue.empty(); });
            if (stopping || failure) return;
            r = queue.front();
            queue.pop_front();

            bool resident = false;
            for (auto &s : slots){
                if (s->state != Slot::FREE && s->imageId == r.imageId && s->level == r.level) resident = true;
            }
            if (resident) continue;

            while (!(slot = reserveSlotLocked())){
                queueCv.wait(lock);
                if (stopping || failure) return;
            }
            slot->state = Slot::DECODING;
            slot->imageId = r.imageId;
            slot->level = r.level;
            slot->frame.reset();
        }

        auto t0 = std::chrono::steady_clock::now();
        std::exception_ptr error;
        try{
            decodeInto(*slot, r);
        }catch (...){
            error = std::current_exception();
        }
        double dt = seconds(t0, std::chrono::steady_clock::now());

        {
            std::lock_guard<std::mutex> lock(m);
            if (error){
                failure = error;
                slot->state = Slot::FREE;
                slot->imageId = -1;
            }else{
                slot->state = Slot::READY;
                slot->lastUse = ++useCounter;
                decodeEma = decodeEma == 0.0 ? dt : 0.9 * decodeEma + 0.1 * dt;
            }
        }
        readyCv.notify_all();
        queueCv.notify_all();
    }
}

void ImagePipeline::decodeInto(Slot &slot, const Request &r){
    Blob bytes = store.cache.get(ImageStore::imageKey(r.imageId, r.level));
    cv::Mat bgr = cv::imdecode(cv::Mat(1, static_cast<int>(bytes->size()), CV_8UC1, const_cast<uint8_t *>(bytes->data())),
                               cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("Cannot decode image " + std::to_string(r.imageId));
    const int H = bgr.rows;
    const int W = bgr.cols;

    const bool pinned = device.is_cuda();
    auto hostOpts = torch::TensorOptions().dtype(torch::kU8).pinned_memory(pinned);
    auto frame = std::make_shared<Frame>();
    frame->imageId = r.imageId;
    frame->level = r.level;
    frame->image = torch::empty({H, W, 3}, hostOpts);
    cv::Mat rgb(H, W, CV_8UC3, frame->image.data_ptr<uint8_t>());
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    if (r.hasMask){
        Blob maskBytes = store.cache.get(ImageStore::maskKey(r.imageId, r.level));
        cv::Mat mask = cv::imdecode(cv::Mat(1, static_cast<int>(maskBytes->size()), CV_8UC1, const_cast<uint8_t *>(maskBytes->data())),
                                    cv::IMREAD_GRAYSCALE);
        if (mask.empty() || mask.rows != H || mask.cols != W){
            throw std::runtime_error("Cannot decode mask for image " + std::to_string(r.imageId));
        }
        frame->mask = torch::empty({H, W}, hostOpts);
        cv::Mat dst(H, W, CV_8UC1, frame->mask.data_ptr<uint8_t>());
        mask.copyTo(dst);
    }
    slot.frame = std::move(frame);
}

FrameLease ImagePipeline::acquire(const Camera &cam, int level, bool wantEdges){
    std::unique_lock<std::mutex> lock(m);
    auto t0 = std::chrono::steady_clock::now();
    bool waited = false;
    Slot *slot = nullptr;
    while (true){
        if (failure) std::rethrow_exception(failure);
        if (stopping) throw std::runtime_error("Image pipeline stopped");
        slot = nullptr;
        bool decoding = false;
        for (auto &s : slots){
            if (s->state == Slot::FREE || s->imageId != cam.imageId || s->level != level) continue;
            if (s->state == Slot::READY){ slot = s.get(); break; }
            decoding = true;
        }
        if (slot) break;

        if (!decoding){
            // A demanded frame takes precedence over speculative prefetch.
            auto queued = std::find_if(queue.begin(), queue.end(), [&](const Request &r){
                return r.imageId == cam.imageId && r.level == level;
            });
            if (queued != queue.end()) queue.erase(queued);
            queue.push_front({ cam.imageId, level, cam.width, cam.height, cam.hasMask });
            queueCv.notify_all();
        }
        waited = true;
        readyCv.wait(lock);
    }

    if (waited){
        waitAccum += seconds(t0, std::chrono::steady_clock::now());
        windowWaits++;
    }
    slot->lastUse = ++useCounter;
    // Keep the decoded CPU buffers alive if a worker replaces this slot.
    std::shared_ptr<Frame> host = slot->frame;
    lock.unlock();

    auto frame = std::make_shared<Frame>();
    frame->imageId = host->imageId;
    frame->level = host->level;
    // Blocking copies finish before the host buffers can be recycled. Workers
    // never upload to the GPU, so prefetch cannot consume training VRAM.
    frame->image = host->image.to(device);
    if (host->mask.defined()){
        frame->mask = host->mask.to(device, torch::kFloat32).div_(255.0f);
    }
    if (wantEdges){
        cv::Mat rgb(host->image.size(0), host->image.size(1), CV_8UC3, host->image.data_ptr<uint8_t>());
        cv::Mat gray, edges;
        cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
        cv::Canny(gray, edges, 50, 150);
        // Conversion to float owns its storage, including on CPU.
        frame->edges = torch::from_blob(edges.data, {edges.rows, edges.cols}, torch::kU8)
                           .to(device, torch::kFloat32).div_(255.0f);
    }
    return frame;
}

void ImagePipeline::noteTrainStep(double sec){
    std::lock_guard<std::mutex> lock(m);
    double compute = (std::max)(sec - waitAccum, 1e-6);
    waitAccum = 0.0;
    trainEma = trainEma == 0.0 ? compute : 0.9 * trainEma + 0.1 * compute;
    stepsTotal++;
    // Adapt quickly at the start, then settle into longer windows
    const int retuneEvery = stepsTotal < 256 ? 16 : RETUNE_EVERY;
    if (++stepsSinceRetune >= retuneEvery){
        stepsSinceRetune = 0;
        retuneLocked(windowWaits > 0);
        windowWaits = 0;
    }
}

// Prefetch depth follows the decode/train latency ratio: it grows whenever the
// loop had to wait for a frame and shrinks only after several quiet windows
void ImagePipeline::retuneLocked(bool waited){
    int base = MIN_PREFETCH;
    if (trainEma > 0.0 && decodeEma > 0.0){
        base = std::clamp(static_cast<int>(std::ceil(decodeEma / trainEma)) + 2, MIN_PREFETCH, MAX_PREFETCH);
    }
    int t = target.load();
    if (waited){
        t = (std::max)(base, t + 1);
        quietWindows = 0;
    }else if (++quietWindows >= 4){
        t = base;
        quietWindows = 0;
    }else{
        t = (std::max)(base, t);
    }
    t = std::clamp(t, MIN_PREFETCH, MAX_PREFETCH);
    target = t;

    // Bound decoded CPU images to the lookahead window and two spare slots,
    // rather than growing a device cache to fill the available VRAM.
    capacity = t + 2;
    // READY frames are immutable and held independently by acquire() callers;
    // DECODING slots must stay alive until their worker finishes.
    for (auto it = slots.begin(); static_cast<int>(slots.size()) > capacity && it != slots.end();){
        if ((*it)->state == Slot::READY) it = slots.erase(it);
        else ++it;
    }
}
