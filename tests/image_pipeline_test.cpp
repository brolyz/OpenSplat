#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "image_pipeline.hpp"
#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

static void check(bool ok, const std::string &message){
    if (!ok) throw std::runtime_error(message);
}

static std::vector<uint8_t> png(const cv::Mat &image){
    std::vector<uint8_t> bytes;
    check(cv::imencode(".png", image, bytes), "PNG encoding failed");
    return bytes;
}

static void cacheTest(){
    BlobCache cache;
    cache.put("held", {1, 2, 3});
    Blob held = cache.get("held");
    cache.put("held", {4});
    check(*held == std::vector<uint8_t>({1, 2, 3}), "Replacing a blob invalidated a reader");
    check(cache.residentBytes() == 1, "Replacement was counted twice");
    for (int i = 0; i < 256; i++) cache.put(std::to_string(i), std::vector<uint8_t>(1024, i));
    check(cache.residentBytes() == 256 * 1024 + 1, "Compressed entries were discarded");
    for (int i = 0; i < 256; i++) check(cache.get(std::to_string(i))->at(0) == static_cast<uint8_t>(i), "Blob changed");
    bool missing = false;
    try { cache.get("missing"); } catch (const std::runtime_error &) { missing = true; }
    check(missing, "A missing blob did not fail");
}

static void pipelineTest(const torch::Device &device){
    ImageStore store;
    std::vector<Camera> cameras(32);
    for (int i = 0; i < static_cast<int>(cameras.size()); i++){
        Camera &cam = cameras[i];
        cam.imageId = i;
        cam.width = 32;
        cam.height = 24;
        cam.hasMask = i % 2 == 0;
        for (int level : {1, 2}){
            cv::Mat bgr(24 / level, 32 / level, CV_8UC3, cv::Scalar(i, 40, 180));
            bgr(cv::Rect(0, 0, bgr.cols / 2, bgr.rows / 2)).setTo(cv::Scalar(0, 0, 0));
            store.cache.put(ImageStore::imageKey(i, level), png(bgr));
            if (cam.hasMask){
                cv::Mat mask(bgr.rows, bgr.cols, CV_8UC1, cv::Scalar(255));
                mask(cv::Rect(0, 0, mask.cols / 2, mask.rows / 2)).setTo(0);
                store.cache.put(ImageStore::maskKey(i, level), png(mask));
            }
        }
    }

#ifdef USE_CUDA
    // Initialize the device before measuring live tensor bytes, not the
    // allocator's reserved pool or CUDA context size.
    if (device.is_cuda()) torch::zeros({1}, torch::TensorOptions().device(device)).cpu();
    auto activeBytes = [](){
        return c10::cuda::CUDACachingAllocator::getDeviceStats(0)
            .allocated_bytes[0].current; // aggregate across allocation pools
    };
    const auto baseline = device.is_cuda() ? activeBytes() : 0;
#endif
    FrameLease retained;
    {
        ImagePipeline pipeline(store, device, 4);
        retained = pipeline.acquire(cameras[0], 1, true);
        torch::Tensor expected = retained->image.cpu().clone();
        // Exceed the pool, retune repeatedly and change pyramid level while an
        // older frame remains in use. Its storage must never be recycled.
        for (int step = 0; step < 192; step++){
            int index = step % cameras.size();
            int level = step < 96 ? 2 : 1;
            for (int k = 1; k <= pipeline.prefetchTarget(); k++){
                pipeline.request(cameras[(index + k) % cameras.size()], level);
            }
            auto frame = pipeline.acquire(cameras[index], level, true);
            check(frame->image.device() == device && frame->image.scalar_type() == torch::kU8, "Wrong image device/type");
            check(frame->image.size(0) == 24 / level && frame->image.size(1) == 32 / level, "Wrong pyramid dimensions");
            auto rgb = frame->image.cpu();
            check(rgb[-1][-1][0].item<uint8_t>() == 180 && rgb[-1][-1][2].item<uint8_t>() == index, "RGB pixels changed");
            check(frame->mask.defined() == cameras[index].hasMask, "Stale or missing mask");
            if (frame->mask.defined()){
                check(frame->mask.device() == device && frame->mask.scalar_type() == torch::kFloat32, "Wrong mask device/type");
                check(frame->mask[0][0].item<float>() == 0 && frame->mask[-1][-1].item<float>() == 1, "Mask normalization changed");
            }
            check(frame->edges.device() == device && frame->edges.scalar_type() == torch::kFloat32, "Wrong edge device/type");
            check(frame->edges.sum().item<float>() > 0, "Expected edges were lost");
            pipeline.noteTrainStep(step < 96 ? 1e-6 : 1.0);
        }
        check(torch::equal(retained->image.cpu(), expected), "A retained frame was overwritten");
        check(!pipeline.acquire(cameras[1], 1)->edges.defined(), "Edge maps leaked into another request");
        retained.reset();
#ifdef USE_CUDA
        if (device.is_cuda()) check(activeBytes() == baseline, "Prefetched frames remain allocated on the GPU");
#endif
        retained = pipeline.acquire(cameras[0], 1, true);
    }
    // A lease must also survive destruction of the worker pool.
    check(retained->image[-1][-1][0].item<uint8_t>() == 180, "Frame lifetime depends on pipeline lifetime");
    check(retained->mask[-1][-1].item<float>() == 1, "Mask storage was released too early");
    retained.reset();
    std::cout << "Frame content, masks, edges, eviction and ownership passed on " << device << std::endl;
}

static void decodeFailureTest(){
    ImageStore store;
    Camera camera;
    camera.imageId = 0;
    store.cache.put(ImageStore::imageKey(0, 1), {1, 2, 3});
    ImagePipeline pipeline(store, torch::kCPU, 2);
    bool failed = false;
    try { pipeline.acquire(camera, 1); } catch (const std::exception &) { failed = true; }
    check(failed, "A decode failure did not reach the caller");
}

int main(int argc, char **argv){
    try{
        bool cuda = argc > 1 && std::string(argv[1]) == "--cuda";
#ifndef USE_CUDA
        if (cuda){
            std::cout << "CUDA residency check requires GPU_RUNTIME=CUDA; skipped" << std::endl;
            return 77;
        }
#endif
        if (cuda && !torch::cuda::is_available()){
            std::cout << "CUDA unavailable; GPU residency check skipped" << std::endl;
            return 77;
        }
        cacheTest();
        pipelineTest(cuda ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU));
        decodeFailureTest();
        std::cout << "Compressed RAM cache and decode failure propagation passed" << std::endl;
        return 0;
    }catch (const std::exception &e){
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
