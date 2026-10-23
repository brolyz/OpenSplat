#ifndef IMAGE_STORE_H
#define IMAGE_STORE_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#include "input_data.hpp"

using Blob = std::shared_ptr<const std::vector<uint8_t>>;

// Thread-safe, RAM-only cache of compressed image bytes. Decoded frames are
// bounded separately by ImagePipeline; compressed blobs never spill to disk.
class BlobCache{
public:
    void put(const std::string &key, std::vector<uint8_t> &&bytes){
        Blob blob = std::make_shared<const std::vector<uint8_t>>(std::move(bytes));
        std::lock_guard<std::mutex> lock(m);
        Blob &entry = map[key];
        if (entry) resident -= entry->size();
        resident += blob->size();
        entry = std::move(blob);
    }
    Blob get(const std::string &key) const{
        std::lock_guard<std::mutex> lock(m);
        auto it = map.find(key);
        if (it == map.end()) throw std::runtime_error("Image blob not found: " + key);
        return it->second;
    }
    uint64_t residentBytes() const{
        std::lock_guard<std::mutex> lock(m);
        return resident;
    }

private:
    mutable std::mutex m;
    std::unordered_map<std::string, Blob> map;
    uint64_t resident = 0;
};

// Preprocesses every camera image once (resize, undistort, downscale levels)
// into compressed blobs and updates the camera intrinsics and dimensions
struct ImageStore{
    void prepare(std::vector<Camera> &cameras, float downscaleFactor, int numDownscales);
    void prepareCamera(Camera &cam, float downscaleFactor);

    static std::string imageKey(int imageId, int level){
        return "i:" + std::to_string(imageId) + ":" + std::to_string(level);
    }
    static std::string maskKey(int imageId, int level){
        return "m:" + std::to_string(imageId) + ":" + std::to_string(level);
    }

    BlobCache cache;
    int numLevels = 1;
};

#endif
