#include <algorithm>
#include <atomic>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "image_store.hpp"
#include "sysinfo.hpp"
#include "undistort.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;

static bool isJpegOrPng(const std::string &path){
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
}

static std::vector<uint8_t> readFileBytes(const std::string &path){
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    std::vector<uint8_t> bytes(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    return bytes;
}

static std::vector<uint8_t> encodeJpeg(const cv::Mat &bgr){
    std::vector<uint8_t> out;
    cv::imencode(".jpg", bgr, out, { cv::IMWRITE_JPEG_QUALITY, 95 });
    return out;
}

static std::vector<uint8_t> encodePng(const cv::Mat &gray){
    std::vector<uint8_t> out;
    cv::imencode(".png", gray, out, { cv::IMWRITE_PNG_COMPRESSION, 1 });
    return out;
}

void ImageStore::prepare(std::vector<Camera> &cameras, float downscaleFactor, int numDownscales){
    numLevels = numDownscales + 1;
    for (size_t i = 0; i < cameras.size(); i++) cameras[i].imageId = static_cast<int>(i);

    // Bound the number of images decoded at once by the available RAM
    uint64_t maxPixels = 1;
    for (const Camera &cam : cameras){
        maxPixels = (std::max)(maxPixels, static_cast<uint64_t>(cam.width) * cam.height);
    }
    const size_t chunk = static_cast<size_t>(std::clamp<uint64_t>(availableRamBytes() / 2 / (maxPixels * 17), 1, 32));

    const size_t total = cameras.size();
    std::atomic<size_t> done{0};
    std::mutex logMutex;
    std::string firstError;

    for (size_t start = 0; start < total; start += chunk){
        size_t end = (std::min)(total, start + chunk);
        parallel_for(cameras.begin() + start, cameras.begin() + end, [&](Camera &cam){
            try{
                prepareCamera(cam, downscaleFactor);
            }catch (const std::exception &e){
                std::lock_guard<std::mutex> lock(logMutex);
                if (firstError.empty()) firstError = e.what();
                return;
            }
            size_t n = ++done;
            if (n % 100 == 0 || n == total){
                std::lock_guard<std::mutex> lock(logMutex);
                std::cout << "Preprocessing images " << n << "/" << total << std::endl;
            }
        });
        if (!firstError.empty()) throw std::runtime_error(firstError);
    }
}

// Decodes, rescales and undistorts one camera image, then stores a compressed
// copy per downscale level and updates the camera intrinsics
void ImageStore::prepareCamera(Camera &cam, float downscaleFactor){
    cv::Mat img = cv::imread(cam.filePath);
    if (img.empty()){
        throw std::runtime_error("Cannot read " + cam.filePath +
                                 "\nMake sure the path to your images is correct");
    }

    cv::Mat mask;
    if (!cam.maskPath.empty()){
        mask = cv::imread(cam.maskPath, cv::IMREAD_GRAYSCALE);
        if (mask.empty()) throw std::runtime_error("Cannot read mask " + cam.maskPath);
    }

    // If camera intrinsics don't match the image dimensions
    if (img.rows != cam.height || img.cols != cam.width){
        float rescaleF = static_cast<float>(img.rows) / static_cast<float>(cam.height);
        cam.fx *= rescaleF;
        cam.fy *= rescaleF;
        cam.cx *= rescaleF;
        cam.cy *= rescaleF;
    }

    bool pixelsUntouched = true;
    if (downscaleFactor > 1.0f){
        float scaleFactor = 1.0f / downscaleFactor;
        cv::resize(img, img, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
        cam.fx *= scaleFactor;
        cam.fy *= scaleFactor;
        cam.cx *= scaleFactor;
        cam.cy *= scaleFactor;
        pixelsUntouched = false;
    }

    if (!mask.empty()){
        cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
        if (mask.rows != img.rows || mask.cols != img.cols){
            cv::resize(mask, mask, cv::Size(img.cols, img.rows), 0.0, 0.0, cv::INTER_LINEAR);
        }
    }

    if (cam.hasDistortionParameters()){
        UndistortParams p = computeUndistortParams(cam.fx, cam.fy, cam.cx, cam.cy, img.cols, img.rows,
                                                   cam.k1, cam.k2, cam.k3, cam.k4, cam.k5, cam.k6, cam.p1, cam.p2);
        cv::Mat mapx, mapy;
        buildUndistortMaps(p, mapx, mapy);
        cv::Mat undistorted;
        cv::remap(img, undistorted, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        img = undistorted;
        if (!mask.empty()){
            cv::Mat remapped;
            cv::remap(mask, remapped, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            mask = remapped;
        }
        cam.fx = p.dstFx;
        cam.fy = p.dstFy;
        cam.cx = p.dstCx;
        cam.cy = p.dstCy;
        pixelsUntouched = false;
    }

    cam.width = img.cols;
    cam.height = img.rows;
    cam.K = cam.getIntrinsicsMatrix();
    cam.hasMask = !mask.empty();

    for (int l = 0; l < numLevels; l++){
        const int level = 1 << l;
        cv::Mat lvlImg = img, lvlMask = mask;
        if (level > 1){
            cv::resize(img, lvlImg, cv::Size(img.cols / level, img.rows / level), 0.0, 0.0, cv::INTER_AREA);
            if (!mask.empty()){
                cv::resize(mask, lvlMask, cv::Size(img.cols / level, img.rows / level), 0.0, 0.0, cv::INTER_LINEAR);
                cv::threshold(lvlMask, lvlMask, 127, 255, cv::THRESH_BINARY);
            }
        }
        // Source bytes are kept verbatim when nothing changed the pixels
        std::vector<uint8_t> bytes = (level == 1 && pixelsUntouched && isJpegOrPng(cam.filePath))
            ? readFileBytes(cam.filePath)
            : encodeJpeg(lvlImg);
        cache.put(imageKey(cam.imageId, level), std::move(bytes));
        if (!lvlMask.empty()){
            cache.put(maskKey(cam.imageId, level), encodePng(lvlMask));
        }
    }
}
