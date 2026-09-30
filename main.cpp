#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Function to check OpenCL errors
void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error during operation '" << operation
                  << "': " << err << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to read a kernel from file
std::string readKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Function to print device info
void printDeviceInfo(cl_device_id device) {
    char deviceName[1024];
    cl_uint computeUnits;
    size_t workGroupSize;
    cl_ulong localMemSize;
    
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workGroupSize), &workGroupSize, NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
    
    std::cout << "Device: " << deviceName << std::endl;
    std::cout << "Compute Units: " << computeUnits << std::endl;
    std::cout << "Max Work Group Size: " << workGroupSize << std::endl;
    std::cout << "Local Memory Size: " << (localMemSize / 1024) << " KB" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

// Struct to hold performance results
struct KernelPerformance {
    std::string name;
    double executionTimeMs;
    size_t localSizeX;
    size_t localSizeY;
};

int main() {
    // Image dimensions
    const int width = 2048;  // Larger image for better performance comparison
    const int height = 2048;
    const size_t bytes = width * height * sizeof(float);
    
    // Create input and output image data
    std::vector<float> inputImage(width * height, 0.0f);
    std::vector<float> outputImage(width * height, 0.0f);
    std::vector<float> outputImageOptimized(width * height, 0.0f);
    
    // Initialize input image with some pattern for testing
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Simple checkerboard pattern
            inputImage[y * width + x] = ((x / 32) + (y / 32)) % 2 ? 1.0f : 0.0f;
        }
    }
    
    // OpenCL variables
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernelSimple, kernelOptimized;
    cl_mem inputBuffer, outputBuffer;
    cl_event event;
    
    // Get platform
    err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err, "getting platform ID");
    
    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    checkError(err, "getting device ID");
    
    // Print device info
    printDeviceInfo(device);
    
    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "creating context");
    
    // Create command queue with profiling enabled
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkError(err, "creating command queue");
    
    // Create memory buffers
    inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                bytes, inputImage.data(), &err);
    checkError(err, "creating input buffer");
    
    outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 bytes, NULL, &err);
    checkError(err, "creating output buffer");
    
    // Read kernel files
    std::string naiveKernelSource = readKernelSource("naive_kernel.cl");
    std::string optKernelSource = readKernelSource("opt_kernel.cl");
    
    // Debug: Print first few characters of each kernel source
    std::cout << "Naive kernel source preview: " << naiveKernelSource.substr(0, 50) << "..." << std::endl;
    std::cout << "Optimized kernel source preview: " << optKernelSource.substr(0, 50) << "..." << std::endl;
    
    // Create and build program with both kernels
    const char* sources[2] = {naiveKernelSource.c_str(), optKernelSource.c_str()};
    size_t lengths[2] = {naiveKernelSource.size(), optKernelSource.size()};
    
    program = clCreateProgramWithSource(context, 2, sources, lengths, &err);
    checkError(err, "creating program");
    
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), NULL);
        std::cerr << "Error building program. Build log:" << std::endl;
        std::cerr << buildLog.data() << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Create kernels
    kernelSimple = clCreateKernel(program, "gaussian_blur_simple", &err);
    checkError(err, "creating simple kernel");
    
    kernelOptimized = clCreateKernel(program, "gaussian_blur_local", &err);
    checkError(err, "creating optimized kernel");
    
    // Prepare performance results storage
    std::vector<KernelPerformance> results;
    
    std::cout << "Starting performance tests..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // Test various work group sizes - adjusted to respect 256 work item limit
    std::vector<std::pair<size_t, size_t>> workGroupSizes = {
        {8, 8},    // 64 work items
        {16, 16},  // 256 work items
        {16, 8}    // 128 work items
    };
    
    // Run simple kernel
    for (auto& wgSize : workGroupSizes) {
        size_t localSize[2] = {wgSize.first, wgSize.second};
        size_t globalSize[2] = {
            ((width + localSize[0] - 1) / localSize[0]) * localSize[0],
            ((height + localSize[1] - 1) / localSize[1]) * localSize[1]
        };
        
        // Set kernel arguments
        err = clSetKernelArg(kernelSimple, 0, sizeof(cl_mem), &inputBuffer);
        checkError(err, "setting simple kernel argument 0");
        
        err = clSetKernelArg(kernelSimple, 1, sizeof(cl_mem), &outputBuffer);
        checkError(err, "setting simple kernel argument 1");
        
        err = clSetKernelArg(kernelSimple, 2, sizeof(int), &width);
        checkError(err, "setting simple kernel argument 2");
        
        err = clSetKernelArg(kernelSimple, 3, sizeof(int), &height);
        checkError(err, "setting simple kernel argument 3");
        
        // Execute kernel with profiling
        err = clEnqueueNDRangeKernel(queue, kernelSimple, 2, NULL, globalSize,
                                     localSize, 0, NULL, &event);
        checkError(err, "enqueuing simple kernel");
        
        // Wait for kernel to finish
        clFinish(queue);
        
        // Get profiling info
        cl_ulong startTime, endTime;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(startTime), &startTime, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(endTime), &endTime, NULL);
        double executionTimeMs = (endTime - startTime) * 1.0e-6; // Convert nanoseconds to milliseconds
        
        // Store result
        std::ostringstream name;
        name << "Simple Kernel (" << localSize[0] << "x" << localSize[1] << ")";
        results.push_back({name.str(), executionTimeMs, localSize[0], localSize[1]});
        
        // Cleanup
        clReleaseEvent(event);
    }
    
    // Run optimized kernel
    for (auto& wgSize : workGroupSizes) {
        // Skip work group sizes larger than our local memory allocation
        if (wgSize.first > 32 || wgSize.second > 32) continue;
        
        size_t localSize[2] = {wgSize.first, wgSize.second};
        size_t globalSize[2] = {
            ((width + localSize[0] - 1) / localSize[0]) * localSize[0],
            ((height + localSize[1] - 1) / localSize[1]) * localSize[1]
        };
        
        // Set kernel arguments
        err = clSetKernelArg(kernelOptimized, 0, sizeof(cl_mem), &inputBuffer);
        checkError(err, "setting optimized kernel argument 0");
        
        err = clSetKernelArg(kernelOptimized, 1, sizeof(cl_mem), &outputBuffer);
        checkError(err, "setting optimized kernel argument 1");
        
        err = clSetKernelArg(kernelOptimized, 2, sizeof(int), &width);
        checkError(err, "setting optimized kernel argument 2");
        
        err = clSetKernelArg(kernelOptimized, 3, sizeof(int), &height);
        checkError(err, "setting optimized kernel argument 3");
        
        // Execute kernel with profiling
        err = clEnqueueNDRangeKernel(queue, kernelOptimized, 2, NULL, globalSize,
                                     localSize, 0, NULL, &event);
        checkError(err, "enqueuing optimized kernel");
        
        // Wait for kernel to finish
        clFinish(queue);
        
        // Read results for verification (only for the last run)
        if (wgSize.first == workGroupSizes.back().first && wgSize.second == workGroupSizes.back().second) {
            err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, bytes,
                                     outputImageOptimized.data(), 0, NULL, NULL);
            checkError(err, "reading output buffer");
        }
        
        // Get profiling info
        cl_ulong startTime, endTime;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(startTime), &startTime, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(endTime), &endTime, NULL);
        double executionTimeMs = (endTime - startTime) * 1.0e-6; // Convert nanoseconds to milliseconds
        
        // Store result
        std::ostringstream name;
        name << "Optimized Kernel (" << localSize[0] << "x" << localSize[1] << ")";
        results.push_back({name.str(), executionTimeMs, localSize[0], localSize[1]});
        
        // Cleanup
        clReleaseEvent(event);
    }
    
    // Print performance results
    std::cout << "\nPerformance Results:\n";
    std::cout << "----------------------------------------\n";
    std::cout << std::left << std::setw(30) << "Kernel"
              << std::right << std::setw(12) << "Time (ms)"
              << std::right << std::setw(15) << "Work Group" << std::endl;
    std::cout << "----------------------------------------\n";
    
    for (const auto& result : results) {
        std::ostringstream wgSize;
        wgSize << result.localSizeX << "x" << result.localSizeY;
        
        std::cout << std::left << std::setw(30) << result.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) << result.executionTimeMs
                  << std::right << std::setw(15) << wgSize.str() << std::endl;
    }
    
    // Find best performance
    auto bestResult = std::min_element(results.begin(), results.end(),
        [](const KernelPerformance& a, const KernelPerformance& b) {
            return a.executionTimeMs < b.executionTimeMs;
        });
    
    std::cout << "\nBest performance: " << bestResult->name
              << " with time " << bestResult->executionTimeMs << " ms" << std::endl;
    
    // Calculate speedup - updated to handle new work group size array
    if (!results.empty() && results.size() >= 4) {
        double simpleTime = results[0].executionTimeMs;  // First result is simple kernel with 8x8
        double optimizedTime = results[3].executionTimeMs;  // Fourth result is optimized kernel with 8x8
        
        double speedup = simpleTime / optimizedTime;
        std::cout << "Speedup (optimized vs simple with same work group size): " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    
    // Clean up
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernelSimple);
    clReleaseKernel(kernelOptimized);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}
