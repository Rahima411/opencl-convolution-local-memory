__kernel void gaussian_blur_simple(
    __global const float* input,
    __global float* output,
    const int width,
    const int height)
{
    // Gaussian kernel weights
    const float k00 = 0.0625f;  const float k01 = 0.1250f;  const float k02 = 0.0625f;
    const float k10 = 0.1250f;  const float k11 = 0.2500f;  const float k12 = 0.1250f;
    const float k20 = 0.0625f;  const float k21 = 0.1250f;  const float k22 = 0.0625f;
    
    int tx = get_global_id(0);
    int ty = get_global_id(1);
    
    // Handle borders by wrapping around
    if (tx < width && ty < height) {
        float result = 0.0f;
        
        // Calculate for each position in the 3x3 kernel manually
        // Top row
        int y = (ty - 1 + height) % height;
        int x = (tx - 1 + width) % width;
        result += input[y * width + x] * k00;
        
        x = (tx + width) % width;
        result += input[y * width + x] * k01;
        
        x = (tx + 1 + width) % width;
        result += input[y * width + x] * k02;
        
        // Middle row
        y = (ty + height) % height;
        x = (tx - 1 + width) % width;
        result += input[y * width + x] * k10;
        
        x = (tx + width) % width;
        result += input[y * width + x] * k11;
        
        x = (tx + 1 + width) % width;
        result += input[y * width + x] * k12;
        
        // Bottom row
        y = (ty + 1 + height) % height;
        x = (tx - 1 + width) % width;
        result += input[y * width + x] * k20;
        
        x = (tx + width) % width;
        result += input[y * width + x] * k21;
        
        x = (tx + 1 + width) % width;
        result += input[y * width + x] * k22;
        
        output[ty * width + tx] = result;
    }
}
