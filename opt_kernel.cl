__kernel void gaussian_blur_local(
    __global const float* input,
    __global float* output,
    const int width,
    const int height)
{
    // Work-item and work-group indices
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);

    const int group_size_x = get_local_size(0);
    const int group_size_y = get_local_size(1);

    // Local tile size: Add 2 extra rows/cols for 3x3 kernel halo
    // Make sure the size is large enough for the work group plus halo
    __local float tile[(16+2)][(16+2)]; // Maximum 16x16 work group plus halo

    // Coordinates in the input including halo
    int tile_x = lx + 1;
    int tile_y = ly + 1;

    // Pre-calculate indices to avoid redundant calculations
    int global_y = (gy + height) % height;
    int global_x = (gx + width) % width;
    int global_index = global_y * width + global_x;

    // Load central pixel directly into register for faster access
    float center_pixel = 0.0f;
    if (gx < width && gy < height) {
        center_pixel = input[global_index];
        tile[tile_y][tile_x] = center_pixel;
    }

    // Load halo (8 surrounding pixels) - minimizing global memory reads
    if (lx == 0) {
        tile[tile_y][0] = input[global_y * width + ((gx - 1 + width) % width)];
    }
    if (lx == group_size_x - 1) {
        tile[tile_y][tile_x + 1] = input[global_y * width + ((gx + 1) % width)];
    }
    if (ly == 0) {
        tile[0][tile_x] = input[((gy - 1 + height) % height) * width + global_x];
    }
    if (ly == group_size_y - 1) {
        tile[tile_y + 1][tile_x] = input[((gy + 1) % height) * width + global_x];
    }

    // Corners
    if (lx == 0 && ly == 0) {
        tile[0][0] = input[((gy - 1 + height) % height) * width + ((gx - 1 + width) % width)];
    }
    if (lx == 0 && ly == group_size_y - 1) {
        tile[tile_y + 1][0] = input[((gy + 1) % height) * width + ((gx - 1 + width) % width)];
    }
    if (lx == group_size_x - 1 && ly == 0) {
        tile[0][tile_x + 1] = input[((gy - 1 + height) % height) * width + ((gx + 1) % width)];
    }
    if (lx == group_size_x - 1 && ly == group_size_y - 1) {
        tile[tile_y + 1][tile_x + 1] = input[((gy + 1) % height) * width + ((gx + 1) % width)];
    }

    // Ensure all threads have loaded tile
    barrier(CLK_LOCAL_MEM_FENCE);

    // Gaussian kernel weights - use exact same weights as naive kernel for accurate comparison
    const float k00 = 0.0625f;  const float k01 = 0.1250f;  const float k02 = 0.0625f;
    const float k10 = 0.1250f;  const float k11 = 0.2500f;  const float k12 = 0.1250f;
    const float k20 = 0.0625f;  const float k21 = 0.1250f;  const float k22 = 0.0625f;

    // Apply convolution only for pixels within image bounds
    if (gx < width && gy < height) {
        // Use a single fused operation for better optimization
        float sum = 0.0f;
        
        // Top row
        sum += tile[tile_y-1][tile_x-1] * k00;
        sum += tile[tile_y-1][tile_x]   * k01;
        sum += tile[tile_y-1][tile_x+1] * k02;
        
        // Middle row
        sum += tile[tile_y][tile_x-1] * k10;
        sum += tile[tile_y][tile_x]   * k11; // Center pixel (heaviest weight)
        sum += tile[tile_y][tile_x+1] * k12;
        
        // Bottom row
        sum += tile[tile_y+1][tile_x-1] * k20;
        sum += tile[tile_y+1][tile_x]   * k21;
        sum += tile[tile_y+1][tile_x+1] * k22;

        // Write result directly
        output[global_index] = sum;
    }
}
