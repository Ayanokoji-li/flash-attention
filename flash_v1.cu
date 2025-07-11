#include <cooperative_groups.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

__global__ void forward_kernel(const float *Q, const float *K, const float *V,
                               const int N, const int d, const int Tc,
                               const int Tr, const int Bc, const int Br,
                               const float softmax_scale, float *l, float *m,
                               float *O) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y; // batch and head index

  // Offset into Q,K,V,O,l,m - different for each batch and head
  int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
  int lm_offset = (bx * gridDim.y * N) + (by * N); // offset for l and m

  // Define SRAM for Q,K,V,S
  extern __shared__ float sram[];
  int tile_size = Bc * d; // size of Qi, Kj, Vj
  float *Qi = sram;
  float *Kj = &sram[tile_size];
  float *Vj = &sram[tile_size * 2];
  float *S = &sram[tile_size * 3];

  for (int j = 0; j < Tc; j++) {

    // Load Kj, Vj to SRAM
    for (int x = 0; x < d; x++) {
      Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
      Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
    }
    __syncthreads(); // such that the inner loop can use the correct Kj, Vj

    for (int i = 0; i < Tr; i++) {

      // Load Qi to SRAM, l and m to registers
      float l_local{}, m_local{};

      for (int x = 0; x < d; x++) {
        Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
      }
      float row_m_prev = m[lm_offset + (Br * i) + tx];
      float row_l_prev = l[lm_offset + (Br * i) + tx];
      // float row_m_prev = m_local;
      // float row_l_prev = l_local;

      // S = QK^T, row_m = rowmax(S)
      float row_m = -INFINITY;
      for (int y = 0; y < Br; y++) {
        float sum = 0;
        for (int x = 0; x < d; x++) {
          sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
        }
        sum *= softmax_scale;
        S[(Bc * tx) + y] = sum;

        if (sum > row_m)
          row_m = sum;
      }

      // P = exp(S - row_m), row_l = rowsum(P)
      float row_l = 0;
      for (int y = 0; y < Bc; y++) {
        S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
        row_l += S[(Bc * tx) + y];
      }

      // Compute new m and l
      float row_m_new = max(row_m_prev, row_m);
      assert(row_m_new != -INFINITY);
      float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) +
                        (__expf(row_m - row_m_new) * row_l);

      // Write O, l, m to HBM
      for (int x = 0; x < d; x++) {
        float pv = 0; // Pij * Vj
        for (int y = 0; y < Bc; y++) {
          pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
        }
        O[qkv_offset + (tile_size * i) + (tx * d) + x] =
            (1 / row_l_new) *
            ((row_l_prev * __expf(row_m_prev - row_m_new) *
              O[qkv_offset + (tile_size * i) + (tx * d) + x]) +
             (__expf(row_m - row_m_new) * pv));
      }
      m[lm_offset + (Br * i) + tx] = row_m_new;
      l[lm_offset + (Br * i) + tx] = row_l_new;
      // m_local = row_m_new;
      // l_local = row_l_new;
    }
    __syncthreads(); // otherwise, thread can use the wrong Kj, Vj in inner loop
  }
}

__global__ void forward_decode_map(const float *Q, const float *K,
                                   const float *V, const int N, const int d,
                                   const int Tc, const int Tr, const int Bc,
                                   const int Br, const float softmax_scale,
                                   float *l, float *m, float *O,
                                   int multithread) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y; // batch and head index
  auto cg_block = cooperative_groups::this_thread_block();

  // Offset into Q,K,V,O,l,m - different for each batch and head
  const int o_offset = (bx * gridDim.y * N * d * multithread) +
                       (by * N * d * multithread); // gridDim.y = nh
  const int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
  const int lm_offset = (bx * gridDim.y * N * multithread) +
                        (by * N * multithread); // offset for l and m
  const int Tr_offset = Tr / multithread * ty;

  const int tile_size = Bc * d; // size of Qi, Kj, Vj
  // const auto tile_gap = Tr / multithread;
  const auto tile_offset = ty * tile_size;
  const auto d_gap = d / multithread;
  const auto j_gap = Tr / multithread;
  const auto j_offset = Tr / multithread * ty;

  // Define SRAM for Q,K,V,S
  extern __shared__ float sram[];
  float *Qi = sram;
  float *Kj = &sram[tile_size + tile_offset];
  float *Vj = &sram[tile_size * (1 + multithread) + tile_offset];
  float *S = &sram[tile_size * (1 + 2 * multithread) + Bc * Br * ty];

  float *lk = &l[lm_offset + ty * N];
  float *mk = &m[lm_offset + ty * N];
  float *Ok = &O[o_offset + ty * d * N];

  for (int j = j_offset; j < Tr && j < j_offset + j_gap; j++) {
    // for (int j = ty; j < Tr; j += multithread) {
    for (int x = 0; x < d; x++) {
      Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
      Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
    }

    // __syncthreads(); // such that the inner loop can use the correct Kj, Vj

    for (int i = 0; i < Tc; i++) {
      cg_block.sync();
      for (int x = d_gap * ty; x < d && x < d_gap * (ty + 1); x++) {
        Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
      }

      // for (int x = ty; x < d; x += multithread) {
      //   Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
      // }

      cg_block.sync();
      // __syncthreads(); // such that the inner loop can use the correct Kj, Vj

      float row_m_prev = mk[i * Bc + tx];
      float row_l_prev = lk[i * Bc + tx];

      float row_m = -INFINITY;
      for (int y = 0; y < Br; y++) {
        float sum = 0;
        for (int x = 0; x < d; x++) {
          sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
        }
        sum *= softmax_scale;
        S[(Bc * tx) + y] = sum;

        if (sum > row_m)
          row_m = sum;
      }

      // P = exp(S - row_m), row_l = rowsum(P)
      float row_l = 0;
      for (int y = 0; y < Bc; y++) {
        S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
        row_l += S[(Bc * tx) + y];
      }

      // Compute new m and l
      float row_m_new = max(row_m_prev, row_m);
      float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) +
                        (__expf(row_m - row_m_new) * row_l);

      for (int x = 0; x < d; x++) {
        float pv = 0;
        for (int y = 0; y < Bc; y++) {
          pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
        }

        Ok[i * tile_size + tx * d + x] =
            (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) *
                                Ok[i * tile_size + tx * d + x]) +
                               __expf(row_m - row_m_new) * pv);
      }

      mk[i * Bc + tx] = row_m_new;
      lk[Br * i + tx] = row_l_new;
      // cg_block.sync();
      // __syncthreads(); // such that the inner loop can use the correct Kj, Vj
    }

    cg_block.sync();
  }
}

__global__ void forward_decode_reduce(const float *O_input, float *O_output,
                                      float *l, float *m, const int N,
                                      const int d, const int Tc, const int Bc,
                                      const int multithread) {
  const auto bx = blockIdx.x;
  const auto by = blockIdx.y;
  const auto bz = blockIdx.z;
  const auto tx = threadIdx.x;

  const int kv_offset = (bx * gridDim.y * N * d * multithread) +
                        (by * N * d * multithread); // gridDim.y = nh
  const int qin_offset = kv_offset;
  const int qout_offset = (bx * gridDim.y * N * d) + (by * N * d);
  const int lm_offset = (bx * gridDim.y * N * multithread) +
                        (by * N * multithread); // offset for l and m
  const auto tile_size = Bc * d;

  for (int i = 0; i < Tc; i++) {
    float local_m = -INFINITY;
    float local_l = 0;
    for (int tid = 0; tid < multithread; tid++) {
      local_m = max(m[lm_offset + tid * N + bz * Bc + tx], local_m);
    }

    for (int tid = 0; tid < multithread; tid++) {
      local_l += __expf(m[lm_offset + tid * N + bz * Bc + tx] - local_m) *
                 l[lm_offset + tid * N + bz * Bc + tx];
    }
    for (int x = 0; x < d; x++) {
      O_output[qout_offset + tile_size * bz + tx * d + x] = 0;
      for (int tid = 0; tid < multithread; tid++) {
        O_output[qout_offset + tile_size * bz + tx * d + x] +=
            (1 / local_l) *
            (l[lm_offset + tid * N + bz * Bc + tx] *
             __expf(m[lm_offset + tid * N + bz * Bc + tx] - local_m) *
             O_input[qin_offset + tid * N * d + (tile_size * bz) + tx * d + x]);
      }
    }
  }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  // TODO: determine Bc, Br dynamically
  const int Bc = 32;
  const int Br = 32;

  const int B = Q.size(0);
  const int nh = Q.size(1);
  const int N = Q.size(2);
  const int d = Q.size(3);

  const int Tc = ceil((float)N / Bc);
  const int Tr = ceil((float)N / Br);
  const float softmax_scale = 1.0 / sqrt(d);

  // Initialize O, l, m to HBM
  auto O = torch::zeros_like(Q);
  auto l = torch::zeros({B, nh, N});
  auto m = torch::full({B, nh, N}, -INFINITY);
  torch::Device device(torch::kCUDA);
  l = l.to(device);
  m = m.to(device);

  // Calculate SRAM size needed per block
  const int sram_size = ((1 + 2 * 2) * Bc * d * sizeof(float)) +
                        (Bc * Br * sizeof(float) * 2) +
                        (Bc * 2 * 2 * sizeof(float));
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size,
         sram_size);

  dim3 grid_dim(B, nh); // batch_size x num_heads
  dim3 block_dim(Bc);   // Bc threads per block

  forward_kernel<<<grid_dim, block_dim, sram_size>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Tc,
      Tr, Bc, Br, softmax_scale, l.data_ptr<float>(), m.data_ptr<float>(),
      O.data_ptr<float>());
  return O;
  // return m;
  // return l;
}

torch::Tensor forward_decode(torch::Tensor Q, torch::Tensor K,
                             torch::Tensor V) {
  const int Bc = 32;         // Q block size
  const int Br = 32;         // K, V block size
  const int multithread = 2; // additional thread for K, V

  const int B = Q.size(0);
  const int nh = Q.size(1);
  const int N = Q.size(2);
  const int d = Q.size(3);

  const int Tc = ceil((float)N / Bc);
  const int Tr = ceil((float)N / Br);
  const float softmax_scale = 1.0 / sqrt(d);

  // Initialize O, l, m to HBM
  auto O_map = torch::zeros({B, nh, N, d, multithread});
  auto O_reduce = torch::zeros_like(Q);
  auto l = torch::zeros({B, nh, N, multithread});
  auto m = torch::full({B, nh, N, multithread}, -INFINITY);
  torch::Device device(torch::kCUDA);
  l = l.to(device);
  m = m.to(device);
  O_map = O_map.to(device);

  // Calculate SRAM size needed per block
  const int sram_size =
      ((1 + multithread * 2) * Bc * d * sizeof(float)) + // Q, K, V cache
      (Bc * Br * sizeof(float) * multithread); // +          // S cache
  // (Bc * 2 * sizeof(float)); // +                         // l, m cache
  // (Bc * multithread * d * sizeof(float));            // O cache
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("Max shared memory: %d, requested shared memory: %d\n", max_sram_size,
         sram_size);

  dim3 map_grid_dim(B, nh);            // batch_size x num_heads
  dim3 map_block_dim(Bc, multithread); // Bc * multithread threads per block
  dim3 reduce_grid_dim(B, nh, Tc);     // batch_size * num_heads * Tc
  dim3 reduce_block_dim(Bc);           // Bc threads per block

  cudaEvent_t map_start, map_end, reduce_start, reduce_end;
  cudaEventCreate(&map_start);
  cudaEventCreate(&map_end);
  cudaEventCreate(&reduce_start);
  cudaEventCreate(&reduce_end);

  cudaEventRecord(map_start);
  forward_decode_map<<<map_grid_dim, map_block_dim, sram_size>>>(
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Tc,
      Tr, Bc, Br, softmax_scale, l.data_ptr<float>(), m.data_ptr<float>(),
      O_map.data_ptr<float>(), multithread);
  cudaEventRecord(map_end);

  cudaEventRecord(reduce_start);
  forward_decode_reduce<<<reduce_grid_dim, reduce_block_dim>>>(
      O_map.data_ptr<float>(), O_reduce.data_ptr<float>(), l.data_ptr<float>(),
      m.data_ptr<float>(), N, d, Tc, Bc, multithread);
  cudaEventRecord(reduce_end);

  cudaDeviceSynchronize();
  float elapse_ms;
  cudaEventElapsedTime(&elapse_ms, map_start, map_end);
  printf("[map] time: %.3f", elapse_ms);

  cudaEventElapsedTime(&elapse_ms, reduce_start, reduce_end);
  printf("[reduce] time: %.3f\n", elapse_ms);

  return O_reduce;
  // return O_map;
  // return m;
  // return l;
}