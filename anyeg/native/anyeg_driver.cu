// Command-line driver for the ORIGINAL AnyEG GPU kernels.
//
// AnyEG ships its algorithm as `compute()` in cudaHelper.cu, called from a
// TTK/VTK filter (ttkExtremeGraphDev.cpp) that needs a full TTK build. This
// driver calls the same, unchanged `compute()` on a neighbour table read from
// a file, so the kernels can be run without TTK and their output compared
// with the Python ports (anyeg/kernels/original.py writes and reads the files).
//
// Usage: anyeg_driver <input.bin> <output.bin>
//
// input.bin  (little endian):
//   char[4]  "AEG1"
//   int32    N              number of vertices
//   int32    M              total number of neighbour entries
//   int32    dims[3]        grid size (only printed by AnyEG)
//   float32  values[N]
//   int32    counts[N]      neighbours per vertex
//   int32    neighbours[M]  all neighbour lists, one after the other
//
// output.bin:
//   char[4]  "AEG1"
//   float64  seconds        wall time of compute() including its copies
//   int32    L              length of AnyEG's result buffer
//   int32    result[L]      the buffer returned by compute(), unchanged:
//            [nCand, cand..., nExt, ext..., FatherLinks_cpy[N], FatherLinks[N], int(Values)[N]]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

int* compute(int _N, float* _Values, int* _NeighborNums, int** _Neighbors, int* _dims);

static bool readAll(FILE* f, void* dst, size_t bytes) { return fread(dst, 1, bytes, f) == bytes; }

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 2;
    }
    FILE* in = fopen(argv[1], "rb");
    if (!in) { perror(argv[1]); return 1; }

    char magic[4];
    int32_t n = 0, m = 0, dims[3];
    if (!readAll(in, magic, 4) || memcmp(magic, "AEG1", 4) != 0 ||
        !readAll(in, &n, 4) || !readAll(in, &m, 4) || !readAll(in, dims, 12)) {
        fprintf(stderr, "bad input header\n");
        return 1;
    }
    std::vector<float> values(n);
    std::vector<int32_t> counts(n), neighbours(m);
    if (!readAll(in, values.data(), 4ull * n) || !readAll(in, counts.data(), 4ull * n) ||
        !readAll(in, neighbours.data(), 4ull * m)) {
        fprintf(stderr, "input file is truncated\n");
        return 1;
    }
    fclose(in);

    // compute() takes one pointer per vertex, as the TTK filter builds them.
    std::vector<int*> rows(n);
    size_t offset = 0;
    for (int32_t i = 0; i < n; ++i) {
        rows[i] = neighbours.data() + offset;
        offset += counts[i];
    }

    auto t0 = std::chrono::steady_clock::now();
    int* result = compute(n, values.data(), counts.data(), rows.data(), dims);
    cudaDeviceSynchronize();
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int32_t nCand = result[0];
    int32_t nExt = result[1 + nCand];
    int32_t length = 2 + nCand + nExt + 3 * n;

    FILE* out = fopen(argv[2], "wb");
    if (!out) { perror(argv[2]); return 1; }
    fwrite("AEG1", 1, 4, out);
    fwrite(&seconds, 8, 1, out);
    fwrite(&length, 4, 1, out);
    fwrite(result, 4, length, out);
    fclose(out);
    return 0;
}
