/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>

/**
 * Custom int32 warp select declarations
 *
 * @author Hugo Phibbs
 */
namespace faiss {
namespace gpu {

WARP_SELECT_DECL(int, true, 1);
WARP_SELECT_DECL(int, false, 1);

WARP_SELECT_DECL(int, true, 32);
WARP_SELECT_DECL(int, false, 32);

WARP_SELECT_DECL(int, true, 64);
WARP_SELECT_DECL(int, false, 64);

WARP_SELECT_DECL(int, true, 128);
WARP_SELECT_DECL(int, false, 128);

WARP_SELECT_DECL(int, true, 256);
WARP_SELECT_DECL(int, false, 256);

WARP_SELECT_DECL(int, true, 512);
WARP_SELECT_DECL(int, false, 512);

WARP_SELECT_DECL(int, true, 1024);
WARP_SELECT_DECL(int, false, 1024);

#if GPU_MAX_SELECTION_K >= 2048
WARP_SELECT_DECL(int, true, 2048);
WARP_SELECT_DECL(int, false, 2048);
#endif

void runWarpSelect(
        Tensor<int, 2, true>& in,
        Tensor<int, 2, true>& outK,
        Tensor<idx_t, 2, true>& outV,
        bool dir,
        int k,
        cudaStream_t stream) {
    FAISS_ASSERT(k <= 2048);

    if (dir) {
        if (k == 1) {
            WARP_SELECT_CALL(int, true, 1);
        } else if (k <= 32 && getWarpSizeCurrentDevice() == 32) {
            WARP_SELECT_CALL(int, true, 32);
        } else if (k <= 64) {
            WARP_SELECT_CALL(int, true, 64);
        } else if (k <= 128) {
            WARP_SELECT_CALL(int, true, 128);
        } else if (k <= 256) {
            WARP_SELECT_CALL(int, true, 256);
        } else if (k <= 512) {
            WARP_SELECT_CALL(int, true, 512);
        } else if (k <= 1024) {
            WARP_SELECT_CALL(int, true, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            WARP_SELECT_CALL(int, true, 2048);
#endif
        }
    } else {
        if (k == 1) {
            WARP_SELECT_CALL(int, false, 1);
        } else if (k <= 32 && getWarpSizeCurrentDevice() == 32) {
            WARP_SELECT_CALL(int, false, 32);
        } else if (k <= 64) {
            WARP_SELECT_CALL(int, false, 64);
        } else if (k <= 128) {
            WARP_SELECT_CALL(int, false, 128);
        } else if (k <= 256) {
            WARP_SELECT_CALL(int, false, 256);
        } else if (k <= 512) {
            WARP_SELECT_CALL(int, false, 512);
        } else if (k <= 1024) {
            WARP_SELECT_CALL(int, false, 1024);
#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            WARP_SELECT_CALL(int, false, 2048);
#endif
        }
    }
}

} // namespace gpu
} // namespace faiss
