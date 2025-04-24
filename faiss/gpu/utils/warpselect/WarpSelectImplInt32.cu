/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>

/**
 * Custom int32 warp select implementations=
 *
 * @author Hugo Phibbs
 */
namespace faiss {
namespace gpu {

WARP_SELECT_IMPL(int, true, 1, 1);
WARP_SELECT_IMPL(int, false, 1, 1);

WARP_SELECT_IMPL(int, true, 32, 2);
WARP_SELECT_IMPL(int, false, 32, 2);

WARP_SELECT_IMPL(int, true, 64, 3);
WARP_SELECT_IMPL(int, false, 64, 3);

WARP_SELECT_IMPL(int, true, 128, 3);
WARP_SELECT_IMPL(int, false, 128, 3);

WARP_SELECT_IMPL(int, true, 256, 3);
WARP_SELECT_IMPL(int, false, 256, 3);

WARP_SELECT_IMPL(int, true, 512, 8);
WARP_SELECT_IMPL(int, false, 512, 8);

WARP_SELECT_IMPL(int, true, 1024, 8);
WARP_SELECT_IMPL(int, false, 1024, 8);

#if GPU_MAX_SELECTION_K >= 2048
WARP_SELECT_IMPL(int, true, 2048, 8);
WARP_SELECT_IMPL(int, false, 2048, 8);
#endif

} // namespace gpu
} // namespace faiss