/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <algorithm>
#include <cmath>
#include <sstream>
#include "nvdsinfer_custom_impl.h"
#include "utils.h"

#include "yoloPlugins.h"

extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

static NvDsInferParseObjectInfo convertBBox(
    const float& bx, const float& by, const float& bw,
    const float& bh, const uint stride, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution
    
    float xCenter = bx * stride;
    float yCenter = by * stride;
    float x0 = xCenter - bw/2;
    float y0 = yCenter - bh/2;
    float ww = bw;
    float hh = bh;

    x0 = clamp(x0, 0, netW);
    y0 = clamp(y0, 0, netH);
    ww = clamp(ww, 0, netW);
    hh = clamp(hh, 0, netH);

    b.left = x0;
    b.width = ww;
    b.top = y0;
    b.height = hh;

    return b;
}

static void addBBoxProposal(
    const float bx, const float by, const float bw, const float bh,
    const uint stride, const uint& netW, const uint& netH, const int maxIndex,
    const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBox(bx, by, bw, bh, stride, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

float sigmoid(float x){ return (1 / (1 + exp(-x)));}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloTensor(
    const float* detections, const std::vector<int> &mask, const std::vector<float> &anchors,
    const uint gridSizeW, const uint gridSizeH, const uint stride, const uint numBBoxes,
    const uint numOutputClasses, const uint& netW,
    const uint& netH)
{

    std::vector<NvDsInferParseObjectInfo> binfo;

    int bbindex = 0;

    for (uint y = 0; y < gridSizeH; ++y) {
        for (uint x = 0; x < gridSizeW; ++x) {
            for (uint b = 0; b < numBBoxes; ++b)
            {
                const int numGridCells = gridSizeH * gridSizeW;
	        int offset =  (x + y * gridSizeW + b * numGridCells) * (5 + numOutputClasses) ;
	        	
                const float pw = anchors[mask[b] * 2];
                const float ph = anchors[mask[b] * 2 + 1];
                
		float bx = sigmoid(detections[offset + 0]);
		bx *=2;
	        bx =bx - 0.5 + x;

                float by = sigmoid(detections[offset + 1]);
		by *=2;
	        by = by	- 0.5 + y;

                float bw =  sigmoid(detections[offset + 2]);
		bw = pow((bw * 2), 2);
		bw *= pw;

                float bh = sigmoid(detections[offset + 3]);
		bh = pow((bh * 2), 2);
		bh *= ph;
	       
		const float objectness
                    = detections[offset + 4];

                float maxProb = 0.0f;
                int maxIndex = -1;

		for (uint i = 0; i < numOutputClasses; ++i)
                {
                    float prob_ = (detections[offset + (5 + i)]);
		    float prob = sigmoid(prob_);
                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                
		maxProb = sigmoid(objectness) * maxProb;
		
	        addBBoxProposal(bx, by, bw, bh, stride, netW, netH, maxIndex, maxProb, binfo);

            }

        }
    }
    return binfo;
}

static inline std::vector<const NvDsInferLayerInfo*>
SortLayers(const std::vector<NvDsInferLayerInfo> & outputLayersInfo)
{
    std::vector<const NvDsInferLayerInfo*> outLayers;
    for (auto const &layer : outputLayersInfo) {
        outLayers.push_back (&layer);
    }
    std::sort(outLayers.begin(), outLayers.end(),
        [](const NvDsInferLayerInfo* a, const NvDsInferLayerInfo* b) {
            return a->inferDims.d[1] < b->inferDims.d[1];
        });
    return outLayers;
}


static bool NvDsInferParseCustomYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList,
    const uint &numBBoxes,
    const uint &numClasses)
{
	const std::vector<float> anchors = {
		10.0, 13.0, 16.0,  30.0,  33.0, 23.0,  30.0,  61.0,  62.0,
		45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
	const std::vector<std::vector<int>> mask = {
		{6, 7, 8},
		{3, 4, 5},
		{0, 1, 2}};


    if (outputLayersInfo.empty())
    {
        std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
        return false;
    }

    if (numClasses != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured: "
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << numClasses << std::endl;
    }
    
    const std::vector<const NvDsInferLayerInfo*> sortedLayers =
        SortLayers (outputLayersInfo);


    std::vector<NvDsInferParseObjectInfo> objects;

    for (uint idx = 0; idx < outputLayersInfo.size(); ++idx)
    {
        const NvDsInferLayerInfo &layer = *sortedLayers[idx];

        assert(layer.inferDims.numDims == 3 || layer.inferDims.numDims == 4);
        const uint gridSizeH = layer.inferDims.d[1];
        const uint gridSizeW = layer.inferDims.d[2];

        const uint stride = DIVUP(networkInfo.width, gridSizeW);
        assert(stride == DIVUP(networkInfo.height, gridSizeH));

        std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeYoloTensor( \
                (const float*)(layer.buffer), mask[idx], anchors,  \
                gridSizeW, gridSizeH, stride, numBBoxes, numClasses, \
                networkInfo.width, networkInfo.height);

        objects.insert(objects.end(), outObjs.begin(), outObjs.end());
    }

    objectList = objects;

    return true;
}

extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    kNUM_BBOXES = 3;
    kNUM_CLASSES = 80;

    uint numBBoxes = kNUM_BBOXES;
    uint numClasses = kNUM_CLASSES;

    return NvDsInferParseCustomYolo (
        outputLayersInfo, networkInfo, detectionParams, objectList, numBBoxes, numClasses);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);
