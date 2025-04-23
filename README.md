# LPCVC 2025 Track 3 -- 3rd Place Solution (Modified DepthAnythingV2)

This repository presents our solution for LPCVC 2025 Track 3, which earned us 3rd place. For more details about this track, please refer to the [official website](https://lpcv.ai/2025LPCVC/monocular-depth/).

---

### 1. Introduction

Our approach builds upon the [official sample solution for Track 3](https://github.com/lpcvai/25LPCVC_Track3_Sample_Solution).

In this challenge, we explored modifications to two variants of the DepthAnythingV2 model that was used in the official sample solution:

- DepthAnythingV2 for Relative Depth Estimation  
- DepthAnythingV2 for Metric Depth Estimation
  
---

### 2. Model Details

#### 2.1 DepthAnythingV2 for Relative Depth Estimation

- According to the task description on the official website (quoted below), the model is expected to output values in the range [0, 1], representing depth from near to far. A smaller value indicates a closer distance to the camera:

    > **Model Output:** The submitted model is expected to predict a relative depth (**ranging from 0 to 1, from near to far**) based on its input. The model should produce a single output with one channel, and the expected output tensor shape is (Batch, 1, Height, Width) = (Batch, 1, 480, 640).

    Upon closely examining the code and architecture of DepthAnythingV2, we noticed that the outputs from the relative depth estimation model should be the disparity rather than depth. It means the smaller the output, the far distance from the camera. Therefore, in addition to the post-processing steps provided in the official sample solution, we reversed the normalized output:

    ``` python
    # Line195 in DepthAnythingV2/depth_anything_v2/dpt.py
    depth = 1 - (depth - depth.min()) / (depth.max() - depth.min()) 
    ```
- To improve inference speed, we explored various optimization strategies. Ultimately, we leveraged the fact that the model is transformer-based and thus largely resolution-agnostic. We reduced the input resolution to accelerate processing:
  
    ``` python
    # Line178 in DepthAnythingV2/depth_anything_v2/dpt.py
    x = F.interpolate(x, (350, 350), mode="bilinear", align_corners=True)
    ```

We got the following result by using such modifications. If you want to duplicate our results, please run the following command:

``` shell
python demo_run_RelativeDepth.py
```

|Rank|Team|Date And Time|Compile Job Id|F-Score|Time(Ms)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|20|Circle|2025-03-15 23:09:34|****o0l8g|71.67294793|26.725|

---

#### 2.2 DepthAnythingV2 for Metric Depth Estimation

In the official sample solution (and the DepthAnythingV2 repository), we also found a  `metric_depth` folder. Upon reviewing its contents in detail, we confirmed that this model directly outputs absolute depth values, so no further conversion is required. However, this model is slightly larger in size. To reduce latency, we adjusted the input resolution. Additionally, we applied the same input normalization and output resizing techniques used in the relative depth model.


``` python
# Line178 in DepthAnythingV2/depth_anything_v2/dpt.py
def forward(self, x):
    x = F.interpolate(x, (336, 336), mode="bilinear", align_corners=True)
    
    # input pre-processing
    mean = torch.tensor([[[[123.6750]], [[116.2800]], [[103.5300]]]])
    std = torch.tensor([[[[58.3950]], [[57.1200]], [[57.3750]]]])
    x = (x - mean)/std

    patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
    
    features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
    
    depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
    
    depth = F.interpolate(depth, (480, 640), mode="bilinear", align_corners=True)
    
    # return depth.squeeze(1)
    return depth
```

The metric depth model gives us a better results. To duplicate our results, please run the following command:

``` shell
python demo_run_MetricDepth.py
```

|Rank|Team|Date And Time|Compile Job Id|F-Score|Time(Ms)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|13|Circle|2025-03-27 03:22:52|****k760p|79.15871095|33.08|

---

### 3. Acknowledgement

We sincerely appreciate the efforts of the organizing team. It was a great honor for our team to participate in this challenge and receive an award.
