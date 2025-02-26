# LPCVC 2025 Track 3 -- Monocular relative depth estimation

This repository takes [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file) as an example to show how participants modify their monocular depth estimation models for the Track 3. 

After reading this repository, you should be able to:
1. Understand the entire pipeline for participating in the challenge.
2. Modify an existing state-of-the-art (SOTA) depth estimation model (Depth-Anything-V2) to meet the requirements of Track 3.
3. Set up the environment for this challenge.
4. Get familiar with the Qualcomm® AI Hub, where your model will be compiled, profiled, and used to infer images on real mobile devices.
5. Submit your model to our challenge server for metric calculation and ranking on the leaderboard.

___

## 1. Participation Pipeline 

In total, we have 4 key steps from beginning to end:

1. **Two Registrations:**
    - Sign up for an account on [Qualcomm® AI Hub](https://aihub.qualcomm.com/) (top right corner). <u>**Every team member can register an account if they want.**</u> The Qualcomm® AI Hub is an easy tool for us to compile, profile, and infer images on real mobile devices.

    - Each team needs to complete a team registration through [this link](https://forms.gle/umXwSXT68ZzRFsfS9). <u>**One team only needs to register once.**</u> We will use this registration information to manage teams and their submissions. <u>P.S.: Team registration will be open on February 3, 2025.</u>

2. **Model Design & Training:** 
    - All participants will contribute their talent here. Participants will try their best to design and optimize their models to meet the 30 FPS running speed requirement and achieve higher accuracy than other teams.

3. **Self-Evaluation:**
    - Before submitting a model for metric calculation and ranking, participants can run their model on real mobile devices (through [Qualcomm® AI Hub](https://app.aihub.qualcomm.com/)) to profile the running time and visualize the model’s outputs.

4. **Submit Models for Ranking:**
    - Once a team has developed good models, the team lead can submit their models for ranking.

Next, we will provide more details for step2 to step4. 


___

## 2. Model Design & Training

Here, we will use [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file) as an example to show participants how to minimize their modifications for this challenge. Specifically, participants only need to manage the input and output of the model based on the [requirements of Track 3](https://lpcv.ai/2025LPCVC/monocular-depth). (finetuning may be needed after changing the input/output of the model)

- **Model Input**: During model evaluation, only the RGB images will be fed into the submitted model. All RGB images are in VGA resolution (640x480), so the input tensor shape will be (Batch, Channels, Height, Width) = (Batch, 3, 480, 640) in the PyTorch format. All images will be loaded in RGB channel order and range from [0, 255] in float. No input normalization will be applied, so each submitted model should include normalization operations, such as (image-mean)/std or image/255, at the beginning of the model if needed.

- **Model Output**: The submitted model is expected to predict a relative depth (ranging from 0 to 1) based on its input. The model should produce a single output with one channel, and the expected output tensor shape is (Batch, 1, Height, Width) = (Batch, 1, 480, 640).


Based on the above requirements, there are several important details to **pay more attention**:
1. Models will take inputs with the sape of (Batch, Channels, Height, Width) = (Batch, 3, 480, 640).
2. Input images are in the range of [0, 255].
3. Input images are in RGB channel order. 
4. Participants need to handle image pre-processing within their models, e.g., (input-mean)/std or input/255.
5. Models are expected to generate a tensor with a 4D output, i.e., (Batch, 1, Height, Width) = (Batch, 1, 480, 640).
6. The output of models should be in the range of [0, 1].


Following the above input and output requirements, we modify the [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file). Specifically, we only need to modify the file **\<Depth-Anything-V2\>/depth_anything_v2/dpt.py** (Line 176 - 184) as follows. We have also cloned Depth-Anything-V2 in this repository. If interested, you can check the corresponding file in this repository for the modifications.

``` diff
def forward(self, x):
+   # resize x (WidthxHeight=640x480) to Depth-Anythig-V2's original input resolution (518x518). 
+   x = F.interpolate(x, (518, 518), mode="bilinear", align_corners=True)
    
+   # input pre-processing
+   mean = torch.tensor([[[[123.6750]], [[116.2800]], [[103.5300]]]])
+   std = torch.tensor([[[[58.3950]], [[57.1200]], [[57.3750]]]])
+   x = (x - mean)/std

    patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
    
    features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
    
    depth = self.depth_head(features, patch_h, patch_w)
    depth = F.relu(depth)

+   # resize model's output to the required resolution (WidthxHeight=640x480)
+   depth = F.interpolate(depth, (480, 640), mode="bilinear", align_corners=True)
+   # normalize the output to ensure the range of [0, 1]
+   depth = (depth - depth.min()) / (depth.max() - depth.min()) 

+   # ensure the model's output is a 4D tensor, (Batch, 1, H, W)
-   return depth.squeeze(1)
+   return depth
```

After making these modifications, the model will take our testing images (640x480, [0,255]) as input, resize them to (518x518), apply input normalization (x-meam)/std inside the model, resize outputs back to (640x480), normalize the output to [0,1], and return a 4D tensor as required. 


## 3. Self-Evaluation

Now, we will demonstrate how to use [Qualcomm® AI Hub](https://app.aihub.qualcomm.com/) to profile the running time of our model and process an image on real mobile devices. 

Before moving forward, please make sure you have registered an account on [Qualcomm® AI Hub](https://app.aihub.qualcomm.com/) and installed the AI Hub Python client. 

``` bash
pip3 install qai-hub

# Once signed in navigate to Qualcomm® AI Hub website
# Account -> Settings -> API Token. 
# This should provide an API token that you can use to configure your client.
qai-hub configure --api_token INSERT_API_TOKEN
```

Then, we can directly run the following Python script to profile our modified Depth-Anything-V2 (you may want to follow the official instruction to set up the environment before running), process an image, and visualize the result.

``` bash
python demo_run.py
```

In this demo_run.py script, we demonstrate how the [Qualcomm® AI Hub](https://app.aihub.qualcomm.com/) works: 
- Step 0. Build our model and load its pre-trained weights.
- Step 1. Trace the model using torch.jit.trace.
- Step 2. Submit the traced model to [Qualcomm® AI Hub](https://app.aihub.qualcomm.com/) for model compilation so that the model can run efficiently on real mobile devices. 
- Step 3. Once the model is compiled, we profile it on a cloud-hosted device. 
- Step 4. Meanwhile, we run an inference on cloud-hosted device and visualize the result. 

Once the script is successfully run, we can check the status and profiling results of our model on [Qualcomm® AI Hub](https://app.aihub.qualcomm.com/). We can also see a visualized result saved at the folder of this repository as "test_tea_vits_518x518.png".
- "COMPILE" page shows the details of the compilation. We can download the compiled model if we desired. The "VISUALIZE" button will display the model structure. 
- "PROFILE" page lists the profiling details, such as inference time and memory usage. We can check "Runtime Layer Analysis" for the running time of each layer of our model, allowing us to optimize it for faster inference speed. 
- "INFERENCE" page corresponds to our image inference job. We can also download the inference result by clicking "OUTPUT DATASET" button for a specific job. 


For our modified Depth-Anything-V2, we should see a similar profiling result on the "PROFILE" page as follows. Since 241.8 ms is longer than the required 30 FPS, we need to optimize this model or design a more efficient one to win the challenge.

| Job Name | Job ID|Status|Inference Time|Peak Memory|Target Device|
|   :---:  | :---:|:---:|:---:|:---:|:---:|
| depth_anything_v2_vits_518x518 | - | Results Ready | 241.8 ms | 0 - 442 MB | Samsung Galaxy S24 (Family) |

Please refer to the [Qualcomm® AI Hub Documents](https://app.aihub.qualcomm.com/docs/index.html) for more details and functions. 


## 4. Submit Models for Ranking

In the previous section, we profiled our model on real devices. Once we are satisfied with its performance and accuracy, we will submit our model for metric calculation and ranking. 


- **Step1.** Share access permission with [lowpowervision@gmail.com](lowpowervision@gmail.com). This ensures that our evaluation server can access submitted models. We have two options to share the access permission: 
    - Option1: Using the [Qualcomm® AI Hub](https://app.aihub.qualcomm.com/) Python client (have included in our demo_run.py).
        ``` python 
        compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])
        ```
    - Option2: On [Qualcomm® AI Hub](https://app.aihub.qualcomm.com/)
        - Nagative to "COMPILE" page.
        - Click the Job Name that we want to submit for ranking.
        - Click "SHARE" at the top right corner.
        - Add the email [lowpowervision@gmail.com](lowpowervision@gmail.com). 
    
- **Step2.** Fill up a submission form at [this link](https://lpcv.ai/2025LPCVC/submission/track3).


Once the submission form is uploaded, the model (specified by the compile job id) will be evaluated and the ranking result will be available on our [leaderboard](https://lpcv.ai/2025LPCVC/leaderboard/track3). 

**P.S.: Our models will not be evaluated and ranked unless we complete a submission form. Each model requires a unique submission form because we will specify the Compile Job Id.**

___

Finally, we hope this repository provides you with sufficient details to quickly tackle this challenge. Next, you will spend more time optimizing your models and boosting your team’s ranking. Good luck and enjoy!
