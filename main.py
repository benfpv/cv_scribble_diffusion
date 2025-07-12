import numpy as np
import time
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import threading

import os

from functions_diffusion_image_s import *

# Path to the local model directory
local_pipe_path = r"stable-diffusion-v1-5"

local_scribble_path = r"sd-controlnet-scribble"

# Keyboard Inputs
# - Spacebar: Reset the canvas to blank
# - Enter: Save the current image (without paint-mask visible) to current working directory
# - Tab: Toggle paint-mask visibility
# - Left: Reduce brush size
# - Right: Increase brush size
# - Esc: Exit app

class Main():
    def __init__(self):
        ### Parameters (all x sizes must be identical to y sizes for now) ###
        self.image_sizes_minmax = [[240,240], [280,280], [320,320], [360,360], [400,400], [440,440], [480,480], [512,512]]
        self.image_size = (512,512) 
        self.present_size = (800,800) # window size, image_size is resized to present size before presenting to window.
        self.use_gpu = True # False for CPU (for artists), True for GPU (for showoffs); disable "enable optimizations
        
        self.brush_thickness = 1
        self.prompt = "colourful beautiful anime manga"
        
        self.min_inference_steps = 2
        self.max_inference_steps = 18
        self.rate_inference_steps_change = 2
        
        ### Init ###
        if torch.cuda.is_available():
            print("CUDA is available.")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        else:
            print("CUDA is not available. Running on CPU.")
            self.use_gpu = False
            #exit()
        print(f"Use GPU?: {self.use_gpu}")
        
        self.drawing = False
        self.mask = np.zeros(self.image_size, dtype="uint8")
        self.mask_active = np.zeros(self.present_size, dtype="uint8")
        self.mask_present = np.zeros((self.present_size[0],self.present_size[1],3), dtype="uint8")
        self.image = Image.new("RGB", self.image_size, (0, 0, 0))
        self.image_present = np.zeros((self.present_size[0],self.present_size[1],3), dtype="uint8")
        
        self.image_sizes_max_index = len(self.image_sizes_minmax)-1
        self.image_size_index = 0
        
        self.brush_point_thickness = self.brush_thickness
        self.brush_stroke_multiplier = 1.5
        self.brush_stroke_thickness = max([2, int(self.brush_thickness * self.brush_stroke_multiplier)])
        
        self.num_inference_steps = -1
        
        self.prev_x = -1
        self.prev_y = -1
        
        self.mask_visibility_toggle = True
        self.image_save_count = 1
        self.image_diffused_this_loop = False
        self.skip_next_image = False
        
        self.image_store_limit_count = 9

        # Load ControlNet model
        self.controlnet = ControlNetModel.from_pretrained(
                                                        local_scribble_path, 
                                                        torch_dtype=torch.float16
                                                        )
        # Load Stable Diffusion pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                                                                    local_pipe_path,
                                                                    controlnet=self.controlnet,
                                                                    torch_dtype=torch.float16
                                                                    )
        if (self.use_gpu):
            self.pipe.to("cuda")
            self.controlnet.to("cuda")
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_sequential_cpu_offload()
            
        # Tweak the pipeline
        self.pipe.safety_checker = None
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_vae_slicing()
        self.pipe.enable_attention_slicing()
        
        cv2.namedWindow("ai_paint_diffusion")
        cv2.setMouseCallback("ai_paint_diffusion", self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        if (event == cv2.EVENT_LBUTTONDOWN):
            self.drawing = True
            cv2.circle(self.mask_active, (x, y), self.brush_point_thickness, 150, -1)
            self.num_inference_steps = self.min_inference_steps
            self.prev_x = x
            self.prev_y = y
        elif (event == cv2.EVENT_MOUSEMOVE):
            if (self.drawing):
                cv2.line(self.mask_active, (self.prev_x, self.prev_y), (x,y), 150, self.brush_stroke_thickness)
                self.num_inference_steps = self.min_inference_steps
                self.prev_x = x
                self.prev_y = y
        elif (event == cv2.EVENT_LBUTTONUP):
            self.drawing = False
            self.prev_x = x
            self.prev_y = y
            self.mask_active = np.zeros(self.present_size, dtype="uint8")
        
    def async_diffusion(self):
        while 1:
            if (self.num_inference_steps > 1):
                image_size = self.image_sizes_minmax[self.image_size_index]
                mask = self.mask.copy()
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                mask = Image.fromarray(mask)
                mask.resize(image_size)
                inference_steps = self.num_inference_steps # prevents real-time change while diffusing
                self.image = Functions_Diffusion_Image.run_cnet_pipe_scribble(self.pipe, mask, self.prompt, inference_steps, image_size=image_size).resize(self.image_size)
                self.image_present = cv2.resize(cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR), self.present_size, interpolation=cv2.INTER_LINEAR)
                self.image_diffused_this_loop = True
                while (self.image_diffused_this_loop):
                    time.sleep(0.1)
                self.image_size_index += 1
                self.image_size_index = np.min([self.image_size_index, self.image_sizes_max_index])
                self.num_inference_steps += self.rate_inference_steps_change
                self.num_inference_steps = np.min([self.num_inference_steps, self.max_inference_steps])
                time.sleep(0.1)
            else:
                time.sleep(1)
            
    def run(self):
        thread = threading.Thread(target=self.async_diffusion, daemon=True)
        thread.start()
        
        while 1:
            ### Present ###
            if (np.any(self.mask_active)):
                self.mask = cv2.add(self.mask, cv2.resize(self.mask_active, self.image_size, interpolation=cv2.INTER_NEAREST))
                self.mask_present = cv2.resize(cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR), self.present_size, interpolation=cv2.INTER_LINEAR)
                if (self.mask_visibility_toggle == False):
                    self.image_present = cv2.add(self.image_present, cv2.cvtColor(self.mask_active, cv2.COLOR_GRAY2BGR))
                self.image_size_index = 0
                self.num_inference_steps = self.min_inference_steps
                
            if (self.mask_visibility_toggle):
                self.image_present = cv2.add(self.image_present, self.mask_present)
                
            if (self.skip_next_image):
                if (self.image_diffused_this_loop):
                    self.image_diffused_this_loop = False
                    self.skip_next_image = False
                    self.image_present = np.zeros((self.present_size[0],self.present_size[1],3), dtype="uint8")
                # Present
                cv2.imshow("ai_paint_diffusion", self.image_present)
                cv2.moveWindow("ai_paint_diffusion", 300, 0)
            else:
                # Present
                cv2.imshow("ai_paint_diffusion", self.image_present)
                cv2.moveWindow("ai_paint_diffusion", 300, 0)
                self.image_diffused_this_loop = False
            
            key = cv2.waitKey(1) & 0xFF  # Waits 1 ms for a key press
            
            if (key == 27): # Esc
                print("...[Esc] Exiting...")
                break
            elif (key == 9): # Tab
                if (self.mask_visibility_toggle):
                    print(f"...[Tab] Toggle Mask Visibility [Off] (changes take effect on next present)...")
                    self.mask_visibility_toggle = False
                else:
                    print(f"...[Tab] Toggle Mask Visibility [On]...")
                    self.mask_visibility_toggle = True
            elif (key == 13): # Enter
                self.image_save_count = 1
                for i in range(self.image_store_limit_count - self.image_save_count + 1):
                    self.image_save_count += 1
                    self.image_save_name = f"saved_image_{self.image_save_count}.png"
                    if (not os.path.isfile(self.image_save_name)):
                        break
                if (self.image_save_count >= (self.image_store_limit_count+1)):
                    print(f"...[Enter] Saving failed - image count ({self.image_save_count}) exceeds limit ({self.image_store_limit_count})...")
                else:
                    print(f"...[Enter] Saving {self.image_save_name}...")
                    image_to_save = self.image.copy()
                    image_to_save = image_to_save.resize(self.image_size, resample=Image.NEAREST)
                    image_to_save = cv2.resize(cv2.cvtColor(np.array(image_to_save), cv2.COLOR_RGB2BGR), self.present_size, interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(self.image_save_name, image_to_save)
            elif (key == 32): # Space Bar
                if (self.skip_next_image == False):
                    print("...[SpaceBar] Reset Canvas (changes take effect on next present)...")
                    self.drawing = False
                    self.mask = np.zeros(self.image_size, dtype="uint8")
                    self.mask_active = np.zeros(self.present_size, dtype="uint8")
                    self.mask_present = np.zeros((self.present_size[0],self.present_size[1],3), dtype="uint8")
                    self.image = Image.new("RGB", self.image_size, (0, 0, 0))
                    self.image_present = np.zeros((self.present_size[0],self.present_size[1],3), dtype="uint8")
                    self.num_inference_steps = -1
                    self.skip_next_image = True
            elif (key == 81): # Left Arrow
                self.brush_thickness -= 1
                self.brush_thickness = np.max([1, self.brush_thickness])
                print(f"...[Left] [-] Brush Thickness to {self.brush_thickness}...")
                self.brush_point_thickness = self.brush_thickness
                self.brush_stroke_thickness = int(self.brush_thickness * self.brush_stroke_multiplier)
            elif (key == 83): # Right Arrow
                self.brush_thickness += 1
                self.brush_thickness = np.min([20, self.brush_thickness])
                print(f"...[Right] [+] Brush Thickness to {self.brush_thickness}...")
                self.brush_point_thickness = self.brush_thickness
                self.brush_stroke_thickness = int(self.brush_thickness * self.brush_stroke_multiplier)
            
            time.sleep(0.01)
                
if (__name__ == "__main__"):
    app = Main()
    app.run()