import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

class InteractiveSegmentation:
    def __init__(self, checkpoint_path, device='cuda:0' if torch.cuda.is_available() else 'cpu', max_display_size=800):
        self.device = device
        self.max_display_size = max_display_size  # Maximum size for display window
        print(f"Using device: {self.device}")
        
        # Initialize SAM
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        # Variables for interactive segmentation
        self.image = None
        self.clicks = []
        self.click_labels = []  # 1 for foreground, 0 for background
        self.current_mask = None
        self.original_image = None
        self.display_image = None
        self.scale_factor = 1.0
        self.window_name = "Interactive Segmentation"
        
    def resize_image_for_display(self, image):
        """Resize image to fit within max_display_size while preserving aspect ratio"""
        h, w = image.shape[:2]
        
        # Calculate scale factor to fit within max_display_size
        scale = min(self.max_display_size / w, self.max_display_size / h)
        
        # Only scale down, not up
        if scale < 1.0:
            self.scale_factor = scale
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))
            return resized
        else:
            self.scale_factor = 1.0
            return image.copy()
    
    def display_to_original_coords(self, x, y):
        """Convert display coordinates to original image coordinates"""
        return int(x / self.scale_factor), int(y / self.scale_factor)
            
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click for foreground
            orig_x, orig_y = self.display_to_original_coords(x, y)
            self.clicks.append([orig_x, orig_y])
            self.click_labels.append(1)
            self.update_mask()
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click for background
            orig_x, orig_y = self.display_to_original_coords(x, y)
            self.clicks.append([orig_x, orig_y])
            self.click_labels.append(0)
            self.update_mask()
    
    def update_mask(self):
        if len(self.clicks) == 0:
            return
        
        input_points = np.array(self.clicks)
        input_labels = np.array(self.click_labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Get the highest-scoring mask
        mask_idx = np.argmax(scores)
        self.current_mask = masks[mask_idx]
        
        # Visualize
        self.display_result()
    
    def display_result(self):
        # Create a visualization with the mask overlay
        display_img = self.original_image.copy()
        
        # Apply mask overlay (semi-transparent red)
        mask_overlay = np.zeros_like(display_img)
        mask_overlay[self.current_mask] = [0, 0, 255]  # Red color for mask
        display_img = cv2.addWeighted(display_img, 1.0, mask_overlay, 0.5, 0)
        
        # Draw points
        for i, (x, y) in enumerate(self.clicks):
            color = (0, 255, 0) if self.click_labels[i] == 1 else (0, 0, 255)  # Green for foreground, Red for background
            cv2.circle(display_img, (x, y), 5, color, -1)
        
        # Resize for display
        display_img = self.resize_image_for_display(display_img)
        
        # Show display image
        cv2.imshow(self.window_name, display_img[:, :, ::-1])  # Convert RGB to BGR for OpenCV
    
    def segment_image(self, image_path, output_dir):
        # Read the image
        self.original_image = cv2.imread(image_path)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Set image in predictor
        self.predictor.set_image(self.original_image)
        
        # Create display version
        self.display_image = self.resize_image_for_display(self.original_image)
        
        # Initialize window and callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Print image information
        h, w = self.original_image.shape[:2]
        disp_h, disp_w = self.display_image.shape[:2]
        print(f"Original image size: {w}x{h}")
        print(f"Display window size: {disp_w}x{disp_h} (scale factor: {self.scale_factor:.2f})")
        
        # Display instructions
        print("Instructions:")
        print("- Left click to select clothing areas (foreground)")
        print("- Right click to mark non-clothing areas (background)")
        print("- Press 's' to save and exit")
        print("- Press 'r' to reset all clicks")
        print("- Press 'q' to quit without saving")
        
        # Initial display
        cv2.imshow(self.window_name, self.display_image[:, :, ::-1])  # Convert RGB to BGR for OpenCV
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save
                if self.current_mask is not None:
                    self.save_results(output_dir)
                break
            elif key == ord('r'):  # Reset
                self.clicks = []
                self.click_labels = []
                self.current_mask = None
                cv2.imshow(self.window_name, self.display_image[:, :, ::-1])
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()
    
    def save_results(self, output_dir):
        """Save the generated mask and agnostic image"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create agnostic image by removing the clothing
        agnostic_image = self.original_image.copy()
        
        # Fill clothing area with a neutral color
        agnostic_image[self.current_mask] = [192, 192, 192]  # Light gray color
        
        # Save original image
        plt.figure(figsize=(10, 10))
        plt.imshow(self.original_image)
        plt.title("Original Image")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "original_image.png"))
        plt.close()
        
        # Save clothing mask
        plt.figure(figsize=(10, 10))
        plt.imshow(self.current_mask, cmap='gray')
        plt.title("Clothing Mask")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "clothing_mask.png"))
        plt.close()
        
        # Save agnostic image
        plt.figure(figsize=(10, 10))
        plt.imshow(agnostic_image)
        plt.title("Agnostic Image")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "agnostic_image.png"))
        plt.close()
        
        # Save binary mask for StableVITON
        binary_mask = (self.current_mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, "clothing_binary_mask.png"), binary_mask)
        
        print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive segmentation for clothing removal")
    parser.add_argument("--image", type=str, required=True, help="Path to the person image")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam_vit_h_4b8939.pth", 
                        help="Path to SAM checkpoint")
    parser.add_argument("--max_display_size", type=int, default=800,
                        help="Maximum size (width/height) for display window")
    
    args = parser.parse_args()
    
    segmenter = InteractiveSegmentation(args.checkpoint, max_display_size=args.max_display_size)
    segmenter.segment_image(args.image, args.output_dir)