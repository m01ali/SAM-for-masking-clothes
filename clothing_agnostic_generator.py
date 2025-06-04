import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

class ClothingAgnosticMapGenerator:
    def __init__(self, checkpoint_path, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize SAM
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
    def generate_masks(self, image_path, clothing_box=None):
        """
        Generate masks for a person image to remove clothing.
        
        Args:
            image_path: Path to the image
            clothing_box: Optional bounding box for the clothing area [x1, y1, x2, y2]
                          If not provided, will attempt to detect automatically
        """
        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image in predictor
        self.predictor.set_image(image)
        
        # If no clothing box is provided, we can try to detect the torso area
        # This is a simplified approach - in production you'd want to use a dedicated 
        # human parsing model or provide accurate boxes
        if clothing_box is None:
            h, w = image.shape[:2]
            # Estimate a general clothing area (center torso region)
            # This is approximate and assumes a centered full-body photo
            clothing_box = [
                int(w * 0.25),  # x1: 25% from left
                int(h * 0.25),  # y1: 25% from top
                int(w * 0.75),  # x2: 75% from left
                int(h * 0.65),  # y2: 65% from top (to cover torso)
            ]
        
        # Generate masks for the clothing area
        input_point = np.array([
            [(clothing_box[0] + clothing_box[2]) // 2, (clothing_box[1] + clothing_box[3]) // 2]
        ])
        input_label = np.array([1])  # 1 for foreground point
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=np.array(clothing_box),
            multimask_output=True,
        )
        
        # Get the highest-scoring mask
        mask_idx = np.argmax(scores)
        clothing_mask = masks[mask_idx]
        
        # Create agnostic image by removing the clothing
        agnostic_image = image.copy()
        
        # Option 1: Fill clothing area with a neutral color (gray)
        agnostic_image[clothing_mask] = [192, 192, 192]  # Light gray color
        
        # Option 2: You could also use inpainting to fill the removed region
        # clothing_mask_uint8 = clothing_mask.astype(np.uint8) * 255
        # agnostic_image = cv2.inpaint(image, clothing_mask_uint8, 3, cv2.INPAINT_TELEA)
        
        return {
            'original_image': image,
            'clothing_mask': clothing_mask,
            'agnostic_image': agnostic_image,
        }
        
    def save_results(self, results, output_dir):
        """Save the generated masks and images"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original image
        plt.figure(figsize=(10, 10))
        plt.imshow(results['original_image'])
        plt.title("Original Image")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "original_image.png"))
        plt.close()
        
        # Save clothing mask
        plt.figure(figsize=(10, 10))
        plt.imshow(results['clothing_mask'], cmap='gray')
        plt.title("Clothing Mask")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "clothing_mask.png"))
        plt.close()
        
        # Save agnostic image
        plt.figure(figsize=(10, 10))
        plt.imshow(results['agnostic_image'])
        plt.title("Agnostic Image")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "agnostic_image.png"))
        plt.close()
        
        # Save binary mask for StableVITON
        binary_mask = (results['clothing_mask'] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, "clothing_binary_mask.png"), binary_mask)
        
        print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"
    generator = ClothingAgnosticMapGenerator(checkpoint_path)
    
    # Put your image path here
    image_path = "E:/SAM-StableVITON/input-test.jpg"
    
    # Optional: specify clothing bounding box [x1, y1, x2, y2]
    # clothing_box = [100, 150, 300, 450]  # Example values
    
    # Generate without specifying box (will use automatic estimation)
    results = generator.generate_masks(image_path)
    
    # Save results
    generator.save_results(results, "output")