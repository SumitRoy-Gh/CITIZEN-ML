from ultralytics import YOLO
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def predict_single_image():
    """Fixed script to predict on a single image - no OpenCV display issues"""
    
    # Your trained model path
    model_path = r"urban_detection_results\yolov11_urban_v13\weights\best.pt"
    
    # Class names
    class_names = {
        0: 'pothole',
        1: 'damaged_streetlight', 
        2: 'water_puddle',
        3: 'garbage'
    }
    
    print("🚀 Urban Detection - Single Image Predictor (FIXED)")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("💡 Please update the model_path variable")
        return
    
    # Load model
    print("📥 Loading model...")
    model = YOLO(model_path)
    print("✅ Model loaded successfully!")
    print("💡 Just type image filename if it's in the same folder (e.g., 'image.png')")
    print("💡 Or provide full path for images elsewhere")
    
    while True:
        # Get image path from user
        image_path = input("\n📸 Enter image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() == 'quit':
            print("👋 Goodbye!")
            break
            
        # Skip if user accidentally pasted command
        if image_path.startswith('python'):
            print("💡 That looks like a command, not an image path. Try again!")
            continue
        
        # Remove quotes if user copied path with quotes
        image_path = image_path.strip('"\'')
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            print("💡 Please check the path and try again")
            continue
        
        try:
            print(f"🔍 Processing: {os.path.basename(image_path)}")
            
            # Run prediction
            results = model(image_path, verbose=False)  # Turn off verbose output
            result = results[0]
            
            # Display results in console
            if len(result.boxes) > 0:
                print(f"✅ Found {len(result.boxes)} detections:")
                print("-" * 50)
                
                for i, box in enumerate(result.boxes, 1):
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    print(f"{i}. {class_name.upper()}")
                    print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
                    print(f"   Bounding Box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                    print(f"   Size: {int(x2-x1)} x {int(y2-y1)} pixels")
                    print()
            else:
                print("❌ No objects detected in this image")
            
            # Save annotated image (no display issues)
            try:
                annotated_img = result.plot()
                
                # Create output filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = f"detected_{base_name}.jpg"
                
                # Convert BGR to RGB for saving
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                # Save using PIL (more reliable)
                pil_image = Image.fromarray(annotated_img_rgb)
                pil_image.save(output_path)
                
                print(f"✅ Annotated image saved as: {output_path}")
                print(f"📁 Check your current folder: {os.getcwd()}")
                
                # Option to show using matplotlib (works better than cv2.imshow)
                show_choice = input("👁️  Display image using matplotlib? (y/n): ").strip().lower()
                if show_choice == 'y':
                    plt.figure(figsize=(12, 8))
                    plt.imshow(annotated_img_rgb)
                    plt.axis('off')
                    plt.title(f'Detections: {os.path.basename(image_path)}', fontsize=14)
                    plt.tight_layout()
                    plt.show()
                    
            except Exception as save_error:
                print(f"⚠️  Could not save image: {save_error}")
                print("But detection results are shown above!")
                
        except Exception as e:
            print(f"❌ Error processing image: {e}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    predict_single_image()