from ultralytics import YOLO
import torch
import os

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📁 Created directory: {path}")

def train_urban_detection_model():
    try:
        print("🔍 Checking directories and files...")
        
        # Setup directory structure
        base_dir = 'urban_detection_results/yolov11_urban_v1'
        weights_dir = f"{base_dir}/weights"
        ensure_dir_exists(base_dir)
        ensure_dir_exists(weights_dir)
        
        checkpoint_path = f"{weights_dir}/last.pt"
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found at: {checkpoint_path}")
            print("💡 Please ensure:")
            print("   1. You have previously trained the model")
            print("   2. The checkpoint file is named 'last.pt'")
            print("   3. The file is in the correct location")
            print(f"\nExpected location: {os.path.abspath(checkpoint_path)}")
            return None
            
        print("🚀 Resuming YOLOv11 training from epoch 47...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"💻 Using device: {device}")
        
        model = YOLO(checkpoint_path)
        print("📊 Model loaded! Continuing training...")
        
        results = model.train(
            data='combined_dataset/data.yaml',
            epochs=100,                         
            imgsz=640,                         
            batch=8,                           
            lr0=0.001,                        # Reduced learning rate for continued training
            device=device,                     
            project='urban_detection_results', 
            name='yolov11_urban_v1',          
            resume=True,                       # This will resume from last checkpoint
            save=True,                         
            plots=True,                        
            verbose=True,                      
            workers=8,                         
            cache=False,
            amp=True                         # Enable mixed precision training (if supported by hardware                        
        )
        
        print("🎉 Training completed!")
        print("📊 Results saved in 'urban_detection_results/yolov11_urban_v1'")
        return results
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")
    return None

if __name__ == "__main__":
    results = train_urban_detection_model()

