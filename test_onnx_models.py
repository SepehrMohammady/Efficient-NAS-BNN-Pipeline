#!/usr/bin/env python3
"""
ONNX Model Testing Script for NAS-BNN Pipeline
Tests exported ONNX models on WakeVision dataset and compares with PyTorch model performance.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import onnxruntime as ort
from PIL import Image
import time
import json
from tqdm import tqdm

class ONNXModelTester:
    def __init__(self, onnx_model_path, data_dir, img_size=128, batch_size=32):
        self.onnx_model_path = onnx_model_path
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Initialize ONNX Runtime session
        self.ort_session = ort.InferenceSession(onnx_model_path)
        
        # Get model input/output info
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        # Setup data transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load validation dataset
        val_dir = os.path.join(data_dir, 'val')
        self.val_dataset = datasets.ImageFolder(val_dir, transform=self.transform)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, 
                                   shuffle=False, num_workers=2)
        
        print(f"✅ ONNX Model loaded: {os.path.basename(onnx_model_path)}")
        print(f"✅ Dataset loaded: {len(self.val_dataset)} validation images")
        print(f"✅ Input shape: {self.ort_session.get_inputs()[0].shape}")
        print(f"✅ Output shape: {self.ort_session.get_outputs()[0].shape}")
        
    def test_single_image(self, image_path):
        """Test ONNX model on a single image"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).numpy()
        
        # Run inference
        start_time = time.time()
        outputs = self.ort_session.run([self.output_name], {self.input_name: input_tensor})
        inference_time = time.time() - start_time
        
        # Get prediction
        probabilities = torch.softmax(torch.tensor(outputs[0]), dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'inference_time': inference_time,
            'probabilities': probabilities[0].numpy()
        }
    
    def test_batch_inference(self, num_batches=10):
        """Test batch inference performance"""
        print(f"\n🔄 Testing batch inference performance ({num_batches} batches)...")
        
        inference_times = []
        batch_count = 0
        
        for batch_idx, (images, targets) in enumerate(self.val_loader):
            if batch_count >= num_batches:
                break
                
            # Convert to numpy
            input_batch = images.numpy()
            
            # Run inference
            start_time = time.time()
            outputs = self.ort_session.run([self.output_name], {self.input_name: input_batch})
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            batch_count += 1
        
        avg_inference_time = np.mean(inference_times)
        avg_time_per_image = avg_inference_time / self.batch_size
        
        print(f"✅ Average batch inference time: {avg_inference_time:.4f}s")
        print(f"✅ Average time per image: {avg_time_per_image:.4f}s")
        print(f"✅ Images per second: {1.0/avg_time_per_image:.2f}")
        
        return {
            'avg_batch_time': avg_inference_time,
            'avg_time_per_image': avg_time_per_image,
            'images_per_second': 1.0/avg_time_per_image
        }
    
    def test_accuracy(self, max_batches=None):
        """Test model accuracy on validation set"""
        print(f"\n📊 Testing model accuracy on validation set...")
        
        correct = 0
        total = 0
        class_correct = {}
        class_total = {}
        
        # Initialize class counters
        for class_idx in range(len(self.val_dataset.classes)):
            class_correct[class_idx] = 0
            class_total[class_idx] = 0
        
        batch_count = 0
        
        with tqdm(total=len(self.val_loader), desc="Evaluating") as pbar:
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                if max_batches and batch_count >= max_batches:
                    break
                
                # Convert to numpy
                input_batch = images.numpy()
                
                # Run inference
                outputs = self.ort_session.run([self.output_name], {self.input_name: input_batch})
                predictions = np.argmax(outputs[0], axis=1)
                
                # Calculate accuracy
                for i in range(len(targets)):
                    predicted = predictions[i]
                    actual = targets[i].item()
                    
                    total += 1
                    class_total[actual] += 1
                    
                    if predicted == actual:
                        correct += 1
                        class_correct[actual] += 1
                
                batch_count += 1
                pbar.update(1)
        
        # Calculate overall accuracy
        overall_accuracy = 100.0 * correct / total
        
        # Calculate per-class accuracy
        class_accuracies = {}
        for class_idx in range(len(self.val_dataset.classes)):
            if class_total[class_idx] > 0:
                class_accuracies[self.val_dataset.classes[class_idx]] = \
                    100.0 * class_correct[class_idx] / class_total[class_idx]
        
        print(f"✅ Overall Accuracy: {overall_accuracy:.2f}%")
        print(f"✅ Total images tested: {total}")
        
        # Print per-class accuracy
        print("\n📋 Per-class accuracy:")
        for class_name, accuracy in class_accuracies.items():
            print(f"  {class_name}: {accuracy:.2f}%")
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_images': total,
            'class_accuracies': class_accuracies,
            'correct_predictions': correct
        }
    
    def compare_with_pytorch(self, pytorch_model_path=None):
        """Compare ONNX model with PyTorch model (if available)"""
        if pytorch_model_path and os.path.exists(pytorch_model_path):
            print(f"\n🔄 Comparing with PyTorch model: {pytorch_model_path}")
            # This would require loading the PyTorch model and comparing
            # For now, just report ONNX results
            pass
        else:
            print(f"\n⚠️  PyTorch model not provided for comparison")
    
    def generate_report(self, output_dir="./onnx_test_results"):
        """Generate comprehensive test report"""
        print(f"\n📊 Generating comprehensive test report...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Test accuracy
        accuracy_results = self.test_accuracy()
        
        # Test performance
        performance_results = self.test_batch_inference()
        
        # Model info
        model_info = {
            'model_path': self.onnx_model_path,
            'model_size_mb': os.path.getsize(self.onnx_model_path) / (1024 * 1024),
            'input_shape': self.ort_session.get_inputs()[0].shape,
            'output_shape': self.ort_session.get_outputs()[0].shape,
            'dataset_size': len(self.val_dataset)
        }
        
        # Combine results
        full_report = {
            'model_info': model_info,
            'accuracy_results': accuracy_results,
            'performance_results': performance_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save report
        model_name = os.path.basename(self.onnx_model_path).replace('.onnx', '')
        report_path = os.path.join(output_dir, f"{model_name}_test_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        print(f"✅ Test report saved to: {report_path}")
        
        # Print summary
        print(f"\n📋 Test Summary for {model_name}:")
        print(f"  Model Size: {model_info['model_size_mb']:.2f} MB")
        print(f"  Accuracy: {accuracy_results['overall_accuracy']:.2f}%")
        print(f"  Inference Speed: {performance_results['images_per_second']:.2f} images/sec")
        print(f"  Average Time per Image: {performance_results['avg_time_per_image']:.4f}s")
        
        return full_report

def main():
    parser = argparse.ArgumentParser(description='Test ONNX models from NAS-BNN pipeline')
    parser.add_argument('onnx_model', help='Path to ONNX model file')
    parser.add_argument('data_dir', help='Path to validation dataset directory')
    parser.add_argument('--img-size', type=int, default=128, help='Input image size')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--max-batches', type=int, help='Maximum number of batches to test')
    parser.add_argument('--output-dir', default='./onnx_test_results', help='Output directory for results')
    parser.add_argument('--single-image', help='Test on a single image')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ONNXModelTester(
        onnx_model_path=args.onnx_model,
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size
    )
    
    if args.single_image:
        # Test single image
        result = tester.test_single_image(args.single_image)
        print(f"\n🔍 Single Image Test Results:")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Inference Time: {result['inference_time']:.4f}s")
    else:
        # Generate comprehensive report
        report = tester.generate_report(args.output_dir)

if __name__ == '__main__':
    main()
