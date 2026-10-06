"""
Setup Pretrained ResNet50 for CIFAR-10
Using weights from https://github.com/huyvnphan/PyTorch_CIFAR10

This script helps you install and verify the pretrained ResNet50 model
from the PyTorch_CIFAR10 repository (93.65% accuracy on CIFAR-10).
"""

import sys
import subprocess

def print_header(text):
    """Print formatted header"""
    print('\n' + '='*70)
    print(text)
    print('='*70 + '\n')

def install_cifar10_models():
    """Install cifar10_models package from GitHub"""
    print_header('Installing PyTorch_CIFAR10 Package')
    
    print('Repository: https://github.com/huyvnphan/PyTorch_CIFAR10')
    print('This package provides pretrained models for CIFAR-10')
    print('\nInstalling...')
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/huyvnphan/PyTorch_CIFAR10.git',
            '--quiet'
        ])
        print('\n✓ Installation successful!')
        return True
    except subprocess.CalledProcessError as e:
        print(f'\n✗ Installation failed: {e}')
        return False

def verify_installation():
    """Verify the installation by loading the model"""
    print_header('Verifying Installation')
    
    try:
        import torch
        from cifar10_models.resnet import resnet50
        
        print('Loading pretrained ResNet50...')
        model = resnet50(pretrained=True)
        model.eval()
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f'✓ Model loaded successfully!')
        print(f'  - Input shape: [1, 3, 32, 32]')
        print(f'  - Output shape: {list(output.shape)}')
        print(f'  - Number of classes: 10')
        print(f'  - Expected accuracy: ~93.65%')
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f'  - Parameters: {num_params/1e6:.3f}M')
        
        return True
        
    except ImportError as e:
        print(f'✗ Import failed: {e}')
        print('\nPlease ensure the package is installed correctly.')
        return False
    except Exception as e:
        print(f'✗ Verification failed: {e}')
        return False

def print_usage():
    """Print usage instructions"""
    print_header('How to Use')
    
    print('1. Test with Spiking-PGD (default - uses pretrained weights):')
    print('\n   python test_resnet50_spiking.py \\')
    print('       --rho 0.02 \\')
    print('       --steps 20 \\')
    print('       --epsilon 8')
    
    print('\n\n2. Test with different configurations:')
    print('\n   # High efficiency (more reuse)')
    print('   python test_resnet50_spiking.py --rho 0.03 --steps 20')
    
    print('\n   # Strong attack')
    print('   python test_resnet50_spiking.py --rho 0.02 --steps 50 --epsilon 16')
    
    print('\n   # Quick test on 1000 samples')
    print('   python test_resnet50_spiking.py --rho 0.02 --num_samples 1000')
    
    print('\n\n3. Run batch experiments:')
    print('\n   bash scripts/run_resnet50_experiments.sh')
    
    print('\n\n4. Use in your own code:')
    print('\n   ```python')
    print('   from cifar10_models.resnet import resnet50')
    print('   from spiking import SpikingModel')
    print('')
    print('   # Load pretrained model')
    print('   net = resnet50(pretrained=True)')
    print('   net.eval()')
    print('')
    print('   # Wrap with spiking mechanism')
    print('   spiking_model = SpikingModel(model=net, rho=0.02)')
    print('   ```')
    
    print('\n' + '='*70)

def print_model_info():
    """Print information about available models"""
    print_header('Available Models from PyTorch_CIFAR10')
    
    models_info = [
        ('resnet18', '93.07%', '11.174M', '43 MB'),
        ('resnet34', '93.34%', '21.282M', '82 MB'),
        ('resnet50', '93.65%', '23.521M', '91 MB'),
        ('vgg16_bn', '94.00%', '33.647M', '129 MB'),
        ('densenet121', '94.06%', '6.956M', '28 MB'),
        ('mobilenet_v2', '93.91%', '2.237M', '9 MB'),
    ]
    
    print('Model         | Accuracy | Parameters | Size')
    print('-' * 50)
    for name, acc, params, size in models_info:
        print(f'{name:13} | {acc:8} | {params:10} | {size}')
    
    print('\nNote: This script focuses on ResNet50')
    print('For other models, modify test_resnet50_spiking.py accordingly')

def main():
    print_header('ResNet50 CIFAR-10 Setup (PyTorch_CIFAR10)')
    
    print('This script will:')
    print('  1. Install cifar10_models package from GitHub')
    print('  2. Download pretrained ResNet50 weights (~91 MB)')
    print('  3. Verify the installation')
    print('')
    
    response = input('Continue? [Y/n]: ').strip().lower()
    if response and response not in ['y', 'yes']:
        print('Setup cancelled.')
        return
    
    # Step 1: Install package
    if not install_cifar10_models():
        print('\n⚠️  Installation failed!')
        print('You can try manual installation:')
        print('  pip install git+https://github.com/huyvnphan/PyTorch_CIFAR10.git')
        return
    
    # Step 2: Verify installation
    if not verify_installation():
        print('\n⚠️  Verification failed!')
        print('Please check your PyTorch installation and try again.')
        return
    
    # Step 3: Show available models
    print_model_info()
    
    # Step 4: Show usage
    print_usage()
    
    print_header('Setup Complete!')
    print('✓ cifar10_models package installed')
    print('✓ Pretrained ResNet50 weights downloaded')
    print('✓ Model verified and ready to use')
    print('\nYou can now run: python test_resnet50_spiking.py --rho 0.02')

if __name__ == '__main__':
    main()
