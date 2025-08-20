import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import UNetGenerator, PatchGANDiscriminator, Pix2PixDataset, create_models, create_small_models
from torchvision.models import vgg16

class Pix2PixLoss:
    """
    Pix2Pix GAN Loss Functions
    """
    def __init__(self, lambda_L1: float = 100.0):
        self.lambda_L1 = lambda_L1
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()
        vgg = vgg16(weights="IMAGENET1K_V1").features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    
    def perceptual_loss(self, generated, real):
        """VGG perceptual loss"""
        gen_features = self.vgg(generated)
        real_features = self.vgg(real)
        return nn.functional.l1_loss(gen_features, real_features)

    def discriminator_loss(self, real_output, fake_output):
        """
        Discriminator loss: maximize log(D(x)) + log(1 - D(G(z)))
        """
        real_labels = torch.ones_like(real_output)
        fake_labels = torch.zeros_like(fake_output)
        
        real_loss = self.criterion_GAN(real_output, real_labels)
        fake_loss = self.criterion_GAN(fake_output, fake_labels)
        
        return real_loss + fake_loss
    
    def generator_loss(self, fake_output, generated_images, real_images):
        """
        Generator loss: minimize log(1 - D(G(z))) + λ * L1(G(z), y)
        """
        # GAN loss
        real_labels = torch.ones_like(fake_output)
        gan_loss = self.criterion_GAN(fake_output, real_labels)
        
        # L1 loss
        l1_loss = self.criterion_L1(generated_images, real_images)
        
         # Perceptual loss
        p_loss = self.perceptual_loss(generated_images, real_images)

        return gan_loss + self.lambda_L1 * l1_loss + 0.5 * p_loss

class Pix2PixTrainer:
    """
    Pix2Pix GAN Trainer
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create models
        self.generator, self.discriminator = create_small_models(self.device)
        
        # Create loss functions
        self.criterion = Pix2PixLoss(lambda_L1=config['lambda_L1'])
        
        # Create optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), 
                                    lr=config['learning_rate'], 
                                    betas=(config['beta1'], config['beta2']))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), 
                                    lr=config['learning_rate'], 
                                    betas=(config['beta1'], config['beta2']))
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, 
                                                    step_size=config['lr_decay_epochs'], 
                                                    gamma=config['lr_decay_factor'])
        self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, 
                                                    step_size=config['lr_decay_epochs'], 
                                                    gamma=config['lr_decay_factor'])
        
        # Create datasets and dataloaders
        self.train_dataset = Pix2PixDataset(config['data_dir'], 
                                          split='train', 
                                          image_size=config['image_size'])
        self.val_dataset = Pix2PixDataset(config['data_dir'], 
                                        split='val', 
                                        image_size=config['image_size'])
        
        self.train_loader = DataLoader(self.train_dataset, 
                                     batch_size=config['batch_size'], 
                                     shuffle=True, 
                                     num_workers=config['num_workers'])
        self.val_loader = DataLoader(self.val_dataset, 
                                   batch_size=config['batch_size'], 
                                   shuffle=False, 
                                   num_workers=config['num_workers'])
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Create output directories
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'samples'), exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'logs'), exist_ok=True)
        
        # Training history
        self.history = {
            'train_g_loss': [],
            'train_d_loss': [],
            'val_g_loss': [],
            'val_d_loss': [],
            'learning_rates': []
        }
        # Add this method inside the Pix2PixTrainer class in train.py
    def validate_and_save_samples(self, epoch, num_samples=8):

        """Save sample images from the validation set"""
        self.generator.eval()
        # Get a fixed batch of validation data to see progress
        if not hasattr(self, 'val_sample_batch'):
            self.val_sample_batch = next(iter(self.val_loader))

        sketches, real_images = self.val_sample_batch
        sketches = sketches.to(self.device)
        
        with torch.no_grad():
            generated_images = self.generator(sketches)

        # Denormalize images
        def denormalize(tensor):
            return (tensor * 0.5) + 0.5 # Correct denormalization

        sketches_d = denormalize(sketches[:num_samples].cpu())
        real_images_d = denormalize(real_images[:num_samples].cpu())
        generated_images_d = denormalize(generated_images[:num_samples].cpu())

        comparison = torch.cat([sketches_d, real_images_d, generated_images_d], dim=0)
        grid = vutils.make_grid(comparison, nrow=num_samples, padding=2, normalize=False)

        val_sample_dir = os.path.join(self.config['output_dir'], 'validation_samples')
        os.makedirs(val_sample_dir, exist_ok=True)
        vutils.save_image(grid, os.path.join(val_sample_dir, f'validation_epoch_{epoch:04d}.png'))
        self.generator.train()
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.current_epoch = checkpoint['epoch']
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def save_samples(self, epoch, sketches, real_images, generated_images, num_samples=8):
        """Save sample images"""
        # Denormalize images
        def denormalize(tensor):
            return (tensor + 1) / 2
        
        sketches = denormalize(sketches[:num_samples])
        real_images = denormalize(real_images[:num_samples])
        generated_images = denormalize(generated_images[:num_samples])
        
        # Create grid
        comparison = torch.cat([sketches, real_images, generated_images], dim=0)
        grid = vutils.make_grid(comparison, nrow=num_samples, padding=2, normalize=False)
        
        # Save image
        vutils.save_image(grid, 
                         os.path.join(self.config['output_dir'], 'samples', f'epoch_{epoch:04d}.png'))
    
    def train_epoch(self):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (sketches, real_images) in enumerate(progress_bar):
            sketches = sketches.to(self.device)
            real_images = real_images.to(self.device)
            
            # Train Discriminator
            self.optimizer_D.zero_grad()
            
            # Generate fake images
            with torch.no_grad():
                fake_images = self.generator(sketches)
            
            # Real images
            real_output = self.discriminator(torch.cat([sketches, real_images], dim=1))
            # Fake images
            fake_output = self.discriminator(torch.cat([sketches, fake_images.detach()], dim=1))
            
            d_loss = self.criterion.discriminator_loss(real_output, fake_output)
            d_loss.backward()
            self.optimizer_D.step()
            
            # Train Generator
            self.optimizer_G.zero_grad()
            
            # Generate fake images again
            fake_images = self.generator(sketches)
            fake_output = self.discriminator(torch.cat([sketches, fake_images], dim=1))
            
            g_loss = self.criterion.generator_loss(fake_output, fake_images, real_images)
            g_loss.backward()
            self.optimizer_G.step()
            
            # Update statistics
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_Loss': f'{g_loss.item():.4f}',
                'D_Loss': f'{d_loss.item():.4f}'
            })
            
            # Save samples periodically
            if batch_idx % self.config['sample_interval'] == 0:
                self.save_samples(self.current_epoch, sketches, real_images, fake_images)
        
        # Update learning rates
        self.scheduler_G.step()
        self.scheduler_D.step()
        
        return total_g_loss / num_batches, total_d_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.generator.eval()
        self.discriminator.eval()
        
        total_g_loss = 0
        total_d_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for sketches, real_images in self.val_loader:
                sketches = sketches.to(self.device)
                real_images = real_images.to(self.device)
                
                # Generate fake images
                fake_images = self.generator(sketches)
                
                # Discriminator outputs
                real_output = self.discriminator(torch.cat([sketches, real_images], dim=1))
                fake_output = self.discriminator(torch.cat([sketches, fake_images], dim=1))
                
                # Calculate losses
                d_loss = self.criterion.discriminator_loss(real_output, fake_output)
                g_loss = self.criterion.generator_loss(fake_output, fake_images, real_images)
                
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
        
        return total_g_loss / num_batches, total_d_loss / num_batches
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_g_loss, train_d_loss = self.train_epoch()
            
            # Validate
            val_g_loss, val_d_loss = self.validate()

            self.validate_and_save_samples(epoch)
            
            # Update history
            self.history['train_g_loss'].append(train_g_loss)
            self.history['train_d_loss'].append(train_d_loss)
            self.history['val_g_loss'].append(val_g_loss)
            self.history['val_d_loss'].append(val_d_loss)
            self.history['learning_rates'].append(self.optimizer_G.param_groups[0]['lr'])
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{self.config['num_epochs']-1}")
            print(f"Train G Loss: {train_g_loss:.4f}, D Loss: {train_d_loss:.4f}")
            print(f"Val G Loss: {val_g_loss:.4f}, D Loss: {val_d_loss:.4f}")
            print(f"Learning Rate: {self.optimizer_G.param_groups[0]['lr']:.6f}")
            print(f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_g_loss < self.best_val_loss:
                self.best_val_loss = val_g_loss
                self.save_checkpoint(os.path.join(self.config['output_dir'], 
                                                'checkpoints', 'best_model.pth'))
                print("Saved best model!")
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(os.path.join(self.config['output_dir'], 
                                                'checkpoints', f'checkpoint_epoch_{epoch}.pth'))
            
            # Save training history
            with open(os.path.join(self.config['output_dir'], 'logs', 'training_history.json'), 'w') as f:
                json.dump(self.history, f, indent=2)
        
        print("Training completed!")
        self.save_checkpoint(os.path.join(self.config['output_dir'], 
                                        'checkpoints', 'final_model.pth'))

def main():
    # Configuration
    # In train.py, modify the config dictionary:
    config = {
        'data_dir': r"E:\1\Code\Python\face_sketch_to_image\sketch-to-image-main\prepared_data",  # Use your augmented data
        'output_dir': r"E:\1\Code\Python\face_sketch_to_image\output_data",
        'image_size': 256,
        'batch_size': 3,  # Reduced from 8 to 2 for memory
        'num_epochs': 400,
        'learning_rate': 0.0002,
        'beta1': 0.5,
        'beta2': 0.999,
        'lambda_L1': 100.0,
        'num_workers': 1,  # <<< CHANGE THIS FROM 2 to 0
        'sample_interval': 200, 
        'save_interval': 10,
        'lr_decay_epochs': 50,
        'lr_decay_factor': 0.5
    }
    
    # Create trainer and start training
    trainer = Pix2PixTrainer(config)
    
    # Uncomment to resume training from checkpoint
    # trainer.load_checkpoint('training_output/checkpoints/checkpoint_epoch_50.pth')
    
    trainer.train()

if __name__ == "__main__":
    main() 