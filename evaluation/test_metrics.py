# test_metrics_optimized.py
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import lpips

def calculate_psnr(predicted, target):
    mse = F.mse_loss(predicted, target)
    return 20 * math.log10(1.0 / torch.sqrt(mse)) if mse > 0 else float('inf')

def create_archaeological_test():
    """Test optimisé pour artefacts archéologiques"""
    print("=== TEST OPTIMISÉ POUR ARTÉFACTS ARCHÉOLOGIQUES ===")
    
    device = torch.device('cpu')
    size = 300  # Résolution plus élevée pour les détails
    
    # Coordonnées
    y, x = torch.meshgrid(torch.linspace(-1, 1, size), 
                         torch.linspace(-1, 1, size), 
                         indexing='xy')
    
    # Simulation d'un artefact archéologique (vase avec texture)
    radius = 0.5
    distance = torch.sqrt(x**2 + y**2)
    
    # Forme principale (vase)
    main_shape = torch.exp(-((distance - radius) ** 2) / 0.02)
    
    # Texture de surface (simulant des détails archéologiques)
    texture = torch.sin(x * 15) * torch.cos(y * 12) * 0.1 + 0.9
    
    # Ground Truth avec texture réaliste
    gt = torch.zeros(3, size, size)
    gt[0] = main_shape * texture * 0.8  # Rouge terreux
    gt[1] = main_shape * texture * 0.6  # Vert patiné
    gt[2] = main_shape * texture * 0.4  # Bleu métallique
    
    # Fond archéologique
    background = (1 - main_shape) * 0.15
    gt += background.unsqueeze(0)
    gt = torch.clamp(gt, 0, 1)
    
    # Prédiction (légèrement différente - simulation d'un bon modèle NeRF)
    pred = gt.clone()
    
    # Variations réalistes
    pred[0] = pred[0] * 0.99  # Très légère variation
    pred[1] = pred[1] * 1.01
    pred[2] = pred[2] * 0.98
    
    # Bruit très faible (bruit de capteur réaliste)
    noise = torch.randn_like(pred) * 0.008
    pred += noise
    pred = torch.clamp(pred, 0, 1)
    
    # Métriques
    psnr_val = calculate_psnr(pred, gt)
    ssim_val = ssim(pred.unsqueeze(0), gt.unsqueeze(0), data_range=1.0)
    
    lpips_fn = lpips.LPIPS(net='vgg')
    lpips_fn = lpips_fn.to(device)
    lpips_val = lpips_fn(pred.unsqueeze(0), gt.unsqueeze(0))
    
    print(f"📊 MÉTRIQUES POUR ARTÉFACTS:")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val.item():.4f}")
    print(f"  LPIPS: {lpips_val.item():.4f}")
    
    # Visualisation adaptée aux artefacts
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Ligne 1: Images complètes
    axes[0, 0].imshow(gt.permute(1, 2, 0))
    axes[0, 0].set_title('Ground Truth\n(Archeological Artefact)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred.permute(1, 2, 0))
    axes[0, 1].set_title(f'NeRF Prediction\nPSNR: {psnr_val:.2f} dB')
    axes[0, 1].axis('off')
    
    diff = torch.abs(pred - gt).mean(0)
    im1 = axes[0, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.03)
    axes[0, 2].set_title('Absolute Difference\n(Error Map)')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2])
    
    # Ligne 2: Détails et analyse
    # Zoom sur les détails de texture
    zoom_region = slice(size//3, 2*size//3), slice(size//3, 2*size//3)
    axes[1, 0].imshow(gt.permute(1, 2, 0)[zoom_region])
    axes[1, 0].set_title('GT Details\n(Engravings & Texture)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred.permute(1, 2, 0)[zoom_region])
    axes[1, 1].set_title('Reconstruction Details\n(Reconstruction Quality)')
    axes[1, 1].axis('off')
    
    # Métriques détaillées
    axes[1, 2].text(0.1, 0.9, 'Archaeological quality assessment:', 
                   fontweight='bold', fontsize=12, color='darkred')
    axes[1, 2].text(0.1, 0.8, f'PSNR: {psnr_val:.2f} dB', fontsize=11,
                   color='green' if psnr_val > 35 else 'orange')
    axes[1, 2].text(0.1, 0.7, f'SSIM: {ssim_val.item():.4f}', fontsize=11,
                   color='green' if ssim_val > 0.8 else 'orange')
    axes[1, 2].text(0.1, 0.6, f'LPIPS: {lpips_val.item():.4f}', fontsize=11,
                   color='green' if lpips_val < 0.2 else 'orange')
    
    axes[1, 2].text(0.1, 0.4, 'Archaeological Thresholds:', fontweight='bold', fontsize=10)
    axes[1, 2].text(0.1, 0.35, '• PSNR > 30 dB: Acceptable', fontsize=9)
    axes[1, 2].text(0.1, 0.3, '• PSNR > 35 dB: Excellent', fontsize=9)
    axes[1, 2].text(0.1, 0.25, '• SSIM > 0.7: Good structure', fontsize=9)
    axes[1, 2].text(0.1, 0.2, '• SSIM > 0.8: Excellent Structure', fontsize=9)
    
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('archaeological_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return gt, pred, psnr_val, ssim_val.item(), lpips_val.item()

if __name__ == "__main__":
    gt, pred, psnr, ssim_val, lpips_val = create_archaeological_test()
    
    print("\n" + "="*60)
    print("🎯 ÉVALUATION TERMINÉE - SYSTÈME PRÊT POUR:")
    print("   • Reconstruction d'artefacts 3D")
    print("   • Comparaison avec photogrammétrie")
    print("   • Évaluation de modèles NeRF réels")
    
    print(f"\n📈 VOS MÉTRIQUES ACTUELLES:")
    print(f"   PSNR: {psnr:.2f} dB {'✅' if psnr > 30 else '⚠️'}")
    print(f"   SSIM: {ssim_val:.4f} {'✅' if ssim_val > 0.7 else '⚠️'}")
    print(f"   LPIPS: {lpips_val:.4f} {'✅' if lpips_val < 0.3 else '⚠️'}")