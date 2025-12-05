# test_blender_dataset.py
import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import lpips
import os
import json
from pathlib import Path
import imageio
from PIL import Image
import zipfile
import urllib.request

class BlenderDatasetEvaluator:
    """Évaluateur spécialisé pour le Blender Synthetic Dataset"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        print(f"🔧 Évaluateur Blender initialisé sur {self.device}")
    
    def calculate_psnr(self, predicted, target):
        """PSNR sécurisé"""
        mse = F.mse_loss(predicted, target)
        return 20 * math.log10(1.0 / torch.sqrt(mse)) if mse > 0 else float('inf')
    
    def download_blender_dataset(self, dataset_path="./nerf_synthetic"):
        """Télécharge le dataset Blender Synthetic"""
        print("📥 Téléchargement du Blender Synthetic Dataset...")
        
        dataset_path = Path(dataset_path)
        dataset_path.mkdir(exist_ok=True)
        
        url = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip"
        zip_path = dataset_path / "nerf_example_data.zip"
        
        try:
            # Téléchargement
            urllib.request.urlretrieve(url, zip_path)
            print("✅ Téléchargement terminé")
            
            # Extraction
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            print("✅ Extraction terminée")
            
            # Nettoyage
            os.remove(zip_path)
            
            return dataset_path / "nerf_example_data"
            
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement: {e}")
            return None
    
    def load_blender_scene(self, scene_path, split='test', max_images=5):
        """Charge une scène spécifique du dataset Blender"""
        scene_path = Path(scene_path)
        images_path = scene_path / split
        
        if not images_path.exists():
            print(f"❌ Dossier non trouvé: {images_path}")
            return []
        
        image_files = sorted([f for f in images_path.iterdir() if f.suffix.lower() in {'.png', '.jpg'}])
        image_files = image_files[:max_images]
        
        print(f"📁 Chargement de {len(image_files)} images depuis {scene_path.name}/{split}")
        
        images = []
        for img_path in image_files:
            try:
                # Chargement avec imageio
                img = imageio.imread(img_path)
                
                # Le dataset Blender utilise RGBA, on convertit en RGB
                if img.shape[2] == 4:
                    # Conversion RGBA -> RGB (supprime canal alpha)
                    img = img[:, :, :3]
                
                # Normalisation [0, 1]
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                
                # Conversion en tensor PyTorch (CHW format)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
                images.append(img_tensor)
                
                print(f"   ✅ {img_path.name}: {img_tensor.shape}")
                
            except Exception as e:
                print(f"   ❌ Erreur avec {img_path.name}: {e}")
        
        return images
    
    def simulate_nerf_predictions(self, gt_images, quality='good'):
        """Simule des prédictions NeRF réalistes avec différents niveaux de qualité"""
        pred_images = []
        
        # Paramètres selon la qualité
        if quality == 'excellent':
            noise_level = 0.01
            color_shift = 0.01
            blur_level = 0
        elif quality == 'good':
            noise_level = 0.02
            color_shift = 0.03
            blur_level = 1
        else:  # 'poor'
            noise_level = 0.05
            color_shift = 0.08
            blur_level = 2
        
        for gt in gt_images:
            pred = gt.clone()
            
            # Variation de couleur réaliste
            color_shift_tensor = torch.randn(3, 1, 1) * color_shift
            pred = pred + color_shift_tensor
            pred = torch.clamp(pred, 0, 1)
            
            # Légère perte de détails (flou gaussien)
            if blur_level > 0 and pred.shape[1] > 64:
                kernel_size = 2 * blur_level + 1
                pred = F.avg_pool2d(pred.unsqueeze(0), kernel_size, 1, blur_level).squeeze(0)
            
            # Bruit réaliste
            noise = torch.randn_like(pred) * noise_level
            pred += noise
            pred = torch.clamp(pred, 0, 1)
            
            pred_images.append(pred)
        
        return pred_images
    
    def evaluate_scene(self, scene_path, scene_name, max_images=5):
        """Évalue une scène complète du dataset Blender"""
        print(f"\n{'='*60}")
        print(f"🎯 ÉVALUATION DE LA SCÈNE: {scene_name}")
        print(f"{'='*60}")
        
        # Charger les images de test
        gt_images = self.load_blender_scene(scene_path, 'test', max_images)
        
        if not gt_images:
            print("❌ Aucune image chargée - test annulé")
            return None
        
        # Tester différentes qualités de prédiction
        quality_levels = ['excellent', 'good', 'poor']
        all_results = {}
        
        for quality in quality_levels:
            print(f"\n🔍 Test avec qualité: {quality.upper()}")
            
            # Simuler les prédictions
            pred_images = self.simulate_nerf_predictions(gt_images, quality)
            
            # Calcul des métriques
            metrics = {'psnr': [], 'ssim': [], 'lpips': []}
            
            for i, (gt, pred) in enumerate(zip(gt_images, pred_images)):
                psnr = self.calculate_psnr(pred, gt)
                ssim_val = ssim(pred.unsqueeze(0), gt.unsqueeze(0), data_range=1.0)
                lpips_val = self.lpips_fn(pred.unsqueeze(0).to(self.device), 
                                        gt.unsqueeze(0).to(self.device))
                
                metrics['psnr'].append(psnr)
                metrics['ssim'].append(ssim_val.item())
                metrics['lpips'].append(lpips_val.item())
            
            # Statistiques
            all_results[quality] = {
                'mean_psnr': np.mean(metrics['psnr']),
                'mean_ssim': np.mean(metrics['ssim']),
                'mean_lpips': np.mean(metrics['lpips']),
                'std_psnr': np.std(metrics['psnr']),
                'std_ssim': np.std(metrics['ssim']),
                'std_lpips': np.std(metrics['lpips'])
            }
            
            print(f"   ✅ PSNR:  {all_results[quality]['mean_psnr']:.2f} ± {all_results[quality]['std_psnr']:.2f} dB")
            print(f"   ✅ SSIM:  {all_results[quality]['mean_ssim']:.4f} ± {all_results[quality]['std_ssim']:.4f}")
            print(f"   ✅ LPIPS: {all_results[quality]['mean_lpips']:.4f} ± {all_results[quality]['std_lpips']:.4f}")
        
        # Créer la visualisation pour la qualité 'good' (représentative)
        pred_images_good = self.simulate_nerf_predictions(gt_images, 'good')
        self.create_scene_visualization(gt_images, pred_images_good, scene_name, all_results['good'])
        
        return all_results
    
    def create_scene_visualization(self, gt_images, pred_images, scene_name, metrics):
        """Crée une visualisation complète pour la scène"""
        n_images = min(4, len(gt_images))
        
        fig, axes = plt.subplots(n_images, 4, figsize=(20, 5*n_images))
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_images):
            gt = gt_images[i]
            pred = pred_images[i]
            
            # Métriques pour cette image
            psnr = self.calculate_psnr(pred, gt)
            ssim_val = ssim(pred.unsqueeze(0), gt.unsqueeze(0), data_range=1.0)
            
            # Affichage
            axes[i, 0].imshow(gt.permute(1, 2, 0))
            axes[i, 0].set_title(f'GT - Vue {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(pred.permute(1, 2, 0))
            axes[i, 1].set_title(f'Pred NeRF\nPSNR: {psnr:.2f} dB')
            axes[i, 1].axis('off')
            
            # Différence
            diff = torch.abs(pred - gt).mean(0)
            im = axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.1)
            axes[i, 2].set_title(f'Différence\nSSIM: {ssim_val.item():.4f}')
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2])
            
            # Histogramme des différences
            diff_flat = torch.abs(pred - gt).flatten().cpu().numpy()
            axes[i, 3].hist(diff_flat, bins=50, alpha=0.7, color='red', density=True)
            axes[i, 3].set_title('Distribution des erreurs')
            axes[i, 3].set_xlabel('Erreur absolue')
            axes[i, 3].set_ylabel('Densité')
        
        plt.suptitle(f'Scène Blender: {scene_name}\n'
                    f'Métriques moyennes: PSNR={metrics["mean_psnr"]:.2f}dB, '
                    f'SSIM={metrics["mean_ssim"]:.4f}, LPIPS={metrics["mean_lpips"]:.4f}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'blender_evaluation_{scene_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, scenes_results):
        """Génère un rapport complet pour toutes les scènes"""
        print(f"\n{'='*70}")
        print("📊 RAPPORT COMPLET BLENDER SYNTHETIC DATASET")
        print(f"{'='*70}")
        
        report = {
            'metadata': {
                'dataset': 'Blender Synthetic',
                'timestamp': str(np.datetime64('now')),
                'device': str(self.device),
                'total_scenes': len(scenes_results)
            },
            'scenes': {},
            'summary': {}
        }
        
        # Analyse par scène
        for scene_name, qualities in scenes_results.items():
            report['scenes'][scene_name] = qualities
            
            print(f"\n🏛️  SCÈNE: {scene_name}")
            for quality, metrics in qualities.items():
                print(f"   {quality.upper():>10}: "
                      f"PSNR={metrics['mean_psnr']:5.2f}dB, "
                      f"SSIM={metrics['mean_ssim']:6.4f}, "
                      f"LPIPS={metrics['mean_lpips']:6.4f}")
        
        # Statistiques globales
        all_psnr = []
        all_ssim = []
        all_lpips = []
        
        for scene_qualities in scenes_results.values():
            if 'good' in scene_qualities:
                all_psnr.append(scene_qualities['good']['mean_psnr'])
                all_ssim.append(scene_qualities['good']['mean_ssim'])
                all_lpips.append(scene_qualities['good']['mean_lpips'])
        
        if all_psnr:
            report['summary'] = {
                'mean_psnr': np.mean(all_psnr),
                'mean_ssim': np.mean(all_ssim),
                'mean_lpips': np.mean(all_lpips),
                'std_psnr': np.std(all_psnr),
                'std_ssim': np.std(all_ssim),
                'std_lpips': np.std(all_lpips)
            }
            
            print(f"\n📈 STATISTIQUES GLOBALES (qualité 'good'):")
            print(f"   PSNR:  {report['summary']['mean_psnr']:.2f} ± {report['summary']['std_psnr']:.2f} dB")
            print(f"   SSIM:  {report['summary']['mean_ssim']:.4f} ± {report['summary']['std_ssim']:.4f}")
            print(f"   LPIPS: {report['summary']['mean_lpips']:.4f} ± {report['summary']['std_lpips']:.4f}")
        
        # Sauvegarde du rapport
        with open('blender_dataset_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Rapport sauvegardé: blender_dataset_report.json")
        return report

def main():
    """Fonction principale"""
    print("🧪 TEST DES MÉTRIQUES SUR BLENDER SYNTHETIC DATASET")
    print("=" * 60)
    
    evaluator = BlenderDatasetEvaluator()
    
    # Télécharger le dataset (si pas déjà fait)
    dataset_path = evaluator.download_blender_dataset()
    
    if dataset_path is None:
        print("❌ Impossible de télécharger le dataset")
        return
    
    # Scènes à tester (les plus représentatives)
    scenes_to_test = [
        "lego",        # Objet complexe avec textures
        "chair",       # Objet avec matériaux variés
        "drums",       # Objet avec réflexions
        "ficus",       # Objet organique
        "mic"          # Objet métallique
    ]
    
    all_results = {}
    
    for scene_name in scenes_to_test:
        scene_path = dataset_path / scene_name
        
        if scene_path.exists():
            results = evaluator.evaluate_scene(scene_path, scene_name, max_images=5)
            if results:
                all_results[scene_name] = results
        else:
            print(f"❌ Scène non trouvée: {scene_path}")
    
    # Rapport final
    if all_results:
        report = evaluator.generate_comprehensive_report(all_results)
        
        print(f"\n✅ VALIDATION TERMINÉE!")
        print(f"📁 Fichiers générés:")
        for scene in all_results.keys():
            print(f"   • blender_evaluation_{scene}.png")
        print(f"   • blender_dataset_report.json")
        
        # Validation par rapport aux attentes
        print(f"\n🎯 COMPARAISON AVEC LES ATTENTES:")
        expected_ranges = {
            'PSNR': {'good': '30-35 dB', 'excellent': '>35 dB'},
            'SSIM': {'good': '0.90-0.95', 'excellent': '>0.95'},
            'LPIPS': {'good': '0.08-0.15', 'excellent': '<0.08'}
        }
        
        print("   Pour un modèle NeRF de qualité 'good' sur Blender Synthetic:")
        for metric, ranges in expected_ranges.items():
            print(f"   {metric}: {ranges['good']}")
    
    else:
        print("❌ Aucun résultat à afficher")

if __name__ == "__main__":
    main()