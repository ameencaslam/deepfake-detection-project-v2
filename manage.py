import os
import argparse
from pathlib import Path
import logging
import zipfile
import shutil
import glob
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectManager:
    def __init__(self, project_path: str = '/content/PROJECT-V2', use_drive: bool = True):
        """Initialize paths.
        
        Args:
            project_path: Path to project directory
            use_drive: Whether to use Google Drive for backup
        """
        self.project_path = Path(project_path)
        self.use_drive = use_drive
        self.drive_path = Path('/content/drive/MyDrive/deepfake-project') if use_drive else None
        
    def backup(self):
        """Backup essential files to Drive."""
        try:
            # Skip if Drive is not enabled
            if not self.use_drive:
                logger.info("Drive backup disabled, skipping backup")
                return
                
            # Check Drive
            if not os.path.exists('/content/drive/MyDrive'):
                raise RuntimeError("Google Drive is not mounted")
                
            # Get files to backup
            files = []
            
            # Checkpoints (*.pth)
            for pth_file in self.project_path.rglob('*.pth'):
                files.append(pth_file)
                
            # Dataset split info
            split_info = Path('/content/dataset/split_info.json')
            if split_info.exists():
                files.append(split_info)
                
            # Results and configs
            for ext in ['*.json', '*.png', '*.jpg', '*.csv']:
                for file in self.project_path.rglob(ext):
                    if 'results' in str(file) or 'configs' in str(file):
                        files.append(file)
                        
            if not files:
                logger.warning("No files to backup")
                return
                
            # Create backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.drive_path / f"project_backup_{timestamp}.zip"
            self.drive_path.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                for file in files:
                    if self.project_path in file.parents:
                        arcname = file.relative_to(self.project_path)
                    elif '/content/dataset' in str(file):
                        arcname = Path('dataset') / file.relative_to('/content/dataset')
                    else:
                        arcname = file.name
                    zip_ref.write(file, arcname)
                    logger.info(f"Added: {arcname}")
                    
            # Keep only latest backup
            old_backups = list(self.drive_path.glob('project_backup_*.zip'))
            old_backups.sort(key=lambda x: x.stat().st_mtime)
            for old_backup in old_backups[:-1]:
                old_backup.unlink()
                
            logger.info(f"Backup completed: {backup_path}")
            
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            raise
            
    def restore(self):
        """Restore from latest backup."""
        try:
            # Skip if Drive is not enabled
            if not self.use_drive:
                logger.info("Drive backup disabled, skipping restore")
                return
                
            # Check Drive
            if not os.path.exists('/content/drive/MyDrive'):
                raise RuntimeError("Google Drive is not mounted")
                
            # Find latest backup
            backup_pattern = str(self.drive_path / "project_backup_*.zip")
            backups = glob.glob(backup_pattern)
            if not backups:
                raise FileNotFoundError("No backups found")
            backup_file = max(backups, key=os.path.getctime)
            
            # Create directories
            self.project_path.mkdir(parents=True, exist_ok=True)
            Path('/content/dataset').mkdir(parents=True, exist_ok=True)
            
            # Extract files
            logger.info(f"Restoring from: {backup_file}")
            with zipfile.ZipFile(backup_file, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    if file.startswith('dataset/'):
                        target_path = Path('/content') / file
                    else:
                        target_path = self.project_path / file
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    zip_ref.extract(file, target_path.parent)
                    logger.info(f"Restored: {file}")
                    
            logger.info("Restore completed")
            
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            raise
            
    def clean(self):
        """Clean temporary and unnecessary files."""
        try:
            # Clean patterns
            patterns = [
                # Python temp files
                '**/*.pyc',
                '**/__pycache__',
                '**/.ipynb_checkpoints',
                
                # System temp files
                '**/.DS_Store',
                '**/Thumbs.db',
                
                # Project temp files
                '**/temp_*',
                '**/logs/*.log',
                '**/results/temp_*',
                
                # Colab temp files
                '**/.config/*',
                '**/.local/*',
                
                # Old backups
                '**/project_backup_*.zip'
            ]
            
            count = 0
            for pattern in patterns:
                for path in self.project_path.glob(pattern):
                    if path.is_file():
                        path.unlink()
                    else:
                        shutil.rmtree(path)
                    count += 1
                    logger.info(f"Cleaned: {path}")
                    
            logger.info(f"Cleaned {count} items")
            
        except Exception as e:
            logger.error(f"Clean failed: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Project management tools')
    parser.add_argument('action', choices=['backup', 'restore', 'clean'],
                       help='Action to perform')
    args = parser.parse_args()
    
    try:
        manager = ProjectManager()
        if args.action == 'backup':
            manager.backup()
        elif args.action == 'restore':
            manager.restore()
        elif args.action == 'clean':
            manager.clean()
    except Exception as e:
        logger.error(f"Action failed: {str(e)}")
        raise

if __name__ == '__main__':
    main() 