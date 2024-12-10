import os
import logging
import shutil
from pathlib import Path
from datetime import datetime
import argparse

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
        """Backup project directory to Drive."""
        try:
            # Skip if Drive is not enabled
            if not self.use_drive:
                logger.info("Drive backup disabled, skipping backup")
                return
                
            # Check Drive
            if not os.path.exists('/content/drive/MyDrive'):
                raise RuntimeError("Google Drive is not mounted")
            
            # Use fixed backup directory name
            backup_dir = self.drive_path / "project_backup"
            
            # Remove existing backup if any
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            
            # Copy entire project directory
            logger.info(f"Backing up project to: {backup_dir}")
            shutil.copytree(self.project_path, backup_dir)
            logger.info("Backup completed")
            
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            raise
            
    def restore(self):
        """Restore from backup if available."""
        try:
            # Skip if Drive is not enabled
            if not self.use_drive:
                logger.info("Drive backup disabled, skipping restore")
                return
            
            # Check Drive
            if not os.path.exists('/content/drive/MyDrive'):
                raise RuntimeError("Google Drive is not mounted")
            
            # Use fixed backup directory name
            backup_dir = self.drive_path / "project_backup"
            
            if not backup_dir.exists():
                logger.info("No backup found, starting fresh")
                return
            
            # Clear existing project directory
            if self.project_path.exists():
                shutil.rmtree(self.project_path)
            
            # Copy from backup
            logger.info(f"Restoring from: {backup_dir}")
            shutil.copytree(backup_dir, self.project_path)
            logger.info("Restore completed")
            
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                logger.info("No backup found, starting fresh")
            else:
                logger.error(f"Restore failed: {str(e)}")
                raise
            
    def clean(self):
        """Clean temporary files."""
        try:
            patterns = [
                '**/*.pyc',
                '**/__pycache__',
                '**/.ipynb_checkpoints',
                '**/.DS_Store',
                '**/Thumbs.db',
                '**/temp_*',
                '**/logs/*.log',
                '**/.config/*',
                '**/.local/*'
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