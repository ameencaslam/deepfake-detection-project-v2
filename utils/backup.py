import os
import zipfile
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging
from google.colab import drive
import glob
from config.paths import DRIVE_PATH

class ProjectBackup:
    def __init__(self, base_path: str, use_drive: bool = True):
        self.base_path = Path(base_path)
        self.use_drive = use_drive
        self.drive_path = Path(DRIVE_PATH)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if self.use_drive:
            self._setup_drive()
            
    def _setup_drive(self):
        """Setup Google Drive connection."""
        try:
            # Check if drive is already mounted
            if not os.path.exists('/content/drive/MyDrive'):
                drive.mount('/content/drive')
            
            # Verify drive access
            if not os.path.exists('/content/drive/MyDrive'):
                self.logger.error("Drive mount point exists but MyDrive is not accessible")
                self.use_drive = False
                return
                
            self.drive_path.mkdir(parents=True, exist_ok=True)
            self.logger.info("Google Drive mounted and verified successfully")
        except Exception as e:
            self.logger.error(f"Failed to setup Google Drive: {str(e)}")
            self.use_drive = False
            
    def _get_important_files(self) -> Dict[str, List[str]]:
        """Get list of important files to backup."""
        files = {
            'code': [],
            'checkpoints': [],
            'configs': [],
            'logs': [],
            'results': []
        }
        
        # Python files
        for py_file in self.base_path.rglob("*.py"):
            files['code'].append(str(py_file))
            
        # Latest and best checkpoints
        checkpoint_path = self.base_path / 'checkpoints'
        if checkpoint_path.exists():
            for model_dir in checkpoint_path.iterdir():
                if model_dir.is_dir():
                    for model_type in ['latest', 'best']:
                        checkpoints = list(model_dir.glob(f"*{model_type}*.pth"))
                        if checkpoints:
                            files['checkpoints'].extend([str(cp) for cp in checkpoints])
                
        # Config files
        for config_file in self.base_path.rglob("*.json"):
            files['configs'].append(str(config_file))
            
        # Log files
        log_path = self.base_path / 'logs'
        if log_path.exists():
            for log_file in log_path.glob("*.log"):
                files['logs'].append(str(log_file))
            
        # Results
        results_path = self.base_path / 'results'
        if results_path.exists():
            for result_file in results_path.rglob("*.*"):
                files['results'].append(str(result_file))
            
        return files
        
    def create_backup(self, include_checkpoints: bool = True) -> str:
        """Create backup of project files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_name = f"deepfake_project_backup_{timestamp}.zip"
        zip_path = self.base_path / zip_name
        
        # Get files to backup
        files = self._get_important_files()
        total_files = sum(len(files[k]) for k in files)
        
        self.logger.info(f"Creating backup with {total_files} files...")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add files by category
            for category, file_list in files.items():
                if category == 'checkpoints' and not include_checkpoints:
                    continue
                    
                for file_path in file_list:
                    rel_path = os.path.relpath(file_path, str(self.base_path))
                    self.logger.info(f"Adding: {rel_path}")
                    zipf.write(file_path, rel_path)
                    
        # Copy to Drive if available
        if self.use_drive:
            drive_zip_path = self.drive_path / zip_name
            shutil.copy(zip_path, drive_zip_path)
            self.logger.info(f"Backup copied to Drive: {drive_zip_path}")
            
        return str(zip_path)
        
    def restore_from_backup(self, zip_path: Optional[str] = None) -> bool:
        """Restore project from backup."""
        # Find latest backup if not specified
        if zip_path is None:
            # Check working directory
            local_backups = list(self.base_path.glob("deepfake_project_backup_*.zip"))
            drive_backups = []
            
            # Check Drive
            if self.use_drive and self.drive_path.exists():
                drive_backups = list(self.drive_path.glob("deepfake_project_backup_*.zip"))
                
            all_backups = local_backups + drive_backups
            if not all_backups:
                self.logger.error("No backup files found")
                return False
                
            # Get latest backup
            zip_path = str(max(all_backups, key=os.path.getctime))
            
        if not os.path.exists(zip_path):
            self.logger.error(f"Backup file not found: {zip_path}")
            return False
            
        self.logger.info(f"Restoring from backup: {zip_path}")
        
        # Create temporary extraction directory
        temp_dir = self.base_path / "temp_restore"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract files
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(temp_dir)
                
            # Move files to correct locations
            for item in temp_dir.iterdir():
                target = self.base_path / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
                
            self.logger.info("Restore completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during restore: {str(e)}")
            return False
            
        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                
    def backup_to_drive(self, file_path: str, category: str, model_name: Optional[str] = None):
        """Backup specific file to Drive."""
        if not self.use_drive:
            return
            
        drive_category_path = self.drive_path / category
        if model_name:
            drive_category_path = drive_category_path / model_name
        drive_category_path.mkdir(parents=True, exist_ok=True)
        
        filename = os.path.basename(file_path)
        drive_path = drive_category_path / filename
        
        shutil.copy(file_path, drive_path)
        self.logger.info(f"Backed up to Drive: {drive_path}")
        
    def clean_old_backups(self, keep_last: int = 3):
        """Clean old backup files, keeping the specified number of recent ones."""
        # Clean local backups
        local_backups = sorted(
            self.base_path.glob("deepfake_project_backup_*.zip"),
            key=os.path.getctime
        )
        
        for backup in local_backups[:-keep_last]:
            backup.unlink()
            
        # Clean Drive backups
        if self.use_drive and self.drive_path.exists():
            # Clean general backups
            drive_backups = sorted(
                self.drive_path.glob("deepfake_project_backup_*.zip"),
                key=os.path.getctime
            )
            
            for backup in drive_backups[:-keep_last]:
                backup.unlink()
                
            # Clean model-specific checkpoints
            checkpoint_path = self.drive_path / 'checkpoints'
            if checkpoint_path.exists():
                for model_dir in checkpoint_path.iterdir():
                    if model_dir.is_dir():
                        model_checkpoints = sorted(
                            model_dir.glob("*.pth"),
                            key=os.path.getctime
                        )
                        for checkpoint in model_checkpoints[:-keep_last]:
                            checkpoint.unlink() 