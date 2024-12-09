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
        """Initialize backup manager."""
        self.base_path = Path(base_path)
        self.use_drive = use_drive
        self.drive_path = Path(DRIVE_PATH)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if self.use_drive:
            self._setup_drive()
            self.restore_latest_backup()
            
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
            
    def restore_latest_backup(self):
        """Restore the latest backup from Drive."""
        if not self.use_drive:
            return
            
        try:
            # Find latest backup zip
            backup_pattern = str(self.drive_path / "deepfake_project_backup_*.zip")
            backup_files = glob.glob(backup_pattern)
            
            if not backup_files:
                self.logger.info("No backup files found in Drive")
                return
                
            # Get latest backup by timestamp in filename
            latest_backup = max(backup_files, key=os.path.getctime)
            self.logger.info(f"Found latest backup: {latest_backup}")
            
            # Create necessary directories
            self.base_path.mkdir(parents=True, exist_ok=True)
            temp_dir = self.base_path / "temp_restore"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Extract backup
                with zipfile.ZipFile(latest_backup, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    
                # Move files to correct locations
                self._restore_from_temp(temp_dir)
                self.logger.info("Successfully restored from latest backup")
                
            finally:
                # Cleanup temp directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            
        except Exception as e:
            self.logger.error(f"Error restoring from backup: {str(e)}")
            
    def _restore_from_temp(self, temp_dir: Path):
        """Restore files from temporary directory to their proper locations."""
        try:
            # Ensure base directories exist
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Restore code files
            for py_file in temp_dir.rglob("*.py"):
                relative_path = py_file.relative_to(temp_dir)
                target_path = self.base_path / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(py_file, target_path)
                self.logger.info(f"Restored: {relative_path}")
                
            # Restore checkpoints
            checkpoint_dir = temp_dir / "checkpoints"
            if checkpoint_dir.exists():
                target_checkpoint_dir = self.base_path / "checkpoints"
                if target_checkpoint_dir.exists():
                    shutil.rmtree(target_checkpoint_dir)
                shutil.copytree(checkpoint_dir, target_checkpoint_dir)
                self.logger.info("Restored: checkpoints/")
                
            # Restore configs
            for config_file in temp_dir.rglob("*.json"):
                relative_path = config_file.relative_to(temp_dir)
                target_path = self.base_path / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(config_file, target_path)
                self.logger.info(f"Restored: {relative_path}")
                
            # Restore dataset info
            dataset_info_dir = temp_dir / "dataset_info"
            if dataset_info_dir.exists():
                dataset_path = Path(os.path.dirname(self.base_path)) / "dataset"
                dataset_path.mkdir(parents=True, exist_ok=True)
                for info_file in dataset_info_dir.glob("*"):
                    target_path = dataset_path / info_file.name
                    shutil.copy2(info_file, target_path)
                    self.logger.info(f"Restored: dataset/{info_file.name}")
                    
        except Exception as e:
            self.logger.error(f"Error during file restoration: {str(e)}")
            raise
        
    def _get_important_files(self) -> Dict[str, List[str]]:
        """Get list of important files to backup."""
        files = {
            'code': [],
            'checkpoints': [],
            'configs': [],
            'logs': [],
            'results': [],
            'dataset_info': []
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
                
        # Dataset split information
        data_path = Path(os.path.join(self.base_path.parent, 'dataset'))
        split_info_file = data_path / 'split_info.json'
        if split_info_file.exists():
            files['dataset_info'].append(str(split_info_file))
            
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