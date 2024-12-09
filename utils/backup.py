import os
import zipfile
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from google.colab import drive
import glob
from config.paths import DRIVE_PATH

class ProjectBackup:
    def __init__(self, base_path: Union[str, Path], use_drive: bool = True):
        """Initialize backup manager."""
        self.base_path = Path(base_path)
        self.use_drive = use_drive
        self.drive_path = Path(DRIVE_PATH) if use_drive else None
        self.logger = logging.getLogger(__name__)
        
    def create_backup(self, include_checkpoints: bool = True):
        """Create a backup of important project files."""
        try:
            # Get list of files to backup
            files = self._get_important_files()
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"deepfake_project_backup_{timestamp}.zip"
            backup_path = self.base_path / backup_name
            
            # Create zip file
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                # Add files by category
                for category, file_list in files.items():
                    if not include_checkpoints and category == 'checkpoints':
                        continue
                    for file_path in file_list:
                        arcname = os.path.relpath(file_path, str(self.base_path))
                        zip_ref.write(file_path, arcname)
                        
            self.logger.info(f"Created backup: {backup_path}")
            
            # Copy to Drive if enabled
            if self.use_drive:
                drive_backup_path = self.drive_path / backup_name
                shutil.copy2(backup_path, drive_backup_path)
                self.logger.info(f"Copied backup to Drive: {drive_backup_path}")
                
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
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