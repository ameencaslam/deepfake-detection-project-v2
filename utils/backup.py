"""Simple backup utility for project files."""

import os
import shutil
import zipfile
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

def get_files_to_backup(project_dir: str, exclude_patterns: Optional[List[str]] = None) -> List[str]:
    """Get list of files to backup, excluding specified patterns."""
    if exclude_patterns is None:
        exclude_patterns = [
            'models/checkpoints/*',  # Exclude all checkpoints
            '*.pyc',                 # Exclude Python cache
            '__pycache__',          # Exclude Python cache directories
            '.git',                 # Exclude git directory
            '*.zip'                 # Exclude zip files
        ]
    
    files_to_backup = []
    project_dir = Path(project_dir)
    
    # Add best and last checkpoints back if they exist
    checkpoint_dir = project_dir / 'models' / 'checkpoints'
    if checkpoint_dir.exists():
        best_checkpoint = checkpoint_dir / 'best_model.pth'
        last_checkpoint = checkpoint_dir / 'last_model.pth'
        if best_checkpoint.exists():
            files_to_backup.append(str(best_checkpoint.relative_to(project_dir)))
        if last_checkpoint.exists():
            files_to_backup.append(str(last_checkpoint.relative_to(project_dir)))
    
    for root, dirs, files in os.walk(project_dir):
        # Convert to Path objects for easier manipulation
        root_path = Path(root)
        rel_path = root_path.relative_to(project_dir)
        
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not any(
            p in str(rel_path / d) for p in exclude_patterns
        )]
        
        # Add non-excluded files
        for file in files:
            file_path = rel_path / file
            if not any(p in str(file_path) for p in exclude_patterns):
                files_to_backup.append(str(file_path))
    
    return files_to_backup

def save_split_info(project_dir: str, zip_file: zipfile.ZipFile):
    """Save dataset split information."""
    project_dir = Path(project_dir)
    data_dir = project_dir / 'data'
    
    if not data_dir.exists():
        return
    
    split_info = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    # Collect file information for each split
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
            
        for label in ['real', 'fake']:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue
                
            # Store filenames and their modification times
            split_info[split][label] = {
                str(f.relative_to(data_dir)): f.stat().st_mtime
                for f in label_dir.rglob('*')
                if f.is_file()
            }
    
    # Save split info to zip
    zip_file.writestr('split_info.json', json.dumps(split_info, indent=2))

def verify_split_info(restore_dir: str, zip_file: zipfile.ZipFile) -> bool:
    """Verify dataset split integrity after restore."""
    try:
        with zip_file.open('split_info.json') as f:
            split_info = json.loads(f.read())
    except KeyError:
        logger.warning("No split information found in backup")
        return True
    
    restore_dir = Path(restore_dir)
    data_dir = restore_dir / 'data'
    
    if not data_dir.exists():
        logger.warning("No data directory found after restore")
        return True
    
    # Verify each file in split info exists
    for split, labels in split_info.items():
        for label, files in labels.items():
            for file_path in files:
                full_path = data_dir / file_path
                if not full_path.exists():
                    logger.error(f"Missing file after restore: {file_path}")
                    return False
    
    return True

def create_backup(project_dir: str, backup_path: Optional[str] = None) -> str:
    """Create a zip backup of the project.
    
    Args:
        project_dir: Directory to backup
        backup_path: Optional path for backup file
        
    Returns:
        Path to created backup file
    """
    project_dir = Path(project_dir)
    
    # Generate backup filename with timestamp if not provided
    if backup_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = project_dir / f'project_backup_{timestamp}.zip'
    
    backup_path = Path(backup_path)
    
    try:
        # Get files to backup
        files_to_backup = get_files_to_backup(project_dir)
        
        # Create zip file
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Save data split information
            save_split_info(project_dir, zf)
            
            # Add project files
            for file in files_to_backup:
                file_path = project_dir / file
                if file_path.exists():
                    zf.write(file_path, file)
        
        logger.info(f"Backup created successfully at: {backup_path}")
        return str(backup_path)
        
    except Exception as e:
        logger.error(f"Failed to create backup: {str(e)}")
        raise

def restore_backup(backup_path: str, restore_dir: str):
    """Restore project from backup zip file.
    
    Args:
        backup_path: Path to backup zip file
        restore_dir: Directory to restore to
    """
    try:
        backup_path = Path(backup_path)
        restore_dir = Path(restore_dir)
        
        # Create restore directory if it doesn't exist
        restore_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(backup_path, 'r') as zf:
            # Extract all files
            zf.extractall(restore_dir)
            
            # Verify data split integrity
            if not verify_split_info(restore_dir, zf):
                raise RuntimeError("Data split verification failed")
            
        logger.info(f"Project restored successfully to: {restore_dir}")
        
    except Exception as e:
        logger.error(f"Failed to restore backup: {str(e)}")
        raise 