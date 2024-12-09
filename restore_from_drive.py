import os
import argparse
import logging
from pathlib import Path
from typing import Optional
from utils.backup import BackupManager

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('restore.log')
        ]
    )
    return logging.getLogger(__name__)

def restore_project(project_dir: str,
                   backup_dir: str,
                   drive_dir: Optional[str] = None,
                   version: Optional[str] = None,
                   target_dir: Optional[str] = None) -> bool:
    """Restore project from backup.
    
    Args:
        project_dir: Project directory
        backup_dir: Local backup directory
        drive_dir: Google Drive backup directory (optional)
        version: Specific version to restore (optional)
        target_dir: Target directory for restoration (optional)
        
    Returns:
        Success status
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize backup manager
        backup_manager = BackupManager(
            project_dir=project_dir,
            backup_dir=backup_dir,
            drive_dir=drive_dir,
            compress=True
        )
        
        # List available backups
        backups = backup_manager.list_backups()
        if not backups:
            logger.error("No backups found")
            return False
            
        # Select version to restore
        if version is None:
            # Get latest version
            version = backups[-1]['version']
            logger.info(f"Using latest backup version: {version}")
        elif version not in {b['version'] for b in backups}:
            logger.error(f"Backup version not found: {version}")
            return False
            
        # Get backup info
        backup_info = backup_manager.get_backup_info(version)
        logger.info("\nRestoring backup:")
        logger.info(f"Version: {backup_info['version']}")
        logger.info(f"Timestamp: {backup_info['timestamp']}")
        logger.info(f"Description: {backup_info['description']}")
        logger.info(f"Size: {backup_info['size'] / 1e6:.1f} MB")
        logger.info(f"Files: {backup_info['num_files']}")
        
        # Verify backup
        logger.info("\nVerifying backup integrity...")
        if not backup_manager.verify_backup(version):
            logger.error("Backup verification failed")
            return False
            
        # Restore backup
        logger.info("\nRestoring files...")
        success = backup_manager.restore_backup(
            version=version,
            target_dir=Path(target_dir) if target_dir else None
        )
        
        if success:
            logger.info("\nRestore completed successfully")
            return True
        else:
            logger.error("\nRestore failed")
            return False
            
    except Exception as e:
        logger.error(f"Error during restore: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Restore project from backup")
    parser.add_argument("--project-dir", type=str, required=True,
                       help="Project directory")
    parser.add_argument("--backup-dir", type=str, required=True,
                       help="Local backup directory")
    parser.add_argument("--drive-dir", type=str,
                       help="Google Drive backup directory")
    parser.add_argument("--version", type=str,
                       help="Specific backup version to restore")
    parser.add_argument("--target-dir", type=str,
                       help="Target directory for restoration")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    success = restore_project(
        project_dir=args.project_dir,
        backup_dir=args.backup_dir,
        drive_dir=args.drive_dir,
        version=args.version,
        target_dir=args.target_dir
    )
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main() 