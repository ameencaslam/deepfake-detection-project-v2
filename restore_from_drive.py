import os
import zipfile
import shutil
from pathlib import Path
import glob
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DriveRestorer:
    def __init__(self):
        """Initialize Drive restoration paths."""
        self.drive_path = Path('/content/drive/MyDrive/deepfake-project')
        self.content_path = Path('/content/PROJECT-V2')
        self.temp_dir = Path('/content/temp_restore')
        
    def find_latest_backup(self) -> Path:
        """Find the latest backup zip in Drive."""
        backup_pattern = str(self.drive_path / "deepfake_project_backup_*.zip")
        backup_files = glob.glob(backup_pattern)
        
        if not backup_files:
            raise FileNotFoundError("No backup files found in Drive")
            
        # Get latest backup by timestamp in filename
        latest_backup = max(backup_files, key=os.path.getctime)
        logger.info(f"Found latest backup: {latest_backup}")
        return Path(latest_backup)
        
    def verify_zip_integrity(self, zip_path: Path) -> bool:
        """Verify the integrity of the zip file."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Test zip file integrity
                test_result = zip_ref.testzip()
                if test_result is not None:
                    logger.error(f"Corrupted file in zip: {test_result}")
                    return False
                    
                # Check for essential directories/files
                contents = zip_ref.namelist()
                required_items = ['config/', 'models/', 'utils/', 'main.py', 'train.py']
                missing = [item for item in required_items if not any(f.startswith(item) for f in contents)]
                
                if missing:
                    logger.error(f"Missing required items in backup: {missing}")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Error verifying zip: {str(e)}")
            return False
            
    def verify_restored_files(self) -> bool:
        """Verify essential files and directories after restoration."""
        required_paths = [
            self.content_path / 'config',
            self.content_path / 'models',
            self.content_path / 'utils',
            self.content_path / 'main.py',
            self.content_path / 'train.py'
        ]
        
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            logger.error(f"Missing restored files/directories: {missing}")
            return False
            
        return True
        
    def restore(self):
        """Restore the latest backup from Drive to content directory."""
        try:
            # Find latest backup
            backup_file = self.find_latest_backup()
            
            # Verify zip integrity
            logger.info("Verifying backup integrity...")
            if not self.verify_zip_integrity(backup_file):
                raise ValueError("Backup verification failed")
                
            # Create/clear temp directory
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(parents=True)
            
            try:
                # Extract to temp directory
                logger.info("Extracting backup...")
                with zipfile.ZipFile(backup_file, 'r') as zip_ref:
                    zip_ref.extractall(self.temp_dir)
                    
                # Remove existing content directory if it exists
                if self.content_path.exists():
                    shutil.rmtree(self.content_path)
                    
                # Move files from temp to content directory
                logger.info("Moving files to project directory...")
                shutil.copytree(self.temp_dir, self.content_path)
                
                # Verify restored files
                logger.info("Verifying restored files...")
                if not self.verify_restored_files():
                    raise ValueError("Restored files verification failed")
                    
                logger.info("Restoration completed successfully")
                
            finally:
                # Clean up temp directory
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                    
        except Exception as e:
            logger.error(f"Error during restoration: {str(e)}")
            raise

def main():
    """Main function to restore from Drive."""
    try:
        # Check if Drive is mounted
        if not os.path.exists('/content/drive/MyDrive'):
            raise RuntimeError("Google Drive is not mounted. Please mount it first.")
            
        restorer = DriveRestorer()
        restorer.restore()
        
    except Exception as e:
        logger.error(f"Restoration failed: {str(e)}")
        raise

if __name__ == '__main__':
    main() 