"""Command line interface for backup and restore operations."""

import argparse
import logging
from pathlib import Path
from utils.backup import create_backup, restore_backup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description='Backup and restore project files')
    parser.add_argument('action', choices=['backup', 'restore'],
                       help='Action to perform')
    parser.add_argument('--project-dir', type=str, default='.',
                       help='Project directory (default: current directory)')
    parser.add_argument('--backup-path', type=str,
                       help='Path for backup file or backup to restore from')
    parser.add_argument('--restore-dir', type=str,
                       help='Directory to restore to (for restore action)')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'backup':
            backup_path = create_backup(args.project_dir, args.backup_path)
            print(f"\nBackup created at: {backup_path}")
            
        else:  # restore
            if not args.backup_path:
                parser.error("--backup-path is required for restore action")
            if not args.restore_dir:
                parser.error("--restore-dir is required for restore action")
                
            restore_backup(args.backup_path, args.restore_dir)
            print(f"\nProject restored to: {args.restore_dir}")
            
    except Exception as e:
        logging.error(f"Operation failed: {str(e)}")
        raise

if __name__ == '__main__':
    main() 