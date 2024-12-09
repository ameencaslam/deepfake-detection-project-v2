"""Main entry point for the deepfake detection system."""

import logging
from core import (
    setup_environment,
    setup_logging,
    setup_hardware,
    load_config,
    get_cli_parser,
    merge_cli_config
)
from train import train
from evaluate import evaluate

def main():
    # Parse arguments
    parser = get_cli_parser()
    args = parser.parse_args()
    
    # Load and merge configuration
    config = load_config()
    config = merge_cli_config(config, args)
    
    # Setup environment
    setup_environment(config)
    logger = setup_logging(config.paths.logs)
    hardware = setup_hardware(config)
    
    try:
        if args.action == 'train':
            logger.info(f"Starting training with {config.model.architecture}...")
            train(config)
            logger.info("Training completed!")
        else:  # evaluate
            if not args.checkpoint:
                logger.error("Please provide checkpoint path for evaluation using --checkpoint")
                return
            logger.info(f"Starting evaluation of {config.model.architecture}...")
            evaluate(config)
            logger.info("Evaluation completed!")

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"\nError occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main() 