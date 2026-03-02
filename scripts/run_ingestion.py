#!/usr/bin/env python3
"""Ingest documents into the RAG system"""
import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import MultimodalRAGSystem
from src.utils.logger import setup_logger

logger = setup_logger()

async def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    parser.add_argument("--file", help="Single file to ingest")
    parser.add_argument("--directory", help="Directory to ingest")
    parser.add_argument("--config", help="Path to config file")
    
    args = parser.parse_args()
    
    if not args.file and not args.directory:
        parser.error("Either --file or --directory is required")
    
    # Initialize system
    system = MultimodalRAGSystem(config_path=args.config)
    await system.initialize()
    
    try:
        if args.file:
            # Process single file
            logger.info(f"Processing file: {args.file}")
            document = await system.process_document(args.file)
            logger.info(f"✅ Processed: {document.id} - Status: {document.status.value}")
            
        elif args.directory:
            # Process directory
            logger.info(f"Processing directory: {args.directory}")
            results = await system.process_directory(args.directory)
            
            # Print summary
            logger.info("\n" + "="*40)
            logger.info("Processing Summary:")
            logger.info("="*40)
            
            for doc_type, documents in results.items():
                successful = sum(1 for d in documents if d.status.value == "completed")
                failed = sum(1 for d in documents if d.status.value == "failed")
                logger.info(f"{doc_type.upper()}: {successful} successful, {failed} failed")
    
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
    finally:
        await system.close()

if __name__ == "__main__":
    asyncio.run(main())