#!/usr/bin/env python3
"""Download and setup required models"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.embedding.bge_service import BGEEmbeddingService
from src.services.reranker.bge_reranker import BGEReranker
from src.utils.logger import setup_logger

logger = setup_logger()

async def main():
    logger.info("="*50)
    logger.info("Setting up models for RAG System")
    logger.info("="*50)
    
    # Setup embedding model
    logger.info("\n1. Downloading BGE embedding model...")
    try:
        embedding_service = BGEEmbeddingService()
        await embedding_service.initialize()
        logger.info("✅ BGE embedding model ready")
    except Exception as e:
        logger.error(f"❌ Failed to download embedding model: {e}")
    
    # Setup reranker (optional)
    logger.info("\n2. Downloading BGE reranker model...")
    try:
        reranker = BGEReranker()
        await reranker.initialize()
        logger.info("✅ BGE reranker model ready")
    except Exception as e:
        logger.warning(f"⚠️ Reranker download failed (optional): {e}")
    
    logger.info("\n" + "="*50)
    logger.info("Model setup complete!")
    logger.info("="*50)

if __name__ == "__main__":
    asyncio.run(main())