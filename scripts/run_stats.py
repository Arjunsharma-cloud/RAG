#!/usr/bin/env python3
"""Get system statistics"""
import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import MultimodalRAGSystem
from src.utils.logger import setup_logger

logger = setup_logger()

async def main():
    parser = argparse.ArgumentParser(description="Get RAG system statistics")
    parser.add_argument("--config", help="Path to config file")
    
    args = parser.parse_args()
    
    # Initialize system
    system = MultimodalRAGSystem(config_path=args.config)
    await system.initialize()
    
    try:
        # Get system info
        info = await system.get_system_info()
        
        print("\n" + "="*40)
        print("📊 System Statistics")
        print("="*40)
        
        print("\n🔧 Configuration:")
        print(f"  Embedding Model: {info['embedding_model']}")
        print(f"  LLM Model: {info['llm_model']}")
        print(f"  Chunk Size: {info['chunk_size']}")
        print(f"  Chunking Strategy: {info['chunking_strategy']}")
        print(f"  Reranker: {'Enabled' if info['reranker_enabled'] else 'Disabled'}")
        
        # Get vector store stats
        stats = await system.vector_store.get_collection_stats()
        print(f"\n🗄️  Vector Store: {stats.get('name', 'Unknown')}")
        print(f"  Total Chunks: {stats.get('count', 0)}")
        
        # Get active sessions
        conversations = await system.memory_store.list_conversations()
        print(f"\n💬 Active Sessions: {len(conversations)}")
        
        print("\n" + "="*40)
        
    finally:
        await system.close()

if __name__ == "__main__":
    asyncio.run(main())