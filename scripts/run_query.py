#!/usr/bin/env python3
"""Query the RAG system"""
import asyncio
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import MultimodalRAGSystem
from src.utils.logger import setup_logger

logger = setup_logger()

async def main():
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("--query", required=True, help="Your question")
    parser.add_argument("--session", default="default", help="Session ID")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--filters", help="JSON string of metadata filters")
    
    args = parser.parse_args()
    
    # Parse filters if provided
    filters = None
    if args.filters:
        try:
            filters = json.loads(args.filters)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid filters JSON: {e}")
            sys.exit(1)
    
    # Initialize system
    system = MultimodalRAGSystem(config_path=args.config)
    await system.initialize()
    
    # Get system info
    info = await system.get_system_info()
    logger.info(f"Using chunking strategy: {info['chunking_strategy']}")
    logger.info(f"Using LLM provider: {info['llm_provider']}")
    
    try:
        # Run query
        result = await system.query(args.query, args.session, filters)
        
        print(f"\n📝 Question: {args.query}")
        print(f"\n💡 Answer: {result['answer']}\n")
        
        if result['sources']:
            print("📚 Sources:")
            for i, source in enumerate(result['sources'], 1):
                filename = source['metadata'].get('file_name', 'Unknown')
                print(f"  {i}. {filename}")
                print(f"     Preview: {source['text'][:100]}...")
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        print(f"\n❌ Error: {e}")
    finally:
        await system.close()

if __name__ == "__main__":
    asyncio.run(main())