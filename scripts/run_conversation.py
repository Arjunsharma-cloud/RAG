#!/usr/bin/env python3
"""Interactive conversation with the RAG system"""
import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import MultimodalRAGSystem
from src.utils.logger import setup_logger

logger = setup_logger()

async def interactive_session(system: MultimodalRAGSystem, session_id: str):
    """Run an interactive conversation session"""
    
    # Get system info
    info = await system.get_system_info()
    
    print("\n" + "="*60)
    print("🤖 Interactive RAG Session Started")
    print("="*60)
    print(f"Session ID: {session_id}")
    print(f"Chunking: {info['chunking_strategy']}")
    print(f"Reranker: {'Enabled' if info['reranker_enabled'] else 'Disabled'}")
    print("="*60)
    print("Commands: /help, /history, /clear, /exit")
    print("="*60 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\n📝 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()
                if command == '/exit':
                    print("👋 Goodbye!")
                    break
                elif command == '/help':
                    print("\nCommands:")
                    print("  /help     - Show this help")
                    print("  /history  - Show conversation history")
                    print("  /clear    - Clear conversation")
                    print("  /exit     - Exit session")
                    continue
                elif command == '/history':
                    conversation = await system.get_conversation_history(session_id)
                    if conversation and conversation.messages:
                        print("\n📜 Conversation History:")
                        for msg in conversation.messages:
                            role = "👤 You" if msg.role == "user" else "🤖 Assistant"
                            print(f"{role}: {msg.content[:100]}...")
                    else:
                        print("\nNo conversation history yet.")
                    continue
                elif command == '/clear':
                    await system.clear_conversation(session_id)
                    print("\n🧹 Conversation cleared!")
                    continue
                else:
                    print(f"Unknown command: {command}")
                    continue
            
            # Process query
            print("🤖 Assistant: ", end="", flush=True)
            result = await system.query(user_input, session_id)
            print(result['answer'])
            
            # Show sources
            if result['sources']:
                print("\n📚 Sources:")
                for i, source in enumerate(result['sources'][:3], 1):
                    filename = source['metadata'].get('file_name', 'Unknown')
                    print(f"  {i}. {filename}")
                    
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Error: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Interactive RAG conversation")
    parser.add_argument("--session", default="default", help="Session ID")
    parser.add_argument("--config", help="Path to config file")
    
    args = parser.parse_args()
    
    # Initialize system
    system = MultimodalRAGSystem(config_path=args.config)
    await system.initialize()
    
    try:
        await interactive_session(system, args.session)
    finally:
        await system.close()

if __name__ == "__main__":
    asyncio.run(main())