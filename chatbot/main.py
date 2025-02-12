import asyncio
import logging
import json
from typing import Dict, List, Any

from helpers.config import Configuration
from helpers.llm_client import LLMClient
from helpers.mcpserver import MCPServer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: List[MCPServer], llm_client: LLMClient) -> None:
        self.servers: List[MCPServer] = servers
        self.llm_client: LLMClient = llm_client

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))
        
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def execute_tool_call(self, tool_call: Dict) -> str:
        """Process the LLM response and execute tools if needed.
        
        Args:
            llm_response: The tool call request from the LLM.
            
        Returns:
            The result of tool execution or the original response.
        """
        try:
            arguments = json.loads(tool_call["arguments"])
            logging.info(f"Requesting execution of tool '{tool_call['name']}' with arguments {tool_call['arguments']}")
                
            for server in self.servers:
                tools = await server.list_tools()
                if any(tool.name == tool_call["name"] for tool in tools):
                    try:
                        result = await server.execute_tool(tool_call["name"], arguments)
                        
                        if isinstance(result, dict) and 'progress' in result:
                            progress = result['progress']
                            total = result['total']
                            logging.info(f"Progress: {progress}/{total} ({(progress/total)*100:.1f}%)")

                        return result.content[0].text
                    except Exception as e:
                        error_msg = f"Error executing tool (inner): {str(e)}"
                        logging.error(error_msg)
                        return error_msg
            
            return f"No server found with tool: {tool_call['tool']}"
        except Exception as e:
            return f"Error executing tool (outer): {str(e)}"

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return
            
            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)
            
            tool_descriptions = []

            for tool in all_tools:
                tool_descriptions.append({
                    "type": "function",
                    "function": tool.format_for_llm()
                })
            
            with open("helpers/main_prompt.txt", "r") as f:
                system_message = f.read()

            messages = [
                {
                    "role": "system",
                    "content": system_message
                }
            ]

            while True:
                try:
                    user_input = input("You: ").strip().lower()
                    if user_input in ['quit', 'exit']:
                        logging.info("Exiting...")
                        break

                    if user_input in ['/messages']:
                        logging.info("Logging %s messages:", len(messages))
                        for message in messages:
                            logging.info(message)
                        continue

                    messages.append({"role": "user", "content": user_input})
                    
                    await self.process_request(messages, tool_descriptions)

                except KeyboardInterrupt:
                    logging.info("Exiting...")
                    break
        
        finally:
            await self.cleanup_servers()

    async def process_request(self, messages, tool_descriptions) -> None:
        llm_response_message = self.llm_client.get_response(messages, tool_descriptions)

        if "tool_calls" in llm_response_message:
            logging.info("Assistant: Assistant has requested to run one or more tools")

            messages.append(llm_response_message)

            for tool_call in llm_response_message["tool_calls"]:
                tool_call_function = tool_call["function"]
                result = await self.execute_tool_call(tool_call_function)
                
                logging.info("Tool execution provided a response of: %s", result)

                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})

            await self.process_request(messages, tool_descriptions)

        else:
            llm_response = llm_response_message['content']
            logging.info("Assistant: %s", llm_response)
            messages.append({"role": "assistant", "content": llm_response})



async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.mcp_servers_config
    servers = [MCPServer(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
    llm_client = LLMClient(config)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()

if __name__ == "__main__":
    asyncio.run(main())