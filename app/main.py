from fastmcp import FastMCP

from app.collection_manager import CollectionManager

mcp = FastMCP("Simple-Embed-MCP")

collection_manager_obj = CollectionManager()
mcp.add_tool(collection_manager_obj.list_collections)
mcp.add_tool(collection_manager_obj.add_collection)

if __name__ == "__main__":
    mcp.run()
