from fastmcp import FastMCP

from app.collection_manager import CollectionManager

mcp = FastMCP("Simple-Embed-MCP")

collection_manager_obj = CollectionManager()
mcp.add_tool(collection_manager_obj.list_collections)
mcp.add_tool(collection_manager_obj.add_collection)
mcp.add_tool(collection_manager_obj.add_value_to_collection)
mcp.add_tool(collection_manager_obj.batch_add_values_to_collection)
mcp.add_tool(collection_manager_obj.search_in_collection_embedding)
mcp.add_tool(collection_manager_obj.search_in_collection_bm25)

if __name__ == "__main__":
    mcp.run()
