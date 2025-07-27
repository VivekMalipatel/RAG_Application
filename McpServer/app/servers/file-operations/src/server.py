# file-operations/src/server.py
from pathlib import Path
import os
import json
from datetime import datetime
import sys

# Add the core directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from core.base_server.src.server import BaseMCPServer

class FileOperationsMCPServer(BaseMCPServer):
    def __init__(self, port: int = 8080, base_path: str = None):
        super().__init__("file-operations", port)
        self.base_path = self._determine_base_path(base_path)
        self.allowed_directories = self._setup_allowed_directories()
        self._ensure_base_directory_exists() 
    
    def _determine_base_path(self, user_base_path: str = None) -> str:
        if user_base_path:
            self.logger.info(f"Using user-provided base path: {user_base_path}")
            return os.path.abspath(user_base_path)
        
        env_base_path = os.getenv("FILE_OPS_BASE_PATH")
        if env_base_path:
            self.logger.info(f"Using environment base path: {env_base_path}")
            return os.path.abspath(env_base_path)
        
        # Check if running in Docker container
        if os.path.exists("/.dockerenv"):
            default_path = "/app/data"
            self.logger.info(f"Docker environment detected, using: {default_path}")
            return default_path
        
        # Local development default
        default_path = os.path.join(os.getcwd(), "data")
        self.logger.info(f"Local environment detected, using: {default_path}")
        return default_path
    
    def _setup_allowed_directories(self) -> list:
        """Setup allowed directories based on base path and environment"""
        allowed_dirs = []
        
        # Always include the base path
        allowed_dirs.append(self.base_path)
        
        # Add additional directories from environment variable
        allowed_dirs_env = os.getenv("FILE_OPS_ALLOWED_DIRS")
        if allowed_dirs_env:
            additional_dirs = [d.strip() for d in allowed_dirs_env.split(":") if d.strip()]
            allowed_dirs.extend(additional_dirs)
        else:
            # Default additional safe directories
            if os.path.exists("/.dockerenv"):
                # Docker environment defaults
                allowed_dirs.extend(["/tmp", "/app/temp"])
            else:
                # Local environment defaults
                allowed_dirs.extend([
                    os.path.join(os.getcwd(), "temp"),
                    "/tmp" if os.name != 'nt' else os.path.join(os.environ.get('TEMP', ''), 'mcp-files')
                ])
        
        # Remove duplicates and normalize paths
        normalized_dirs = list(set([os.path.abspath(d) for d in allowed_dirs]))
        
        self.logger.info(f"Configured allowed directories: {normalized_dirs}")
        return normalized_dirs
    
    def _ensure_base_directory_exists(self):
        """Ensure the base directory exists and is writable"""
        try:
            base_dir = Path(self.base_path)
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = base_dir / ".mcp_write_test"
            test_file.write_text("test")
            test_file.unlink()
            
            self.logger.info(f"✅ Base directory ready: {self.base_path}")
        except Exception as e:
            self.logger.error(f"❌ Cannot setup base directory {self.base_path}: {e}")
            raise RuntimeError(f"Base directory setup failed: {e}")
    
    def _resolve_path(self, path: str, use_base_path: bool = True) -> str:
        """Resolve a path, optionally relative to base_path"""
        if not use_base_path or os.path.isabs(path):
            return os.path.abspath(path)
        
        # Resolve relative to base_path
        return os.path.abspath(os.path.join(self.base_path, path))
    
    def register_tools(self):
        
        @self.app.tool()
        @self.require_auth("read")
        async def get_base_path() -> dict:
            """Get current base path and configuration information
            
            Returns:
                dict: Configuration details including base path and allowed directories
                
            Example response:
                {
                    "base_path": "/app/data",
                    "allowed_directories": ["/app/data", "/tmp"],
                    "is_docker": true
                }
            """
            return {
                "base_path": self.base_path,
                "allowed_directories": self.allowed_directories,
                "is_docker": os.path.exists("/.dockerenv"),
                "current_working_dir": os.getcwd()
            }
        
        @self.app.tool()
        @self.require_auth("read")
        async def read_file(path: str, use_base_path: bool = True) -> str:
            """Read the complete contents of a text file
            
            Args:
                path (str): File path to read. Can be relative like "myfile.txt" or absolute like "/tmp/file.txt"
                use_base_path (bool, optional): If True, relative paths are resolved from the base directory. Defaults to True.
            
            Returns:
                str: File contents as text, or error message if file cannot be read
                
            Examples:
                - read_file("notes.txt") -> Reads from base_path/notes.txt
                - read_file("docs/readme.md", True) -> Reads from base_path/docs/readme.md  
                - read_file("/tmp/config.json", False) -> Reads absolute path
            """
            try:
                resolved_path = self._resolve_path(path, use_base_path)
                
                if not self._is_path_allowed(resolved_path):
                    return f"❌ Error: Access denied to path {resolved_path}. Allowed directories: {self.allowed_directories}"
                
                file_path = Path(resolved_path)
                if not file_path.exists():
                    return f"❌ Error: File {resolved_path} does not exist"
                
                if not file_path.is_file():
                    return f"❌ Error: {resolved_path} is not a regular file"
                
                content = file_path.read_text(encoding='utf-8')
                return f"📄 File: {resolved_path}\n📏 Size: {len(content)} characters\n\n{content}"
            except Exception as e:
                return f"❌ Error reading file: {str(e)}"
        
        @self.app.tool()
        @self.require_auth("write")
        async def write_file(path: str, content: str = "", use_base_path: bool = True) -> str:
            """Write text content to a file with automatic directory creation
            
            Args:
                path (str): File path to write to. Examples: "myfile.txt", "docs/readme.md", "/absolute/path.txt"
                content (str, optional): Text content to write. Can be empty string. Defaults to "".
                use_base_path (bool, optional): If True, relative paths are resolved from base directory. Defaults to True.
            
            Returns:
                str: Success message with details, or error message if write fails
                
            Examples:
                - write_file("notes.txt", "Hello World") -> Creates base_path/notes.txt
                - write_file("docs/api.md", "# API Documentation", True) -> Creates base_path/docs/api.md with auto-created dirs
                - write_file("/tmp/temp.txt", "temporary data", False) -> Writes to absolute path
            """
            try:
                # Input validation
                if not path or not path.strip():
                    return "❌ Error: 'path' parameter is required and cannot be empty. Example: 'myfile.txt'"
                
                resolved_path = self._resolve_path(path, use_base_path)
                
                if not self._is_path_allowed(resolved_path):
                    return f"❌ Error: Access denied to path {resolved_path}. Allowed directories: {self.allowed_directories}"
                
                file_path = Path(resolved_path)
                # Create directory if it doesn't exist
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_path.write_text(content, encoding='utf-8')
                return f"✅ Successfully wrote {len(content)} characters to {resolved_path}"
            except Exception as e:
                return f"❌ Error writing file: {str(e)}"
        
        @self.app.tool()
        async def list_files(directory: str = "", use_base_path: bool = True) -> dict:
            """List all files and directories in the specified path
            
            Args:
                directory (str, optional): Directory to list. Empty string lists the base directory. Examples: "", "subfolder", "/absolute/path"
                use_base_path (bool, optional): If True, relative paths are resolved from base directory. Defaults to True.
            
            Returns:
                dict: Directory listing with files and subdirectories
                
            Response structure:
                {
                    "directory": "/full/resolved/path",
                    "base_path": "/configured/base/path",
                    "files": [{"name": "file.txt", "path": "/full/path", "size": 123}],
                    "directories": [{"name": "subfolder", "path": "/full/path"}],
                    "total_files": 2,
                    "total_directories": 1
                }
                
            Examples:
                - list_files() -> Lists base directory
                - list_files("docs") -> Lists base_path/docs/
                - list_files("/tmp", False) -> Lists absolute path
            """
            try:
                if not directory:  # Use base path if no directory specified
                    resolved_path = self.base_path
                else:
                    resolved_path = self._resolve_path(directory, use_base_path)
                
                if not self._is_path_allowed(resolved_path):
                    return {"error": f"❌ Access denied to path {resolved_path}. Allowed: {self.allowed_directories}"}
                
                dir_path = Path(resolved_path)
                if not dir_path.exists():
                    return {"error": f"❌ Directory {resolved_path} does not exist"}
                
                if not dir_path.is_dir():
                    return {"error": f"❌ {resolved_path} is not a directory"}
                
                files = []
                directories = []
                
                for item in dir_path.iterdir():
                    try:
                        if item.is_file():
                            files.append({
                                "name": item.name,
                                "path": str(item),
                                "size": item.stat().st_size,
                                "size_human": self._human_readable_size(item.stat().st_size),
                                "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                            })
                        elif item.is_dir():
                            directories.append({
                                "name": item.name,
                                "path": str(item)
                            })
                    except (OSError, PermissionError):
                        # Skip items we can't access
                        continue
                
                return {
                    "directory": resolved_path,
                    "base_path": self.base_path,
                    "files": files,
                    "directories": directories,
                    "total_files": len(files),
                    "total_directories": len(directories)
                }
            except Exception as e:
                return {"error": f"❌ Error listing directory: {str(e)}"}
        
        @self.app.tool()
        async def file_stats(path: str, use_base_path: bool = True) -> dict:
            """Get detailed statistics and metadata for a file or directory
            
            Args:
                path (str): Path to file or directory to analyze. Examples: "myfile.txt", "docs/", "/absolute/path"
                use_base_path (bool, optional): If True, relative paths are resolved from base directory. Defaults to True.
            
            Returns:
                dict: Detailed file statistics including size, dates, permissions
                
            Response includes:
                - path, name, size (bytes and human-readable)
                - file type (file/directory)
                - creation, modification, access timestamps
                - file permissions
                
            Examples:
                - file_stats("config.json") -> Stats for base_path/config.json
                - file_stats("/tmp/data", False) -> Stats for absolute path
            """
            try:
                resolved_path = self._resolve_path(path, use_base_path)
                
                if not self._is_path_allowed(resolved_path):
                    return {"error": f"❌ Access denied to path {resolved_path}"}
                
                file_path = Path(resolved_path)
                if not file_path.exists():
                    return {"error": f"❌ Path {resolved_path} does not exist"}
                
                stat = file_path.stat()
                return {
                    "path": str(file_path),
                    "name": file_path.name,
                    "size_bytes": stat.st_size,
                    "size_human": self._human_readable_size(stat.st_size),
                    "is_file": file_path.is_file(),
                    "is_directory": file_path.is_dir(),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                    "permissions": oct(stat.st_mode)[-3:],
                    "owner_readable": bool(stat.st_mode & 0o400),
                    "owner_writable": bool(stat.st_mode & 0o200),
                    "owner_executable": bool(stat.st_mode & 0o100)
                }
            except Exception as e:
                return {"error": f"❌ Error getting file stats: {str(e)}"}
        
        @self.app.tool()
        async def create_directory(path: str, use_base_path: bool = True) -> str:
            """Create a new directory with automatic parent directory creation
            
            Args:
                path (str): Directory path to create. Examples: "new_folder", "docs/api", "/absolute/path"
                use_base_path (bool, optional): If True, relative paths are resolved from base directory. Defaults to True.
            
            Returns:
                str: Success message or error description
                
            Features:
                - Creates parent directories automatically if they don't exist
                - Safe to call on existing directories (no error)
                - Respects security restrictions
                
            Examples:
                - create_directory("uploads") -> Creates base_path/uploads/
                - create_directory("docs/api/v1", True) -> Creates nested dirs base_path/docs/api/v1/
                - create_directory("/tmp/myapp", False) -> Creates absolute path
            """
            try:
                if not path or not path.strip():
                    return "❌ Error: 'path' parameter is required and cannot be empty"
                
                resolved_path = self._resolve_path(path, use_base_path)
                
                if not self._is_path_allowed(resolved_path):
                    return f"❌ Error: Access denied to path {resolved_path}. Allowed: {self.allowed_directories}"
                
                dir_path = Path(resolved_path)
                dir_path.mkdir(parents=True, exist_ok=True)
                
                if dir_path.exists():
                    return f"✅ Successfully created directory {resolved_path}"
                else:
                    return f"❌ Failed to create directory {resolved_path}"
            except Exception as e:
                return f"❌ Error creating directory: {str(e)}"
        
        @self.app.tool()
        @self.require_auth("write")
        async def delete_file(path: str, use_base_path: bool = True) -> str:
            """Delete a file or empty directory with safety checks
            
            Args:
                path (str): Path to file or directory to delete. Examples: "old_file.txt", "empty_folder", "/tmp/temp_file"
                use_base_path (bool, optional): If True, relative paths are resolved from base directory. Defaults to True.
            
            Returns:
                str: Success message or error description
                
            Safety features:
                - Only deletes empty directories (prevents accidental data loss)
                - Confirms file/directory exists before deletion
                - Respects security restrictions
                - Clear error messages for all failure cases
                
            Examples:
                - delete_file("old_notes.txt") -> Deletes base_path/old_notes.txt
                - delete_file("empty_folder") -> Deletes base_path/empty_folder/ if empty
                - delete_file("/tmp/temp_file.log", False) -> Deletes absolute path
            """
            try:
                if not path or not path.strip():
                    return "❌ Error: 'path' parameter is required and cannot be empty"
                
                resolved_path = self._resolve_path(path, use_base_path)
                
                if not self._is_path_allowed(resolved_path):
                    return f"❌ Error: Access denied to path {resolved_path}"
                
                file_path = Path(resolved_path)
                if not file_path.exists():
                    return f"❌ Error: Path {resolved_path} does not exist"
                
                if file_path.is_file():
                    file_path.unlink()
                    return f"✅ Successfully deleted file {resolved_path}"
                elif file_path.is_dir():
                    # Only delete empty directories for safety
                    if any(file_path.iterdir()):
                        return f"❌ Error: Directory {resolved_path} is not empty. Only empty directories can be deleted for safety."
                    file_path.rmdir()
                    return f"✅ Successfully deleted empty directory {resolved_path}"
                else:
                    return f"❌ Error: {resolved_path} is neither a regular file nor directory"
            except Exception as e:
                return f"❌ Error deleting: {str(e)}"
    
    def get_capabilities(self) -> list:
        """Return capabilities of this server"""
        return [
            "file-read",
            "file-write", 
            "directory-list",
            "file-stats",
            "directory-create",
            "file-delete",
            "base-path-management",
            "basic"
        ]
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if the path is within allowed directories"""
        abs_path = os.path.abspath(path)
        return any(abs_path.startswith(os.path.abspath(allowed)) 
                  for allowed in self.allowed_directories)
    
    def _human_readable_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

if __name__ == "__main__":
    server = FileOperationsMCPServer()
    print(f"🚀 Starting File Operations MCP Server on port 8080")
    print(f"📁 Base path: {server.base_path}")
    print(f"📁 Allowed directories: {server.allowed_directories}")
    server.run()
