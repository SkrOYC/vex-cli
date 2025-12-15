"""Configuration bridge to map VibeConfig to DeepAgents parameters."""

from langchain.agents.middleware import InterruptOnConfig
from langchain_core.language_models import BaseChatModel

from deepagents.backends import FilesystemBackend

from vibe.core.config import VibeConfig


class DeepAgentsConfig:
    """Bridge between VibeConfig and DeepAgents parameters."""
    
    @staticmethod
    def create_model(config: VibeConfig) -> BaseChatModel:
        """Create LangChain model from Vibe config."""
        # Import here to avoid circular imports
        from vibe.core.engine.models import create_model_from_config
        return create_model_from_config(config)
    
    @staticmethod
    def create_backend(config: VibeConfig) -> FilesystemBackend:
        """Create filesystem backend from Vibe config."""
        return FilesystemBackend(
            root_dir=config.effective_workdir,
            virtual_mode=False,
        )
    
    @staticmethod
    def create_interrupt_config(config: VibeConfig) -> dict[str, bool | InterruptOnConfig]:
        """Create interrupt config from Vibe tool permissions."""
        interrupt_on = {}
        
        # Map Vibe permissions to DeepAgents interrupts
        for tool_name, tool_config in config.tools.items():
            if tool_config.permission == "ask":
                interrupt_on[tool_name] = True
            # "always" = no interrupt, "never" = tool filtered out during loading
        
        return interrupt_on
