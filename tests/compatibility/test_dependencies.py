def test_import_compatibility():
    """Test all imports work together."""
    from deepagents import create_deep_agent
    from langchain_openai import ChatOpenAI
    from vibe.core.config import VibeConfig
    # Verify no conflicts