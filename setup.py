from setuptools import setup, find_packages

setup(
    name="agentvault",
    version="0.1.0",
    description="Secure MCP Gateway + Action-Level Agent Firewall",
    author="AgentVault",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "aiosqlite>=0.19.0",
        "pyyaml>=6.0.1",
        "pydantic>=2.5.0",
        "openai>=1.6.0",
        "anthropic>=0.18.0",
        "google-genai>=1.0.0",
        "httpx>=0.25.0",
        "websockets>=12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.23.0",
        ]
    },
)
