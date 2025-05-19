# Conversation History

### Remember your conversation

![remember](https://github.com/user-attachments/assets/9b199f61-2eeb-4a54-a86d-03907f002a05)

### Recall your conversation (persistent)

![recall](https://github.com/user-attachments/assets/29c80d05-ebd8-4fcb-931d-5634e0d65a2d)

A specialized client using [Chroma DB](https://www.trychroma.com/) designed to help remember and manage conversations with an MCP (Multi-Channel Platform) client. This tool makes it easy to query, explore, and filter conversation history stored in your local Chroma DB instance, especially when used alongside the companion browser scraper repository.

---

## Prerequisites

- [Chroma DB](https://www.trychroma.com/) running locally
- Data ingested into Chroma DB via the companion browser scraper repository
- Python 3.8+ (or specify your language/environment if different)
- (Optional) Virtual environment for Python projects

---

## Setup

1. Download the repo, run `uv sync`
2. Copy paste the contents of `claude-config.json` and set data dir to whereever.

```
"conversation_history": {
        "command": "uv",
        "args": [
          "--directory",
          "/THIS_REPO_PATH",
          "run",
          "server.py",
          "--data-dir",
          "/THIS_REPO_PATH/chroma_data"
        ]
}
```

3. same config should also work in any other mcp client.

---

## License

[MIT License](LICENSE)

---

## Acknowledgements

- [Chroma DB](https://www.trychroma.com/)
