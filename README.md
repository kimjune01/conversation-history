# Chroma DB Query Tool

A specialized client for [Chroma DB](https://www.trychroma.com/) designed to help remember and manage conversations with an MCP (Multi-Channel Platform) client. This tool makes it easy to query, explore, and filter conversation history stored in your local Chroma DB instance, especially when used alongside the companion browser scraper repository.

---

## Prerequisites

- [Chroma DB](https://www.trychroma.com/) running locally
- Data ingested into Chroma DB via the companion browser scraper repository
- Python 3.8+ (or specify your language/environment if different)
- (Optional) Virtual environment for Python projects

---

## Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure your Chroma DB is running locally and populated with data from the browser scraper.**

---

## Usage

1. **Start the query tool:**

   ```bash
   python main.py
   ```

   _(Replace `main.py` with your entry point if different)_

2. **Query collections:**

   - List available collections
   - Search or filter data by your desired criteria

3. **Example query:**

   ```python
   # Example Python code to query a collection
   from chromadb import Client

   client = Client()
   collection = client.get_collection("your_collection_name")
   results = collection.query("your search term")
   print(results)
   ```

---

## Data Source

- Data is provided by a separate browser scraper repository.
- Make sure to run the scraper and ingest data into your local Chroma DB before querying.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

---

## License

[MIT License](LICENSE)  
_(Or specify your license)_

---

## Acknowledgements

- [Chroma DB](https://www.trychroma.com/)
- The browser scraper project for data ingestion
