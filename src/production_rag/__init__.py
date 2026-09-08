"""Production RAG over DuckDB, dbt and Dagster documentation.

Retrieval is measured, not asserted: BM25, dense and hybrid arms are evaluated
against the same ground truth, and the generation model is a swappable component.
"""

__version__ = "0.1.0"
