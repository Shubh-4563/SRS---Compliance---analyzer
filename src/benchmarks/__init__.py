"""
V2 Declarative Benchmark Manifest Package.
"""
from src.benchmarks.manifest_validator import (
    validate_schema,
    validate_cross_field_rules,
    validate_document_integrity,
    load_and_validate_manifest,
    get_schema_path,
    BenchmarkValidationError,
    BenchmarkSchemaError,
    BenchmarkCrossFieldError,
    BenchmarkIntegrityError,
)

__all__ = [
    "validate_schema",
    "validate_cross_field_rules",
    "validate_document_integrity",
    "load_and_validate_manifest",
    "get_schema_path",
    "BenchmarkValidationError",
    "BenchmarkSchemaError",
    "BenchmarkCrossFieldError",
    "BenchmarkIntegrityError",
]

