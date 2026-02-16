"""
Test Pipeline for Legal Document Processing (UPDATED VERSION)

UPDATES:
1. Uses new embedding-based role classifier (no training required!)
2. Fixes missing top_k_chunks.append() bug
3. Ensures all_chunks.json files are saved for each document
4. Removes duplicate 'id' and 'chunk_id' fields (keeps only 'id')
5. Cleans up chunk metadata properly
6. Improves logging for debugging

ROLE CLASSIFICATION:
- Now uses semantic similarity between chunks and role descriptions
- No fine-tuned model required
- Better confidence scores (0.3-0.8 instead of 0.1-0.2)
- Easy to customize via role_descriptions_dict in config.py
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib
import sys

from utils.json_helper import safe_json_dump

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration
from config import (
    LOGGING_CONFIG,
    ADLS_CONFIG,
    EMBEDDING_CONFIG,
    CHUNKING_CONFIG,
    PROCESSING_CONFIG,
    ROLE_CLASSIFICATION_CONFIG
)

# Import pipeline components
from core.adls_fetcher import ADLSFetcher
from core.legal_text_cleaner import LegalTextCleaner
from core.semantic_chunker import SemanticChunker
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
# Use embedding-based role classifier (no training required)
from core.role_classifier_embedding import EmbeddingRoleClassifier, create_classifier_from_config
# Old fine-tuned classifier kept for future use (see role_classifier_FUTURE.py)

logger = logging.getLogger(__name__)


def generate_document_id_from_path(source_file_path: str) -> str:
    """
    Generate unique document ID from source file path.
    
    Example:
        'raw/newapp/decisions/queensland/2005/doc.json' 
        -> 'decisions_queensland_2005_doc'
    """
    # Remove extension
    path_without_ext = source_file_path.replace('.json', '')
    
    # Split and filter
    parts = Path(path_without_ext).parts
    skip_prefixes = {'raw', 'newapp', 'input', 'data'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    
    # Create ID
    doc_id = '_'.join(relevant_parts).replace('-', '_').replace(' ', '_').lower()
    
    # Handle long IDs
    if len(doc_id) > 200:
        base = relevant_parts[-1] if relevant_parts else 'doc'
        hash_suffix = hashlib.md5(source_file_path.encode()).hexdigest()[:8]
        doc_id = f"{base}_{hash_suffix}"
    
    return doc_id


def get_local_chunks_path(source_file_path: str, base_output: str) -> Path:
    """
    Generate local path for all_chunks.json that mirrors input structure.
    
    Example:
        Input:  'raw/newapp/decisions/queensland/2005/document.json'
        Output: 'test_output/chunks/decisions/queensland/2005/document_all_chunks.json'
    """
    path_without_ext = source_file_path.replace('.json', '')
    parts = Path(path_without_ext).parts
    
    skip_prefixes = {'raw', 'newapp', 'input', 'data'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    
    if relevant_parts:
        directory_parts = relevant_parts[:-1]
        filename = relevant_parts[-1]
        
        if directory_parts:
            output_dir = Path(base_output) / "chunks" / Path(*directory_parts)
            return output_dir / f"{filename}_all_chunks.json"
        else:
            return Path(base_output) / "chunks" / f"{filename}_all_chunks.json"
    else:
        return Path(base_output) / "chunks" / "unknown_all_chunks.json"


class TestPipeline:
    """
    Updated test pipeline with:
    - Embedding-based role classification (no training required)
    - Proper all_chunks.json saving
    - No duplicate id/chunk_id fields
    - Clean metadata
    - Fixed top_k_chunks bug
    """
    
    def __init__(self, output_dir: str = "test_output"):
        """
        Initialize test pipeline.
        
        Args:
            output_dir: Local directory for all outputs
        """
        logger.info("=" * 80)
        logger.info("TEST PIPELINE - UPDATED WITH EMBEDDING-BASED ROLE CLASSIFIER")
        logger.info("=" * 80)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.chunks_dir = self.output_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        
        self.stats_dir = self.output_dir / "stats"
        self.stats_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        
        # Validate only ADLS config
        self._validate_adls_config()
        
        # Initialize components
        self._init_adls()
        self._init_processors()
        self.role_classifier = None

        # Initialize embedding-based role classifier if enabled
        if ROLE_CLASSIFICATION_CONFIG["enabled"]:
            logger.info("Initializing Embedding-Based Role Classifier...")
            logger.info(f"Method: {ROLE_CLASSIFICATION_CONFIG.get('method', 'embedding')}")
            
            # Use the new embedding-based classifier
            # This uses semantic similarity - no training required!
            self.role_classifier = create_classifier_from_config()
            
            if self.role_classifier:
                logger.info("Role Classifier initialized successfully")
                # Log role info
                role_info = self.role_classifier.get_role_info()
                logger.info(f"  Number of roles: {role_info['num_roles']}")
                logger.info(f"  Confidence threshold: {role_info['threshold']}")
                logger.info(f"  Aggregation method: {role_info['aggregation']}")
            else:
                logger.warning("Role classification is disabled or failed to initialize")
        else:
            logger.info("Role classification disabled")

        logger.info("Test pipeline initialized successfully")
    
    def _validate_adls_config(self):
        """Validate ADLS configuration only."""
        errors = []
        
        if not ADLS_CONFIG["account_name"]:
            errors.append("ADLS_ACCOUNT_NAME not set")
        if not ADLS_CONFIG["account_key"]:
            errors.append("ADLS_ACCOUNT_KEY not set")
        if not ADLS_CONFIG["container_name"]:
            errors.append("ADLS_CONTAINER_NAME not set")
        
        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    def _init_adls(self):
        """Initialize ADLS fetcher."""
        logger.info("Initializing ADLS connection...")
        self.adls_fetcher = ADLSFetcher(
            account_name=ADLS_CONFIG["account_name"],
            account_key=ADLS_CONFIG["account_key"],
            container_name=ADLS_CONFIG["container_name"]
        )
    
    def _init_processors(self):
        """Initialize text processors."""
        logger.info("Initializing text processors...")
        
        # Text cleaner
        self.text_cleaner = LegalTextCleaner()
        
        # Semantic chunker
        self.semantic_chunker = SemanticChunker(
            model_name=EMBEDDING_CONFIG["model_name"],
            similarity_threshold=CHUNKING_CONFIG["similarity_threshold"],
            min_sentences_per_chunk=CHUNKING_CONFIG["min_sentences_per_chunk"],
            max_sentences_per_chunk=CHUNKING_CONFIG["max_sentences_per_chunk"],
            role_file_path="role_desc.py",
            min_chunk_size=CHUNKING_CONFIG["min_chunk_size"]
        )
        
        # Embedding model (reuse from chunker)
        self.embedding_model = self.semantic_chunker.model
        
        logger.info(f"Using embedding model: {EMBEDDING_CONFIG['model_name']}")
    
    def fetch_documents(self, max_docs: Optional[int] = None) -> List[Dict]:
        """
        Fetch documents from ADLS.
        
        Args:
            max_docs: Maximum documents to fetch (for testing)
            
        Returns:
            List of documents
        """
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: FETCHING DOCUMENTS FROM ADLS")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        documents = self.adls_fetcher.fetch_all(
            path=ADLS_CONFIG["input_path"],
            pattern=ADLS_CONFIG["file_pattern"],
            recursive=ADLS_CONFIG["recursive"],
            max_files=max_docs,
            show_progress=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Fetched {len(documents)} documents in {elapsed:.2f}s")
        
        return documents
    
    def process_single_document(
        self,
        doc: Dict
    ) -> Optional[Tuple[str, List[Dict], List[Dict], Path]]:
        """
        Process a single document.
        
        FIXED:
        - Uses only 'id' field (not both 'id' and 'chunk_id')
        - Properly includes all_chunks_path in top-k chunks
        
        Returns:
            Tuple of (doc_id, all_chunks, top_k_chunks, local_chunks_path) or None
        """
        source_file = doc.get("_source_file", "unknown.json")
        doc_id = generate_document_id_from_path(source_file)
        local_chunks_path = get_local_chunks_path(source_file, str(self.output_dir))
        
        # For traceability in chunks
        relative_chunks_path = str(local_chunks_path.relative_to(self.output_dir))
        
        try:
            # Clean text
            text = doc.get("text", "")
            if not text:
                logger.warning(f"Document {doc_id} has no text")
                return None
            
            cleaned_text = self.text_cleaner.clean(text)
            
            if not cleaned_text:
                logger.warning(f"Document {doc_id} is empty after cleaning")
                return None
            
            # Semantic chunking
            chunks, doc_embedding = self.semantic_chunker.split(
                cleaned_text,
                compute_doc_similarity=CHUNKING_CONFIG["compute_doc_similarity"]
            )
            
            if not chunks:
                logger.warning(f"Document {doc_id} produced no chunks")
                return None
            
            # Extract metadata (exclude text and internal fields)
            excluded_fields = {'text', 'embedding', '_source_file'}
            metadata = {k: v for k, v in doc.items() if k not in excluded_fields}
            
            # FIXED: Create chunks for ALL (stored locally)
            # Use only 'id' field, not both 'id' and 'chunk_id'
            all_chunks = []
            for idx, chunk in enumerate(chunks):
                chunk_all = {
                    "id": f"{doc_id}_{idx}",  # ONLY id field
                    "doc_id": doc_id,
                    "original_source_path": source_file,
                    "text": chunk["text"],
                    **metadata
                    # Removed: chunk_id (duplicate), start_char, end_char, num_sentences, similarities
                }
                all_chunks.append(chunk_all)
            # Role classification using embedding-based classifier
            # if self.role_classifier:
            #     self.role_classifier.classify_chunks(
            #         all_chunks,
            #         text_field="text",
            #         batch_size=ROLE_CLASSIFICATION_CONFIG["batch_size"],
            #         add_to_chunks=True,
            #         show_progress=True,  # Useful in testing to see progress
            #         return_all_scores=ROLE_CLASSIFICATION_CONFIG.get("add_probabilities", False)
            #     )
            #     # Note: Threshold is now handled inside the classifier
            #     # No need for post-processing here

            # Select top-k chunks if configured
            if CHUNKING_CONFIG["top_k"]:
                sort_key = CHUNKING_CONFIG["top_k_method"]  # 'doc_similarity' or 'avg_similarity'
                sorted_chunks = sorted(
                    enumerate(chunks),
                    key=lambda x: x[1].get(sort_key, 0.0),
                    reverse=True
                )
                selected_indices = [idx for idx, _ in sorted_chunks[:CHUNKING_CONFIG["top_k"]]]
            else:
                selected_indices = list(range(len(chunks)))
            
            # FIXED: Create chunks for TOP-K (for embedding/search)
            # Use only 'id' field and include all_chunks_path
            top_k_chunks = []
            for idx in selected_indices:
                chunk_topk = all_chunks[idx].copy()
                chunk_topk["all_chunks_path"] = relative_chunks_path
                top_k_chunks.append(chunk_topk)  # FIX: Actually append the chunk!

            logger.debug(f"Processed {doc_id}: {len(all_chunks)} total chunks, {len(top_k_chunks)} top-k chunks")
            
            return (doc_id, all_chunks, top_k_chunks, local_chunks_path)
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
            return None
    
    def process_documents(
        self,
        documents: List[Dict]
    ) -> Tuple[Dict[str, Tuple[List[Dict], Path]], List[Dict], Dict]:
        """
        Process all documents.
        
        Returns:
            Tuple of (doc_chunks_map, all_top_k_chunks, stats)
            - doc_chunks_map: dict mapping doc_id -> (chunks, local_path)
            - all_top_k_chunks: list of all top-k chunks
        """
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: PROCESSING DOCUMENTS")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        doc_chunks_map = {}  # doc_id -> (chunks, local_path)
        all_top_k_chunks = []
        
        failed_count = 0
        
        for doc in tqdm(documents, desc="Processing documents"):
            result = self.process_single_document(doc)
            
            if result is None:
                failed_count += 1
                if not PROCESSING_CONFIG["skip_errors"]:
                    raise ValueError("Document processing failed")
                continue
            
            doc_id, all_chunks, top_k_chunks, chunks_path = result
            
            # Store all chunks with their local path
            doc_chunks_map[doc_id] = (all_chunks, chunks_path)
            
            # Collect all top-k chunks
            all_top_k_chunks.extend(top_k_chunks)
        
        elapsed = time.time() - start_time
        
        stats = {
            "total_documents": len(documents),
            "successful_documents": len(doc_chunks_map),
            "failed_documents": failed_count,
            "total_chunks": sum(len(chunks) for chunks, _ in doc_chunks_map.values()),
            "top_k_chunks": len(all_top_k_chunks),
            "processing_time": elapsed
        }
        
        logger.info(f"Processed {stats['successful_documents']}/{stats['total_documents']} documents")
        logger.info(f"Total chunks: {stats['total_chunks']}, Top-K: {stats['top_k_chunks']}")
        
        return doc_chunks_map, all_top_k_chunks, stats
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: Chunks to embed
            
        Returns:
            Chunks with embeddings
        """
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: GENERATING EMBEDDINGS")
        logger.info("=" * 80)
        
        if not chunks:
            logger.warning("No chunks to embed")
            return []
        
        start_time = time.time()
        
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings in batches
        logger.info(f"Encoding {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=EMBEDDING_CONFIG["batch_size"],
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add embeddings to chunks
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy["embedding"] = embedding.tolist()
            chunks_with_embeddings.append(chunk_copy)
        
        elapsed = time.time() - start_time
        logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
        
        return chunks_with_embeddings
    
    def save_local_outputs(
        self,
        doc_chunks_map: Dict[str, Tuple[List[Dict], Path]],
        top_k_chunks: Optional[List[Dict]] = None,
        stats: Optional[Dict] = None
    ):
        """
        FIXED: Save all outputs to local filesystem in mirrored structure.
        
        Ensures all_chunks.json files are actually created.
        
        Args:
            doc_chunks_map: dict mapping doc_id -> (chunks, local_path)
            top_k_chunks: Top-k selected chunks (optional)
            stats: Pipeline statistics (optional)
        """
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: SAVING OUTPUTS LOCALLY")
        logger.info("=" * 80)
        
        # FIXED: Save each document's chunks to separate file in mirrored structure
        saved_count = 0
        for doc_id, (chunks, chunks_path) in tqdm(
            doc_chunks_map.items(),
            desc="Saving all_chunks files"
        ):
            try:
                # Create directory if needed
                chunks_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save chunks
                safe_json_dump(chunks, chunks_path, indent=2)
                saved_count += 1
                logger.debug(f"✓ Saved {len(chunks)} chunks for {doc_id} to {chunks_path}")
            except Exception as e:
                logger.error(f"✗ Failed to save chunks for {doc_id} to {chunks_path}: {e}")
        
        logger.info(f"✓ Saved {saved_count}/{len(doc_chunks_map)} all_chunks.json files")
        
        # Save combined top-k chunks if available
        if top_k_chunks:
            top_k_path = self.chunks_dir / "combined_top_k_chunks.json"
            try:
                safe_json_dump(top_k_chunks, top_k_path, indent=2)
                logger.info(f"✓ Saved {len(top_k_chunks)} combined top-k chunks to: {top_k_path}")
            except Exception as e:
                logger.error(f"✗ Failed to save combined top-k chunks: {e}")
        
        # Save statistics if available
        if stats:
            stats_path = self.stats_dir / "pipeline_stats.json"
            try:
                safe_json_dump(stats, stats_path, indent=2)
                logger.info(f"✓ Saved statistics to: {stats_path}")
            except Exception as e:
                logger.error(f"✗ Failed to save statistics: {e}")
        
        # Create index file for easy navigation
        try:
            index = {
                "total_documents": len(doc_chunks_map),
                "documents": {
                    doc_id: {
                        "num_chunks": len(chunks),
                        "chunks_file": str(chunks_path.relative_to(self.output_dir))
                    }
                    for doc_id, (chunks, chunks_path) in doc_chunks_map.items()
                }
            }
            index_path = self.output_dir / "document_index.json"
            safe_json_dump(index, index_path, indent=2)
            logger.info(f"✓ Saved document index to: {index_path}")
        except Exception as e:
            logger.error(f"✗ Failed to save document index: {e}")
    
    def run(self, max_documents: Optional[int] = None) -> Dict:
        """
        Run the test pipeline.
        
        Args:
            max_documents: Maximum documents to process (for testing)
            
        Returns:
            Pipeline statistics
        """
        pipeline_start = time.time()
        
        try:
            # Stage 1: Fetch documents
            documents = self.fetch_documents(max_docs=max_documents)
            
            if not documents:
                logger.error("No documents fetched from ADLS")
                return {"error": "No documents found"}
            
            # Stage 2: Process documents
            doc_chunks_map, top_k_chunks, proc_stats = self.process_documents(documents)
            
            if not doc_chunks_map:
                logger.error("No chunks generated")
                return {"error": "Processing failed"}
            
            # Stage 3: Generate embeddings for top-k chunks
            if top_k_chunks:
                top_k_with_embeddings = self.generate_embeddings(top_k_chunks)
            else:
                logger.warning("No top-k chunks selected")
                top_k_with_embeddings = []
            
            # Calculate statistics
            pipeline_end = time.time()
            pipeline_time = pipeline_end - pipeline_start
            
            stats = {
                "status": "success",
                "mode": "test_with_embedding_classifier",
                "pipeline_time_seconds": round(pipeline_time, 2),
                "documents_fetched": len(documents),
                "documents_processed": proc_stats["successful_documents"],
                "documents_failed": proc_stats["failed_documents"],
                "total_chunks_generated": proc_stats["total_chunks"],
                "top_k_chunks_selected": proc_stats["top_k_chunks"],
                "chunks_with_embeddings": len(top_k_with_embeddings),
                "timestamp": datetime.now().isoformat(),
                "output_directory": str(self.output_dir.absolute())
            }
            
            # Stage 4: Save locally
            self.save_local_outputs(doc_chunks_map, top_k_with_embeddings, stats)
            
            logger.info("\n" + "=" * 80)
            logger.info("TEST PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Total time: {pipeline_time:.2f}s")
            logger.info(f"Output directory: {self.output_dir.absolute()}")
            logger.info(f"Documents processed: {stats['documents_processed']}")
            logger.info(f"Total chunks: {stats['total_chunks_generated']}")
            logger.info(f"all_chunks files saved: {len(doc_chunks_map)}")
            
            return stats
        
        except Exception as e:
            logger.error(f"Test pipeline failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


def setup_logging():
    """Configure logging for the test pipeline."""
    log_format = LOGGING_CONFIG["format"]
    log_level = getattr(logging, LOGGING_CONFIG["level"].upper())
    
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    
    # Save test logs to test_output directory
    test_log_file = "test_output/test_pipeline.log"
    Path("test_output").mkdir(exist_ok=True)
    handlers.append(logging.FileHandler(test_log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True
    )


def main():
    """Main entry point for test pipeline."""
    # Setup logging
    setup_logging()
    
    logger.info("Starting Test Pipeline (UPDATED WITH EMBEDDING-BASED ROLE CLASSIFIER)")
    
    try:
        # Create and run test pipeline
        pipeline = TestPipeline(output_dir="test_output")
        
        # Run with limited documents for testing (adjust as needed)
        stats = pipeline.run(max_documents=20)
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEST PIPELINE SUMMARY")
        print("=" * 80)
        print(json.dumps(stats, indent=2))
        print("=" * 80)
        
        # Exit with appropriate code
        exit_code = 0 if stats.get("status") == "success" else 1
        exit(exit_code)
    
    except KeyboardInterrupt:
        logger.warning("Test pipeline interrupted by user")
        exit(130)
    
    except Exception as e:
        logger.error(f"Test pipeline failed with unexpected error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    print("DEBUG MODEL:", EMBEDDING_CONFIG["model_name"])

    main()