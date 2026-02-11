"""
Production Pipeline for Legal Document Processing (FIXED VERSION)
- Generates unique document IDs from source paths
- Removes unnecessary metadata from chunks
- Adds traceability (original_source_path, all_chunks_path)
- Saves ALL chunks to ADLS in mirrored directory structure
- Uploads TOP-K chunks to Azure AI Search
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration
from config import (
    LOGGING_CONFIG,
    ADLS_CONFIG,
    SEARCH_CONFIG,
    EMBEDDING_CONFIG,
    CHUNKING_CONFIG,
    PROCESSING_CONFIG,
    OUTPUT_CONFIG,
    PIPELINE_CONFIG,
    validate_config
)

# Import pipeline components
from adls_fetcher import ADLSFetcher
from adls_uploader import ADLSUploader
from legal_text_cleaner import LegalTextCleaner
from semantic_chunker import SemanticChunker
from sentence_transformers import SentenceTransformer
from search_uploader import SearchIndexManager, SearchUploader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_document_id_from_path(source_file_path: str) -> str:
    """
    Generate unique document ID from source file path.
    
    Example:
        'raw/newapp/decisions/queensland/2005/doc.json' -> 'decisions_queensland_2005_doc'
    """
    import hashlib
    
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


def get_all_chunks_adls_path(source_file_path: str, base_output: str = "processed") -> str:
    """
    Generate ADLS path for all_chunks.json that mirrors input structure.
    
    Example:
        Input:  'raw/newapp/decisions/queensland/2005/document.json'
        Output: 'processed/decisions/queensland/2005/document_all_chunks.json'
    """
    path_without_ext = source_file_path.replace('.json', '')
    parts = Path(path_without_ext).parts
    
    skip_prefixes = {'raw', 'newapp', 'input', 'data'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    
    if relevant_parts:
        directory_parts = relevant_parts[:-1]
        filename = relevant_parts[-1]
        
        if directory_parts:
            output_dir = '/'.join(directory_parts)
            return f"{base_output}/{output_dir}/{filename}_all_chunks.json"
        else:
            return f"{base_output}/{filename}_all_chunks.json"
    else:
        return f"{base_output}/unknown_all_chunks.json"


class ProductionPipeline:
    """
    Production pipeline with:
    - Unique document IDs from source paths
    - Cleaned metadata (no similarity scores, char positions, etc.)
    - Traceability (original_source_path, all_chunks_path)
    - Mirrored ADLS output structure
    """
    
    def __init__(self, base_output_path: str = "processed"):
        """Initialize production pipeline."""
        logger.info("=" * 80)
        logger.info("PRODUCTION PIPELINE (FIXED VERSION)")
        logger.info("=" * 80)
        
        # Validate configuration
        validate_config()
        
        self.base_output_path = base_output_path
        
        # Initialize components
        self._init_adls()
        self._init_processors()
        self._init_search()
        
        logger.info("Production pipeline initialized successfully")
    
    def _init_adls(self):
        """Initialize ADLS fetcher and uploader."""
        logger.info("Initializing ADLS connections...")
        
        self.adls_fetcher = ADLSFetcher(
            account_name=ADLS_CONFIG["account_name"],
            account_key=ADLS_CONFIG["account_key"],
            container_name=ADLS_CONFIG["container_name"]
        )
        
        self.adls_uploader = ADLSUploader(
            account_name=ADLS_CONFIG["account_name"],
            account_key=ADLS_CONFIG["account_key"],
            container_name=ADLS_CONFIG["container_name"]
        )
    
    def _init_processors(self):
        """Initialize text processors."""
        logger.info("Initializing text processors...")
        
        self.text_cleaner = LegalTextCleaner()
        
        self.semantic_chunker = SemanticChunker(
            model_name=EMBEDDING_CONFIG["model_name"],
            similarity_threshold=CHUNKING_CONFIG["similarity_threshold"],
            min_sentences_per_chunk=CHUNKING_CONFIG["min_sentences_per_chunk"],
            max_sentences_per_chunk=CHUNKING_CONFIG["max_sentences_per_chunk"],
            min_chunk_size=CHUNKING_CONFIG["min_chunk_size"]
        )
        
        self.embedding_model = self.semantic_chunker.model
        
        logger.info(f"Using embedding model: {EMBEDDING_CONFIG['model_name']}")
    
    def _init_search(self):
        """Initialize Azure Search components."""
        logger.info("Initializing Azure Search components...")
        
        self.index_manager = SearchIndexManager(
            endpoint=SEARCH_CONFIG["endpoint"],
            key=SEARCH_CONFIG["key"]
        )
        
        self.search_uploader = SearchUploader(
            endpoint=SEARCH_CONFIG["endpoint"],
            key=SEARCH_CONFIG["key"],
            index_name=SEARCH_CONFIG["index_name"],
            batch_size=SEARCH_CONFIG["upload_batch_size"],
            max_retries=SEARCH_CONFIG["max_retries"],
            retry_delay=SEARCH_CONFIG["retry_delay"]
        )
    
    def fetch_documents(self) -> List[Dict]:
        """Fetch documents from ADLS."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: FETCHING DOCUMENTS FROM ADLS")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        documents = self.adls_fetcher.fetch_all(
            path=ADLS_CONFIG["input_path"],
            pattern=ADLS_CONFIG["file_pattern"],
            recursive=ADLS_CONFIG["recursive"],
            max_files=PIPELINE_CONFIG["max_documents"],
            show_progress=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Fetched {len(documents)} documents in {elapsed:.2f}s")
        
        return documents
    
    def process_single_document(
        self,
        doc: Dict
    ) -> Optional[Tuple[str, List[Dict], List[Dict], str]]:
        """
        Process a single document.
        
        Returns:
            Tuple of (doc_id, all_chunks, top_k_chunks, all_chunks_path) or None
        """
        source_file = doc.get("_source_file", "unknown.json")
        doc_id = generate_document_id_from_path(source_file)
        all_chunks_path = get_all_chunks_adls_path(source_file, self.base_output_path)
        
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
            
            # Create chunks for ALL (stored in ADLS)
            all_chunks = []
            for idx, chunk in enumerate(chunks):
                chunk_all = {
                    "id": f"{doc_id}_{idx}",
                    "chunk_id": f"{doc_id}_{idx}",
                    "doc_id": doc_id,
                    "original_source_path": source_file,
                    **metadata,
                    "text": chunk["text"]
                    # Removed: start_char, end_char, num_sentences, avg_similarity
                }
                all_chunks.append(chunk_all)
            
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
            
            # Create chunks for TOP-K (sent to search)
            top_k_chunks = []
            for idx in selected_indices:
                chunk_topk = {
                    "id": f"{doc_id}_{idx}",
                    "chunk_id": f"{doc_id}_{idx}",
                    "doc_id": doc_id,
                    "original_source_path": source_file,
                    "all_chunks_path": all_chunks_path,
                    **metadata,
                    "text": chunks[idx]["text"]
                    # Removed: similarity scores, char positions
                }
                top_k_chunks.append(chunk_topk)
            
            return (doc_id, all_chunks, top_k_chunks, all_chunks_path)
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
            return None
    
    def process_documents(
        self,
        documents: List[Dict]
    ) -> Tuple[Dict[str, Dict], List[Dict], Dict]:
        """
        Process all documents.
        
        Returns:
            Tuple of (doc_chunks_map, all_top_k_chunks, stats)
            - doc_chunks_map: dict mapping doc_id -> {chunks, path}
            - all_top_k_chunks: list of all top-k chunks for search
        """
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: PROCESSING DOCUMENTS")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        doc_chunks_map = {}  # doc_id -> {"chunks": [...], "path": "..."}
        all_top_k_chunks = []
        
        failed_count = 0
        batch_size = PROCESSING_CONFIG["batch_size"]
        skip_errors = PROCESSING_CONFIG["skip_errors"]
        
        for doc in tqdm(documents, desc="Processing documents"):
            result = self.process_single_document(doc)
            
            if result is None:
                failed_count += 1
                if not skip_errors:
                    raise ValueError("Document processing failed")
                continue
            
            doc_id, all_chunks, top_k_chunks, chunks_path = result
            
            # Store all chunks with their ADLS path
            doc_chunks_map[doc_id] = {
                "chunks": all_chunks,
                "path": chunks_path
            }
            
            # Collect all top-k chunks
            all_top_k_chunks.extend(top_k_chunks)
        
        elapsed = time.time() - start_time
        
        stats = {
            "total_documents": len(documents),
            "successful_documents": len(doc_chunks_map),
            "failed_documents": failed_count,
            "total_chunks": sum(len(v["chunks"]) for v in doc_chunks_map.values()),
            "top_k_chunks": len(all_top_k_chunks),
            "processing_time": elapsed
        }
        
        logger.info(f"Processed {stats['successful_documents']}/{stats['total_documents']} documents")
        logger.info(f"Total chunks: {stats['total_chunks']}, Top-K: {stats['top_k_chunks']}")
        
        return doc_chunks_map, all_top_k_chunks, stats
    
    def generate_embeddings(
        self,
        chunks: List[Dict]
    ) -> List[Dict]:
        """Generate embeddings for chunks."""
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
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
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
    
    def upload_chunks_to_adls(
        self,
        doc_chunks_map: Dict[str, Dict]
    ) -> Dict[str, bool]:
        """
        Upload all chunks to ADLS in mirrored directory structure.
        Each document's chunks go to a separate file.
        
        Args:
            doc_chunks_map: dict mapping doc_id -> {"chunks": [...], "path": "..."}
            
        Returns:
            Upload results dict
        """
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: UPLOADING CHUNKS TO ADLS")
        logger.info("=" * 80)
        
        upload_results = {}
        
        for doc_id, doc_data in tqdm(
            doc_chunks_map.items(),
            desc="Uploading to ADLS"
        ):
            chunks = doc_data["chunks"]
            adls_path = doc_data["path"]
            
            # Upload this document's chunks to its specific path
            success = self.adls_uploader.upload_json_file(
                data=chunks,
                adls_path=adls_path,
                overwrite=True
            )
            
            upload_results[doc_id] = success
            
            if not success:
                logger.warning(f"Failed to upload chunks for {doc_id} to {adls_path}")
        
        successful_uploads = sum(1 for v in upload_results.values() if v)
        logger.info(
            f"Upload complete: {successful_uploads}/{len(upload_results)} "
            f"documents uploaded successfully"
        )
        
        return upload_results
    
    def create_search_index(self):
        """Create or update Azure Search index."""
        if not PIPELINE_CONFIG["create_index"]:
            logger.info("Index creation disabled")
            return
        
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 5: CREATING/UPDATING SEARCH INDEX")
        logger.info("=" * 80)
        
        success = self.index_manager.create_legal_documents_index(
            index_name=SEARCH_CONFIG["index_name"],
            vector_dimensions=EMBEDDING_CONFIG["dimensions"],
            force_recreate=False
        )
        
        if success:
            logger.info(f"✓ Index '{SEARCH_CONFIG['index_name']}' is ready")
        else:
            logger.error("✗ Failed to create/update index")
            raise RuntimeError("Index creation failed")
    
    def upload_topk_to_search(self, top_k_chunks: List[Dict]) -> Dict:
        """Upload TOP-K chunks to Azure AI Search."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 6: UPLOADING TOP-K CHUNKS TO AZURE AI SEARCH")
        logger.info("=" * 80)
        
        if not top_k_chunks:
            logger.warning("No top-k chunks to upload")
            return {"skipped": True, "reason": "no_topk_chunks"}
        
        stats = self.search_uploader.upload_chunks(
            chunks=top_k_chunks,
            show_progress=True
        )
        
        logger.info(f"Upload statistics: {stats}")
        return stats
    
    def save_stats_to_adls(self, stats: Dict, timestamp: str) -> bool:
        """Save pipeline statistics to ADLS."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        stats_path = f"output/stats/{date_str}/{timestamp}/pipeline_stats.json"
        
        logger.info(f"Saving statistics to ADLS: {stats_path}")
        
        return self.adls_uploader.upload_json_file(
            data=stats,
            adls_path=stats_path,
            overwrite=True
        )
    
    def run(self) -> Dict:
        """Run the complete production pipeline."""
        pipeline_start = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Stage 1: Fetch documents
            documents = self.fetch_documents()
            
            if not documents:
                logger.error("No documents fetched")
                return {"error": "No documents found"}
            
            # Stage 2: Process documents
            doc_chunks_map, top_k_chunks, proc_stats = self.process_documents(documents)
            
            if not doc_chunks_map:
                logger.error("No chunks generated")
                return {"error": "Processing failed"}
            
            # Stage 3: Generate embeddings ONLY for top-k chunks
            if top_k_chunks:
                top_k_with_embeddings = self.generate_embeddings(top_k_chunks)
            else:
                logger.warning("No top-k chunks selected")
                top_k_with_embeddings = []
            
            # Stage 4: Upload ALL chunks to ADLS (mirrored structure)
            upload_results = self.upload_chunks_to_adls(doc_chunks_map)
            
            # Stage 5: Create/update search index
            if PIPELINE_CONFIG["create_index"]:
                self.create_search_index()
            
            # Stage 6: Upload TOP-K chunks to Azure AI Search
            if PIPELINE_CONFIG["upload_to_search"] and top_k_with_embeddings:
                search_stats = self.upload_topk_to_search(top_k_with_embeddings)
            else:
                search_stats = {"skipped": True}
            
            # Calculate final statistics
            pipeline_end = time.time()
            pipeline_time = pipeline_end - pipeline_start
            
            stats = {
                "status": "success",
                "mode": "production_fixed",
                "pipeline_time_seconds": round(pipeline_time, 2),
                "documents_fetched": len(documents),
                "documents_processed": proc_stats["successful_documents"],
                "documents_failed": proc_stats["failed_documents"],
                "total_chunks_generated": proc_stats["total_chunks"],
                "top_k_chunks_selected": proc_stats["top_k_chunks"],
                "adls_uploads_successful": sum(1 for v in upload_results.values() if v),
                "adls_uploads_failed": sum(1 for v in upload_results.values() if not v),
                "chunks_uploaded_to_search": search_stats.get("uploaded", 0),
                "search_upload_stats": search_stats,
                "timestamp": timestamp,
                "base_output_path": self.base_output_path,
                "search_index_name": SEARCH_CONFIG["index_name"]
            }
            
            # Save statistics
            self.save_stats_to_adls(stats, timestamp)
            
            logger.info("\n" + "=" * 80)
            logger.info("PRODUCTION PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Total time: {pipeline_time:.2f}s")
            logger.info(f"✓ {stats['adls_uploads_successful']} documents saved to ADLS (mirrored structure)")
            logger.info(f"✓ {stats['chunks_uploaded_to_search']} chunks indexed in Azure AI Search")
            
            return stats
        
        except Exception as e:
            logger.error(f"Production pipeline failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": timestamp
            }


def setup_logging():
    """Configure logging for the production pipeline."""
    log_format = LOGGING_CONFIG["format"]
    log_level = getattr(logging, LOGGING_CONFIG["level"].upper())
    
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    
    if LOGGING_CONFIG["log_file"]:
        handlers.append(logging.FileHandler(LOGGING_CONFIG["log_file"]))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True
    )


def main():
    """Main entry point for the production pipeline."""
    setup_logging()
    
    logger.info("Starting Production Pipeline (FIXED VERSION)")
    
    try:
        pipeline = ProductionPipeline(base_output_path="processed")
        stats = pipeline.run()
        
        # Print summary
        print("\n" + "=" * 80)
        print("PRODUCTION PIPELINE SUMMARY")
        print("=" * 80)
        print(json.dumps(stats, indent=2))
        print("=" * 80)
        
        exit_code = 0 if stats.get("status") == "success" else 1
        exit(exit_code)
    
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        exit(130)
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()