"""
Production Pipeline for Australian Legal Document Processing

Usage:
    python production_pipeline.py --index_type 0
        index_type: 0 = AI Assistant, 1 = Precedent Finder
"""
import argparse
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from config import (
    LOGGING_CONFIG,
    ADLS_CONFIG,
    SEARCH_CONFIG,
    EMBEDDING_CONFIG,
    CHUNKING_CONFIG,
    PROCESSING_CONFIG,
    ROLE_CLASSIFICATION_CONFIG,
    PIPELINE_CONFIG,
    INDEX_TYPE_CONFIG,
    validate_config
)

from core.adls_fetcher import ADLSFetcher
from core.adls_uploader import ADLSUploader
from core.legal_text_cleaner import LegalTextCleaner
from core.semantic_chunker import SemanticChunker
from core.search_uploader import SearchIndexManager, SearchUploader
from core.role_classifier import create_classifier_from_config
from utils.weighted_selector import weighted_topk_selection
from tqdm import tqdm

logger = logging.getLogger(__name__)


# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def generate_document_id_from_path(source_file_path: str) -> str:
    path_without_ext = source_file_path.replace('.json', '')
    parts = Path(path_without_ext).parts
    skip_prefixes = {'raw', 'newapp', 'input', 'data', 'app'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    doc_id = "_".join(relevant_parts).replace("-", "_").replace(" ", "_").lower()
    if len(doc_id) > 200:
        base = relevant_parts[-1] if relevant_parts else "doc"
        hash_suffix = hashlib.md5(source_file_path.encode()).hexdigest()[:8]
        doc_id = f"{base}_{hash_suffix}"
    return doc_id


def get_all_chunks_adls_path(source_file_path: str, base_output: str = "processed") -> str:
    path_without_ext = source_file_path.replace('.json', '')
    parts = Path(path_without_ext).parts
    skip_prefixes = {'raw', 'newapp', 'input', 'data', 'app'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    if relevant_parts:
        directory_parts = relevant_parts[:-1]
        filename = relevant_parts[-1]
        if directory_parts:
            output_dir = "/".join(directory_parts)
            return f"{base_output}/{output_dir}/{filename}_all_chunks.json"
        return f"{base_output}/{filename}_all_chunks.json"
    return f"{base_output}/unknown_all_chunks.json"


def get_done_marker_path(source_file_path: str, base_output: str = "processed", index_type: int = 0) -> str:
    path_without_ext = source_file_path.replace('.json', '')
    parts = Path(path_without_ext).parts
    skip_prefixes = {'raw', 'newapp', 'input', 'data', 'app'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    if relevant_parts:
        directory_parts = relevant_parts[:-1]
        filename = relevant_parts[-1]
        if directory_parts:
            output_dir = "/".join(directory_parts)
            return f"{base_output}/{output_dir}/{filename}_done_{index_type}.json"
        return f"{base_output}/{filename}_done_{index_type}.json"
    return f"{base_output}/unknown_done_{index_type}.json"


def attach_same_role_chunk_ids(chunks: List[Dict]) -> None:
    role_to_ids: Dict = defaultdict(list)
    for chunk in chunks:
        role_to_ids[chunk.get("role", "Others")].append(chunk["id"])
    for chunk in chunks:
        role = chunk.get("role", "Others")
        chunk["same_role_chunk_ids"] = [
            cid for cid in role_to_ids[role] if cid != chunk["id"]
        ]


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

class ProductionPipeline:

    def __init__(self, index_type: int, base_output_path: str = "processed"):
        logger.info("=" * 80)
        logger.info(
            f"PRODUCTION PIPELINE — "
            f"Australian Legislation / "
            f"{INDEX_TYPE_CONFIG[index_type]['name']}"
        )
        logger.info("=" * 80)

        self.index_type = index_type
        self.base_output_path = base_output_path

        self.index_name = INDEX_TYPE_CONFIG[index_type]["index_name"]
        self.role_weights = INDEX_TYPE_CONFIG[index_type]["role_weights"]

        validate_config()

        self._init_adls()
        self._init_processors()
        self._init_search()

    def _init_adls(self):
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
        self.text_cleaner = LegalTextCleaner()
        self.semantic_chunker = SemanticChunker(
            model_name=EMBEDDING_CONFIG["model_name"],
            similarity_threshold=CHUNKING_CONFIG["similarity_threshold"],
            min_sentences_per_chunk=CHUNKING_CONFIG["min_sentences_per_chunk"],
            max_sentences_per_chunk=CHUNKING_CONFIG["max_sentences_per_chunk"],
            min_chunk_size=CHUNKING_CONFIG["min_chunk_size"]
        )
        self.embedding_model = self.semantic_chunker.model

        self.role_classifier = None
        if ROLE_CLASSIFICATION_CONFIG["enabled"]:
            logger.info("Initializing role classifier")
            self.role_classifier = create_classifier_from_config()

    def _init_search(self):
        self.index_manager = SearchIndexManager(
            endpoint=SEARCH_CONFIG["endpoint"],
            key=SEARCH_CONFIG["key"]
        )
        self.search_uploader = SearchUploader(
            endpoint=SEARCH_CONFIG["endpoint"],
            key=SEARCH_CONFIG["key"],
            index_name=self.index_name,
            batch_size=SEARCH_CONFIG["upload_batch_size"],
            max_retries=SEARCH_CONFIG["max_retries"],
            retry_delay=SEARCH_CONFIG["retry_delay"]
        )

    def fetch_documents(self) -> List[Dict]:
        return self.adls_fetcher.fetch_all(
            path=ADLS_CONFIG["input_path"],
            pattern=ADLS_CONFIG["file_pattern"],
            recursive=ADLS_CONFIG["recursive"],
            max_files=PIPELINE_CONFIG["max_documents"],
            show_progress=True
        )

    def process_single_document(self, doc: Dict) -> Optional[Tuple]:
        source_file = doc.get("_source_file", "unknown.json")
        doc_id = generate_document_id_from_path(source_file)
        all_chunks_path = get_all_chunks_adls_path(source_file, self.base_output_path)

        try:
            text = doc.get("text") or doc.get("full_text") or doc.get("judgment_text", "")
            if not text:
                return None

            cleaned_text = self.text_cleaner.clean(text)
            if not cleaned_text:
                return None

            chunks, _ = self.semantic_chunker.split(
                cleaned_text,
                compute_doc_similarity=CHUNKING_CONFIG["compute_doc_similarity"]
            )
            if not chunks:
                return None

            excluded_fields = {"text", "full_text", "judgment_text", "embedding", "_source_file"}
            metadata = {k: v for k, v in doc.items() if k not in excluded_fields}

            all_chunks = []
            for idx, chunk in enumerate(chunks):
                chunk_all = {
                    "id":                   f"{doc_id}_{idx}",
                    "chunk_id":             f"{doc_id}_{idx}",
                    "doc_id":               doc_id,
                    "original_source_path": source_file,
                    "text":                 chunk["text"],
                    "start_char":           chunk.get("start_char", 0),
                    "end_char":             chunk.get("end_char", 0),
                    "num_sentences":        chunk.get("num_sentences", 0),
                    "doc_similarity":       float(chunk.get("doc_similarity", 0.0)),
                    "avg_similarity":       float(chunk.get("avg_similarity", 0.0)),
                    **metadata
                }
                all_chunks.append(chunk_all)

            if self.role_classifier:
                self.role_classifier.classify_chunks(
                    all_chunks,
                    text_field="text",
                    batch_size=ROLE_CLASSIFICATION_CONFIG["batch_size"],
                    add_to_chunks=True,
                    show_progress=False
                )
                for chunk in all_chunks:
                    if "role_prediction" in chunk:
                        chunk["role"] = chunk["role_prediction"]["role"]
                        chunk["confidence"] = float(chunk["role_prediction"]["confidence"])
                        del chunk["role_prediction"]

            attach_same_role_chunk_ids(all_chunks)

            if CHUNKING_CONFIG["top_k"]:
                selected_indices = weighted_topk_selection(
                    chunks=all_chunks,
                    top_k=CHUNKING_CONFIG["top_k"],
                    similarity_key="doc_similarity",
                    role_weights=self.role_weights
                )
            else:
                selected_indices = list(range(len(all_chunks)))

            top_k_chunks = []
            for idx in selected_indices:
                chunk_topk = all_chunks[idx].copy()
                chunk_topk["all_chunks_path"] = all_chunks_path
                top_k_chunks.append(chunk_topk)

            return (doc_id, all_chunks, top_k_chunks, all_chunks_path)

        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
            return None

    def _partition_documents(self, documents: List[Dict]) -> Tuple[List[Dict], List[Tuple], int]:
        """Split documents into pending, resume candidates, and already-done (skipped)."""
        pending = []
        resume_candidates = []
        skipped = 0

        for doc in tqdm(documents, desc="Checking document status"):
            source_file = doc.get("_source_file", "unknown.json")
            all_chunks_path = get_all_chunks_adls_path(source_file, self.base_output_path)
            done_marker_path = get_done_marker_path(source_file, self.base_output_path, self.index_type)

            if self.adls_uploader.file_exists(done_marker_path):
                skipped += 1
            elif self.adls_uploader.file_exists(all_chunks_path):
                resume_candidates.append((source_file, all_chunks_path, done_marker_path))
            else:
                pending.append(doc)

        return pending, resume_candidates, skipped

    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=EMBEDDING_CONFIG["batch_size"],
            show_progress_bar=True,
            convert_to_numpy=True
        )
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            c = chunk.copy()
            c["embedding"] = embedding.tolist()
            chunks_with_embeddings.append(c)
        return chunks_with_embeddings

    def upload_chunks_to_adls(self, doc_chunks_map: Dict) -> Dict:
        results = {}
        for doc_id, doc_data in doc_chunks_map.items():
            results[doc_id] = self.adls_uploader.upload_json_file(
                data=doc_data["chunks"],
                adls_path=doc_data["path"],
                overwrite=True
            )
        return results

    def upload_topk_to_search(self, top_k_chunks: List[Dict]) -> Dict:
        return self.search_uploader.upload_chunks(top_k_chunks, show_progress=True)

    def _resume_pending_search(self, resume_candidates: List[Tuple]) -> int:
        """Re-upload to Search for docs where ADLS completed but Search didn't."""
        if not resume_candidates:
            return 0
        logger.info(f"Resuming search upload for {len(resume_candidates)} pending documents")
        completed = 0
        for source_file, all_chunks_path, done_marker_path in resume_candidates:
            try:
                all_chunks_data = self.adls_fetcher.read_json_file(all_chunks_path)
                all_chunks = all_chunks_data if isinstance(all_chunks_data, list) else []
                if CHUNKING_CONFIG["top_k"]:
                    selected = weighted_topk_selection(
                        all_chunks, CHUNKING_CONFIG["top_k"], "doc_similarity", self.role_weights
                    )
                else:
                    selected = list(range(len(all_chunks)))
                top_k = [all_chunks[i] for i in selected]
                top_k_with_emb = self.generate_embeddings(top_k)
                upload_stats = self.search_uploader.upload_chunks(top_k_with_emb, show_progress=False)
                if upload_stats.get("failed", 0) == 0:
                    self.adls_uploader.write_marker(done_marker_path)
                    completed += 1
                else:
                    logger.warning(f"Upload had failures for {source_file}, skipping done marker")
            except Exception as e:
                logger.error(f"Failed to resume {source_file}: {e}", exc_info=True)
        logger.info(f"Resumed {completed}/{len(resume_candidates)} documents")
        return completed

    def _write_done_markers(self, doc_chunks_map: Dict, adls_results: Dict, search_had_failures: bool) -> int:
        """Write _done markers only when both ADLS and Search succeeded."""
        if search_had_failures:
            logger.warning("Search upload had failures — skipping _done markers (will resume on next run)")
            return 0
        written = 0
        for doc_id, doc_data in doc_chunks_map.items():
            if adls_results.get(doc_id):
                done_path = doc_data["path"].replace("_all_chunks.json", f"_done_{self.index_type}.json")
                if self.adls_uploader.write_marker(done_path):
                    written += 1
        logger.info(f"Wrote {written} _done markers")
        return written

    def run(self) -> Dict:
        start = time.time()
        batch_size = PIPELINE_CONFIG["processing_batch_size"]

        documents = self.fetch_documents()
        pending_docs, resume_candidates, skipped = self._partition_documents(documents)

        logger.info(
            f"Documents — pending: {len(pending_docs)}, "
            f"resume: {len(resume_candidates)}, skipped: {skipped}"
        )

        if PIPELINE_CONFIG["upload_to_search"]:
            self._resume_pending_search(resume_candidates)

        if PIPELINE_CONFIG["create_index"]:
            self.index_manager.create_legal_documents_index(
                index_name=self.index_name,
                vector_dimensions=EMBEDDING_CONFIG["dimensions"]
            )

        total_processed = 0
        total_chunks = 0
        total_top_k = 0
        all_search_stats: Dict = {"uploaded": 0, "failed": 0}

        num_batches = max(1, (len(pending_docs) + batch_size - 1) // batch_size)

        for batch_num, batch_start in enumerate(range(0, len(pending_docs), batch_size), 1):
            batch = pending_docs[batch_start:batch_start + batch_size]
            logger.info(f"--- Batch {batch_num}/{num_batches} ({len(batch)} documents) ---")

            doc_chunks_map: Dict = {}
            top_k_chunks: List[Dict] = []

            for doc in tqdm(batch, desc=f"Processing batch {batch_num}"):
                result = self.process_single_document(doc)
                if result is None:
                    continue
                doc_id, all_chunks, top_k_chunks_doc, chunks_path = result
                doc_chunks_map[doc_id] = {"chunks": all_chunks, "path": chunks_path}
                top_k_chunks.extend(top_k_chunks_doc)

            if not doc_chunks_map:
                logger.warning(f"Batch {batch_num}: no documents processed successfully, skipping upload")
                continue

            top_k_with_embeddings = self.generate_embeddings(top_k_chunks)
            adls_results = self.upload_chunks_to_adls(doc_chunks_map)

            if PIPELINE_CONFIG["upload_to_search"] and top_k_with_embeddings:
                batch_search_stats = self.upload_topk_to_search(top_k_with_embeddings)
                search_had_failures = batch_search_stats.get("failed", 0) > 0
                self._write_done_markers(doc_chunks_map, adls_results, search_had_failures)
                all_search_stats["uploaded"] += batch_search_stats.get("uploaded", 0)
                all_search_stats["failed"] += batch_search_stats.get("failed", 0)

            total_processed += len(doc_chunks_map)
            total_chunks += sum(len(v["chunks"]) for v in doc_chunks_map.values())
            total_top_k += len(top_k_chunks)

            logger.info(
                f"Batch {batch_num} done — "
                f"processed: {len(doc_chunks_map)}, chunks: {total_chunks}, top-k: {total_top_k}"
            )

        stats = {
            "status":               "success",
            "index_type":           INDEX_TYPE_CONFIG[self.index_type]["name"],
            "index_name":           self.index_name,
            "pipeline_time_seconds": round(time.time() - start, 2),
            "documents_processed":  total_processed,
            "documents_skipped":    skipped,
            "documents_resumed":    len(resume_candidates),
            "total_chunks":         total_chunks,
            "top_k_chunks":         total_top_k,
            "search_upload":        all_search_stats
        }

        logger.info("PIPELINE COMPLETE")
        return stats


# --------------------------------------------------

def setup_logging():
    log_format = LOGGING_CONFIG["format"]
    log_level = getattr(logging, LOGGING_CONFIG["level"].upper())
    logging.basicConfig(level=log_level, format=log_format,
                        handlers=[logging.StreamHandler()], force=True)


def main():
    parser = argparse.ArgumentParser(description="Production pipeline for Australian legal document indexing")
    parser.add_argument("--index_type", type=int, choices=[0, 1], required=True,
                        help="Index type: 0 = AI Assistant, 1 = Precedent Finder")
    args = parser.parse_args()

    setup_logging()

    pipeline = ProductionPipeline(
        index_type=args.index_type,
        base_output_path="processed"
    )
    stats = pipeline.run()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
