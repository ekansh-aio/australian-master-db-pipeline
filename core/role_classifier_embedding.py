"""
Embedding-Based Role Classifier for Legal Document Chunks

This module provides role classification using semantic similarity between
chunk embeddings and role description embeddings. Uses the same embedding
model as the main pipeline (sentence-transformers/all-MiniLM-L6-v2).

No fine-tuning required - works out of the box with good role descriptions.

ADVANTAGES:
- No training data or fine-tuning required
- Uses existing embedding model from pipeline
- Easy to add/modify role definitions
- Transparent and explainable (similarity scores)
- Consistent with existing semantic chunking approach

USAGE:
    from role_classifier_embedding import EmbeddingRoleClassifier
    
    # Define roles with descriptions
    role_descriptions = {
        "case_metadata": [
            "Court name, case number, filing date, parties involved",
            "Jurisdictional information and case identifiers"
        ],
        "procedural_history": [
            "Timeline of court proceedings and motions",
            "Previous hearings, appeals, and procedural steps"
        ]
    }
    
    # Create classifier
    classifier = EmbeddingRoleClassifier(
        role_descriptions=role_descriptions,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Classify chunks
    chunks = [{"text": "The case was filed on..."}, ...]
    classified_chunks = classifier.classify_chunks(chunks)
"""
import logging
import numpy as np
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


class EmbeddingRoleClassifier:
    """
    Role classifier using semantic similarity between chunk and role description embeddings.
    
    How it works:
    1. Each role has one or more description sentences
    2. Role descriptions are embedded using the same model as chunks
    3. For each chunk, compute cosine similarity with all role descriptions
    4. Assign the role with highest similarity (if above threshold)
    """
    
    def __init__(
        self,
        role_descriptions: Dict[str, List[str]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        confidence_threshold: float = 0.3,
        aggregation_method: str = "max",
        device: Optional[str] = None
    ):
        """
        Initialize the embedding-based role classifier.
        
        Args:
            role_descriptions: Dict mapping role names to list of description sentences.
                Example: {
                    "case_metadata": [
                        "Court name, case number, filing date",
                        "Jurisdictional information"
                    ],
                    "facts": [
                        "Factual background and circumstances of the case",
                        "Events that led to the legal dispute"
                    ]
                }
            model_name: Sentence transformer model name
            confidence_threshold: Minimum similarity score to assign a role (0.0-1.0)
                If max similarity < threshold, role will be "unclassified"
            aggregation_method: How to combine multiple descriptions per role:
                - "max": Use maximum similarity across descriptions
                - "mean": Use average similarity across descriptions
            device: Device to run on (None for auto-detect, 'cuda', 'cpu')
        """
        self.role_descriptions = role_descriptions
        self.role_names = list(role_descriptions.keys())
        self.confidence_threshold = confidence_threshold
        self.aggregation_method = aggregation_method
        
        logger.info(f"Initializing EmbeddingRoleClassifier with {len(self.role_names)} roles")
        logger.info(f"Roles: {self.role_names}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        
        # Load embedding model
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Loaded embedding model: {model_name}")
        
        # Pre-compute role description embeddings
        self._compute_role_embeddings()
    
    def _compute_role_embeddings(self):
        """Pre-compute embeddings for all role descriptions."""
        logger.info("Computing role description embeddings...")
        
        # Flatten all descriptions with their role names
        self.role_embedding_map = {}
        
        for role_name, descriptions in self.role_descriptions.items():
            # Encode all descriptions for this role
            embeddings = self.model.encode(
                descriptions,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            self.role_embedding_map[role_name] = embeddings
            logger.debug(f"  {role_name}: {len(descriptions)} descriptions encoded")
        
        logger.info("Role embeddings computed successfully")
    
    def predict(
        self,
        texts: Union[str, List[str]],
        return_all_scores: bool = False
    ) -> Union[Dict, List[Dict]]:
        """
        Predict roles for text chunks.
        
        Args:
            texts: Single text or list of texts
            return_all_scores: Whether to return similarity scores for all roles
        
        Returns:
            Single prediction dict or list of prediction dicts with:
            - role: predicted role name (or "unclassified")
            - confidence: confidence score (0-1)
            - all_scores: dict of role -> score (if return_all_scores=True)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Encode input texts
        text_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        predictions = []
        
        # Ensure we're working with numpy arrays
        if not isinstance(text_embeddings, np.ndarray):
            text_embeddings = np.array(text_embeddings)
        
        for text_embedding in text_embeddings:
            # Compute similarity with each role's descriptions
            role_scores = {}
            
            for role_name, role_embeddings in self.role_embedding_map.items():
                # Compute cosine similarity with all descriptions for this role
                # Ensure both inputs are numpy arrays
                text_emb_2d = np.array(text_embedding).reshape(1, -1)
                role_emb_2d = np.array(role_embeddings)
                
                similarities = cosine_similarity(text_emb_2d, role_emb_2d)[0]
                
                # Aggregate similarities
                if self.aggregation_method == "max":
                    role_scores[role_name] = float(np.max(similarities))
                elif self.aggregation_method == "mean":
                    role_scores[role_name] = float(np.mean(similarities))
                else:
                    raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
            
            # Find best role
            best_role = max(role_scores.items(), key=lambda x: x[1])
            role_name, confidence = best_role
            
            # Apply threshold
            if confidence < self.confidence_threshold:
                role_name = "unclassified"
            
            result = {
                'role': role_name,
                'confidence': confidence
            }
            
            if return_all_scores:
                result['all_scores'] = role_scores
            
            predictions.append(result)
        
        return predictions[0] if single_input else predictions
    
    def classify_chunks(
        self,
        chunks: List[Dict],
        text_field: str = 'text',
        batch_size: int = 32,
        add_to_chunks: bool = True,
        show_progress: bool = True,
        return_all_scores: bool = False
    ) -> List[Dict]:
        """
        Classify a list of chunk dictionaries.
        
        Args:
            chunks: List of chunk dicts
            text_field: Field name containing the text
            batch_size: Batch size for encoding (not used for similarity, kept for compatibility)
            add_to_chunks: Whether to add predictions to chunk dicts (modifies in-place)
            show_progress: Show progress bar
            return_all_scores: Include similarity scores for all roles
        
        Returns:
            List of chunks with added 'role_prediction' field
        """
        texts = [chunk.get(text_field, '') for chunk in chunks]
        
        logger.info(f"Classifying {len(chunks)} chunks...")
        
        # Predict all at once (batch encoding handled internally by sentence-transformers)
        if show_progress:
            iterator = tqdm(range(0, len(texts), batch_size), desc="Classifying chunks")
        else:
            iterator = range(0, len(texts), batch_size)
        
        all_predictions = []
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_predictions = self.predict(batch_texts, return_all_scores=return_all_scores)
            all_predictions.extend(batch_predictions)
        
        # Add predictions to chunks
        if add_to_chunks:
            for chunk, prediction in zip(chunks, all_predictions):
                chunk['role_prediction'] = prediction
        
        # Log statistics
        role_counts = {}
        for pred in all_predictions:
            role = pred['role']
            role_counts[role] = role_counts.get(role, 0) + 1
        
        logger.info("Classification complete!")
        logger.info("Role distribution:")
        for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
            percentage = (count / len(chunks)) * 100
            logger.info(f"  {role}: {count} ({percentage:.1f}%)")
        
        return chunks
    
    def update_role_descriptions(self, role_descriptions: Dict[str, List[str]]):
        """
        Update role descriptions and re-compute embeddings.
        
        Useful for experimenting with different descriptions without recreating the classifier.
        
        Args:
            role_descriptions: New role descriptions dict
        """
        logger.info("Updating role descriptions...")
        self.role_descriptions = role_descriptions
        self.role_names = list(role_descriptions.keys())
        self._compute_role_embeddings()
        logger.info("Role descriptions updated successfully")
    
    def get_role_info(self) -> Dict:
        """
        Get information about configured roles.
        
        Returns:
            Dict with role names, descriptions, and statistics
        """
        info = {
            'num_roles': len(self.role_names),
            'roles': {},
            'threshold': self.confidence_threshold,
            'aggregation': self.aggregation_method
        }
        
        for role_name, descriptions in self.role_descriptions.items():
            info['roles'][role_name] = {
                'num_descriptions': len(descriptions),
                'descriptions': descriptions
            }
        
        return info
    
    def save_config(self, filepath: str):
        """
        Save role descriptions and configuration to JSON file.
        
        Args:
            filepath: Path to save configuration
        """
        config = {
            'role_descriptions': self.role_descriptions,
            'confidence_threshold': self.confidence_threshold,
            'aggregation_method': self.aggregation_method,
            'model_name': self.model.get_sentence_embedding_dimension()  # Save for reference
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_config(
        cls,
        filepath: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Load role descriptions from JSON file.
        
        Args:
            filepath: Path to configuration file
            model_name: Embedding model to use
            device: Device to run on
        
        Returns:
            EmbeddingRoleClassifier instance
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        return cls(
            role_descriptions=config['role_descriptions'],
            model_name=model_name,
            confidence_threshold=config.get('confidence_threshold', 0.3),
            aggregation_method=config.get('aggregation_method', 'max'),
            device=device
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_default_legal_role_descriptions() -> Dict[str, List[str]]:
    """
    Create default role descriptions for legal documents.
    
    These are generic descriptions that work well for common legal document structures.
    Customize these based on your specific document types and needs.
    
    Returns:
        Dict mapping role names to description lists
    """
    return {
        "case_metadata": [
            "Court name, case number, filing date, and parties involved in the case",
            "Jurisdictional information and case identifiers",
            "Judge names, attorney information, and case citation details"
        ],
        "procedural_history": [
            "Timeline of court proceedings, hearings, and motions filed",
            "Previous rulings, appeals, and procedural steps in the case",
            "Dates of hearings, filings, and procedural events",
            "Lower court decisions and procedural background"
        ],
        "factual_background": [
            "Factual circumstances and events that led to the legal dispute",
            "Description of what happened, when, where, and involving whom",
            "Background facts and context of the case",
            "Narrative of events and relevant factual details"
        ],
        "legal_issues": [
            "Legal questions presented to the court for decision",
            "Issues and matters under consideration by the court",
            "Questions of law that need to be resolved"
        ],
        "legal_analysis": [
            "Legal reasoning, analysis of statutes, precedents, and case law",
            "Application of legal principles to the facts",
            "Discussion of relevant laws, regulations, and previous court decisions",
            "Interpretation of statutes and constitutional provisions",
            "Analysis of legal arguments presented by parties"
        ],
        "holdings_and_conclusions": [
            "Court's decision, ruling, or determination on the issues",
            "Final conclusions reached by the court",
            "Holdings on specific legal questions presented"
        ],
        "orders_and_disposition": [
            "Final orders, judgments, and relief granted",
            "Disposition of the case and remedies awarded",
            "Court's directions and mandates to parties",
            "Affirmation, reversal, or remand decisions"
        ],
        "dissenting_or_concurring": [
            "Dissenting opinions disagreeing with the majority",
            "Concurring opinions agreeing with result but different reasoning",
            "Separate opinions by individual judges"
        ],
        "other": [
            "Miscellaneous content not fitting other categories",
            "Administrative or procedural notes",
            "General information or boilerplate text"
        ]
    }


def create_classifier_from_config(config: Optional[Dict] = None) -> Optional[EmbeddingRoleClassifier]:
    """
    Create an embedding-based classifier using settings from config.py.
    
    Args:
        config: Configuration dictionary. If None, will import from config.py
    
    Returns:
        EmbeddingRoleClassifier instance configured from config, or None if disabled
    """
    if config is None:
        # Import config from config.py
        try:
            from config import ROLE_CLASSIFICATION_CONFIG, EMBEDDING_CONFIG
            role_config = ROLE_CLASSIFICATION_CONFIG
            embedding_config = EMBEDDING_CONFIG
        except ImportError:
            logger.warning("Could not import config.py, using defaults")
            return EmbeddingRoleClassifier(
                role_descriptions=create_default_legal_role_descriptions()
            )
    else:
        role_config = config.get('role_classification', {})
        embedding_config = config.get('embedding', {})
    
    # Check if role classification is enabled
    if not role_config.get('enabled', True):
        logger.warning("Role classification is disabled in config")
        return None
    
    # Get role descriptions from config
    role_descriptions_dict = role_config.get('role_descriptions_dict')
    
    # If not provided as dict, use defaults
    if not role_descriptions_dict:
        logger.info("No role_descriptions_dict in config, using defaults")
        role_descriptions_dict = create_default_legal_role_descriptions()
    
    # Get model name from embedding config (use same model as main pipeline)
    model_name = embedding_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
    
    # Get other settings
    confidence_threshold = role_config.get('confidence_threshold', 0.3)
    aggregation_method = role_config.get('aggregation_method', 'max')
    
    logger.info(f"Creating embedding-based classifier with model: {model_name}")
    logger.info(f"Confidence threshold: {confidence_threshold}")
    logger.info(f"Aggregation method: {aggregation_method}")
    
    return EmbeddingRoleClassifier(
        role_descriptions=role_descriptions_dict,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        aggregation_method=aggregation_method,
        device=role_config.get('device', None)
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create classifier with custom role descriptions
    role_descriptions = {
        "facts": [
            "Factual background and what happened",
            "Events and circumstances of the case"
        ],
        "law": [
            "Legal analysis and application of statutes",
            "Discussion of relevant case law and precedents"
        ],
        "decision": [
            "Court's ruling and final decision",
            "Orders and disposition of the case"
        ]
    }
    
    classifier = EmbeddingRoleClassifier(
        role_descriptions=role_descriptions,
        confidence_threshold=0.3
    )
    
    # Example chunks
    test_chunks = [
        {"text": "The plaintiff filed a motion on January 15, 2024, seeking summary judgment."},
        {"text": "According to Section 102(a) of the statute, employers must provide reasonable accommodations."},
        {"text": "The court finds in favor of the defendant and dismisses the case with prejudice."},
    ]
    
    # Classify
    classified = classifier.classify_chunks(test_chunks)
    
    # Print results
    for chunk in classified:
        print(f"\nText: {chunk['text'][:80]}...")
        print(f"Role: {chunk['role_prediction']['role']}")
        print(f"Confidence: {chunk['role_prediction']['confidence']:.3f}")