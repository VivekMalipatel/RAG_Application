import logging
import asyncio
from typing import Dict, Any, List, Optional, Type, Union, Tuple
from pydantic import BaseModel, Field
from app.core.agent.base_agent import BaseAgent
from app.core.agent.lang_graph_executer import OmniGraph
from app.services.agents.graph_search_agent import GraphSearchAgent
from app.services.agents.hybrid_search_agent import HybridSearchAgent
from app.core.embedding.embedding_handler import EmbeddingHandler
from app.config import settings

class QueryAnalysisSchema(BaseModel):
    """
    Schema for query analysis output.
    """
    is_entity_focused: bool = Field(description="Whether the query focuses on specific entities")
    is_relationship_focused: bool = Field(description="Whether the query focuses on relationships between entities")
    is_semantic_conceptual: bool = Field(description="Whether the query is about broader concepts/ideas")
    is_factual_specific: bool = Field(description="Whether the query asks for specific factual information")
    rewritten_query: str = Field(description="Optimized version of the query for search")
    search_strategy: str = Field(description="Recommended search strategy: 'graph', 'vector', or 'both'")
    sub_queries: List[str] = Field(default=[], description="Additional sub-queries to explore for complex questions")

class ResultVerificationSchema(BaseModel):
    """
    Schema for search result verification output.
    """
    is_adequate: bool = Field(description="Whether current results adequately answer the query")
    information_gaps: List[str] = Field(default=[], description="Areas where information is missing or incomplete")
    reformulated_query: Optional[str] = Field(default=None, description="Suggested query reformulation if results are inadequate")
    alternative_strategy: Optional[str] = Field(default=None, description="Alternative search strategy to try ('graph', 'vector', or 'both')")
    follow_up_queries: List[str] = Field(default=[], description="Specific follow-up queries to fill information gaps")

class SearchResultSchema(BaseModel):
    """
    Schema for structured search results.
    """
    relevance_score: float = Field(description="Overall relevance score (0-1)")
    source_type: str = Field(description="Source of this result: 'graph', 'vector', or 'combined'")
    content: str = Field(description="Main content of the result")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata about the result")

class CompiledResultsSchema(BaseModel):
    """
    Schema for final compiled search results.
    """
    query: str = Field(description="Original user query")
    rewritten_query: str = Field(description="Rewritten search query used")
    summary: str = Field(description="Executive summary of findings")
    main_findings: List[str] = Field(description="Key information points discovered")
    detailed_results: List[SearchResultSchema] = Field(description="Detailed search results")
    source_distribution: Dict[str, int] = Field(description="Distribution of results by source type")

class SearchOrchestrationState(BaseModel):
    """
    State schema for the search orchestration workflow.
    """
    user_id: int
    query_text: str
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None
    top_k: int = 10
    # Dynamic state fields added during workflow execution
    query_analysis: Optional[Dict[str, Any]] = None
    current_query: Optional[str] = None
    search_strategy: Optional[str] = None
    sub_queries: List[str] = []
    iteration: int = 1
    all_search_results: List[Dict[str, Any]] = []
    current_results: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None
    is_adequate: bool = False
    queries_to_run: List[str] = []
    final_results: Optional[Dict[str, Any]] = None

class SearchEnhancementParams(BaseModel):
    """
    Parameters for enhancing search based on initial results.
    """
    entity_constraints: List[str] = Field(default=[], description="Entities to constrain the search")
    keyword_constraints: List[str] = Field(default=[], description="Keywords to constrain the search")
    expanded_depth: Optional[int] = Field(default=None, description="Expanded search depth for graph search")
    expanded_limit: Optional[int] = Field(default=None, description="Expanded result limit")
    focus_entities: List[str] = Field(default=[], description="Entities to focus on")
    focus_relationships: List[str] = Field(default=[], description="Relationships to focus on")
    source_filter: Optional[str] = Field(default=None, description="Filter to specific source types")

class SearchStrengthAssessment(BaseModel):
    """
    Assessment of search result strength.
    """
    entity_coverage: float = Field(description="Coverage of relevant entities (0-1)")
    relationship_coverage: float = Field(description="Coverage of relevant relationships (0-1)")
    semantic_relevance: float = Field(description="Semantic relevance of results (0-1)")
    factual_specificity: float = Field(description="Specificity of factual information (0-1)")
    overall_strength: float = Field(description="Overall strength of results (0-1)")
    strength_areas: List[str] = Field(default=[], description="Areas where the search performed well")
    weakness_areas: List[str] = Field(default=[], description="Areas where the search performed poorly")

class SearchOrchestrationAgent(BaseAgent):
    """
    Advanced orchestration agent that manages multiple search agents,
    rewrites queries, verifies results, and optimizes search strategies.
    """
    
    def __init__(self):
        system_prompt = """
        You are an advanced search orchestration agent responsible for managing multiple specialized search systems.
        Your goal is to provide the most comprehensive and accurate search results by intelligently:
        
        1. Analyzing and reshaping user queries to make them more searchable
        2. Coordinating specialized search agents (Graph Search and Vector Search)
        3. Verifying result quality and initiating follow-up searches when needed
        4. Intelligently merging results from multiple sources
        
        For each query, you'll determine:
        - If the query is better suited for knowledge graph search (entities and relationships)
        - If the query is better suited for semantic hybrid search (conceptual understanding)
        - If the query requires both approaches in parallel
        - How to reformulate or decompose the query for optimal retrieval
        
        When verifying results, you'll consider:
        - Relevance to the original query
        - Comprehensiveness of the information
        - Need for additional context or clarification
        
        Based on this analysis, you'll either:
        - Return the final compiled results
        - Generate follow-up queries to fill information gaps
        - Reshape the original query based on initial findings
        
        Your expertise in search optimization ensures users receive the most accurate and comprehensive information possible.
        """
        
        super().__init__(
            agent_name="SearchOrchestrationAgent", 
            system_prompt=system_prompt, 
            temperature=0.7, 
            top_p=0.95
        )
        
        # Initialize search agents
        self.graph_search_agent = GraphSearchAgent()
        self.hybrid_search_agent = HybridSearchAgent()
        
        # Initialize embedding handler for query embeddings
        self.embedding_handler = EmbeddingHandler()
        
        # For tracking query iterations
        self.max_search_iterations = 3
        self.max_preliminary_results = 5  # For initial quick assessment
        
        logging.info("[SearchOrchestrationAgent] Initialized with graph and hybrid search capabilities")

    async def analyze_and_rewrite_query(self, query_text: str, context_keywords: List[str] = None) -> Dict[str, Any]:
        """
        Analyzes the query and reshapes it for optimal searchability.
        
        Args:
            query_text (str): Original user query
            context_keywords (List[str], optional): Keywords from context to help guide rewriting
            
        Returns:
            Dict[str, Any]: Analysis results including rewritten queries
        """
        context_section = ""
        if context_keywords and len(context_keywords) > 0:
            context_section = f"""
            Consider these context keywords when rewriting the query:
            {', '.join(context_keywords)}
            """
        
        prompt = f"""
        Analyze and optimize the following search query:
        
        <QUERY>
        {query_text}
        </QUERY>
        
        {context_section}
        
        Your task is to:
        
        1. Determine query characteristics:
           - Is this query entity-focused (asking about specific people, organizations, concepts)?
           - Is this query relationship-focused (asking how entities relate to each other)?
           - Is this query semantic/conceptual (asking about broader topics or ideas)?
           - Is this query factual/specific (asking for precise information)?
        
        2. Rewrite the query to enhance searchability:
           - Create a clear, specific rewritten version
           - Remove ambiguities and unnecessary words
           - Include key terms that will improve search relevance
           - Prioritize precise entity names and relationships
           - Format factual questions in a direct, searchable way
           
        3. Determine if this query should be:
           - Sent to graph search (entity/relationship focused)
           - Sent to vector search (semantic/conceptual)
           - Sent to both search systems
           
        4. If the query is complex, provide up to 3 sub-queries that could be used to gather more complete information.
        
        Return a structured output with your analysis.
        """
        
        analysis = await self.generate_structured_response(prompt, schema=QueryAnalysisSchema)
        
        if not analysis:
            # Fallback to default analysis if LLM fails
            logging.warning("[SearchOrchestrationAgent] LLM query analysis failed, using fallback defaults")
            return {
                "is_entity_focused": "entity" in query_text.lower() or any(term in query_text.lower() for term in ["who", "person", "organization", "company"]),
                "is_relationship_focused": any(term in query_text.lower() for term in ["between", "related", "connection", "relationship", "linked", "associated"]),
                "is_semantic_conceptual": len(query_text.split()) > 8,
                "is_factual_specific": "?" in query_text or any(term in query_text.lower() for term in ["who", "what", "when", "where", "how", "why"]),
                "rewritten_query": query_text,
                "search_strategy": "both",
                "sub_queries": []
            }
        
        return analysis.model_dump()

    async def execute_preliminary_search(self, user_id: int, query_text: str, 
                                        dense_vector: Optional[List[float]] = None, 
                                        sparse_vector: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Executes a quick preliminary search with limited results for strategy assessment.
        
        Args:
            user_id (int): User ID
            query_text (str): The query to search
            dense_vector (Optional[List[float]]): Pre-computed dense vector
            sparse_vector (Optional[Dict[str, float]]): Pre-computed sparse vector
            
        Returns:
            Dict[str, Any]: Preliminary search results
        """
        # Generate embeddings if not provided
        if dense_vector is None or sparse_vector is None:
            try:
                if dense_vector is None:
                    dense_vector = await self.embedding_handler.encode_dense([query_text])
                    if dense_vector and len(dense_vector) > 0:
                        dense_vector = dense_vector[0]
                
                if sparse_vector is None:
                    sparse_vector = await self.embedding_handler.encode_sparse([query_text])
                    if sparse_vector and len(sparse_vector) > 0:
                        sparse_vector = sparse_vector[0]
            except Exception as e:
                logging.error(f"[SearchOrchestrationAgent] Error generating embeddings: {str(e)}")
        
        # Execute both search types with limited parameters
        results = {"graph_results": None, "vector_results": None}
        tasks = []
        
        # Graph search with limited parameters
        graph_search_task = asyncio.create_task(
            self.graph_search_agent.execute({
                "user_id": user_id,
                "query_text": query_text
            })
        )
        tasks.append(("graph", graph_search_task))
        
        # Vector search with limited parameters
        hybrid_search_task = asyncio.create_task(
            self.hybrid_search_agent.execute({
                "user_id": user_id,
                "query_text": query_text,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
                "top_k": self.max_preliminary_results
            })
        )
        tasks.append(("vector", hybrid_search_task))
        
        # Wait for all tasks to complete
        for result_type, task in tasks:
            try:
                result = await task
                if result_type == "graph":
                    results["graph_results"] = result
                else:
                    results["vector_results"] = result
            except Exception as e:
                logging.error(f"[SearchOrchestrationAgent] Error in preliminary {result_type} search: {str(e)}")
                results[f"{result_type}_error"] = str(e)
        
        return results

    async def assess_search_strength(self, search_results: Dict[str, Any], query_analysis: Dict[str, Any]) -> SearchStrengthAssessment:
        """
        Assess the strength of search results based on query characteristics.
        
        Args:
            search_results (Dict[str, Any]): Results from search agents
            query_analysis (Dict[str, Any]): Query analysis output
            
        Returns:
            SearchStrengthAssessment: Assessment of search result strength
        """
        is_entity_focused = query_analysis.get("is_entity_focused", False)
        is_relationship_focused = query_analysis.get("is_relationship_focused", False)
        is_semantic_conceptual = query_analysis.get("is_semantic_conceptual", False)
        is_factual_specific = query_analysis.get("is_factual_specific", False)
        
        # Default values
        entity_coverage = 0.0
        relationship_coverage = 0.0
        semantic_relevance = 0.0
        factual_specificity = 0.0
        strength_areas = []
        weakness_areas = []
        
        # Assess graph search results
        if search_results.get("graph_results"):
            graph_results = search_results["graph_results"]
            
            # Extract relevant data
            extracted_data = graph_results.get("extracted_data", {})
            entities = extracted_data.get("entities", [])
            relationships = extracted_data.get("relationships", [])
            
            search_results_data = graph_results.get("search_results", {})
            knowledge_paths = search_results_data.get("knowledge_paths", [])
            
            # Calculate metrics
            entity_count = len(entities)
            rel_count = len(relationships)
            path_count = len(knowledge_paths)
            
            # Calculate entity coverage
            if is_entity_focused:
                entity_coverage = min(1.0, entity_count / 5.0)  # Normalize: 5+ entities → 1.0
                if entity_coverage > 0.7:
                    strength_areas.append("entity_identification")
                elif entity_coverage < 0.3:
                    weakness_areas.append("entity_identification")
            
            # Calculate relationship coverage
            if is_relationship_focused:
                relationship_coverage = min(1.0, rel_count / 3.0)  # Normalize: 3+ relationships → 1.0
                if relationship_coverage > 0.7:
                    strength_areas.append("relationship_mapping")
                elif relationship_coverage < 0.3:
                    weakness_areas.append("relationship_mapping")
            
            # Calculate factual specificity from knowledge paths
            if is_factual_specific:
                factual_specificity = min(1.0, path_count / 2.0)  # Normalize: 2+ paths → 1.0
                if factual_specificity > 0.7:
                    strength_areas.append("factual_connections")
                elif factual_specificity < 0.3:
                    weakness_areas.append("factual_connections")
        
        # Assess vector search results
        if search_results.get("vector_results"):
            vector_results = search_results["vector_results"]
            result_docs = vector_results.get("results", [])
            
            # Calculate semantic relevance
            if is_semantic_conceptual:
                # Check number of results and average score
                result_count = len(result_docs)
                avg_score = 0.0
                if result_count > 0:
                    avg_score = sum(doc.get("score", 0) for doc in result_docs) / result_count
                
                semantic_relevance = min(1.0, (result_count / 3.0) * avg_score)  # Normalize
                if semantic_relevance > 0.7:
                    strength_areas.append("semantic_understanding")
                elif semantic_relevance < 0.3:
                    weakness_areas.append("semantic_understanding")
            
            # Factual specificity can also be derived from vector search
            if is_factual_specific and factual_specificity < 0.5:
                # Check if vector results contain specific facts
                fact_indicators = ["date", "number", "percentage", "statistic", "measurement"]
                fact_count = 0
                for doc in result_docs:
                    text = doc.get("metadata", {}).get("text", "").lower()
                    if any(indicator in text for indicator in fact_indicators):
                        fact_count += 1
                
                vector_factual = min(1.0, fact_count / 2.0)  # Normalize
                factual_specificity = max(factual_specificity, vector_factual)
                if vector_factual > 0.7:
                    strength_areas.append("factual_evidence")
                elif vector_factual < 0.3 and is_factual_specific:
                    weakness_areas.append("factual_evidence")
        
        # Calculate overall strength
        weights = {
            "entity_coverage": 0.25 if is_entity_focused else 0.1,
            "relationship_coverage": 0.25 if is_relationship_focused else 0.1,
            "semantic_relevance": 0.25 if is_semantic_conceptual else 0.1,
            "factual_specificity": 0.25 if is_factual_specific else 0.1
        }
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            for key in weights:
                weights[key] /= weight_sum
        
        # Calculate weighted overall strength
        overall_strength = (
            weights["entity_coverage"] * entity_coverage +
            weights["relationship_coverage"] * relationship_coverage +
            weights["semantic_relevance"] * semantic_relevance +
            weights["factual_specificity"] * factual_specificity
        )
        
        return SearchStrengthAssessment(
            entity_coverage=entity_coverage,
            relationship_coverage=relationship_coverage,
            semantic_relevance=semantic_relevance,
            factual_specificity=factual_specificity,
            overall_strength=overall_strength,
            strength_areas=strength_areas,
            weakness_areas=weakness_areas
        )

    async def extract_enhancement_context(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts context from initial search results to enhance subsequent searches.
        
        Args:
            search_results (Dict[str, Any]): Results from search agents
            
        Returns:
            Dict[str, Any]: Extracted context for search enhancement
        """
        entity_constraints = []
        keyword_constraints = []
        focus_entities = []
        focus_relationships = []
        
        # Extract from graph results
        if search_results.get("graph_results"):
            graph_results = search_results["graph_results"]
            
            # Extract entity constraints from entities
            extracted_data = graph_results.get("extracted_data", {})
            entities = extracted_data.get("entities", [])
            if entities:
                # Get top entities sorted by confidence
                sorted_entities = sorted(entities, key=lambda e: e.get("confidence", 0), reverse=True)
                for entity in sorted_entities[:min(5, len(sorted_entities))]:
                    entity_text = entity.get("text", "")
                    if entity_text and entity_text not in entity_constraints:
                        entity_constraints.append(entity_text)
                        focus_entities.append({
                            "text": entity_text,
                            "type": entity.get("entity_type", "Unknown")
                        })
            
            # Extract relationship constraints
            relationships = extracted_data.get("relationships", [])
            if relationships:
                # Get top relationships sorted by confidence
                sorted_relationships = sorted(relationships, key=lambda r: r.get("confidence", 0), reverse=True)
                for rel in sorted_relationships[:min(3, len(sorted_relationships))]:
                    source = rel.get("source", "")
                    target = rel.get("target", "")
                    rel_type = rel.get("relation_type", "")
                    if source and target and rel_type:
                        focus_relationships.append({
                            "source": source,
                            "target": target,
                            "type": rel_type
                        })
                        # Add entities from relationships if not already included
                        if source not in entity_constraints:
                            entity_constraints.append(source)
                        if target not in entity_constraints:
                            entity_constraints.append(target)
        
        # Extract from vector results
        if search_results.get("vector_results"):
            vector_results = search_results["vector_results"]
            result_docs = vector_results.get("results", [])
            
            # Extract keywords from top documents
            if result_docs:
                top_docs = result_docs[:min(3, len(result_docs))]
                doc_texts = []
                for doc in top_docs:
                    text = doc.get("metadata", {}).get("text", "")
                    if text:
                        doc_texts.append(text)
                
                if doc_texts:
                    # Extract keywords from document texts
                    keywords = await self._extract_key_concepts(doc_texts)
                    for keyword in keywords:
                        if keyword not in keyword_constraints:
                            keyword_constraints.append(keyword)
        
        return {
            "entity_constraints": entity_constraints,
            "keyword_constraints": keyword_constraints,
            "focus_entities": focus_entities,
            "focus_relationships": focus_relationships
        }

    async def determine_optimal_strategy(self, 
                                        strength_assessment: SearchStrengthAssessment, 
                                        query_analysis: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Determines the optimal search strategy based on preliminary assessment.
        
        Args:
            strength_assessment (SearchStrengthAssessment): Assessment of search strengths
            query_analysis (Dict[str, Any]): Query analysis output
            
        Returns:
            Tuple[str, Dict[str, Any]]: Search strategy and enhancement parameters
        """
        is_entity_focused = query_analysis.get("is_entity_focused", False) 
        is_relationship_focused = query_analysis.get("is_relationship_focused", False)
        
        # Strategy decision parameters
        entity_coverage = strength_assessment.entity_coverage
        relationship_coverage = strength_assessment.relationship_coverage
        semantic_relevance = strength_assessment.semantic_relevance
        factual_specificity = strength_assessment.factual_specificity
        overall_strength = strength_assessment.overall_strength
        weakness_areas = strength_assessment.weakness_areas
        
        # Enhancement parameters
        enhancement_params = {
            "expanded_depth": None,
            "expanded_limit": None,
            "source_filter": None
        }
        
        # Default to both initially
        strategy = "both"
        
        # Decision tree for optimal strategy
        if overall_strength > 0.7:
            # If results are already strong, focus on the best-performing method
            if entity_coverage > 0.7 and relationship_coverage > 0.7:
                strategy = "graph"
                enhancement_params["expanded_depth"] = 3  # Deeper graph exploration
            elif semantic_relevance > 0.7:
                strategy = "vector"
                enhancement_params["expanded_limit"] = 20  # More vector results
            else:
                strategy = "both"  # Both methods performed well
        else:
            # If results are weak, strategically enhance
            if (is_entity_focused or is_relationship_focused) and entity_coverage < 0.3:
                # Entity-focused queries with poor entity coverage need graph search with expanded parameters
                strategy = "graph" if semantic_relevance < 0.5 else "both"
                enhancement_params["expanded_depth"] = 4  # Much deeper graph exploration
                enhancement_params["expanded_limit"] = 30  # More results
            elif not (is_entity_focused or is_relationship_focused) and semantic_relevance < 0.3:
                # Conceptual queries with poor semantic relevance need better vector search
                strategy = "vector" if entity_coverage < 0.5 else "both"
                enhancement_params["expanded_limit"] = 30  # Many more vector results
            else:
                # For mixed results, use both with balanced enhancements
                strategy = "both"
                enhancement_params["expanded_depth"] = 3
                enhancement_params["expanded_limit"] = 20
        
        # Override if certain areas are particularly weak
        if "entity_identification" in weakness_areas and "relationship_mapping" in weakness_areas:
            enhancement_params["expanded_depth"] = 4  # Maximum depth for graph search
        
        if "semantic_understanding" in weakness_areas:
            enhancement_params["expanded_limit"] = 30  # Maximum limit for vector search
        
        # If factual information is needed but weak, adjust strategy
        if factual_specificity < 0.3 and query_analysis.get("is_factual_specific", False):
            strategy = "both"  # Ensure we try both methods
            enhancement_params["expanded_depth"] = 4
            enhancement_params["expanded_limit"] = 25
        
        return strategy, enhancement_params

    async def coordinate_search_agents(self, 
                                      graph_results: Dict[str, Any], 
                                      vector_results: Dict[str, Any]) -> SearchEnhancementParams:
        """
        Coordinates between graph and vector search agents to enhance results.
        
        Args:
            graph_results (Dict[str, Any]): Results from graph search
            vector_results (Dict[str, Any]): Results from vector search
            
        Returns:
            SearchEnhancementParams: Enhanced search parameters for both agents
        """
        enhancement_params = SearchEnhancementParams()
        
        # Extract entities from graph results to enhance vector search
        if graph_results and "extracted_data" in graph_results:
            entities = graph_results["extracted_data"].get("entities", [])
            if entities:
                # Get entity texts
                entity_texts = [e.get("text") for e in entities if e.get("text")]
                enhancement_params.entity_constraints = entity_texts[:min(5, len(entity_texts))]
                
                # Extract focused entities for graph search enhancement
                top_entities = sorted(entities, key=lambda e: e.get("confidence", 0), reverse=True)
                enhancement_params.focus_entities = [e.get("text") for e in top_entities[:min(3, len(top_entities))]]
        
        # Extract keywords from vector results to enhance graph search
        if vector_results and "results" in vector_results:
            top_docs = vector_results["results"][:min(3, len(vector_results.get("results", [])))]
            if top_docs:
                # Extract document texts
                doc_texts = [doc.get("metadata", {}).get("text", "") for doc in top_docs]
                doc_texts = [text for text in doc_texts if text]
                
                if doc_texts:
                    # Extract keywords from document texts
                    keywords = await self._extract_key_concepts(doc_texts)
                    enhancement_params.keyword_constraints = keywords[:min(10, len(keywords))]
        
        return enhancement_params

    async def _extract_key_concepts(self, texts: List[str]) -> List[str]:
        """
        Extracts key concepts and keywords from texts.
        
        Args:
            texts (List[str]): List of text passages
            
        Returns:
            List[str]: Extracted key concepts
        """
        combined_text = "\n\n".join(texts)
        
        prompt = f"""
        Extract the top 10 most important keywords, entities, and concepts from the following text:
        
        <TEXT>
        {combined_text}
        </TEXT>
        
        Focus on:
        1. Named entities (people, organizations, products, etc.)
        2. Technical terms and domain-specific concepts
        3. Action verbs and significant nouns
        
        Return the keywords as a comma-separated list.
        """
        
        try:
            response = await self.generate_response(prompt)
            # Parse the comma-separated list
            keywords = [kw.strip() for kw in response.split(",") if kw.strip()]
            return keywords
        except Exception as e:
            logging.error(f"[SearchOrchestrationAgent] Error extracting key concepts: {str(e)}")
            return []

    async def execute_optimized_search(self, 
                                     user_id: int, 
                                     query_text: str, 
                                     search_strategy: str,
                                     enhancement_params: Dict[str, Any] = None,
                                     dense_vector: Optional[List[float]] = None, 
                                     sparse_vector: Optional[Dict[str, float]] = None,
                                     top_k: int = 10) -> Dict[str, Any]:
        """
        Executes an optimized search with enhanced parameters based on preliminary results.
        
        Args:
            user_id (int): User ID
            query_text (str): The query to search
            search_strategy (str): "graph", "vector", or "both"
            enhancement_params (Dict[str, Any]): Enhanced search parameters
            dense_vector (Optional[List[float]]): Pre-computed dense vector
            sparse_vector (Optional[Dict[str, float]]): Pre-computed sparse vector
            top_k (int): Number of results to return
            
        Returns:
            Dict[str, Any]: Optimized search results
        """
        results = {"graph_results": None, "vector_results": None}
        tasks = []
        
        # Generate embeddings if not provided
        if (search_strategy in ["vector", "both"]) and (dense_vector is None or sparse_vector is None):
            try:
                # Enhance query with entity constraints if available
                enhanced_query = query_text
                if enhancement_params and "entity_constraints" in enhancement_params and enhancement_params["entity_constraints"]:
                    entity_str = " ".join(enhancement_params["entity_constraints"])
                    enhanced_query = f"{query_text} {entity_str}"
                
                if dense_vector is None:
                    dense_vector = await self.embedding_handler.encode_dense([enhanced_query])
                    if dense_vector and len(dense_vector) > 0:
                        dense_vector = dense_vector[0]
                
                if sparse_vector is None:
                    sparse_vector = await self.embedding_handler.encode_sparse([enhanced_query])
                    if sparse_vector and len(sparse_vector) > 0:
                        sparse_vector = sparse_vector[0]
            except Exception as e:
                logging.error(f"[SearchOrchestrationAgent] Error generating embeddings: {str(e)}")
        
        # Prepare enhanced parameters for graph search
        graph_input = {
            "user_id": user_id,
            "query_text": query_text
        }
        
        if enhancement_params:
            expanded_depth = enhancement_params.get("expanded_depth")
            focus_entities = enhancement_params.get("focus_entities", [])
            focus_relationships = enhancement_params.get("focus_relationships", [])
            
            if expanded_depth:
                graph_input["search_depth"] = expanded_depth
            
            if focus_entities:
                graph_input["focus_entities"] = focus_entities
            
            if focus_relationships:
                graph_input["focus_relationships"] = focus_relationships
            
            # Enhance query with keyword constraints if available
            if "keyword_constraints" in enhancement_params and enhancement_params["keyword_constraints"]:
                keywords = enhancement_params["keyword_constraints"]
                graph_input["query_text"] = f"{query_text} {' '.join(keywords)}"
        
        # Prepare enhanced parameters for vector search
        vector_input = {
            "user_id": user_id,
            "query_text": query_text,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
            "top_k": top_k
        }
        
        if enhancement_params:
            expanded_limit = enhancement_params.get("expanded_limit")
            entity_constraints = enhancement_params.get("entity_constraints", [])
            
            if expanded_limit:
                vector_input["top_k"] = expanded_limit
            
            if entity_constraints:
                vector_input["entity_filters"] = entity_constraints
        
        # Execute in parallel based on strategy
        if search_strategy in ["graph", "both"]:
            graph_search_task = asyncio.create_task(
                self.graph_search_agent.execute(graph_input)
            )
            tasks.append(("graph", graph_search_task))
        
        if search_strategy in ["vector", "both"]:
            hybrid_search_task = asyncio.create_task(
                self.hybrid_search_agent.execute(vector_input)
            )
            tasks.append(("vector", hybrid_search_task))
        
        # Wait for all tasks to complete
        for result_type, task in tasks:
            try:
                result = await task
                if result_type == "graph":
                    results["graph_results"] = result
                else:
                    results["vector_results"] = result
            except Exception as e:
                logging.error(f"[SearchOrchestrationAgent] Error in {result_type} search: {str(e)}")
                results[f"{result_type}_error"] = str(e)
        
        return results

    async def execute_search_with_agents(self, user_id: int, query_text: str, search_strategy: str, 
                                        dense_vector: Optional[List[float]] = None, 
                                        sparse_vector: Optional[Dict[str, float]] = None,
                                        top_k: int = 10) -> Dict[str, Any]:
        """
        Executes search using the appropriate agent(s) based on search strategy.
        
        Args:
            user_id (int): User ID
            query_text (str): The query to search
            search_strategy (str): "graph", "vector", or "both"
            dense_vector (Optional[List[float]]): Pre-computed dense vector (if available)
            sparse_vector (Optional[Dict[str, float]]): Pre-computed sparse vector (if available)
            top_k (int): Number of results to return
            
        Returns:
            Dict[str, Any]: Search results
        """
        results = {"graph_results": None, "vector_results": None}
        tasks = []
        
        # Generate embeddings if not provided
        if (search_strategy in ["vector", "both"]) and (dense_vector is None or sparse_vector is None):
            try:
                if dense_vector is None:
                    dense_vector = await self.embedding_handler.encode_dense([query_text])
                    if dense_vector and len(dense_vector) > 0:
                        dense_vector = dense_vector[0]  # Get first embedding
                
                if sparse_vector is None:
                    sparse_vector = await self.embedding_handler.encode_sparse([query_text])
                    if sparse_vector and len(sparse_vector) > 0:
                        sparse_vector = sparse_vector[0]  # Get first embedding
            except Exception as e:
                logging.error(f"[SearchOrchestrationAgent] Error generating embeddings: {str(e)}")
        
        # Execute in parallel when using both search strategies
        if search_strategy in ["graph", "both"]:
            graph_search_task = asyncio.create_task(
                self.graph_search_agent.execute({
                    "user_id": user_id,
                    "query_text": query_text
                })
            )
            tasks.append(("graph", graph_search_task))
        
        if search_strategy in ["vector", "both"]:
            hybrid_search_task = asyncio.create_task(
                self.hybrid_search_agent.execute({
                    "user_id": user_id,
                    "query_text": query_text,
                    "dense_vector": dense_vector,
                    "sparse_vector": sparse_vector,
                    "top_k": top_k
                })
            )
            tasks.append(("vector", hybrid_search_task))
        
        # Wait for all tasks to complete
        for result_type, task in tasks:
            try:
                result = await task
                if result_type == "graph":
                    results["graph_results"] = result
                else:
                    results["vector_results"] = result
            except Exception as e:
                logging.error(f"[SearchOrchestrationAgent] Error in {result_type} search: {str(e)}")
                results[f"{result_type}_error"] = str(e)
        
        return results

    async def verify_search_results(self, query_text: str, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifies if the search results are adequate or if further querying is needed.
        
        Args:
            query_text (str): The original query
            search_results (Dict[str, Any]): Results from search agents
            
        Returns:
            Dict[str, Any]: Verification results including whether additional searches are needed
        """
        # Extract relevant portions of the results for LLM to analyze
        graph_summary = "No graph results available."
        vector_summary = "No vector results available."
        
        if search_results.get("graph_results"):
            graph_results = search_results["graph_results"]
            
            extracted_data = graph_results.get("extracted_data", {})
            entities = extracted_data.get("entities", [])
            relationships = extracted_data.get("relationships", [])
            
            search_results_data = graph_results.get("search_results", {})
            knowledge_paths = search_results_data.get("knowledge_paths", [])
            
            entity_count = len(entities)
            rel_count = len(relationships)
            path_count = len(knowledge_paths)
            
            # Summarize some top entities and relationships if available
            entity_examples = ""
            if entities and len(entities) > 0:
                top_entities = entities[:min(3, len(entities))]
                entity_names = [e.get("text", "Unknown") for e in top_entities]
                entity_examples = f"Top entities: {', '.join(entity_names)}. "
            
            rel_examples = ""
            if relationships and len(relationships) > 0:
                top_rels = relationships[:min(3, len(relationships))]
                rel_descriptions = []
                for r in top_rels:
                    source = r.get("source", "Unknown")
                    target = r.get("target", "Unknown")
                    rel_type = r.get("relation_type", "related to")
                    rel_descriptions.append(f"{source} {rel_type} {target}")
                rel_examples = f"Example relationships: {'; '.join(rel_descriptions)}. "
            
            graph_summary = f"Graph search found {entity_count} entities, {rel_count} relationships, and {path_count} knowledge paths. {entity_examples}{rel_examples}"
        
        if search_results.get("vector_results"):
            vector_results = search_results["vector_results"]
            result_docs = vector_results.get("results", [])
            result_count = len(result_docs)
            
            doc_examples = ""
            if result_docs and len(result_docs) > 0:
                top_docs = result_docs[:min(3, len(result_docs))]
                doc_excerpts = []
                for doc in top_docs:
                    text = doc.get("metadata", {}).get("text", "No text available")
                    # Truncate long texts
                    if len(text) > 100:
                        text = text[:100] + "..."
                    doc_excerpts.append(text)
                doc_examples = f"Example documents: {'; '.join(doc_excerpts)}"
            
            vector_summary = f"Vector search found {result_count} relevant documents. {doc_examples}"
        
        prompt = f"""
        Evaluate the adequacy of search results for the following query:
        
        <QUERY>
        {query_text}
        </QUERY>
        
        <SEARCH_RESULTS_SUMMARY>
        {graph_summary}
        {vector_summary}
        </SEARCH_RESULTS_SUMMARY>
        
        Your task is to:
        
        1. Determine if the results appear adequate to answer the query
        2. Identify any information gaps that would require additional searches
        3. Suggest how to reformulate the query if results are insufficient
        4. Recommend if we should try an alternative search strategy
        
        Return a structured evaluation.
        """
        
        verification = await self.generate_structured_response(prompt, schema=ResultVerificationSchema)
        
        if not verification:
            # Default verification if LLM fails
            logging.warning("[SearchOrchestrationAgent] LLM verification failed, using fallback defaults")
            return {
                "is_adequate": True,
                "information_gaps": [],
                "reformulated_query": None,
                "alternative_strategy": None,
                "follow_up_queries": []
            }
        
        return verification.model_dump()

    async def extract_follow_up_queries(self, search_results: Dict[str, Any], original_query: str) -> List[str]:
        """
        Extracts potential follow-up queries from search results.
        
        Args:
            search_results (Dict[str, Any]): Results from search agents
            original_query (str): The original user query
        
        Returns:
            List[str]: Suggested follow-up queries
        """
        if not search_results:
            return []
        
        # Extract context from results
        context = []
        
        # Extract entities and relationships from graph results
        if search_results.get("graph_results"):
            graph_results = search_results["graph_results"]
            extracted_data = graph_results.get("extracted_data", {})
            
            entities = extracted_data.get("entities", [])
            if entities:
                for entity in entities[:min(5, len(entities))]:
                    entity_text = entity.get("text", "")
                    entity_type = entity.get("entity_type", "")
                    if entity_text and entity_type:
                        context.append(f"{entity_text} ({entity_type})")
            
            relationships = extracted_data.get("relationships", [])
            if relationships:
                for rel in relationships[:min(5, len(relationships))]:
                    source = rel.get("source", "")
                    target = rel.get("target", "")
                    rel_type = rel.get("relation_type", "")
                    if source and target and rel_type:
                        context.append(f"{source} {rel_type} {target}")
        
        # Extract document snippets from vector results
        if search_results.get("vector_results"):
            vector_results = search_results["vector_results"]
            results = vector_results.get("results", [])
            
            if results:
                for result in results[:min(3, len(results))]:
                    text = result.get("metadata", {}).get("text", "")
                    if text:
                        # Truncate long texts
                        if len(text) > 200:
                            text = text[:200] + "..."
                        context.append(text)
        
        # Use the LLM to generate follow-up queries
        if not context:
            return []
        
        prompt = f"""
        Based on the original query and search context, generate 2-3 follow-up queries that would help gather additional relevant information.
        
        Original Query: {original_query}
        
        Context from search results:
        {"- " + chr(10) + "- ".join(context)}
        
        Suggest follow-up queries that:
        1. Address aspects not covered in the original results
        2. Explore related topics that would complement the information
        3. Provide more specific details on key entities or concepts
        
        Return 2-3 follow-up queries as a comma-separated list.
        """
        
        try:
            response = await self.generate_response(prompt)
            # Parse the comma-separated list
            follow_ups = [q.strip() for q in response.split(",") if q.strip()]
            # Remove any leading numbers or bullets
            follow_ups = [q.lstrip("0123456789.-) ") for q in follow_ups]
            return follow_ups[:3]  # Return at most 3
        except Exception as e:
            logging.error(f"[SearchOrchestrationAgent] Error generating follow-up queries: {str(e)}")
            return []

    async def generate_fallback_strategy(self, query_text: str, failed_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates fallback search strategies when initial searches fail.
        
        Args:
            query_text (str): The original query
            failed_results (Dict[str, Any]): Results from failed searches
            
        Returns:
            Dict[str, Any]: Fallback search strategy
        """
        prompt = f"""
        The following search query returned insufficient results:
        
        <QUERY>
        {query_text}
        </QUERY>
        
        Generate a fallback search strategy by:
        1. Identifying broader concepts if the query is too specific
        2. Breaking down the query into simpler sub-queries
        3. Suggesting alternative search approaches
        
        Return your suggestions in the following format:
        - Broader query: (a more general version of the query)
        - Sub-queries: (comma-separated list of 2-3 simpler sub-queries)
        - Alternative approach: (brief description of a different search approach)
        """
        
        try:
            response = await self.generate_response(prompt)
            
            # Parse the response
            broader_query = None
            sub_queries = []
            alternative_approach = None
            
            lines = response.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("- Broader query:") or line.startswith("Broader query:"):
                    broader_query = line.split(":", 1)[1].strip()
                elif line.startswith("- Sub-queries:") or line.startswith("Sub-queries:"):
                    sub_queries_text = line.split(":", 1)[1].strip()
                    sub_queries = [q.strip() for q in sub_queries_text.split(",") if q.strip()]
                elif line.startswith("- Alternative approach:") or line.startswith("Alternative approach:"):
                    alternative_approach = line.split(":", 1)[1].strip()
            
            return {
                "broader_query": broader_query if broader_query else query_text,
                "sub_queries": sub_queries[:3],  # Limit to top 3
                "alternative_approach": alternative_approach
            }
        except Exception as e:
            logging.error(f"[SearchOrchestrationAgent] Error generating fallback strategy: {str(e)}")
            return {
                "broader_query": query_text,
                "sub_queries": [],
                "alternative_approach": None
            }

    async def rank_merged_results(self, graph_results: Dict[str, Any], vector_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Implements a sophisticated ranking system for merged results.
        
        Args:
            graph_results (Dict[str, Any]): Results from graph search
            vector_results (Dict[str, Any]): Results from vector search
            
        Returns:
            List[Dict[str, Any]]: Ranked and scored combined results
        """
        ranked_results = []
        
        # Process graph results
        if graph_results and "search_results" in graph_results:
            # Process entity results
            entity_results = graph_results["search_results"].get("entities", [])
            for entity_result in entity_results:
                # Calculate graph confidence score based on multiple factors
                confidence = self._calculate_graph_confidence(entity_result, "entity")
                ranked_results.append({
                    "source_type": "graph",
                    "result_type": "entity",
                    "relevance_score": confidence,
                    "content": f"Entity: {entity_result.get('text', '')} ({entity_result.get('entity_type', '')})",
                    "metadata": entity_result
                })
            
            # Process relationship results
            relationship_results = graph_results["search_results"].get("relationships", [])
            for rel_result in relationship_results:
                confidence = self._calculate_graph_confidence(rel_result, "relationship")
                ranked_results.append({
                    "source_type": "graph",
                    "result_type": "relationship",
                    "relevance_score": confidence,
                    "content": f"Relationship: {rel_result.get('source', '')} {rel_result.get('relation_type', '')} {rel_result.get('target', '')}",
                    "metadata": rel_result
                })
            
            # Process knowledge paths
            knowledge_paths = graph_results["search_results"].get("knowledge_paths", [])
            for path in knowledge_paths:
                confidence = self._calculate_graph_confidence(path, "path")
                ranked_results.append({
                    "source_type": "graph",
                    "result_type": "path",
                    "relevance_score": confidence,
                    "content": path.get("path_summary", ""),
                    "metadata": path
                })
        
        # Process vector results
        if vector_results and "results" in vector_results:
            for doc in vector_results["results"]:
                # Use the vector score and apply additional weighting factors
                score = doc.get("score", 0) * self._calculate_vector_weight(doc)
                ranked_results.append({
                    "source_type": "vector",
                    "result_type": "document",
                    "relevance_score": score,
                    "content": doc.get("metadata", {}).get("text", ""),
                    "metadata": doc
                })
        
        # Sort by score in descending order
        ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return ranked_results

    def _calculate_graph_confidence(self, item: Dict[str, Any], item_type: str) -> float:
        """
        Calculates a confidence score for graph search items.
        
        Args:
            item (Dict[str, Any]): The graph item (entity, relationship, or path)
            item_type (str): Type of the item ('entity', 'relationship', or 'path')
            
        Returns:
            float: Confidence score (0-1)
        """
        base_confidence = item.get("confidence", 0.5)
        
        if item_type == "entity":
            # Entities with profile information are more valuable
            profile_weight = 0.2 if item.get("entity_profile") else 0
            return min(1.0, base_confidence + profile_weight)
        
        elif item_type == "relationship":
            # Relationships with specific types are more valuable than generic ones
            specificity_weight = 0.0
            relation_type = item.get("relation_type", "").lower()
            generic_relations = ["related to", "connected to", "associated with", "linked to"]
            if relation_type and relation_type not in generic_relations:
                specificity_weight = 0.2
            return min(1.0, base_confidence + specificity_weight)
        
        elif item_type == "path":
            # Shorter paths and paths with more specific relationships are more valuable
            path_length = item.get("path_length", 3)
            length_weight = max(0, 0.3 - (path_length * 0.05))  # Shorter paths get higher weight
            return min(1.0, base_confidence + length_weight)
        
        return base_confidence

    def _calculate_vector_weight(self, doc: Dict[str, Any]) -> float:
        """
        Calculates additional weighting factors for vector search results.
        
        Args:
            doc (Dict[str, Any]): Vector search result document
            
        Returns:
            float: Weight modifier (typically 0.8-1.2)
        """
        # Base weight
        weight = 1.0
        
        # Check document metadata for quality indicators
        metadata = doc.get("metadata", {})
        
        # Documents with titles are typically more structured and valuable
        if metadata.get("title"):
            weight += 0.1
        
        # Check if document contains factual elements like dates, numbers
        text = metadata.get("text", "").lower()
        fact_indicators = ["date:", "year:", "percent", "%", "number:", "$", "€", "£"]
        if any(indicator in text for indicator in fact_indicators):
            weight += 0.1
        
        # Longer documents may contain more comprehensive information
        if len(text) > 500:
            weight += 0.05
        
        # Check for source authority (if available)
        if metadata.get("source_authority"):
            authority_score = metadata.get("source_authority")
            if isinstance(authority_score, (int, float)):
                weight += authority_score * 0.1
        
        return weight

    def _deduplicate_and_rerank(self, merged_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes duplicates and reranks the merged results.
        
        Args:
            merged_results (List[Dict[str, Any]]): Merged results from all search iterations
            
        Returns:
            List[Dict[str, Any]]: Deduplicated and reranked results
        """
        # Track seen content to avoid duplicates
        seen_content = set()
        deduplicated_results = []
        
        for result in merged_results:
            content = result.get("content", "")
            # Create a simplified representation for duplication check
            simple_content = ' '.join(content.lower().split()[:10])  # First 10 words
            
            if simple_content and simple_content not in seen_content:
                seen_content.add(simple_content)
                deduplicated_results.append(result)
        
        # Apply diversity ranking - ensure a mix of result types
        graph_results = [r for r in deduplicated_results if r.get("source_type") == "graph"]
        vector_results = [r for r in deduplicated_results if r.get("source_type") == "vector"]
        
        # Sort each group by relevance score
        graph_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        vector_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Interleave results based on score ratios
        final_results = []
        g_idx, v_idx = 0, 0
        
        while g_idx < len(graph_results) or v_idx < len(vector_results):
            # Get next graph result if available
            if g_idx < len(graph_results):
                g_score = graph_results[g_idx].get("relevance_score", 0)
            else:
                g_score = -1
            
            # Get next vector result if available
            if v_idx < len(vector_results):
                v_score = vector_results[v_idx].get("relevance_score", 0)
            else:
                v_score = -1
            
            # Compare scores and add the higher-scoring result
            if g_score >= v_score and g_score > 0:
                final_results.append(graph_results[g_idx])
                g_idx += 1
            elif v_score > 0:
                final_results.append(vector_results[v_idx])
                v_idx += 1
            else:
                # No more valid results
                break
        
        return final_results

    async def compile_final_results_with_ranking(self, query_text: str, ranked_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compiles final results from ranked search results.
        
        Args:
            query_text (str): The original query
            ranked_results (List[Dict[str, Any]]): Ranked search results
            
        Returns:
            Dict[str, Any]: Compiled final results
        """
        # Prepare result summary for the LLM
        result_summary = "Search Results:\n\n"
        
        # Count result types for distribution
        source_distribution = {"graph": 0, "vector": 0}
        
        for i, result in enumerate(ranked_results[:min(15, len(ranked_results))]):
            source_type = result.get("source_type", "unknown")
            result_type = result.get("result_type", "unknown")
            score = result.get("relevance_score", 0)
            content = result.get("content", "No content available")
            
            result_summary += f"Result {i+1} [{source_type.upper()} - {result_type} - Score: {score:.2f}]:\n{content}\n\n"
            
            # Update distribution count
            if source_type in source_distribution:
                source_distribution[source_type] += 1
        
        prompt = f"""
        Compile comprehensive search results for the following query:
        
        <QUERY>
        {query_text}
        </QUERY>
        
        <SEARCH_RESULTS>
        {result_summary}
        </SEARCH_RESULTS>
        
        Your task is to:
        
        1. Analyze all search results
        2. Prioritize the most relevant information
        3. Organize results coherently
        4. Ensure comprehensive coverage of the query topic
        5. Remove redundant information
        
        Provide a structure with the following sections:
        1. A short executive summary (2-3 sentences)
        2. A list of main findings (4-6 bullet points)
        3. Detailed explanation that directly answers the original query
        
        Be specific, use concrete information from the search results, and maintain a balanced view.
        """
        
        compiled_response = await self.generate_response(prompt)
        
        # Create a structured result
        result_entities = []
        for result in ranked_results:
            if result.get("source_type") == "graph" and result.get("result_type") == "entity":
                entity_data = result.get("metadata", {})
                if entity_data and "text" in entity_data:
                    result_entities.append(entity_data)
        
        # Extract main findings
        main_findings = []
        for line in compiled_response.split("\n"):
            if line.strip().startswith("- ") or line.strip().startswith("* "):
                main_findings.append(line.strip()[2:])
        
        # If no bullet points were found, try to extract key sentences
        if not main_findings:
            sentences = compiled_response.split(".")
            potential_findings = [s.strip() for s in sentences if 10 < len(s.strip()) < 100]
            main_findings = potential_findings[:min(5, len(potential_findings))]
        
        # Extract summary (first paragraph)
        summary = ""
        paragraphs = [p.strip() for p in compiled_response.split("\n\n") if p.strip()]
        if paragraphs:
            summary = paragraphs[0]
        
        # Create detailed results list
        detailed_results = []
        for result in ranked_results[:min(10, len(ranked_results))]:
            detailed_results.append({
                "relevance_score": result.get("relevance_score", 0),
                "source_type": result.get("source_type", "unknown"),
                "content": result.get("content", "No content available"),
                "metadata": {
                    "result_type": result.get("result_type", "unknown")
                }
            })
        
        return {
            "query": query_text,
            "compiled_results": compiled_response,
            "structured_output": {
                "summary": summary,
                "main_findings": main_findings,
                "detailed_results": detailed_results,
                "source_distribution": source_distribution,
                "entities": result_entities
            }
        }

    async def compile_final_results(self, query_text: str, all_search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compiles final results from multiple search iterations.
        
        Args:
            query_text (str): The original query
            all_search_results (List[Dict[str, Any]]): Results from all search iterations
            
        Returns:
            Dict[str, Any]: Compiled final results
        """
        # Prepare a summary of all gathered information for the LLM
        result_summary = "Search Results:\n\n"
        
        for i, results in enumerate(all_search_results):
            result_summary += f"--- Iteration {i+1} ---\n"
            
            if results.get("graph_results"):
                graph_results = results["graph_results"]
                
                extracted_data = graph_results.get("extracted_data", {})
                entities = extracted_data.get("entities", [])
                relationships = extracted_data.get("relationships", [])
                
                search_results_data = graph_results.get("search_results", {})
                entity_results = search_results_data.get("entities", [])
                relation_results = search_results_data.get("relationships", [])
                knowledge_paths = search_results_data.get("knowledge_paths", [])
                
                result_summary += f"Graph Search Results:\n"
                result_summary += f"- Extracted {len(entities)} entities and {len(relationships)} relationships\n"
                
                # Add entity details
                if entities:
                    result_summary += "- Key entities:\n"
                    for e in entities[:min(5, len(entities))]:
                        entity_text = e.get("text", "Unknown")
                        entity_type = e.get("entity_type", "Unknown")
                        entity_profile = e.get("entity_profile", "")[:100]
                        result_summary += f"  * {entity_text} ({entity_type}): {entity_profile}...\n"
                
                # Add relationship details
                if relationships:
                    result_summary += "- Key relationships:\n"
                    for r in relationships[:min(5, len(relationships))]:
                        source = r.get("source", "Unknown")
                        relation = r.get("relation_type", "related to")
                        target = r.get("target", "Unknown")
                        result_summary += f"  * {source} {relation} {target}\n"
                
                # Add knowledge path details
                if knowledge_paths:
                    result_summary += "- Knowledge paths found:\n"
                    for path in knowledge_paths[:min(3, len(knowledge_paths))]:
                        path_summary = path.get("path_summary", "No path summary available")
                        result_summary += f"  * {path_summary}\n"
                
                result_summary += "\n"
            
            if results.get("vector_results"):
                vector_results = results["vector_results"]
                result_docs = vector_results.get("results", [])
                
                result_summary += f"Vector Search Results:\n"
                result_summary += f"- Found {len(result_docs)} relevant documents\n"
                
                if result_docs:
                    result_summary += "- Top document excerpts:\n"
                    for doc in result_docs[:min(5, len(result_docs))]:
                        score = doc.get("score", 0)
                        text = doc.get("metadata", {}).get("text", "No text available")
                        # Truncate long texts
                        if len(text) > 150:
                            text = text[:150] + "..."
                        result_summary += f"  * [{score:.2f}] {text}\n"
                
                result_summary += "\n"
        
        prompt = f"""
        Compile comprehensive search results for the following query:
        
        <QUERY>
        {query_text}
        </QUERY>
        
        <SEARCH_RESULTS_SUMMARY>
        {result_summary}
        </SEARCH_RESULTS_SUMMARY>
        
        Your task is to:
        
        1. Analyze all search results across iterations
        2. Prioritize the most relevant information
        3. Organize results coherently
        4. Ensure comprehensive coverage of the query topic
        5. Remove redundant information
        
        Provide a complete compilation of the search results that directly answers the original query.
        """
        
        compiled_response = await self.generate_response(prompt)
        
        return {
            "query": query_text,
            "compiled_results": compiled_response,
            "raw_search_data": all_search_results
        }

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core execution function for the orchestration agent.
        
        Args:
            inputs (Dict[str, Any]): Input data containing user query and metadata
            
        Returns:
            Dict[str, Any]: Final compiled search results
        """
        user_id = inputs.get("user_id")
        query_text = inputs.get("query_text")
        dense_vector = inputs.get("dense_vector")
        sparse_vector = inputs.get("sparse_vector")
        top_k = inputs.get("top_k", 10)
        
        if not query_text:
            logging.error("[SearchOrchestrationAgent] No query text provided")
            return {"error": "No query text provided"}
        
        logging.info(f"[SearchOrchestrationAgent] Processing query: {query_text}")
        
        # Track all search iterations for final compilation
        all_search_results = []
        
        # 1. QUERY ANALYSIS AND STRATEGY SELECTION
        logging.info("[SearchOrchestrationAgent] Analyzing and rewriting query")
        analysis = await self.analyze_and_rewrite_query(query_text)
        current_query = analysis.get("rewritten_query", query_text)
        initial_strategy = analysis.get("search_strategy", "both")
        sub_queries = analysis.get("sub_queries", [])
        
        logging.info(f"[SearchOrchestrationAgent] Rewritten query: {current_query}")
        logging.info(f"[SearchOrchestrationAgent] Initial strategy: {initial_strategy}")
        
        # 2. PRELIMINARY SEARCH EXECUTION - Quick assessment with limited parameters
        logging.info("[SearchOrchestrationAgent] Executing preliminary search")
        preliminary_results = await self.execute_preliminary_search(
            user_id=user_id,
            query_text=current_query,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )
        
        # 3. POST-RETRIEVAL DECISION - Assess preliminary results and determine strategy
        logging.info("[SearchOrchestrationAgent] Assessing preliminary results")
        strength_assessment = await self.assess_search_strength(preliminary_results, analysis)
        
        # Extract context from preliminary results for search enhancement
        enhancement_context = await self.extract_enhancement_context(preliminary_results)
        
        # Determine optimal strategy and enhancement parameters
        optimal_strategy, enhancement_params = await self.determine_optimal_strategy(
            strength_assessment=strength_assessment,
            query_analysis=analysis
        )
        
        # Enhance parameters with extracted context
        enhancement_params.update(enhancement_context)
        
        logging.info(f"[SearchOrchestrationAgent] Optimal strategy: {optimal_strategy}")
        logging.info(f"[SearchOrchestrationAgent] Enhancement parameters: {enhancement_params}")
        
        # 4. EXECUTE OPTIMIZED SEARCH - Use the optimal strategy and enhanced parameters
        logging.info("[SearchOrchestrationAgent] Executing optimized search")
        search_results = await self.execute_optimized_search(
            user_id=user_id,
            query_text=current_query,
            search_strategy=optimal_strategy,
            enhancement_params=enhancement_params,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k
        )
        all_search_results.append(search_results)
        
        # 5. VERIFICATION AND ITERATIVE REFINEMENT
        logging.info("[SearchOrchestrationAgent] Verifying search results")
        verification = await self.verify_search_results(query_text, search_results)
        
        # Extract keywords from results for query refinement
        result_keywords = []
        if not verification.get("is_adequate", True):
            if search_results.get("vector_results", {}).get("results"):
                vector_docs = search_results["vector_results"]["results"][:3]
                vector_texts = [doc.get("metadata", {}).get("text", "") for doc in vector_docs]
                if vector_texts:
                    result_keywords = await self._extract_key_concepts(vector_texts)
        
        # Follow-up iterations if needed (up to max_search_iterations)
        iterations = 1
        while iterations < self.max_search_iterations:
            # If results are adequate, no need for more searches
            if verification.get("is_adequate", True):
                logging.info("[SearchOrchestrationAgent] Results are adequate, no further iterations needed")
                break
            
            iterations += 1
            logging.info(f"[SearchOrchestrationAgent] Starting iteration {iterations}")
            
            # Get follow-up queries or reformulations
            follow_up_queries = verification.get("follow_up_queries", [])
            reformulated_query = verification.get("reformulated_query")
            new_strategy = verification.get("alternative_strategy")
            
            # If no follow-up queries or reformulation, try to extract them from results
            if not follow_up_queries and not reformulated_query:
                follow_up_queries = await self.extract_follow_up_queries(search_results, query_text)
            
            # If still no queries, try fallback strategy
            if not follow_up_queries and not reformulated_query:
                fallback = await self.generate_fallback_strategy(query_text, search_results)
                broader_query = fallback.get("broader_query")
                if broader_query and broader_query != query_text:
                    reformulated_query = broader_query
                
                fallback_sub_queries = fallback.get("sub_queries", [])
                if fallback_sub_queries and not follow_up_queries:
                    follow_up_queries = fallback_sub_queries
            
            # If still no queries, use sub-queries from initial analysis
            if not follow_up_queries and not reformulated_query and sub_queries:
                follow_up_queries = sub_queries
            
            # Determine which queries to run next
            if reformulated_query:
                # Rewrite query with context keywords for better results
                context_rewrite = await self.analyze_and_rewrite_query(
                    reformulated_query, 
                    result_keywords
                )
                current_query = context_rewrite.get("rewritten_query", reformulated_query)
                strategy = new_strategy if new_strategy else optimal_strategy
                logging.info(f"[SearchOrchestrationAgent] Using reformulated query: {current_query}")
                
                # Execute search with reformulated query
                iteration_results = await self.execute_optimized_search(
                    user_id=user_id,
                    query_text=current_query,
                    search_strategy=strategy,
                    enhancement_params=enhancement_params,
                    dense_vector=None,  # Generate new embeddings
                    sparse_vector=None,
                    top_k=top_k
                )
                all_search_results.append(iteration_results)
                
                # Re-verify with new results
                verification = await self.verify_search_results(current_query, iteration_results)
                
            elif follow_up_queries:
                # Execute searches for follow-up queries
                for query in follow_up_queries[:min(2, len(follow_up_queries))]:
                    logging.info(f"[SearchOrchestrationAgent] Using follow-up query: {query}")
                    
                    # Rewrite follow-up query with context for better results
                    context_rewrite = await self.analyze_and_rewrite_query(
                        query, 
                        result_keywords
                    )
                    follow_up_rewritten = context_rewrite.get("rewritten_query", query)
                    follow_up_strategy = context_rewrite.get("search_strategy", optimal_strategy)
                    
                    # Execute search with follow-up query
                    iteration_results = await self.execute_optimized_search(
                        user_id=user_id,
                        query_text=follow_up_rewritten,
                        search_strategy=follow_up_strategy,
                        enhancement_params=enhancement_params,
                        dense_vector=None,  # Generate new embeddings
                        sparse_vector=None,
                        top_k=top_k
                    )
                    all_search_results.append(iteration_results)
                
                # Verify with the latest results
                if all_search_results:
                    verification = await self.verify_search_results(
                        follow_up_queries[-1],
                        all_search_results[-1]
                    )
            else:
                logging.info("[SearchOrchestrationAgent] No additional queries to run")
                break
        
        # 6. RANKING AND MERGING - Merge and rank all results
        logging.info("[SearchOrchestrationAgent] Ranking and merging results")
        merged_results = []
        
        for results in all_search_results:
            ranked_batch = await self.rank_merged_results(
                results.get("graph_results"), 
                results.get("vector_results")
            )
            merged_results.extend(ranked_batch)
        
        # Deduplicate and rerank
        final_ranked_results = self._deduplicate_and_rerank(merged_results)
        
        # 7. COMPILE FINAL RESULTS - Comprehensive answer from all results
        logging.info("[SearchOrchestrationAgent] Compiling final results")
        final_results = await self.compile_final_results_with_ranking(query_text, final_ranked_results)
        
        return final_results

    def create_search_workflow(self, state_schema: Type[BaseModel] = SearchOrchestrationState) -> OmniGraph:
        """
        Creates an OmniGraph workflow for search orchestration.
        
        Args:
            state_schema (Type[BaseModel]): Pydantic schema for graph state
            
        Returns:
            OmniGraph: Configured graph for search workflow
        """
        graph = OmniGraph(state_schema=state_schema, graph_name="SearchOrchestrationWorkflow")
        
        # Define nodes for the graph (steps in the workflow)
        async def analyze_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Analyzes and rewrites the query."""
            query_text = state["query_text"]
            analysis = await self.analyze_and_rewrite_query(query_text)
            
            state["query_analysis"] = analysis
            state["current_query"] = analysis.get("rewritten_query", query_text)
            state["search_strategy"] = analysis.get("search_strategy", "both")
            state["sub_queries"] = analysis.get("sub_queries", [])
            
            return state
        
        async def preliminary_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Executes a preliminary search for strategy assessment."""
            user_id = state["user_id"]
            current_query = state["current_query"]
            dense_vector = state.get("dense_vector")
            sparse_vector = state.get("sparse_vector")
            
            preliminary_results = await self.execute_preliminary_search(
                user_id=user_id,
                query_text=current_query,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector
            )
            
            # Assess search strength
            strength_assessment = await self.assess_search_strength(
                preliminary_results, 
                state["query_analysis"]
            )
            
            # Extract context for enhancement
            enhancement_context = await self.extract_enhancement_context(preliminary_results)
            
            # Determine optimal strategy
            optimal_strategy, enhancement_params = await self.determine_optimal_strategy(
                strength_assessment=strength_assessment,
                query_analysis=state["query_analysis"]
            )
            
            # Update enhancement params with context
            enhancement_params.update(enhancement_context)
            
            # Update state
            state["search_strategy"] = optimal_strategy
            state["enhancement_params"] = enhancement_params
            
            return state
        
        async def optimized_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Executes optimized search with enhanced parameters."""
            user_id = state["user_id"]
            current_query = state["current_query"]
            search_strategy = state["search_strategy"]
            enhancement_params = state.get("enhancement_params", {})
            dense_vector = state.get("dense_vector")
            sparse_vector = state.get("sparse_vector")
            top_k = state.get("top_k", 10)
            
            results = await self.execute_optimized_search(
                user_id=user_id,
                query_text=current_query,
                search_strategy=search_strategy,
                enhancement_params=enhancement_params,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=top_k
            )
            
            state["current_results"] = results
            state["all_search_results"].append(results)
            
            return state
        
        async def verify_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Verifies search results and decides if more iterations are needed."""
            query_text = state["query_text"]
            current_query = state["current_query"]
            current_results = state["current_results"]
            
            verification = await self.verify_search_results(current_query, current_results)
            
            state["verification"] = verification
            state["is_adequate"] = verification.get("is_adequate", True)
            
            # Extract keywords from results for query refinement
            result_keywords = []
            if not verification.get("is_adequate", True):
                if current_results.get("vector_results", {}).get("results"):
                    vector_docs = current_results["vector_results"]["results"][:3]
                    vector_texts = [doc.get("metadata", {}).get("text", "") for doc in vector_docs]
                    if vector_texts:
                        result_keywords = await self._extract_key_concepts(vector_texts)
            
            state["result_keywords"] = result_keywords
            
            # Prepare for potential next iteration
            reformulated_query = verification.get("reformulated_query")
            follow_up_queries = verification.get("follow_up_queries", [])
            
            # If no follow-up queries or reformulation, try to extract them
            if not follow_up_queries and not reformulated_query:
                follow_up_queries = await self.extract_follow_up_queries(current_results, current_query)
            
            # If still no queries, try fallback strategy
            if not follow_up_queries and not reformulated_query:
                fallback = await self.generate_fallback_strategy(current_query, current_results)
                broader_query = fallback.get("broader_query")
                if broader_query and broader_query != current_query:
                    reformulated_query = broader_query
                
                fallback_sub_queries = fallback.get("sub_queries", [])
                if fallback_sub_queries and not follow_up_queries:
                    follow_up_queries = fallback_sub_queries
            
            # If still no queries, use sub-queries from initial analysis
            if not follow_up_queries and not reformulated_query and state["sub_queries"]:
                follow_up_queries = state["sub_queries"]
            
            if reformulated_query:
                state["queries_to_run"] = [reformulated_query]
                if verification.get("alternative_strategy"):
                    state["search_strategy"] = verification["alternative_strategy"]
            elif follow_up_queries:
                state["queries_to_run"] = follow_up_queries[:min(2, len(follow_up_queries))]
            else:
                state["queries_to_run"] = []
            
            return state
        
        async def iteration_router(state: Dict[str, Any]) -> str:
            """Routes to either additional searches or final compilation."""
            is_adequate = state["is_adequate"]
            iteration = state["iteration"]
            queries_to_run = state["queries_to_run"]
            
            if is_adequate or iteration >= self.max_search_iterations or not queries_to_run:
                return "rank_and_compile"
            else:
                return "follow_up_search"
        
        async def follow_up_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Executes searches for follow-up queries."""
            user_id = state["user_id"]
            search_strategy = state["search_strategy"]
            enhancement_params = state.get("enhancement_params", {})
            top_k = state.get("top_k", 10)
            queries_to_run = state["queries_to_run"]
            result_keywords = state.get("result_keywords", [])
            
            for query in queries_to_run:
                # Rewrite with context for better results
                context_rewrite = await self.analyze_and_rewrite_query(query, result_keywords)
                rewritten_query = context_rewrite.get("rewritten_query", query)
                
                state["current_query"] = rewritten_query
                
                results = await self.execute_optimized_search(
                    user_id=user_id,
                    query_text=rewritten_query,
                    search_strategy=search_strategy,
                    enhancement_params=enhancement_params,
                    dense_vector=None,
                    sparse_vector=None,
                    top_k=top_k
                )
                state["current_results"] = results
                state["all_search_results"].append(results)
            
            # Increment iteration counter
            state["iteration"] += 1
            
            return state
        
        async def rank_and_compile_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Ranks, merges, and compiles final results."""
            query_text = state["query_text"]
            all_search_results = state["all_search_results"]
            
            # Merge and rank all results
            merged_results = []
            for results in all_search_results:
                ranked_batch = await self.rank_merged_results(
                    results.get("graph_results"),
                    results.get("vector_results")
                )
                merged_results.extend(ranked_batch)
            
            # Deduplicate and rerank
            final_ranked_results = self._deduplicate_and_rerank(merged_results)
            
            # Compile final results
            final_results = await self.compile_final_results_with_ranking(
                query_text, 
                final_ranked_results
            )
            
            state["final_results"] = final_results
            
            return state
        
        # Add nodes to the graph
        graph.add_node("analyze_query", analyze_node)
        graph.add_node("preliminary_search", preliminary_search_node)
        graph.add_node("optimized_search", optimized_search_node)
        graph.add_node("verify_results", verify_node)
        graph.add_node("follow_up_search", follow_up_search_node)
        graph.add_node("rank_and_compile", rank_and_compile_node)
        
        # Define edges between nodes
        graph.add_edge("analyze_query", "preliminary_search")
        graph.add_edge("preliminary_search", "optimized_search")
        graph.add_edge("optimized_search", "verify_results")
        
        # Add conditional routing
        graph.builder.add_conditional_edges(
            "verify_results",
            iteration_router,
            {
                "follow_up_search": "follow_up_search",
                "rank_and_compile": "rank_and_compile"
            }
        )
        
        graph.add_edge("follow_up_search", "verify_results")
        
        # Set entry and exit points
        graph.set_entry_point("analyze_query")
        graph.set_exit_point("rank_and_compile")
        
        # Compile the graph
        graph.compile()
        
        return graph
