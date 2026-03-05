"""
RagChatView - Graph-RAG Django API View
========================================

UPDATED FOR NEWEST LANGCHAIN VERSION
- MultiQueryRetriever import changed from langchain to langchain-classic
- Document import changed from langchain.schema to langchain_core.documents  
- Method changed from get_relevant_documents() to invoke()

This view implements a hybrid Graph-RAG pipeline combining:
1. Vector search for seed chunks (Pinecone)
2. Entity extraction from seed chunks
3. Graph expansion via Neo4j (Entity->MENTIONED_IN->Chunk)
4. Combined retrieval for enhanced context

INSTALLATION REQUIREMENTS:
- pip install langchain
- pip install langchain-classic
- pip install langchain-openai
- pip install langchain-pinecone
- pip install langchain-core
"""

from rest_framework.views import APIView
from rest_framework.response import Response
import os
import json
import re
import time
import traceback
from typing import List, Set, Dict
from rest_framework import status
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from neo4j import GraphDatabase
from openai import OpenAI
from algaebackend.models.Chat import ChatSession, ChatMessage
from algaebackend.models.ResponseMetrics import ResponseMetrics


class ResponseTimeTracker:
    """
    Utility class to track response times for different pipeline stages.
    
    Usage:
        tracker = ResponseTimeTracker()
        tracker.start('vector_search')
        # ... do vector search ...
        tracker.stop('vector_search')
        
        timings = tracker.get_timings()  # {'vector_search_ms': 1234, 'total_ms': 5678}
    """
    
    def __init__(self):
        self._start_times: Dict[str, float] = {}
        self._timings: Dict[str, int] = {}
        self._total_start = time.time()
    
    def start(self, stage: str):
        """Start timing a stage"""
        self._start_times[stage] = time.time()
    
    def stop(self, stage: str):
        """Stop timing a stage and record the duration in milliseconds"""
        if stage in self._start_times:
            elapsed_ms = int((time.time() - self._start_times[stage]) * 1000)
            self._timings[f"{stage}_ms"] = elapsed_ms
            del self._start_times[stage]
    
    def get_timings(self) -> Dict[str, int]:
        """Get all recorded timings including total time"""
        self._timings['total_ms'] = int((time.time() - self._total_start) * 1000)
        return self._timings.copy()


class EntityExtractor:
    """Extract entity names from text"""
    
    def __init__(self, openai_client: OpenAI = None, use_llm: bool = True):
        self.openai = openai_client
        self.use_llm = use_llm
    
    def extract_from_text(self, text: str, query: str = "") -> List[str]:
        """
        Extract entity names from text
        
        Returns:
            List of entity names (flat list, no categorization)
        """
        if self.use_llm and self.openai:
            return self._llm_extraction(text, query)
        else:
            return self._pattern_extraction(text)
    
    def _llm_extraction(self, text: str, query: str = "") -> List[str]:
        """Use LLM to extract entity names"""
        
        prompt = f"""Extract important entity names from this scientific text about algae.
Include: algae species, compounds, genes, methods, systems, etc.

TEXT: {text}

Return a simple JSON list of entity names:
{{"entities": ["Chlorella vulgaris", "omega-3", "photobioreactor"]}}

Only include entities explicitly mentioned in the text."""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract entity names and return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result_text = result_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(result_text)
            
            return result.get('entities', [])
        
        except Exception as e:
            print(f"LLM extraction failed: {e}, using patterns")
            return self._pattern_extraction(text)
    
    def _pattern_extraction(self, text: str) -> List[str]:
        """Fallback: Extract using patterns"""
        
        entities = set()
        text_lower = text.lower()
        
        # Algae species (scientific names)
        algae_genera = ['Chlorella', 'Spirulina', 'Dunaliella', 'Haematococcus',
                       'Nannochloropsis', 'Arthrospira', 'Scenedesmus', 'Chondrus']
        species_pattern = r'\b([A-Z][a-z]+)\s+([a-z]+)\b'
        for match in re.finditer(species_pattern, text):
            genus, species = match.group(1), match.group(2)
            if genus in algae_genera and len(species) > 3 and \
               species not in ['and', 'or', 'the', 'also', 'from']:
                entities.add(f"{genus} {species}")
        
        # Compounds
        compounds = ['omega-3', 'omega-6', 'EPA', 'DHA', 'beta-carotene', 'astaxanthin',
                    'phycocyanin', 'chlorophyll', 'carotenoid', 'lipid', 'protein']
        for compound in compounds:
            if compound.lower() in text_lower:
                entities.add(compound)
        
        # Systems
        systems = ['photobioreactor', 'open pond', 'raceway', 'fermenter']
        for system in systems:
            if system in text_lower:
                entities.add(system)
        
        # Methods
        methods = ['HPLC', 'chromatography', 'mass spectrometry', 'PCR']
        for method in methods:
            if method in text:
                entities.add(method)
        
        return list(entities)


class GraphExpander:
    """
    Expand from entities to chunks via Neo4j
    
    Schema: Entity --[MENTIONED_IN]--> Chunk
    """
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def find_chunks_via_entities(self, entity_names: List[str], limit: int = 10) -> List[Document]:
        """
        Find chunks via entities in Neo4j graph
        
        Args:
            entity_names: List of entity names to search for
            limit: Max chunks to return
        
        Returns:
            List of Documents containing Chunk text
        """
        
        if not entity_names:
            return []
        
        print(f"Searching for {len(entity_names)} entities in graph...")
        
        with self.driver.session() as session:
            query = """
            // Step 1: Find Entity nodes matching any of the entity names
            MATCH (e:Entity)
            WHERE ANY(search_name IN $entity_names WHERE 
                toLower(toString(e.name)) CONTAINS toLower(search_name)
                OR toLower(search_name) CONTAINS toLower(toString(e.name))
            )
            
            // Step 2: Follow MENTIONED_IN relationship to Chunks
            MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
            
            // Step 3: Return Chunk data
            RETURN DISTINCT
                c.text as chunk_text,
                c.chunk_id as chunk_id,
                c.document_id as document_id,
                c.page_number as page_number,
                collect(DISTINCT e.name)[0..5] as matched_entities
            
            LIMIT $limit
            """
            
            result = session.run(
                query,
                entity_names=entity_names,
                limit=limit
            )
            
            documents = []
            for record in result:
                chunk_text = record.get('chunk_text', '')
                
                if chunk_text:
                    doc = Document(
                        page_content=chunk_text,
                        metadata={
                            'chunk_id': record.get('chunk_id', ''),
                            'document_id': record.get('document_id', ''),
                            'page_number': record.get('page_number', 0),
                            'retrieval_method': 'graph_expansion',
                            'source': 'neo4j',
                            'via_entities': ', '.join(record.get('matched_entities', [])[:3])
                        }
                    )
                    documents.append(doc)
            
            print(f"Found {len(documents)} chunks via graph expansion")
            return documents


class RagChatAPIView(APIView):
    # Configuration for chat history sliding window
    CHAT_HISTORY_WINDOW_SIZE = 3  # Number of previous Q&A pairs to include

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize graph expander as None, will be created per request
        self.graph_expander = None

    def _get_or_create_session(self, request):
        """
        Get existing session or create new one
        Returns (session, is_new)
        """
        session_id = request.data.get('session_id')
        user = request.user
        
        if session_id:
            try:
                session = ChatSession.objects.get(id=session_id, user=user)
                return session, False
            except ChatSession.DoesNotExist:
                print(f"Session {session_id} not found, creating new one")
        
        # Create new session
        session = ChatSession.objects.create(
            user=user,
            title=f"Chat {ChatSession.objects.filter(user=user).count() + 1}"
        )
        return session, True

    def _get_chat_history(self, session, window_size=3):
        """
        Retrieve the last N Q&A pairs (sliding window)
        
        Args:
            session: ChatSession object
            window_size: Number of Q&A pairs to retrieve
        
        Returns:
            List of ChatMessage objects in chronological order
        """
        # Get last (window_size * 2) messages (each Q&A pair = 2 messages)
        messages = ChatMessage.objects.filter(
            session=session
        ).order_by('-created_at')[:(window_size * 2)]
        
        # Reverse to get chronological order
        return list(reversed(messages))

    def _format_chat_history(self, messages):
        """
        Format chat messages into a string for the LLM prompt
        
        Args:
            messages: List of ChatMessage objects
        
        Returns:
            Formatted string of conversation history
        """
        if not messages:
            return "No previous conversation."
        
        formatted = []
        for msg in messages:
            role = "User" if msg.message_type == "user" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted)

    def _get_all_chat_history_for_response(self, session):
        """
        Get ALL messages in the session for returning to frontend
        
        Returns:
            List of message dictionaries
        """
        messages = ChatMessage.objects.filter(session=session).order_by('created_at')
        
        return [{
            "id": msg.id,
            "message_type": msg.message_type,
            "content": msg.content,
            "created_at": msg.created_at.isoformat(),
            "metadata": msg.metadata
        } for msg in messages]

    def _save_messages(self, session, question, answer, rag_metadata=None):
        """
        Save user question and assistant answer to database
        
        Args:
            session: ChatSession object
            question: User's question text
            answer: Assistant's answer text
            rag_metadata: Optional dict with RAG info (entities, docs, etc.)
        
        Returns:
            The assistant ChatMessage instance (for linking ResponseMetrics)
        """
        # Save user message
        ChatMessage.objects.create(
            session=session,
            message_type='user',
            content=question
        )
        
        # Save assistant message with metadata
        assistant_message = ChatMessage.objects.create(
            session=session,
            message_type='assistant',
            content=answer,
            metadata=rag_metadata
        )
        
        # Update session timestamp
        session.save(update_fields=['updated_at'])
        
        return assistant_message

    def post(self, request, *args, **kwargs):
        # Initialize response time tracker
        tracker = ResponseTimeTracker()
        
        try:
            query = request.data.get('question')
            
            # Step 0: Get or create chat session
            tracker.start('session_handling')
            session, is_new_session = self._get_or_create_session(request)
            print(f"Using session {session.id} (new: {is_new_session})")
            
            # Retrieve chat history (sliding window)
            chat_history_messages = self._get_chat_history(session, self.CHAT_HISTORY_WINDOW_SIZE)
            formatted_history = self._format_chat_history(chat_history_messages)
            print(f"Retrieved {len(chat_history_messages)} messages from history")
            tracker.stop('session_handling')
            
            load_dotenv()
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            index_name = os.getenv("PINECONE_INDEX_NAME")
            NEO4J_URI = os.getenv("NEO4J_URI")
            NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
            NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
            
            # Configuration for GraphRAG
            TOP_K_VECTOR = 5  # Number of seed chunks from vector search
            TOP_K_GRAPH = 15  # Number of expanded chunks from graph
            USE_LLM_EXTRACTION = True  # Use LLM for entity extraction (set False for regex only)
            
            # Initialize OpenAI client for entity extraction
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Initialize LLM and Vectorstore
            model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="chatgpt-4o")
            parser = StrOutputParser()
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)
            
            # Step 1: MultiQuery Retriever for vector search (seed chunks)
            tracker.start('vector_search')
            llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=vectorstore.as_retriever(search_kwargs={'k': TOP_K_VECTOR}),
                llm=llm
            )
            
            print(f"Step 1: Vector search for seed chunks...")
            seed_docs = retriever_from_llm.invoke(query)[:TOP_K_VECTOR]
            
            # Add metadata to seed documents
            for doc in seed_docs:
                doc.metadata['retrieval_method'] = 'vector'
                doc.metadata['source'] = 'pinecone'
            
            print(f"Got {len(seed_docs)} seed chunks from vector search")
            tracker.stop('vector_search')
            
            # Step 2: Extract entity names from seed chunks
            tracker.start('entity_extraction')
            print(f"Step 2: Extracting entity names from seed chunks...")
            entity_extractor = EntityExtractor(openai_client=openai_client, use_llm=USE_LLM_EXTRACTION)
            
            all_entity_names = set()
            for i, doc in enumerate(seed_docs, 1):
                print(f"Processing seed chunk {i}/{len(seed_docs)}...")
                entity_names = entity_extractor.extract_from_text(doc.page_content, query)
                all_entity_names.update(entity_names)
            
            entity_names_list = list(all_entity_names)
            print(f"Extracted {len(entity_names_list)} unique entity names")
            tracker.stop('entity_extraction')
            
            # Step 3: Graph expansion - find more chunks via entities
            tracker.start('graph_expansion')
            expanded_docs = []
            if entity_names_list and NEO4J_URI and NEO4J_PASSWORD:
                print(f"Step 3: Graph expansion via Neo4j...")
                try:
                    self.graph_expander = GraphExpander(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
                    expanded_docs = self.graph_expander.find_chunks_via_entities(
                        entity_names=entity_names_list,
                        limit=TOP_K_GRAPH
                    )
                except Exception as neo4j_error:
                    print(f"Neo4j graph expansion failed: {neo4j_error}")
                    print("Continuing with vector search results only...")
                    expanded_docs = []
                finally:
                    if self.graph_expander:
                        try:
                            self.graph_expander.close()
                        except:
                            pass
            else:
                print("Step 3: Skipping graph expansion (no entities or Neo4j not configured)")
            tracker.stop('graph_expansion')
            
            # Step 4: Combine and deduplicate chunks
            print(f"Step 4: Combining results...")
            seen_ids = set()
            combined_docs = []
            
            # Add seed chunks first
            for doc in seed_docs:
                chunk_id = doc.metadata.get('chunk_id', doc.page_content[:50])
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    combined_docs.append(doc)
            
            # Add expanded chunks
            for doc in expanded_docs:
                chunk_id = doc.metadata.get('chunk_id', doc.page_content[:50])
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    combined_docs.append(doc)
            
            vector_count = len(seed_docs)
            graph_count = len(combined_docs) - vector_count
            
            print(f"Combined Results: {vector_count} vector + {graph_count} graph = {len(combined_docs)} total chunks")
            
            # Build context from combined documents
            context = self._format_docs(combined_docs)
            
            # Build prompt and chain
            prompt = self.buildPrompt()
            prompt_text = prompt.format(chat_history=formatted_history, context=context, question=query)
            
            # Build question chain with combined documents and chat history
            chain = (
                {
                    "chat_history": lambda x: formatted_history,
                    "context": lambda x: self._format_docs(combined_docs),
                    "question": RunnablePassthrough()
                }
                | prompt
                | model
                | parser
            )
            
            # Invoke answer
            tracker.start('llm_generation')
            answer = chain.invoke(query)
            tracker.stop('llm_generation')
            
            # Get final timings
            response_times = tracker.get_timings()
            print(f"Response times: {response_times}")
            
            # Save messages to database
            rag_metadata = {
                "entities_extracted": entity_names_list[:20],
                "vector_count": vector_count,
                "graph_count": graph_count,
                "total_chunks": len(combined_docs)
            }
            assistant_message = self._save_messages(session, query, answer, rag_metadata)
            
            # Save response metrics
            ResponseMetrics.create_from_timings(assistant_message, response_times)
            
            # Get complete chat history for response
            full_chat_history = self._get_all_chat_history_for_response(session)
        
            # Build response to be handled in the frontend
            data = {
                "session_id": session.id,
                "is_new_session": is_new_session,
                "question": query,
                "answer": answer,
                "chat_history": full_chat_history,  # Complete conversation history
                "chat_history_window_size": len(chat_history_messages),  # How many used in context
                "docs": context,
                "docs_expanded": {
                    "vector_chunks": [self._doc_to_dict(doc) for doc in seed_docs],
                    "graph_chunks": [self._doc_to_dict(doc) for doc in expanded_docs],
                    "total_chunks": len(combined_docs),
                    "vector_count": vector_count,
                    "graph_count": graph_count
                },
                "entities_extracted": entity_names_list[:20],  # Limit to first 20 for response
                "prompt": prompt_text,
                "response_times": response_times,  # Timing metrics for each pipeline stage
            }

            # Return with data and 200 response
            return Response(data, status=status.HTTP_200_OK)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response(
                {"error": "An error occurred while processing your request", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for context"""
        if not docs:
            return "No relevant context found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            method = doc.metadata.get('retrieval_method', 'unknown')
            source = doc.metadata.get('source', 'unknown')
            
            # Show which entities led to this chunk (for graph docs)
            via_entities = doc.metadata.get('via_entities', '')
            if via_entities:
                formatted.append(f"[Chunk {i} - {method} via entities: {via_entities}]")
            else:
                formatted.append(f"[Chunk {i} - {method} from {source}]")
            
            formatted.append(doc.page_content)
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _doc_to_dict(self, doc: Document) -> dict:
        """Convert Document to dictionary for JSON serialization"""
        return {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
    
    def buildPrompt(self):
        template = """
        You are an expert algae research assistant with access to conversation history and research documents.

        Previous Conversation:
        {chat_history}

        Current Research Context:
        The context below includes chunks from both vector search and graph-based entity expansion, providing 
        comprehensive coverage of relevant information.

        {context}

        Current Question: {question}

        Instructions:
        - First, provide a direct and concise answer to the current question.
        - Use both the conversation history and the research context to inform your answer.
        - If the question references previous messages ("it", "that", "the previous"), use the conversation history.
        - If the answer cannot be found in either source, respond with "I don't know."
        - After your direct answer, elaborate with relevant details from the context.
        - If you could not find an answer, state that you could not find any references in the provided context. And then answer from your own knowledge. Tell the user that you are answering from your own knowledge.

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt