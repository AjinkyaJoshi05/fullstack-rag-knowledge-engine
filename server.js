import express from 'express';
import cors from 'cors';
import pg from 'pg';
import multer from 'multer';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const pdfParse = require('pdf-parse');
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Configure Multer to intercept file uploads and keep them in volatile memory as raw binary buffers
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Initialize Database Connection Pool to Supabase PostgreSQL
const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
});

// Initialize our core Gemini models
const embeddings = new GoogleGenerativeAIEmbeddings({ model: "gemini-embedding-001" });
const llm = new ChatGoogleGenerativeAI({ model: "gemini-2.5-flash", temperature: 0 });

// Construct the LangChain Supabase integration manager
const vectorStore = new SupabaseVectorStore(embeddings, {
  client: pool,
  tableName: "documents",
  queryName: "match_documents",
});

/**
 * ENDPOINT 1: DYNAMIC MULTIPART PDF UPLOAD AND SEEDING
 * Accepts an actual file upload, extracts text chunks, embeds them, and persists them to Supabase pgvector.
 */
app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: "No file uploaded. Please send a PDF using form-data key 'file'" });
    }

    console.log(`Received binary PDF upload: ${req.file.originalname}`);

    // 1. Parse raw text content directly out of the binary memory buffer
    const pdfData = await pdfParse(req.file.buffer);
    const rawText = pdfData.text;

    if (!rawText || rawText.trim().length === 0) {
      return res.status(400).json({ success: false, error: "Could not extract readable text strings from this PDF." });
    }

    console.log(`Successfully parsed ${pdfData.numpages} pages. Slicing into text chunks...`);

    // 2. Chunk the text logically using optimized hyperparameters
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 600, chunkOverlap: 120 });
    
    // Create base Document objects using LangChain standards, injecting file metadata
    const docs = await splitter.createDocuments(
      [rawText], 
      [{ source_file: req.file.originalname, total_pages: pdfData.numpages }]
    );

    console.log(`Generated ${docs.length} semantic vectors. Bulk inserting into Supabase via pgvector...`);

    // 3. Perform automated mass-vectorization and SQL database population
    await SupabaseVectorStore.fromDocuments(docs, embeddings, {
      client: pool,
      tableName: "documents",
    });

    return res.status(200).json({
      success: true,
      message: "PDF Ingestion Engine successfully synchronized data with Supabase.",
      filename: req.file.originalname,
      chunks_created: docs.length,
      pages_parsed: pdfData.numpages
    });

  } catch (error) {
    console.error("Ingestion Pipeline Failure:", error);
    return res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * ENDPOINT 2: CHAT & CONTEXTUAL RETRIEVAL
 * Takes a user query, searches Supabase via vector math, injects context, and queries Gemini.
 */
app.post('/api/chat', async (req, res) => {
  const { question } = req.body;
  if (!question) return res.status(400).json({ error: "Question parameter is required" });

  try {
    const retriever = vectorStore.asRetriever({ searchType: "similarity", k: 3 });
    
    const prompt = ChatPromptTemplate.fromTemplate(`
      You are a professional full-stack knowledge assistant. Answer the user query using ONLY the technical data provided inside the text context blocks below. 
      If the solution cannot be derived from the context data explicitly, state clearly that the document information is insufficient to formulate an answer. Do not hallucinate.

      Context Blocks:
      {context}

      User Question: {question}

      Formatted Technical Answer:
    `);

    const formatDocs = (docs) => docs.map(doc => doc.pageContent).join("\n\n");

    const ragChain = RunnableSequence.from([
      {
        context: retriever.pipe(formatDocs),
        question: new RunnablePassthrough(),
      },
      prompt,
      llm,
      new StringOutputParser(),
    ]);

    // Fetch documents independently to extract text citations and pass them to our client frontend interface
    const sourceDocuments = await retriever.invoke(question);
    const aiAnswer = await ragChain.invoke(question);

    return res.status(200).json({
      answer: aiAnswer,
      citations: sourceDocuments.map(doc => ({
        snippet: doc.pageContent,
        source: doc.metadata.source_file || "Unknown Document"
      }))
    });

  } catch (error) {
    console.error("Retrieval Engine Failure:", error);
    return res.status(500).json({ error: error.message });
  }
});

const PORT = 5000;
app.listen(PORT, () => console.log(`🚀 AI RAG Production API Layer listening on http://localhost:${PORT}`));