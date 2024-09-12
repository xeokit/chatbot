import { ChatOllama } from '@langchain/ollama';
import { RetrievalQAChain } from 'langchain/chains';
import { OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Document } from 'langchain/document';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { GithubRepoLoader } from "@langchain/community/document_loaders/web/github";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from '@qdrant/js-client-rest';
import { Runnable, RunnableSequence } from "@langchain/core/runnables";
import { PromptTemplate } from "@langchain/core/prompts";

import fs from 'fs/promises';
import path from 'path';
import * as cheerio from 'cheerio';
import pLimit from 'p-limit';

const COLLECTION_NAME = 'xeokit-docs';
const OLLAMA_MODEL = 'llama3.1';
const OLLAMA_EMBEDDING_MODEL = 'nomic-embed-text';
const OLLAMA_BASEURL = 'http://localhost:11434';
const QDRANT_BASEURL = 'http://localhost:6333';

// Function to load documents from a directory
async function loadDocumentsFromDirectory(directoryPath: string): Promise<Document[]> {
    const documents: Document[] = [];
    const files = await fs.readdir(directoryPath, { recursive: true });
  
    for (const file of files) {
      const filePath = path.join(directoryPath, file);
      const ext = path.extname(file).toLowerCase();
  
      try {
        let content: string;
        let metadata: Record<string, any> = { source: file };
  
        switch (ext) {
          case '.txt':
          case '.md':
            content = await fs.readFile(filePath, 'utf-8');
            break;
          case '.html':
            console.log(`Parsing ${filePath}`);
            const htmlContent = await fs.readFile(filePath, 'utf-8');
            const $ = cheerio.load(htmlContent);
  
            // Extract relevant content from ESDoc-generated HTML
            const title = $('title').text();
            const description = $('meta[name="description"]').attr('content') || '';
            const mainContent = $('.content').text();
  
            // Combine extracted content
            content = `${title}\n\n${description}\n\n${mainContent}`;
  
            // Add more metadata
            metadata.title = title;
            metadata.description = description;
            break;
          default:
            console.warn(`Unsupported file type: ${file}`);
            continue;
        }
  
        documents.push(new Document({ pageContent: content, metadata }));
      } catch (error) {
        console.error(`Error processing file ${file}:`, error);
      }
    }
  
    return documents;
  }

// Function to load documents from a URL
async function loadDocumentsFromUrl(url: string): Promise<Document[]> {
  const loader = new CheerioWebBaseLoader(url);
  //const loader = new RecursiveUrlLoader(url, { maxDepth: 5 });
  return loader.load();
}

// Function to load documents from a GitHub repository
async function loadDocumentsFromGithub(repoUrl: string): Promise<Document[]> {
  const loader = new GithubRepoLoader(repoUrl, {
      branch: "master",
      recursive: true,
      unknown: "warn",
  });
  return loader.load();
}

async function initializeVectorStore(docs: Document[]) {
    const client = new QdrantClient({ url: QDRANT_BASEURL, timeout: 60000 }); // Increased timeout

    const embeddings = new OllamaEmbeddings({
        model: OLLAMA_EMBEDDING_MODEL,
        baseUrl: OLLAMA_BASEURL,
        keepAlive: "15m"
    });

    console.log({client, embeddings});

    // Implement batching
    const batchSize = 50; // Adjust this value based on your system's capabilities
    const limit = pLimit(5); // Limit concurrent requests, adjust as needed

    for (let i = 0; i < docs.length; i += batchSize) {
        const batch = docs.slice(i, i + batchSize);
        await Promise.all(
            batch.map(doc => limit(() => retryWithBackoff(() =>
                QdrantVectorStore.fromDocuments([doc], embeddings, {
                    client,
                    collectionName: COLLECTION_NAME,
                })
            )))
        );
        console.log(`Processed ${i + batch.length} out of ${docs.length} documents`);
    }
}

// Helper function to retry with exponential backoff
async function retryWithBackoff(fn: () => Promise<any>, maxRetries = 3, initialDelay = 1000) {
    let retries = 0;
    while (retries < maxRetries) {
        try {
            return await fn();
        } catch (error) {
            retries++;
            if (retries === maxRetries) throw error;
            const delay = initialDelay * Math.pow(2, retries);
            console.log(`Retry ${retries} after ${delay}ms`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
}

async function loadVectorStore() {
    const client = new QdrantClient({ url: QDRANT_BASEURL });

    const embeddings = new OllamaEmbeddings({
       model: OLLAMA_EMBEDDING_MODEL,
        baseUrl: OLLAMA_BASEURL,
    });

    return await QdrantVectorStore.fromExistingCollection(embeddings, {
        client,
        collectionName: COLLECTION_NAME,
    });
}

async function initializeChatbot(source: string, sourceType: 'directory' | 'url' | 'github') {
    let rawDocs: Document[];

    switch (sourceType) {
        case 'directory':
            rawDocs = await loadDocumentsFromDirectory(source);
            break;
        case 'url':
            rawDocs = await loadDocumentsFromUrl(source);
            break;
        case 'github':
            rawDocs = await loadDocumentsFromGithub(source);
            break;
        default:
            throw new Error('Invalid source type');
    }

    console.log({rawDocs: rawDocs.length});

    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const docs = await textSplitter.splitDocuments(rawDocs);

    await initializeVectorStore(docs);

    console.log("Vector store initialized with documents.");
}

async function createChatbot() {
    const vectorStore = await loadVectorStore();

    const ollama = new ChatOllama({
      baseUrl: OLLAMA_BASEURL,
      model: OLLAMA_MODEL
    });

    const retriever = vectorStore.asRetriever();
    const prompt = PromptTemplate.fromTemplate(
        `Answer the following question based on the context:
        Context: {context}
        Question: {question}
        Answer:`
    );

    const chain = RunnableSequence.from([
        {
            context: async (input: { question: string }) => {
                const docs = await retriever.getRelevantDocuments(input.question);
                return docs.map(doc => doc.pageContent).join('\n');
            },
            question: (input: { question: string }) => input.question,
        },
        prompt,
        ollama,
    ]);

    return chain;
}

async function askQuestion(chain: Runnable, question: string) {
    const response = await chain.invoke({ question });
    return response;
}

async function main() {

    // Initialize the vector store with documents (do this once)
    // Generate embeddings from docs or upload collection snapsot from snapshots directory to your local qdrant database at http://localhost:6333/
    // await initializeChatbot('./docs', 'directory');

    // or
    //await initializeChatbot('https://xeokit.github.io/xeokit-sdk/docs/identifiers.html', 'url');
    //console.log('initializeChatbot end');

    // or
    // await initializeChatbot('https://github.com/xeokit/xeokit-sdk/', 'github');

    // Create chatbot (do this for each session or server start)
    const chatbot = await createChatbot();

    const start = performance.now();
    console.log('askQuestion start', start);

    // Ask questions (do this for each user query)
    const question = "What is the main feature of xeokit ? What is xeokit advantage of alternatives ? How do i load LAS file into Viewer ? Please give me running javascript code example.";
    const answer = await askQuestion(chatbot, question);

    console.log('askQuestion end', performance.now() - start);

    console.log(`Q: ${question}`);
    console.log(`A: ${answer.content}`);
}

main().catch(console.error);