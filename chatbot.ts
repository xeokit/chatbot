import { ChatOllama } from '@langchain/ollama';
import { RetrievalQAChain } from 'langchain/chains';
import { OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Document } from 'langchain/document';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { GithubRepoLoader } from "@langchain/community/document_loaders/web/github";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from '@qdrant/js-client-rest';
import fs from 'fs/promises';
import path from 'path';
import * as cheerio from 'cheerio';

const COLLECTION_NAME = 'xeokit-docs';
const OLLAMA_MODEL = 'llama3.1';
const OLLAMA_BASEURL = 'http://localhost:11434';
const QDRANT_BASEURL = 'http://localhost:6333';

// Function to load documents from a directory
async function loadDocumentsFromDirectory(directoryPath: string): Promise<Document[]> {
  const documents: Document[] = [];
  const files = await fs.readdir(directoryPath, { recursive: true } );

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
                  content = $('.content').text();
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
    const client = new QdrantClient({ url: QDRANT_BASEURL, timeout: 3000 });

    const embeddings = new OllamaEmbeddings({
       model: OLLAMA_MODEL,
        baseUrl: OLLAMA_BASEURL,
        keepAlive: "15m"
    });

    console.log({client, embeddings});

    await QdrantVectorStore.fromDocuments(docs, embeddings, {
        client,
        collectionName: COLLECTION_NAME,
    });
}

async function loadVectorStore() {
    const client = new QdrantClient({ url: QDRANT_BASEURL });

    const embeddings = new OllamaEmbeddings({
       model: OLLAMA_MODEL,
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

    const chain = RetrievalQAChain.fromLLM(ollama, vectorStore.asRetriever());

    return chain;
}

async function askQuestion(chain: RetrievalQAChain, question: string) {
    const response = await chain.call({ query: question });
    return response.text;
}

async function main() {

    // Initialize the vector store with documents (do this once)
    await initializeChatbot('./docs', 'directory');

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
    const question = "What is the main feature of our product? Can you give me example in on how to use WebIFCLoaderPlugin ?";
    const answer = await askQuestion(chatbot, question);

    console.log('askQuestion end', performance.now() - start);

    console.log(`Q: ${question}`);
    console.log(`A: ${answer}`);
}

main().catch(console.error);