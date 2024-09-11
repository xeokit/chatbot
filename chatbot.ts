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
import cheerio from 'cheerio';
//import pdf from 'pdf-parse';

const COLLECTION_NAME = 'docs';
const OLLAMA_MODEL = 'llama3:8b';
const OLLAMA_BASEURL = 'http://localhost:11434';

// Function to load documents from a directory
async function loadDocumentsFromDirectory(directoryPath: string): Promise<Document[]> {
  const documents: Document[] = [];
  const files = await fs.readdir(directoryPath);

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
                  const htmlContent = await fs.readFile(filePath, 'utf-8');
                  const $ = cheerio.load(htmlContent);
                  content = $('body').text();
                  break;
              // case '.pdf':
              //     const pdfBuffer = await fs.readFile(filePath);
              //     const pdfData = await pdf(pdfBuffer);
              //     content = pdfData.text;
              //     metadata = { ...metadata, ...pdfData.info };
              //     break;
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
  console.log('loadDocumentsFromUrl');
  const loader = new CheerioWebBaseLoader(url);
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
    const client = new QdrantClient({ url: 'http://localhost:6333' });

    const embeddings = new OllamaEmbeddings({
       model: OLLAMA_MODEL,
        baseUrl: OLLAMA_BASEURL,
    });

    console.log({client, embeddings});

    await QdrantVectorStore.fromDocuments(docs, embeddings, {
        client,
        collectionName: COLLECTION_NAME,
    });
}

async function loadVectorStore() {
    const client = new QdrantClient({ url: 'http://localhost:6333' });

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

    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const docs = await textSplitter.splitDocuments(rawDocs);

    console.log({docs});

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

// Usage example
async function main() {
  console.log('start chatbot');
    // Initialize the vector store with documents (do this once)
    // await initializeChatbot('./docs', 'directory');
    // or
    await initializeChatbot('https://xeokit.github.io/xeokit-sdk/docs/', 'url');
    // or
    // await initializeChatbot('https://github.com/xeokit/xeokit-sdk/', 'github');

    // Create chatbot (do this for each session or server start)
    const chatbot = await createChatbot();

    // Ask questions (do this for each user query)
    const question = "What is the main feature of our product?";
    const answer = await askQuestion(chatbot, question);
    console.log(`Q: ${question}`);
    console.log(`A: ${answer}`);
}

main().catch(console.error);