import * as pdfjsLib from 'pdfjs-dist/build/pdf.mjs';
import Tesseract from 'tesseract.js';
import WebLLMEngine from '../engines/WebLLMEngine.js';
import OutputParser from '../utils/OutputParser.js';
import SystemPrompts from '../utils/SystemPrompts.js';
import ConfigValidator from '../utils/ConfigValidator.js';

// Import tiktoken for accurate token-based splitting
let tokenizer = null;
try {
  // Try to import the browser version of tiktoken (WASM)
  const { getTokenizer } = await import('tiktoken');
  tokenizer = await getTokenizer('gpt2'); // Use GPT-2 tokenizer as a default
} catch (e) {
  console.warn('Failed to load tiktoken. Falling back to word-based splitting.');
}

export class Extract2MDConverter {
    constructor(config = {}) {
        // Validate and normalize configuration
        this.config = ConfigValidator.validate(config);
        
        // Initialize components
        this.webllmEngine = null;
        this.outputParser = new OutputParser();
        
        // Setup PDF.js worker
        if (this.config.pdfJsWorkerSrc) {
            import(this.config.pdfJsWorkerSrc).catch(err => console.error('Failed to load PDF.js worker:', err));
        }
    }

    static async quickConvertOnly(pdfFile, options = {}) {
        const converter = new Extract2MDConverter(options);
        return await converter._performQuickConvert(pdfFile);
    }

    static async highAccuracyConvertOnly(pdfFile, options = {}) {
        const converter = new Extract2MDConverter(options);
        return await converter._performHighAccuracyConvert(pdfFile);
    }

    static async quickConvertWithLLM(pdfFile, options = {}) {
        const converter = new Extract2MDConverter(options);
        return await converter._performQuickConvertWithLLM(pdfFile);
    }

    static async highAccuracyConvertWithLLM(pdfFile, options = {}) {
        const converter = new Extract2MDConverter(options);
        return await converter._performHighAccuracyConvertWithLLM(pdfFile);
    }

    static async combinedConvertWithLLM(pdfFile, options = {}) {
        const converter = new Extract2MDConverter(options);
        return await converter._performCombinedConvertWithLLM(pdfFile);
    }

    _performQuickConvert(pdfFile) {
        // Implementation for quick conversion
    }

    _performHighAccuracyConvert(pdfFile) {
        // Implementation for high accuracy OCR
    }

    async _performQuickConvertWithLLM(pdfFile) {
        const text = await this._extractTextFromPDF(pdfFile);
        return await this._processWithLLM(text);
    }

    async _performHighAccuracyConvertWithLLM(pdfFile) {
        const text = await this._extractOCRText(pdfFile);
        return await this._processWithLLM(text);
    }

    async _performCombinedConvertWithLLM(pdfFile) {
        const text1 = await this._extractTextFromPDF(pdfFile);
        const text2 = await this._extractOCRText(pdfFile);
        const combinedText = `${text1}\n\n${text2}`;
        return await this._processWithLLM(combinedText);
    }

    async _extractTextFromPDF(pdfFile) {
        // Implementation to extract text using PDF.js
        return "Extracted text from PDF...";
    }

    async _extractOCRText(pdfFile) {
        // Implementation for OCR extraction with Tesseract
        return "OCR extracted text...";
    }

    async _processWithLLM(text) {
        if (!this.webllmEngine) {
            this.webllmEngine = new WebLLMEngine(this.config.llm);
        }

        const systemPrompt = this.config.systemPrompts?.combinedExtraction || '';
        const systemTokens = this._getTokenCount(systemPrompt);

        // Calculate safe chunk size with buffer
        const contextWindowSize = this.config.llm.options?.maxTokens || 4096;
        const maxChunkTokens = Math.max(1, contextWindowSize - systemTokens - 500); // Add buffer

        const chunks = this._splitTextIntoChunks(text, maxChunkTokens);
        let results = [];

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            try {
                this.progressCallback({
                    stage: 'llm_chunk_start',
                    message: `Processing LLM chunk ${i + 1} of ${chunks.length}`,
                    currentPage: i + 1,
                    totalPages: chunks.length
                });

                // Ensure the final prompt (chunk + system prompts) is under context window size
                const fullPrompt = this._buildFullPrompt(chunk);
                if (this._getTokenCount(fullPrompt) > contextWindowSize) {
                    throw new Error(`Prompt too long: ${this._getTokenCount(fullPrompt)} tokens`);
                }

                const result = await this.webllmEngine.generate(fullPrompt, this.config.llm.options);
                results.push(result);

                this.progressCallback({
                    stage: 'llm_chunk_complete',
                    message: `LLM chunk ${i + 1} processed`,
                    currentPage: i + 1,
                    totalPages: chunks.length
                });
            } catch (error) {
                console.error(`LLM chunk ${i + 1} processing failed:`, error);
                throw new Error(`LLM chunk processing failed: ${error.message}`);
            }
        }

        return results.join('\n\n');
    }

    _buildFullPrompt(chunk) {
        const systemPrompt = this.config.systemPrompts?.combinedExtraction || '';
        return `${systemPrompt}\n\n${chunk}`;
    }

    _getTokenCount(text) {
        if (tokenizer) {
            return tokenizer.encode(text).length;
        } else {
            // Fallback: assume 1 token per word
            return text.split(' ').length;
        }
    }

    _splitTextIntoChunks(text, maxTokens = 2000) {
        if (tokenizer) {
            // Use tiktoken for accurate token-based splitting
            const tokens = tokenizer.encode(text);
            const chunks = [];
            let start = 0;

            while (start < tokens.length) {
                const end = Math.min(start + maxTokens, tokens.length);
                const chunkTokens = tokens.slice(start, end);
                const chunkText = tokenizer.decode(chunkTokens);
                chunks.push(chunkText);
                start = end;
            }

            return chunks;
        } else {
            // Fallback to word-based splitting with smaller chunk size
            const words = text.split(' ');
            let chunks = [];
            let currentChunk = [];

            for (const word of words) {
                currentChunk.push(word);
                if (currentChunk.length >= 512) { // Reduced from 1024 to 512
                    chunks.push(currentChunk.join(' '));
                    currentChunk = [];
                }
            }

            if (currentChunk.length > 0) {
                chunks.push(currentChunk.join(' '));
            }

            return chunks;
        }
    }
}
