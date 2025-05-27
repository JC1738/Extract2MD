import * as pdfjsLib from 'pdfjs-dist/build/pdf.mjs';
import Tesseract from 'tesseract.js';
import WebLLMEngine from '../engines/WebLLMEngine.js';
import OutputParser from '../utils/OutputParser.js';
import SystemPrompts from '../utils/SystemPrompts.js';
import ConfigValidator from '../utils/ConfigValidator.js';

// Import tiktoken for accurate token-based splitting
let tokenizer = null;
try {
  const { getTokenizer } = await import('tiktoken');
  tokenizer = getTokenizer('gpt2');
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

    static async quickConvertOnly(pdfFile, config) {
        const converter = new Extract2MDConverter(config);
        return await converter._performQuickConvert(pdfFile);
    }

    static async highAccuracyConvertOnly(pdfFile, config) {
        const converter = new Extract2MDConverter(config);
        return await converter._performHighAccuracyConvert(pdfFile);
    }

    static async quickConvertWithLLM(pdfFile, config) {
        const converter = new Extract2MDConverter(config);
        return await converter._processWithLLM(pdfFile);
    }

    static async highAccuracyConvertWithLLM(pdfFile, config) {
        const converter = new Extract2MDConverter(config);
        return await converter._processWithLLM(pdfFile);
    }

    static async combinedConvertWithLLM(pdfFile, config) {
        const converter = new Extract2MDConverter(config);
        return await converter._processWithLLM(pdfFile);
    }

    _performQuickConvert(pdfFile) {
        // Implementation for quick conversion
    }

    _performHighAccuracyConvert(pdfFile) {
        // Implementation for OCR-based conversion
    }

    async _processWithLLM(pdfFile) {
        if (!this.webllmEngine) {
            this.webllmEngine = new WebLLMEngine(this.config.llm);
        }

        const systemPrompt = this.config.systemPrompts.combinedExtraction || '';
        const systemTokens = this._getTokenCount(systemPrompt);

        // Calculate safe chunk size with buffer
        const contextWindowSize = this.config.llm.options?.maxTokens || 4096;
        const maxChunkTokens = Math.max(1, contextWindowSize - systemTokens - 500); // Conservative buffer

        const text = await this._extractText(pdfFile);
        const chunks = this._splitTextIntoChunks(text, maxChunkTokens);

        let results = [];

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            try {
                const fullPrompt = `${systemPrompt}\n\n${chunk}`;
                if (this._getTokenCount(fullPrompt) > contextWindowSize) {
                    throw new Error(`Chunk ${i} exceeds token limit: ${this._getTokenCount(fullPrompt)} tokens`);
                }
                const result = await this.webllmEngine.generate(fullPrompt, this.config.llm.options);
                results.push(result);
            } catch (error) {
                console.error(`LLM chunk ${i} processing failed:`, error);
                throw new Error(`LLM chunk ${i} processing failed: ${error.message}`);
            }
        }

        return results.join('\n\n');
    }

    _extractText(pdfFile) {
        // Implementation for text extraction
    }

    _splitTextIntoChunks(text, maxTokens = 2000) {
        if (tokenizer) {
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
            // Fallback to word-based splitting
            const words = text.split(' ');
            const chunks = [];
            let currentChunk = [];

            for (const word of words) {
                currentChunk.push(word);
                if (currentChunk.length >= 512) { // Smaller chunk size as fallback
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

    _getTokenCount(text) {
        if (tokenizer) {
            return tokenizer.encode(text).length;
        } else {
            return text.split(' ').length;
        }
    }
}

export default Extract2MDConverter;
